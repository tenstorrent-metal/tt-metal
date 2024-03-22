// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/scan/scan_op.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt::tt_metal {

void Scan::validate(const std::vector<Tensor> &input_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);

    const Shape &input_shape = input_tensor.get_legacy_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Scan: Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Scan: Expect input tensor to be allocated on a device buffer.");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Scan: Expect input tensor in tile layout.");
    TT_FATAL(input_tensor.is_sharded(), "Scan: Expect input tensor to be sharded.");

    auto core_grid = input_tensor.shard_spec()->grid;
    TT_FATAL(core_grid.ranges().size() == 1, "Scan: Expect input tensor to be sharded along a single core range.");
}

operation::ProgramWithCallbacks scan_impl(const Tensor &input, Tensor &output, ScanOpDirection direction) {
    /*
    The scan op performs the in-place inclusive prefix scan along the specified dimension of the input 2D tensor.
    Currently we only support the bottom-to-top scan direction (COLS_REVERSED). Moreover, we only support
    multiplication as the scan aggregation operation.

    Each core stores a subset of a tensor we call a shard. Within shards, the tensor is further subdivided into tiles.
    Each tile is 32 x 32 elements. Our hardware processes the tiles sequentially. The tiles are stored contiguously
    in a circular buffer. There's currently no low-level kernel that can perform the scan operation on a single tile.

    To circumvent this limitation, we need to repackage the tile data, so that each tiles stores elements from a single
    row. This way we can use element-wise multiplication to compute the scan operation along the columns. Here's how we
    can do this:

    - Prepare 3 auxiliary buffers: 8-tiles buffer (A), 32-tiles buffer (C), and W / 1024 tiles buffer (C), where W is
    the width of the input tensor.
    - C has to be initialized with 1's.
    - For each row of tiles in the input tensor, we will untilize a block of 8 tiles into the A buffer, then copy A into
    the B with correct offset.
        - We will do this 4 times to fill B.
        - Each row of B now contains 1024 elements of the input tensor. We can treat them as one "tile" in the math
    kernel and run element-wise mul on them. We do this to perform a sequential scan on the rows of B. Each tile also
    has to be multiplied by a corresponding tile in C.
        - We copy aside the last "tile" of B into C.
        - Then we run reverse tilize operation to copy from B to A to input tensor.
        - Continue this until the row has been exhausted then move to the next row.

    This operation runs in parallel across all cores of the device. Since the columns of the input tensor are split
    across the core columns, we need to communicate the total product of the columns from the previous core to the next
    core in a core-column.
    */

    Program program = Program();
    tt_metal::Device *device = output.device();
    Buffer *src_buffer = input.buffer();
    auto all_cores = output.shard_spec()->grid;

    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t tile_size = tt_metal::detail::TileSize(data_format);
    uint32_t total_size = output.shard_spec()->numel() / TILE_HW * tile_size;
    uint32_t in_tiles_per_row = input.shard_spec()->shape[1] / TILE_WIDTH;
    uint32_t in_tiles_per_col = input.shard_spec()->shape[0] / TILE_HEIGHT;
    uint32_t out_tiles_per_row = output.shard_spec()->shape[1] / TILE_HW;

    log_info(tt::LogOp, "total_size: {}", total_size);
    log_info(
        tt::LogOp,
        "in_tiles_per_row: {}, in_tiles_per_col: {}, out_tiles_per_row: {}",
        in_tiles_per_row,
        in_tiles_per_col,
        out_tiles_per_row);

    // input CB
    uint8_t cb_src0_index = CB::c_in0;
    auto src_cb_config = CircularBufferConfig(total_size, {{cb_src0_index, data_format}})
                             .set_page_size(cb_src0_index, tile_size)
                             .set_globally_allocated_address(*src_buffer);
    auto cb_src0 = CreateCircularBuffer(program, all_cores, src_cb_config);

    // auxiliary CBs
    uint8_t cb_32_tiles_index = CB::c_intermed1;
    auto cb_32_tiles_config = CircularBufferConfig(32 * tile_size, {{cb_32_tiles_index, data_format}})
                                  .set_page_size(cb_32_tiles_index, tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_32_tiles_config);

    uint8_t cb_8_tiles_index = CB::c_intermed0;
    auto cb_8_tiles_config = CircularBufferConfig(8 * tile_size, {{cb_8_tiles_index, data_format}})
                                 .set_page_size(cb_8_tiles_index, tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_8_tiles_config);

    uint8_t cb_1_row_tiles_index = CB::c_intermed2;
    auto cb_1_row_tiles_config =
        CircularBufferConfig(out_tiles_per_row * tile_size, {{cb_1_row_tiles_index, data_format}})
            .set_page_size(cb_1_row_tiles_index, tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_1_row_tiles_config);

    // Reader kernel
    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/dataflow/reader_scan_sharded.cpp",
        all_cores,
        WriterDataMovementConfig({cb_src0_index, cb_32_tiles_index, cb_8_tiles_index, cb_1_row_tiles_index}));

    // Reader run-time args
    SetRuntimeArgs(program, reader_kernel_id, all_cores, {in_tiles_per_row, in_tiles_per_col, out_tiles_per_row});

    tt_metal::KernelHandle compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/scan/kernels/compute/untilize_scan_tilize.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .compile_args = {cb_src0_index, cb_32_tiles_index, cb_8_tiles_index, cb_1_row_tiles_index}});

    tt_metal::SetRuntimeArgs(
        program, compute_kernel_id, all_cores, {in_tiles_per_row, in_tiles_per_col, out_tiles_per_row});

    auto override_runtime_args_callback = [](const void *operation,
                                             Program &program,
                                             const std::vector<Tensor> &input_tensors,
                                             const std::vector<std::optional<const Tensor>> &optional_input_tensors,
                                             const std::vector<Tensor> &output_tensors) {};

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks Scan::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);
    Tensor &output_tensor = output_tensors.at(0);

    return scan_impl(input_tensor, output_tensor, direction);
}

Tensor scan(const Tensor &a) {
    uint32_t n_tile_columns = a.shard_spec()->shape[1] / TILE_WIDTH;
    return operation::run(Scan{.n_tile_columns = n_tile_columns}, {a}).at(0);
}

}  // namespace tt::tt_metal
