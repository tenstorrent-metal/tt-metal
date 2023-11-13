// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/split/split_last_dim_two_chunks_tiled.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/auto_format.hpp"

#include "common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/util.hpp"

#include <iostream>

using namespace tt::constants;
using namespace tt::tt_metal;



namespace tt {

namespace tt_metal {

void setup_runtime(
    const Program &program,
    const uint32_t &core_offset,
    const uint32_t &num_cores_r,
    const uint32_t &num_cores_c,
    const uint32_t &z,
    const uint32_t &num_cores_x,
    const uint32_t &per_core_tiles_y,
    const uint32_t &per_core_tiles_x,
    const uint32_t &num_tiles_per_z,
    tt_metal::Buffer *in0_buffer,
    tt_metal::Buffer *out0_buffer,
    tt_metal::Buffer *out1_buffer,
    tt_metal::KernelID reader_kernel_id,
    tt_metal::KernelID writer_kernel_id) {
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    if(num_cores_c > 1){
        TT_ASSERT(num_cores_c %2 == 0, "Must be even number of cores");
    }
    uint32_t idc_outer_limit = 1;
    uint32_t idc_inner_limit = num_cores_c;


    for (int id_r_outer = 0; id_r_outer < z; id_r_outer++) {
        for(int id_r_inner = 0; id_r_inner < num_cores_x; id_r_inner++){
            uint32_t id_r = id_r_outer*num_cores_x + id_r_inner;

            uint32_t id_r_reader = id_r_outer*num_tiles_per_z + id_r_inner*per_core_tiles_y*num_cores_c*per_core_tiles_x;
            uint32_t id_r_writer = id_r_reader/2;
            if(num_cores_c > 1){
                idc_outer_limit = 2;
                idc_inner_limit = num_cores_c/2;
            }
            for(int id_c_outer = 0; id_c_outer < idc_outer_limit; id_c_outer++){
                for (int id_c_inner = 0; id_c_inner < idc_inner_limit; id_c_inner++) {
                    uint32_t id_c = id_c_outer*idc_inner_limit + id_c_inner;
                    CoreCoord core = {(std::size_t)start_core_x + id_r, (std::size_t)start_core_y + id_c};

                    uint32_t reader_core_id = id_c*per_core_tiles_y;
                    reader_core_id += id_r_reader;


                    std::vector<uint32_t> reader_runtime_args = {
                        (std::uint32_t)reader_core_id,
                        (std::uint32_t)(in0_buffer->address()),  // in0_tensor_addr
                        (std::uint32_t) 0 //split on last dim
                    };
                    bool out0_only = false;
                    bool out1_only = false;
                    if(num_cores_c > 1){
                        out0_only = (id_c_outer == 0);
                        out1_only = (id_c_outer == 1);
                    }

                    uint32_t writer_core_id = id_c_inner*per_core_tiles_y + (id_r_writer);

                    std::vector<uint32_t> writer_runtime_args = {
                        writer_core_id,
                        (std::uint32_t)out0_buffer->address(),  // first base addr
                        (std::uint32_t)out1_buffer->address(),  // second base addr
                        (std::uint32_t) out0_only,
                        (std::uint32_t) out1_only
                    };
                    tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
                    tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
                }
            }
        }
    }
}

operation::ProgramWithCallbacks split_last_dim_two_chunks_tiled(
    const Tensor &input_tensor, std::vector<Tensor> &output_tensors, const MemoryConfig &mem_config) {
    SplitLastDimTwoChunksTiled op(mem_config);
    uint32_t dim = op.dim;
    uint32_t num_chunks = op.num_chunks;

    auto input_shape = input_tensor.shape();

    Program program{};
    tt_metal::Device *device = input_tensor.device();
    op.boiler_plate_asserts(input_tensor);
    op.shape_asserts(input_tensor);

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    tt_metal::Buffer *in0_buffer = input_tensor.buffer();

    // Output buffers
    TT_ASSERT(output_tensors.size() == num_chunks);
    tt_metal::Tensor &out0 = output_tensors[0];
    tt_metal::Tensor &out1 = output_tensors[1];

    tt_metal::Buffer *out0_buffer = out0.buffer();
    TT_ASSERT(out0_buffer != nullptr, "Output 0 buffer should be allocated on device!");
    tt_metal::Buffer *out1_buffer = out1.buffer();
    TT_ASSERT(out1_buffer != nullptr, "Output 1 buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t z = input_shape[1];
    uint32_t num_tiles_dim_2 = input_shape[2] / TILE_HEIGHT;
    uint32_t num_tiles_dim_3 = input_shape[3] / TILE_WIDTH;
    uint32_t num_cores_x_limit = device->compute_with_storage_grid_size().x;
    uint32_t num_cores_y_limit = device->compute_with_storage_grid_size().y;

    // parallelize z
    auto num_cores_z = z;

    // parallelize y
    auto [num_cores_y, per_core_tiles_y] =
        get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_dim_3, num_cores_y_limit, /*request_even=*/true);

    // parallelize x
    auto [num_cores_x, per_core_tiles_x] = get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_dim_2,
                                                                                        num_cores_x_limit/ num_cores_z);

    uint32_t per_core_tiles = per_core_tiles_x * per_core_tiles_y * (z / num_cores_z);

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t num_cores_c = num_cores_y;
    uint32_t num_cores_r = num_cores_x * num_cores_z;

    CoreRange all_cores{
        .start = {(std::size_t)start_core_x, (std::size_t)start_core_y},
        .end = {(std::size_t)start_core_x + num_cores_r - 1, (std::size_t)start_core_y + num_cores_c - 1},
    };

    bool tile_dtype_is_bfloat16 = input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool in0_is_dram = in0_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    bool out_is_dram = out0_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    TT_ASSERT(out0_buffer->buffer_storage() == out1_buffer->buffer_storage(), "Output buffers should be the same type");

    uint32_t num_tiles_per_z = (per_core_tiles_x * num_cores_x) * (per_core_tiles_y * num_cores_y);
    uint32_t z_stride_read = num_tiles_per_z;
    uint32_t y_stride_read = per_core_tiles_y * num_cores_y;

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)tile_dtype_is_bfloat16,
        // by default in dram
        (std::uint32_t)in0_is_dram,

        // READER COMPILE TIME ARGS
        (std::uint32_t)(z / num_cores_z),
        (std::uint32_t) per_core_tiles_x,  // out_num_tiles_per_tensor
        (std::uint32_t) per_core_tiles_y,  // out_num_tiles_per_tensor
        (std::uint32_t)z_stride_read,
        (std::uint32_t)y_stride_read};

    uint32_t z_stride_write = num_tiles_per_z / num_chunks;
    uint32_t y_stride_write = per_core_tiles_y * (num_cores_c/num_chunks);
    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)tile_dtype_is_bfloat16,
        (std::uint32_t)out_is_dram,

        (std::uint32_t) per_core_tiles_x,  // out_num_tiles_per_tensor
        (std::uint32_t) per_core_tiles_y,  // out_num_tiles_per_tensor

        (std::uint32_t)(z / num_cores_z),
        (std::uint32_t)z_stride_write,
        (std::uint32_t)y_stride_write

    };

    auto reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_tm_tile_layout_split_two_chunks.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    auto writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_tm_tile_layout_split_two_chunks.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    // Dummy compute kernel
    std::vector<uint32_t> compute_args = {0};  // dummy
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto dummy_compute_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/blank.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compute_args});

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    setup_runtime(
        program,
        0,
        num_cores_r,
        num_cores_c,
        num_cores_z,
        num_cores_x,
        per_core_tiles_y,
        per_core_tiles_x,
        num_tiles_per_z,
        in0_buffer,
        out0_buffer,
        out1_buffer,
        reader_kernel_id,
        writer_kernel_id);

    auto override_runtime_args_callback =
        [reader_kernel_id, writer_kernel_id, num_cores_r, num_cores_c, start_core_x, start_core_y](
            const Program &program, const std::vector<Buffer *> &input_buffers, const std::vector<Buffer *> &output_buffers) {
            auto src_dram_buffer = input_buffers.at(0);

            auto dst_0_dram_buffer = output_buffers.at(0);
            auto dst_1_dram_buffer = output_buffers.at(0);

            for (int core_idx_y = 0; core_idx_y < num_cores_c; core_idx_y++) {
                for (int core_idx_x = 0; core_idx_x < num_cores_r; core_idx_x++) {
                    CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

                    {
                        auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                        runtime_args[1] = src_dram_buffer->address();
                        SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
                    }

                    {
                        auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                        runtime_args[1] = dst_0_dram_buffer->address();
                        runtime_args[2] = dst_1_dram_buffer->address();
                        SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
                    }
                }
            }
        };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks SplitLastDimTwoChunksTiled::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    return split_last_dim_two_chunks_tiled(input_tensor, output_tensors, this->output_mem_config);
}

std::vector<Tensor> split_last_dim_two_chunks_tiled(const Tensor &input_tensor, const MemoryConfig &mem_config) {
    SplitLastDimTwoChunksTiled op(mem_config);

    tt_metal::Device *device;
    // Get the device
    if (input_tensor.storage_type() == StorageType::OWNED) {
        device = AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    auto input_shape = input_tensor.shape();
    TT_ASSERT(input_shape[-1] % TILE_WIDTH == 0, "Split last dim currently only supported tile sized widths");

    auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_shape);
    if (AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape)) {
        return operation::run(op, {input_tensor});
    } else {
        auto device = input_tensor.device();
        auto output_shape = op.compute_output_shapes({input_tensor}).at(0);
        const auto padded_tensor = AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, 0.0, Layout::TILE);
        auto output_tensors = operation::run(op, {padded_tensor});
        for (auto &output_tensor : output_tensors) {
            output_tensor = AutoFormat::format_output_tensor(output_tensor, output_shape, device, Layout::TILE);
        }
        return output_tensors;
    }
}

}  // namespace tt_metal

}  // namespace tt
