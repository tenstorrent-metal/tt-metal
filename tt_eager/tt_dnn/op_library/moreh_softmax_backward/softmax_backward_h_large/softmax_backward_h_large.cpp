// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_softmax_backward/moreh_softmax_backward_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_softmax_backward/helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include <optional>

using u32 = std::uint32_t;
using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

operation::ProgramWithCallbacks moreh_softmax_backward_h_large(const Tensor &output, const Tensor &output_grad, Tensor &input_grad, const CoreRange core_range) {
    // split work
    auto shape = input_grad.shape();
    auto N = shape[0];
    auto C = shape[1];
    auto H = shape[2];
    auto W = shape[3];
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;

    uint32_t num_cols_tiles = N * C * Wt;
    uint32_t core_w = core_range.end.x - core_range.start.x + 1;
    uint32_t core_h = core_range.end.y - core_range.start.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(core_range, num_cols_tiles);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_grad.dtype());

    CreateCircularBuffers(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 2},        // output
            {CB::c_in1, 2},        // output_grad
            {CB::c_in2, 1},        // scaler
            {CB::c_in3, 1},        // mask
            {CB::c_out0, 2},       // input_grad
            {CB::c_intermed0, 1},  // output * output_grad
            {CB::c_intermed1, 1},  // reduce
            {CB::c_intermed2, 1},  // dy - sum
            {CB::c_intermed3, 2},  // add(output * output_grad)
        });

    // create read/wrtie kernel
    bool y_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dy_is_dram = output_grad.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dx_is_dram = input_grad.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    auto reader_kernel_id = CreateReadKernel(
        program, "reader_moreh_softmax_backward_h_large.cpp", all_cores, {y_is_dram, dy_is_dram}, reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program, "writer_moreh_softmax_h.cpp", all_cores, {dx_is_dram}, writer_defines);

    // create compute kernel
    CreateComputeKernel(
        program,
        "moreh_softmax_backward_h_large.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1, Ht}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2, Ht}},
        });

    // Set Runtime Args
    auto core_x_offset = core_range.start.x;
    auto core_y_offset = core_range.start.y;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        float scaler = 1.0f;
        uint32_t mask_h = shape.without_padding()[2] % TILE_HEIGHT;
        if(mask_h == 0) mask_h = TILE_HEIGHT;
        vector<u32> reader_args = {
            output.buffer()->address(),
            output_grad.buffer()->address(),
            num_tiles_per_core, tile_offset, Ht, Wt, *reinterpret_cast<uint32_t *>(&scaler), mask_h};

        vector<u32> writer_args = {input_grad.buffer()->address(), num_tiles_per_core, tile_offset, Ht, Wt};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core;
    }

    CoreGridDesc grid(input_grad.device());

    auto override_runtime_args_callback = [
            reader_kernel_id=reader_kernel_id,
            writer_kernel_id=writer_kernel_id,
            num_cores,
            grid
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {
        TT_ASSERT(input_buffers.size() == 2);
        TT_ASSERT(output_buffers.size() == 1);

        auto output_dram_buffer = input_buffers.at(0);
        auto output_grad_dram_buffer = input_buffers.at(1);
        auto input_grad_dram_buffer = output_buffers.at(0);

        for (uint32_t icore = 0; icore < num_cores; icore++) {
            auto core = grid.wrap_core(icore);

            // CoreCoord core = {icore / core_h + core_x_offset, icore % core_h + core_y_offset};

            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = output_dram_buffer->address();
                runtime_args[1] = output_grad_dram_buffer->address();
                SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = input_grad_dram_buffer->address();
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
