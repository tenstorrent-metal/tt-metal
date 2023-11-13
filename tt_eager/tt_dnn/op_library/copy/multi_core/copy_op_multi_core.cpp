// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks copy_multi_core(const Tensor &input, const Tensor &output, bool backwards) {
    tt_metal::Program program{};

    bool tilized = output.layout() == Layout::TILE;

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = tilized ? tt_metal::detail::TileSize(input_cb_data_format) : input.shape()[-1] * input.element_size();
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_unit_size = tilized ? tt_metal::detail::TileSize(output_cb_data_format) : output.shape()[-1] * output.element_size();

    uint32_t num_units = tilized ? output.volume() / TILE_HW : output.volume() / output.shape()[-1];

    tt_metal::Device *device = output.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_units);

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_units = 2;
    uint32_t aligned_input_unit_size = round_up_to_mul32(input_unit_size);
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_units * aligned_input_unit_size, {{src0_cb_index, input_cb_data_format}})
		.set_page_size(src0_cb_index, aligned_input_unit_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = src0_cb_index; // same as input cb
    /* If we need dataformat conversion, use output buffer + compute kernel
    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
		.set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    */

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;
    if (tilized) {
        reader_compile_time_args = {(uint32_t)src_is_dram};
        writer_compile_time_args = {
            (std::uint32_t) output_cb_index,
            (std::uint32_t) dst_is_dram
        };
    } else {
        bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(input_unit_size);
        uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(input_unit_size) : 0;
        reader_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) src_is_dram,
            (std::uint32_t) src_stick_size_is_power_of_two,
            (std::uint32_t) src_log2_stick_size
        };
        bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_unit_size);
        uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(output_unit_size) : 0;
        writer_compile_time_args = {
            (std::uint32_t) output_cb_index,
            (std::uint32_t) dst_is_dram,
            (std::uint32_t) dst_stick_size_is_power_of_two,
            (std::uint32_t) dst_log2_stick_size
        };
    }
    std::map<string, string> kernel_defines;
    if (backwards) {
        kernel_defines["BACKWARDS"] = "1";
    }
    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        tilized ? "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp" : "tt_metal/kernels/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args, .defines = kernel_defines});

    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        tilized ? "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp" : "tt_metal/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args, .defines = kernel_defines});

    /* If we need dataformat conversion, use compute kernel
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;

    vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1
    };

    auto eltwise_unary_kernel_group_1 = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        core_group_1,
        compute_kernel_args_group_1,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    if(!core_group_2.ranges().empty()){
        vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2
        };

        auto eltwise_unary_kernel_group_2 = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_copy.cpp",
            core_group_2,
            compute_kernel_args_group_2,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
    }
    */
   uint32_t start_id = 0;
   if (backwards) {
        start_id = num_units - 1;
   }

    for (uint32_t i = 0; i < num_cores; ++i){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_units_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_units_per_core = num_units_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_units_per_core = num_units_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        if (tilized) {
            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {
                    src_buffer->address(),
                    num_units_per_core,
                    start_id
                }
            );

            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {
                    dst_buffer->address(),
                    num_units_per_core,
                    start_id
                }
            );
        } else {
            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {
                    src_buffer->address(),
                    input_unit_size,
                    num_units_per_core,
                    start_id
                }
            );

            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {
                    dst_buffer->address(),
                    output_unit_size,
                    num_units_per_core,
                    start_id
                }
            );
        }
        if (backwards) {
            start_id -= num_units_per_core;
        } else {
            start_id += num_units_per_core;
        }
    }

    auto override_runtime_args_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            num_cores,
            num_cores_y
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = input_buffers.size() == 2 ? input_buffers.at(1) : output_buffers.at(0);

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
                SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
                SetRuntimeArgs(program, unary_writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
