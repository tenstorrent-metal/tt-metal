// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_dot_backward/moreh_dot_backward_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_dot_backward_single_core(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &other,
    const std::optional<const Tensor> &input_grad,
    const std::optional<const Tensor> &other_grad) {
    Program program{};
    CoreRange core = {.start = {0, 0}, .end = {0, 0}};

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    Buffer *src0_buffer = output_grad.buffer();
    Buffer *src1_buffer = input.buffer();
    Buffer *src2_buffer = other.buffer();

    uint32_t num_tiles = input.volume() / TILE_HW;
    float scaler = 1.0f;
    const auto &a_shape_wo_padding = input.shape().without_padding();
    uint32_t pad_h = a_shape_wo_padding[2] % TILE_HEIGHT;
    uint32_t pad_w = a_shape_wo_padding[3] % TILE_WIDTH;
    uint32_t mask_h = (pad_h == 0) ? (TILE_HEIGHT) : (pad_h);
    uint32_t mask_w = (pad_w == 0) ? (TILE_WIDTH) : (pad_w);

    // This should allocate a DRAM buffer on the device
    Device *device = input.device();

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t src2_cb_index = CB::c_in2;
    CircularBufferConfig cb_src2_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src2_cb_index, cb_data_format}})
            .set_page_size(src2_cb_index, single_tile_size);
    auto cb_src2 = CreateCircularBuffer(program, core, cb_src2_config);

    uint32_t output0_cb_index = CB::c_out0;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output0_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output0_cb_index, cb_data_format}})
            .set_page_size(output0_cb_index, single_tile_size);
    auto cb_output0 = CreateCircularBuffer(program, core, cb_output0_config);

    uint32_t output1_cb_index = CB::c_out1;
    CircularBufferConfig cb_output1_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output1_cb_index, cb_data_format}})
            .set_page_size(output1_cb_index, single_tile_size);
    auto cb_output1 = CreateCircularBuffer(program, core, cb_output1_config);

    bool has_input_grad = input_grad.has_value();
    bool has_other_grad = other_grad.has_value();

    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool src2_is_dram = src2_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_is_dram, (std::uint32_t)src1_is_dram, (std::uint32_t)src2_is_dram};

    bool dst0_is_dram = false;
    bool dst1_is_dram = false;
    uint32_t dst0_address = 0;
    uint32_t dst1_address = 0;

    if (has_input_grad) {
        const auto &input_grad_tensor = input_grad.value();
        Buffer *dst0_buffer = input_grad_tensor.buffer();
        TT_ASSERT(dst0_buffer != nullptr, "input_grad buffer should be allocated on device!");
        dst0_is_dram = dst0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
        dst0_address = dst0_buffer->address();
    }

    if (has_other_grad) {
        const auto &other_grad_tensor = other_grad.value();
        Buffer *dst1_buffer = other_grad_tensor.buffer();
        TT_ASSERT(dst1_buffer != nullptr, "other_grad buffer should be allocated on device!");
        dst1_is_dram = dst1_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
        dst1_address = dst1_buffer->address();
    }

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output0_cb_index,
        (std::uint32_t)output1_cb_index,
        (std::uint32_t)dst0_is_dram,
        (std::uint32_t)dst1_is_dram,
    };

    KernelID binary_reader_kernel_id = CreateDataMovementKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_dot_backward/kernels/reader_moreh_dot_backward.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    KernelID binary_writer_kernel_id = CreateDataMovementKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_dot_backward/kernels/writer_moreh_dot_backward.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_kernel_args = {};
    std::map<string, string> defines;

    auto dot_backward_kernel = CreateComputeKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_dot_backward/kernels/moreh_dot_backward.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args, .defines = defines});

    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {(std::uint32_t)has_input_grad,
         (std::uint32_t)has_other_grad,
         src0_buffer->address(),
         src1_buffer->address(),
         src2_buffer->address(),
         num_tiles,
         0});

    SetRuntimeArgs(
        program, dot_backward_kernel, core, {(std::uint32_t)has_input_grad, (std::uint32_t)has_other_grad, num_tiles});

    SetRuntimeArgs(
        program,
        binary_writer_kernel_id,
        core,
        {(std::uint32_t)has_input_grad, (std::uint32_t)has_other_grad, dst0_address, dst1_address, num_tiles, 0});

    auto override_runtime_arguments_callback =
        [binary_reader_kernel_id, binary_writer_kernel_id, dot_backward_kernel](
            const void *operation,
            const Program &program,
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<Tensor> &output_tensors) {
            const auto &output_grad = input_tensors.at(0);
            const auto &input = input_tensors.at(1);
            const auto &other = input_tensors.at(2);
            const auto &input_grad = optional_input_tensors.at(0);
            const auto &other_grad = optional_input_tensors.at(1);

            bool has_input_grad = input_grad.has_value();
            bool has_other_grad = other_grad.has_value();

            Buffer *src0_buffer = output_grad.buffer();
            Buffer *src1_buffer = input.buffer();
            Buffer *src2_buffer = other.buffer();

            uint32_t dst0_address = 0;
            uint32_t dst1_address = 0;

            if (has_input_grad) {
                const auto &input_grad_tensor = input_grad.value();
                Buffer *dst0_buffer = input_grad_tensor.buffer();
                dst0_address = dst0_buffer->address();
            }

            if (has_other_grad) {
                const auto &other_grad_tensor = other_grad.value();
                Buffer *dst1_buffer = other_grad_tensor.buffer();
                dst1_address = dst1_buffer->address();
            }

            CoreCoord core = {0, 0};
            {
                auto runtime_args = GetRuntimeArgs(program, binary_reader_kernel_id, core);
                runtime_args[0] = (std::uint32_t)has_input_grad;
                runtime_args[1] = (std::uint32_t)has_input_grad;
                runtime_args[2] = src0_buffer->address();
                runtime_args[3] = src1_buffer->address();
                runtime_args[4] = src2_buffer->address();
                SetRuntimeArgs(program, binary_reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, dot_backward_kernel, core);
                runtime_args[0] = (std::uint32_t)has_input_grad;
                runtime_args[1] = (std::uint32_t)has_input_grad;
                SetRuntimeArgs(program, dot_backward_kernel, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, binary_writer_kernel_id, core);
                runtime_args[0] = (std::uint32_t)has_input_grad;
                runtime_args[1] = (std::uint32_t)has_input_grad;
                runtime_args[2] = dst0_address;
                runtime_args[3] = dst1_address;
                SetRuntimeArgs(program, binary_writer_kernel_id, core, runtime_args);
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
