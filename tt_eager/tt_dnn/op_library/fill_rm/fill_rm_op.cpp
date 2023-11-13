// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/fill_rm/fill_rm_op.hpp"
#include "common/test_tiles.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using u32 = uint32_t;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks fill_rm_single_core(const Tensor& any, Tensor &output, uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, float val_hi, float val_lo) {

    tt_metal::Device *device = any.device();
    tt_metal::Program program = tt_metal::Program();
    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(any.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_cb_tiles = 16;
    TT_ASSERT(W < 1024*num_cb_tiles); // Limitation for simplifying the kernel

    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{0, cb_data_format}})
		.set_page_size(0, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{1, cb_data_format}})
		.set_page_size(1, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    bool dst_is_dram = dst_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t) dst_is_dram};

    tt_metal::KernelID binary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/fill_rm_interleaved.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args=reader_compile_time_args});

    tt_metal::SetRuntimeArgs(
        program, binary_reader_kernel_id, core,
        { dst_buffer->address(), u32(N*C), u32(H), u32(W), u32(hFill), u32(wFill), u32(bfloat16(val_hi).to_uint16()), u32(bfloat16(val_lo).to_uint16()) }
    );

    auto override_runtime_args_callback = [kernel_id=binary_reader_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(program, kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void FillRM::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT((this->N > 0 && this->C > 0 && this-> H > 0 && this-> W > 0));
    TT_ASSERT((this->hFill <= this->H && this->wFill <= this->W));
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16);
}

std::vector<Shape> FillRM::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    Shape output_shape = {this->N, this->C, this->H, this->W};
    return {output_shape};
}

std::vector<Tensor> FillRM::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::ROW_MAJOR, this->output_mem_config);
}

operation::ProgramWithCallbacks FillRM::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return fill_rm_single_core(input_tensor, output_tensor, this->N, this->C, this->H, this->W, this->hFill, this-> wFill, this->val_hi, this->val_lo);

}

tt::stl::reflection::Attributes FillRM::attributes() const {
    return {
        {"N", this->N},
        {"C", this->C},
        {"H", this->H},
        {"W", this->W},
        {"hFill", this->hFill},
        {"wFill", this->wFill},
        {"val_hi", this->val_hi},
        {"val_lo", this->val_lo},
        {"output_mem_config", this->output_mem_config},
    };
}

tt_metal::Tensor fill_rm(uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, const tt_metal::Tensor& any, float val_hi, float val_lo, const MemoryConfig& output_mem_config) {
    return operation::run_without_autoformat(FillRM{N, C, H, W, hFill, wFill, val_hi, val_lo, output_mem_config}, {any}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
