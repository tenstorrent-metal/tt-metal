// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"

#include <optional>

using u32 = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

inline bool is_dram(const Tensor& input_tensor) { return input_tensor.memory_config().buffer_storage == BufferStorage::DRAM; }
inline bool is_dram(const std::optional<const Tensor> input_tensor) {
     return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_storage() == BufferStorage::DRAM; }

// implementation of softmax with optional scale/mask (see the header for input_tensor more detailed description)
operation::ProgramWithCallbacks scale_mask_softmax_(const Tensor &input_tensor, const Tensor &output_tensor, const std::optional<const Tensor> mask, std::optional<float> scale) {

    const auto shape = input_tensor.shape();
    u32 W = shape[-1], H = (input_tensor.volume() / (shape[0] * shape[-1])), NC = shape[0];
    u32 HW = H*W;

    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    Program program = Program();

    uint32_t scalar_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    tt::DataFormat in0_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t in0_tile_size = tt_metal::detail::TileSize(in0_cb_data_format);

    tt::DataFormat out0_cb_data_format = tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t out0_tile_size = tt_metal::detail::TileSize(out0_cb_data_format);

    tt::DataFormat mask_cb_data_format = mask.has_value() ? tt_metal::datatype_to_dataformat_converter(mask.value().dtype()) : tt::DataFormat::Float16_b;
    uint32_t mask_tile_size = tt_metal::detail::TileSize(mask_cb_data_format);

    auto src0_buffer = input_tensor.buffer();
    auto out0_buffer = output_tensor.buffer();

    uint32_t num_tiles = input_tensor.volume()/TILE_HW;

    // This should allocate input_tensor DRAM buffer on the device
    Device *device = input_tensor.device();

    uint32_t block_size = find_max_divisor(Wt, 8);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_t  = block_size*2;
    uint32_t out0_t = block_size*2;
    uint32_t im1_t  = 1; // 1/sum(exp(x))
    uint32_t in2_t  = 1; // scaler for reduce coming from reader
    uint32_t in3_t  = 1; // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    uint32_t in4_t  = div_up(Wt, block_size)*block_size; // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled

    // cb_exps - keeps exps in CB in L1 to avoid recomputing
    uint32_t im0_t  = block_size*div_up(Wt, block_size);
    TT_ASSERT(im0_t == Wt);

    // used for buffering scale-mask
    // can't easily reuse im0_t because cumulative wait for Wt needs to have Wt tiles contiguous free
    uint32_t im3_t  = block_size*(div_up(Wt, block_size)+1);
    TT_ASSERT(im3_t == Wt+block_size);

    TT_ASSERT(Wt % block_size == 0);
    TT_ASSERT((block_size != -1) && "Wt must be divisible by one of the numbers in the range from 8 to 1.");
    TT_ASSERT(im0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(out0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in4_t % block_size == 0);
    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");

    uint32_t num_tile_rows = NC * Ht;
    auto grid_size = device->compute_with_storage_grid_size();
    auto all_device_cores = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] = split_work_to_cores(grid_size, num_tile_rows, true);

    bool src0_is_dram = src0_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    bool out0_is_dram = out0_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        src0_is_dram
    };
    if (mask.has_value()) {
        bool mask_is_dram = mask.value().buffer()->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
        reader_compile_time_args.push_back(mask_is_dram);
    }

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        out0_is_dram
    };
    std::map<string, string> softmax_defines;
    if (mask.has_value()) {
        softmax_defines["FUSED_SCALE_MASK"] = "1";
    }
    auto reader_kernels_id = CreateDataMovementKernel(
        program, "tt_eager/tt_dnn/op_library/softmax/kernels/reader_unary_interleaved_sm.cpp", all_device_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
            .defines = softmax_defines
    });
    //DataMovementProcessor::RISCV_1, core.x < 6 ? NOC::RISCV_1_default : NOC::RISCV_0_default);

    auto writer_kernels_id = CreateDataMovementKernel(
        program, "tt_eager/tt_dnn/op_library/softmax/kernels/writer_unary_interleaved_start_id_blocked_sm.cpp", all_device_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args
    });
    //DataMovementProcessor::RISCV_0, core.x < 6 ? NOC::RISCV_0_default : NOC::RISCV_1_default);

    // for broadcasting in H direction we need to
    // NCHt, Nt, Wt
    // if wtpc < Ht then since we pass tpc to the kernel as Ht, the broadcasts should be correct
    // if wtpc >= Ht then tpc should be a multiple of Ht
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto softmax_kernels_id = CreateComputeKernel(
        program, "kernels/compute/softmax.cpp", all_device_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode,
            .compile_args = {},
            .defines = softmax_defines
    });

    // Create circular buffers
    // see softmax.cpp for which buffers are needed

    auto c_in0_config = CircularBufferConfig(in0_t * in0_tile_size, {{CB::c_in0, in0_cb_data_format}}).set_page_size(CB::c_in0, in0_tile_size);
    auto cb_in0_id = CreateCircularBuffer( program, all_device_cores, c_in0_config);
    auto c_out0_config = CircularBufferConfig(out0_t * out0_tile_size, {{CB::c_out0, out0_cb_data_format}}).set_page_size(CB::c_out0, out0_tile_size);
    auto cb_out0_id = CreateCircularBuffer( program, all_device_cores, c_out0_config );
    auto c_intermed1_config = CircularBufferConfig(im1_t * in0_tile_size, {{CB::c_intermed1, in0_cb_data_format}}).set_page_size(CB::c_intermed1, in0_tile_size);
    auto cb_intermed1_id = CreateCircularBuffer( program, all_device_cores, c_intermed1_config );
    auto c_in2_config = CircularBufferConfig(in2_t * scalar_tile_size, {{CB::c_in2, DataFormat::Float16_b}}).set_page_size(CB::c_in2, scalar_tile_size);
    auto cb_in2_id = CreateCircularBuffer( program, all_device_cores, c_in2_config );
    auto c_intermed0_config = CircularBufferConfig(im0_t * in0_tile_size, {{CB::c_intermed0, in0_cb_data_format}}).set_page_size(CB::c_intermed0, in0_tile_size);
    auto cb_intermed0_id = CreateCircularBuffer( program, all_device_cores, c_intermed0_config );
    std::optional<CircularBufferID> cb_intermed3_id;
    std::optional<CircularBufferID> cb_in3_id;
    std::optional<CircularBufferID> cb_in4_id;
    if (mask.has_value()) {
        CircularBufferConfig c_intermed3_config = CircularBufferConfig(im3_t * in0_tile_size, {{CB::c_intermed3, in0_cb_data_format}}).set_page_size(CB::c_intermed3, in0_tile_size);
        cb_intermed3_id = CreateCircularBuffer( program, all_device_cores, c_intermed3_config );
        CircularBufferConfig c_in3_config = CircularBufferConfig(in3_t * scalar_tile_size, {{CB::c_in3, DataFormat::Float16_b}}).set_page_size(CB::c_in3, scalar_tile_size);
        cb_in3_id = CreateCircularBuffer( program, all_device_cores, c_in3_config );
        CircularBufferConfig c_in4_config = CircularBufferConfig(in4_t * mask_tile_size, {{CB::c_in4, mask_cb_data_format}}).set_page_size(CB::c_in4, mask_tile_size);
        cb_in4_id = CreateCircularBuffer( program, all_device_cores, c_in4_config);
    }
    uint32_t src_addr = src0_buffer->address();
    uint32_t mask_addr = mask.has_value() ? mask.value().buffer()->address() : 0;
    uint32_t out_addr = out0_buffer->address();

    uint32_t curr_row = 0;
    union { float f; uint32_t u; } s; s.f = scale.value_or(1.0f); // scale for fused scale-mask-softmax
    for (uint32_t i = 0; i < grid_size.x * grid_size.y; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        if (i >= num_cores) {
            SetRuntimeArgs(program, reader_kernels_id, core, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }); // [8]=1.0f is scaler
            SetRuntimeArgs(program, softmax_kernels_id, core, { 0, 0, 0, 0, 0 });
            SetRuntimeArgs(program, writer_kernels_id, core, { 0, 0, 0, 0 });
            continue;
        }
        uint32_t num_tile_rows_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t tile_offset = curr_row * Wt;
        uint32_t curr_ht = curr_row % Ht;
        uint32_t mask_id = curr_row / Ht * Wt;

        SetRuntimeArgs(program, reader_kernels_id, core, { src_addr, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_addr, curr_ht, mask_id, 0x3f803f80 }); // [10]=1.0f is scaler
        SetRuntimeArgs(program, softmax_kernels_id, core, { num_tile_rows_per_core, Ht, Wt, block_size, curr_ht });
        SetRuntimeArgs(program, writer_kernels_id, core, { out_addr, num_tile_rows_per_core * Wt, tile_offset, block_size });
        curr_row += num_tile_rows_per_core;
    }

    auto override_runtime_arguments_callback = [
            reader_kernels_id,
            writer_kernels_id,
            softmax_kernels_id,
            grid_size,
            scalar_tile_size,
            in0_tile_size,
            out0_tile_size,
            mask_tile_size,
            cb_in0_id,
            cb_out0_id,
            cb_intermed1_id,
            cb_in2_id,
            cb_intermed0_id,
            cb_intermed3_id,
            cb_in3_id,
            cb_in4_id
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        const auto scale = static_cast<const Softmax*>(operation)->scale;

        auto src_buffer_address = input_tensors.at(0).buffer()->address();
        auto mask_buffer_address = optional_input_tensors.at(0).has_value() ? optional_input_tensors.at(0).value().buffer()->address() : 0;
        auto dst_buffer_address = output_tensors.size() == 1 ? output_tensors.at(0).buffer()->address() : src_buffer_address;

        const auto shape = input_tensors.at(0).shape();
        u32 W = shape[-1], H = (input_tensors.at(0).volume() / (shape[0] * shape[-1])), NC = shape[0];
        u32 HW = H*W;

        u32 Wt = W/TILE_WIDTH;
        u32 Ht = H/TILE_HEIGHT;

        int32_t num_tiles = input_tensors.at(0).volume()/TILE_HW;
        uint32_t block_size = find_max_divisor(Wt, 8);

        // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
        uint32_t in0_t  = block_size*2;
        uint32_t out0_t = block_size*2;
        uint32_t im1_t  = 1; // 1/sum(exp(x))
        uint32_t in2_t  = 1; // scaler for reduce coming from reader
        uint32_t in3_t  = 1; // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
        uint32_t in4_t  = div_up(Wt, block_size)*block_size; // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled

        // cb_exps - keeps exps in CB in L1 to avoid recomputing
        uint32_t im0_t  = block_size*div_up(Wt, block_size);
        TT_ASSERT(im0_t == Wt);

        // used for buffering scale-mask
        // can't easily reuse im0_t because cumulative wait for Wt needs to have Wt tiles contiguous free
        uint32_t im3_t  = block_size*(div_up(Wt, block_size)+1);
        TT_ASSERT(im3_t == Wt+block_size);

        TT_ASSERT(Wt % block_size == 0);
        TT_ASSERT((block_size != -1) && "Wt must be divisible by one of the numbers in the range from 8 to 1.");
        TT_ASSERT(im0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
        TT_ASSERT(out0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
        TT_ASSERT(in4_t % block_size == 0);
        TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");

        uint32_t NCHt = NC*Ht;
        uint32_t num_tile_rows = NC * Ht;
        auto all_device_cores = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
        auto [num_cores, all_cores, core_group_1, core_group_2, num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] = split_work_to_cores(grid_size, num_tile_rows, true);

        GetCircularBufferConfig(program, cb_in0_id).set_total_size(in0_t * in0_tile_size);
        GetCircularBufferConfig(program, cb_out0_id).set_total_size(out0_t * out0_tile_size);
        GetCircularBufferConfig(program, cb_intermed1_id).set_total_size(im1_t * in0_tile_size);
        GetCircularBufferConfig(program, cb_in2_id).set_total_size(in2_t * scalar_tile_size);
        GetCircularBufferConfig(program, cb_intermed0_id).set_total_size(im0_t * in0_tile_size);

        if (optional_input_tensors.at(0).has_value()) {
            GetCircularBufferConfig(program, cb_intermed3_id.value()).set_total_size(im3_t * in0_tile_size);
            GetCircularBufferConfig(program, cb_in3_id.value()).set_total_size(in3_t * scalar_tile_size);
            GetCircularBufferConfig(program, cb_in4_id.value()).set_total_size(in4_t * mask_tile_size);
        }

        uint32_t curr_row = 0;
        union { float f; uint32_t u; } s; s.f = scale.value_or(1.0f); // scale for fused scale-mask-softmax
        for (uint32_t i = 0; i < grid_size.x * grid_size.y; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};
            if (i >= num_cores) {
                SetRuntimeArgs(program, reader_kernels_id, core, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }); // [8]=1.0f is scaler
                SetRuntimeArgs(program, softmax_kernels_id, core, { 0, 0, 0, 0, 0 });
                SetRuntimeArgs(program, writer_kernels_id, core, { 0, 0, 0, 0 });
                continue;
            }

            uint32_t num_tile_rows_per_core;
            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_tile_rows_per_core = num_tile_rows_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
                num_tile_rows_per_core = num_tile_rows_per_core_group_2;
            } else {
                TT_ASSERT(false, "Core not in specified core ranges");
            }

            uint32_t tile_offset = curr_row * Wt;
            uint32_t curr_ht = curr_row % Ht;
            uint32_t mask_id = curr_row / Ht * Wt;

            SetRuntimeArgs(program, reader_kernels_id, core, { src_buffer_address, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_buffer_address, curr_ht, mask_id, 0x3f803f80 }); // [10]=1.0f is scaler
            SetRuntimeArgs(program, softmax_kernels_id, core, { num_tile_rows_per_core, Ht, Wt, block_size, curr_ht });
            SetRuntimeArgs(program, writer_kernels_id, core, { dst_buffer_address, num_tile_rows_per_core * Wt, tile_offset, block_size });
            curr_row += num_tile_rows_per_core;
        }
    };

    return {std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
} // scale_mask_softmax_


void Softmax::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 1 and optional_input_tensors.size() <= 1, "Must have 1 or 2 input tensors");
    auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr , "Operands to softmax need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor.layout() == Layout::TILE), "Inputs to softmax must be tilized");
    TT_ASSERT(input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B);
    if (optional_input_tensors.size() == 1) {
        if (optional_input_tensors.at(0).has_value()) {
            auto& mask = optional_input_tensors.at(0).value();
            TT_ASSERT(mask.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
            TT_ASSERT(input_tensor.device() == mask.device());
            TT_ASSERT(input_tensor.dtype() == mask.dtype());
            TT_ASSERT(input_tensor.layout() == mask.layout());
            TT_ASSERT(input_tensor.shape()[-1] == mask.shape()[-1]);
            TT_ASSERT(input_tensor.shape()[0] == mask.shape()[0]);
            TT_ASSERT(mask.shape()[-2] == TILE_HEIGHT);
            for (uint32_t i = 1; i < input_tensor.shape().rank() - 2; i++) {
                TT_ASSERT(mask.shape()[i] == 1);
            }
        } else {
            TT_ASSERT(not this->scale.has_value());
        }
    } else {
        TT_ASSERT(not this->scale.has_value());
    }


}

std::vector<Shape> Softmax::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    // Do nothing because it's an in-place operation
    if (this->inplace) {
        return {};
    } else {
        return {input_tensors.at(0).shape()};
    }
}

std::vector<Tensor> Softmax::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    // Do nothing because it's an in-place operation
    if (this->inplace) {
        return {};
    }  else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensors.at(0).dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Softmax::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = this->inplace ? input_tensors.at(0) : output_tensors.at(0);
    const auto& mask = optional_input_tensors.at(0);
    return scale_mask_softmax_(input_tensor, output_tensor, mask, this->scale);

}

tt::stl::reflection::Attributes Softmax::attributes() const {
    return {
        {"scale", this->scale},
        {"inplace", this->inplace},
        {"output_mem_config", this->output_mem_config},
    };
}


const operation::Hash Softmax::compute_program_hash(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return fmt::format(
        "Softmax_{}_{}_{}_{}_{}",
        input_tensors.at(0).memory_config(),
        input_tensors.at(0).dtype(),
        optional_input_tensors.at(0).has_value() ? std::optional{optional_input_tensors.at(0).value().memory_config()} : std::nullopt,
        optional_input_tensors.at(0).has_value() ? std::optional{optional_input_tensors.at(0).value().dtype()} : std::nullopt,
        this->output_mem_config
    );
}

Tensor softmax_in_place(Tensor& input_tensor) {
    return transformers::scale_mask_softmax_in_place(input_tensor, std::nullopt, std::nullopt);
}

namespace transformers {
Tensor scale_mask_softmax_in_place(Tensor& input_tensor, std::optional<float> scale, std::optional<const Tensor> mask) {
    operation::run(Softmax{.scale=scale, .inplace=true, .output_mem_config=input_tensor.memory_config()}, {input_tensor}, {mask});
    return input_tensor;
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations

namespace tt_metal {
Tensor softmax(const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    return transformers::scale_mask_softmax(input_tensor, std::nullopt, std::nullopt, output_mem_config);
}

namespace transformers {
Tensor scale_mask_softmax(const Tensor& input_tensor, std::optional<float> scale, std::optional<const Tensor> mask, const MemoryConfig& output_mem_config) {
    Shape input_pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape());
    FormatParams input_format_params = {.pad_shape=input_pad_shape, .pad_value=-std::numeric_limits<float>::infinity(), .target_layout=Layout::TILE};
    std::optional<FormatParams> mask_format_params = std::nullopt;
    if (mask.has_value()) {
        TT_ASSERT(input_tensor.shape()[-1] == mask.value().shape()[-1]);
        TT_ASSERT(input_tensor.shape()[0] == mask.value().shape()[0]);
        TT_ASSERT(mask.value().shape()[-2] == 1);
        for (uint32_t i = 1; i < input_tensor.shape().rank() - 2; i++) {
            TT_ASSERT(mask.value().shape()[i] == 1);
        }
        Shape mask_pad_shape = AutoFormat::pad_to_tile_shape(mask.value().shape());
        mask_format_params = {.pad_shape=mask_pad_shape, .pad_value=-std::numeric_limits<float>::infinity(), .target_layout=Layout::TILE};
    }
    return operation::run_with_autoformat(tt::operations::primary::Softmax{.scale=scale, .inplace=false, .output_mem_config=output_mem_config}, {input_tensor}, {input_format_params}, {Layout::TILE}, {mask}, {mask_format_params}).at(0);
}
}  // namespace transformers
}  // namespace tt_metal
}  // namespace tt
