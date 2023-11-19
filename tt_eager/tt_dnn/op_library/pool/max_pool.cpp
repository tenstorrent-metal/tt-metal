// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>

#include "tt_dnn/op_library/pool/max_pool.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"   // for reduce_op_utils
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "detail/util.hpp"

namespace tt {
namespace tt_metal {

void MaxPool::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input.buffer() != nullptr , "Operands to reshape need to be allocated in buffers on device!");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Only BFLOAT16 supported for now");
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR supported for now");

    // NOTE: This is not a hard requirement. If need to support non-power-of-2, simply change the address generator in reader to generic one.
    uint32_t in_nbytes_c = (input.shape()[3]) * (input.dtype() == DataType::BFLOAT16 ? 2 : 1);
    bool is_pow2 = (in_nbytes_c & (in_nbytes_c - 1)) == 0;
    TT_FATAL(is_pow2, "Row size (nchannels * bytes = {}) should be power of 2 ({}).", in_nbytes_c, is_pow2);

    TT_FATAL(2 * pad_h_ < kernel_size_h_ && 2 * pad_w_ < kernel_size_w_,
              "Total padding along a dim should be less than kernel/window size along same dim");
    TT_FATAL(out_w_ % nblocks_ == 0, "Make sure out_w is divisible by nblocks for now.");

    if (input.memory_config().is_sharded()) {
        TT_FATAL(input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(this->use_multicore_);
    } else {
        TT_FATAL(input.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
    if (this->out_mem_config_.is_sharded()) {
        TT_FATAL(this->out_mem_config_.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(this->use_multicore_);
    } else {
        TT_FATAL(this->out_mem_config_.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<Shape> MaxPool::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    // NOTE: Only for RM
    // NOTE2: Assuming { N, 1, H * W, C }
    // NOTE3: Assuming output data type is same as input
    const auto& input = input_tensors.at(0);
    const auto input_shape = input.shape().without_padding();
    // confirm that the output size supplied to the function matches
    TT_ASSERT(out_h_ == ((in_h_ + 2 * pad_h_ - (dilation_h_ * kernel_size_h_ - 1) - 1) / stride_h_) + 1);
    TT_ASSERT(out_w_ == ((in_w_ + 2 * pad_w_ - (dilation_w_ * kernel_size_w_ - 1) - 1) / stride_w_) + 1);
    uint32_t out_h = out_h_;
    uint32_t out_w = out_w_;
    // need to pad the last dim to TILE_WIDTH
    uint32_t out_c = input_shape[3];
    uint32_t out_c_padded = ceil_multiple_of(out_c, constants::TILE_WIDTH);
    uint32_t out_pagesize = out_c_padded * datum_size(datatype_to_dataformat_converter(input.dtype()));
    uint32_t out_hw = out_h * out_w;
    uint32_t out_hw_padded = (uint32_t) ceil_multiple_of(out_hw, constants::TILE_HEIGHT);

    // {N, 1, H * W, C}
    const auto out_dims = std::vector<uint32_t>({ in_n_, 1, out_hw, out_c });
    const auto padding = Padding({{0, 0},
                                  {0, 0},
                                  {0, out_hw_padded - out_hw},
                                  {0, out_c_padded - out_c}},
                                 Padding::PadValue::NegativeInfinity);

    auto out_shape = Shape{out_dims, padding};

    return {out_shape};
}

std::vector<Tensor> MaxPool::create_output_tensors(const std::vector<Tensor> &inputs) const {
    const auto& input = inputs.at(0);
    if (this->out_mem_config_.is_sharded()) {
        Shape output_shape = this->compute_output_shapes(inputs).at(0);
        uint32_t nbatch = in_n_;
        uint32_t out_hw = this->out_h_ * this->out_w_;
        uint32_t out_nhw = out_hw * nbatch;
        uint32_t ncores = max_pool_helpers::get_num_cores(input.device().compute_with_storage_grid_size(), out_nhw);
        uint32_t out_nhw_per_core = out_nhw / ncores;
        CoreRangeSet shard_grid = num_cores_to_corerange_set(ncores, input.device().compute_with_storage_grid_size(), true);
        std::array<uint32_t, 2> shard_shape = {out_nhw_per_core, input.shape()[-1]};
        auto shard_spec = ShardSpec{.shard_grid=shard_grid, .shard_shape=shard_shape, .shard_orientation=ShardOrientation::ROW_MAJOR, .halo = false};
        return {create_sharded_device_tensor(output_shape, input.dtype(), input.layout(), input.device(), this->out_mem_config_, shard_spec)};
    } else {
        return operation::generic_create_output_tensors(*this, inputs, input.dtype(), input.layout(), out_mem_config_);
    }
}

operation::ProgramWithCallbacks MaxPool::create_program(const std::vector<Tensor>& inputs, std::vector<Tensor> &outputs) const {
    const auto& input = inputs.at(0);
    auto& output = outputs.at(0);
    if (!use_multicore_) {
        return {max_pool_2d_single_core(input, output,
                                        in_h_, in_w_,
                                        out_h_, out_w_,
                                        kernel_size_h_, kernel_size_w_,
                                        stride_h_, stride_w_,
                                        pad_h_, pad_w_,
                                        dilation_h_, dilation_w_,
                                        out_mem_config_,
                                        nblocks_)};
    } else {
        if (input.memory_config().is_sharded()) {
            auto shard_spec = input.shard_spec().value();
            if (shard_spec.halo) {
                log_debug(LogOp, "Using sharded with halo");
                return {max_pool_2d_multi_core_sharded_with_halo(input, output,
                                            in_n_, in_h_, in_w_,
                                            out_h_, out_w_,
                                            kernel_size_h_, kernel_size_w_,
                                            stride_h_, stride_w_,
                                            pad_h_, pad_w_,
                                            dilation_h_, dilation_w_,
                                            out_mem_config_,
                                            nblocks_)};
            } else {
                log_debug(LogOp, "Using sharded");
                return {max_pool_2d_multi_core_generic(input, output,
                                                        in_h_, in_w_,
                                                        out_h_, out_w_,
                                                        kernel_size_h_, kernel_size_w_,
                                                        stride_h_, stride_w_,
                                                        pad_h_, pad_w_,
                                                        dilation_h_, dilation_w_,
                                                        out_mem_config_,
                                                        nblocks_)};
            }
        } else {
            log_debug(LogOp, "Using generic");
            return {max_pool_2d_multi_core_generic(input, output,
                                        in_h_, in_w_,
                                        out_h_, out_w_,
                                        kernel_size_h_, kernel_size_w_,
                                        stride_h_, stride_w_,
                                        pad_h_, pad_w_,
                                        dilation_h_, dilation_w_,
                                        out_mem_config_,
                                        nblocks_)};
        }
    }
}

Tensor max_pool2d(const Tensor &input,
                  uint32_t in_n, uint32_t in_h, uint32_t in_w,
                  uint32_t kernel_size_h, uint32_t kernel_size_w,
                  uint32_t stride_h, uint32_t stride_w,
                  uint32_t pad_h, uint32_t pad_w,
                  uint32_t dilation_h, uint32_t dilation_w,
                  const MemoryConfig& out_mem_config,
                  uint32_t nblocks,
                  bool use_multicore) {
    TT_ASSERT(dilation_h == 1 && dilation_w == 1 && "Dilation not yet supported in max_pool2d.");
    TT_ASSERT(pad_h < 2 && pad_w < 2 && "Padding > 1 not yet supported.");
    TT_ASSERT(stride_h == stride_w && "Stride should be equal for both H and W for now.");
    // calculate the H and W dims for output
    uint32_t out_h = ((in_h + 2 * pad_h - (dilation_h * kernel_size_h - 1) - 1) / stride_h) + 1;   // floor
    uint32_t out_w = ((in_w + 2 * pad_w - (dilation_w * kernel_size_w - 1) - 1) / stride_w) + 1;   // floor
    return operation::run_without_autoformat(MaxPool{in_n, in_h, in_w,
                                                     out_h, out_w,
                                                     kernel_size_h, kernel_size_w,
                                                     stride_h, stride_w,
                                                     pad_h, pad_w,
                                                     dilation_h, dilation_w,
                                                     out_mem_config,
                                                     nblocks,
                                                     use_multicore},
                                             {input}).at(0);
}

} // namespace tt_metal
} // namespace tt
