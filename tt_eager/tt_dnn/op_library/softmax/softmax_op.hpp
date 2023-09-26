/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_eager/tensor/tensor.hpp"

#include "tt_dnn/op_library/operation.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

struct Softmax {
    const std::optional<float> scale;
    uint32_t dim;
    bool in_place;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    tt::stl::reflection::Attributes attributes() const;
};

// const ref prevents in-place
Tensor softmax_in_place(Tensor& input_tensor);

namespace transformers {
// computes
// tmp1 = bcast_hw_mul(scale, x)  ; shape of scale is [1,1,32,32]
// tmp2 = bcast_add_w->h(tmp1, mask) ; shape of attn mask is [1,N,32,W]
// y = softmax(tmp2)              ; r=result
// If scale == 1.0f then just y = softmax(x) is computed
Tensor scale_mask_softmax_in_place(Tensor& input_tensor, std::optional<float> scale = std::nullopt, std::optional<const Tensor> mask = std::nullopt);
}  // namespace transformers

}  // namespace primary
}  // namespace operations

namespace tt_metal {
    Tensor softmax(const Tensor& input_tensor, const std::int64_t dim, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
    Tensor scale_mask_softmax(const Tensor& input_tensor, const std::int64_t dim, std::optional<float> scale = std::nullopt, std::optional<const Tensor> mask = std::nullopt, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
}  // namespace tt_metal
}  // namespace tt
