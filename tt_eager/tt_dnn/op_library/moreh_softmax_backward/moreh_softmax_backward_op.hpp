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

enum class MorehSoftmaxBackwardOpParallelizationStrategy {
    SMALL_W = 0,
    SMALL_H = 1,
    LARGE_W = 2,
    LARGE_H = 3,
    LARGE_C = 4
};

bool is_moreh_softmax_backward_w_small_available(const Tensor &tensor);
bool is_moreh_softmax_backward_h_small_available(const Tensor &tensor);

operation::ProgramWithCallbacks moreh_softmax_backward_w_small(const Tensor &output, const Tensor &output_grad, Tensor& input_grad, const CoreRange core_range);
operation::ProgramWithCallbacks moreh_softmax_backward_w_large(const Tensor &output, const Tensor &output_grad, Tensor& input_grad, const CoreRange core_range);
operation::ProgramWithCallbacks moreh_softmax_backward_h_small(const Tensor &output, const Tensor &output_grad, Tensor& input_grad, const CoreRange core_range);
operation::ProgramWithCallbacks moreh_softmax_backward_h_large(const Tensor &output, const Tensor &output_grad, Tensor& input_grad, const CoreRange core_range);
operation::ProgramWithCallbacks moreh_softmax_backward_c_large(const Tensor &output, const Tensor &output_grad, Tensor& input_grad, uint32_t dim, const CoreRange core_range);

struct MorehSoftmaxBackward {
    const uint32_t dim;
    const MemoryConfig output_mem_config;
    const CoreRange core_range; // unused for now

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    MorehSoftmaxBackwardOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

// const ref prevents
Tensor moreh_softmax_backward(const Tensor& output_tensor, const Tensor& output_grad_tensor, uint32_t dim, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary
}  // namespace operations
}  // namespace tt
