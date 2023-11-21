/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <optional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

/*
 * dot product
 */
operation::ProgramWithCallbacks moreh_dot_single_core(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor);

struct MorehDot {
    const MemoryConfig output_mem_config;
    const DataType output_dtype;  // TODO: Uplift output_dtype as an option for general dot/bmm
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline Tensor moreh_dot(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return operation::run(
               MorehDot{.output_mem_config = mem_config, .output_dtype = input_tensor_a.dtype()},
               {input_tensor_a, input_tensor_b})
        .at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
