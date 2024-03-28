/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace operations{

namespace primary{

using namespace constants;
using namespace tt_metal;
/*
 * prod product
 */

struct Prod_op {
    const MemoryConfig output_mem_config;
    const DataType output_dtype;  // TODO: Uplift output_dtype as an option for general dot/bmm
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names =
        std::make_tuple("output_mem_config", "output_dtype");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->output_mem_config), std::cref(this->output_dtype));
    }
};

operation::ProgramWithCallbacks prod_single_core(const Tensor &input_tensor_a, const Tensor &output_tensor);

Tensor prod_all(
    const Tensor &input,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
}

}
}
