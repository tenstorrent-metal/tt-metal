// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

struct MorehLayerNorm {
    uint32_t normalized_dims;
    float eps;
    tt_metal::MemoryConfig output_mem_config;

    void validate(
        const std::vector<tt_metal::Tensor> &input_tensors,
        const std::vector<std::optional<const tt_metal::Tensor>> &optional_input_tensors) const;
    std::vector<tt_metal::Shape> compute_output_shapes(const std::vector<tt_metal::Tensor> &input_tensors) const;
    std::vector<tt_metal::Tensor> create_output_tensors(const std::vector<tt_metal::Tensor> &input_tensors) const;
    tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<tt_metal::Tensor> &input_tensors,
        const std::vector<std::optional<const tt_metal::Tensor>> &optional_input_tensors,
        std::vector<tt_metal::Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

tt_metal::operation::ProgramWithCallbacks moreh_layernorm_impl(
    const tt_metal::Tensor &input,
    uint32_t normalized_dims,
    float eps,
    tt_metal::Tensor &output,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> gamma,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> beta,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> mean,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> rstd);

tt_metal::Tensor moreh_layernorm(
    const tt_metal::Tensor &input,
    uint32_t normalized_dims,
    float eps,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> gamma = std::nullopt,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> beta = std::nullopt,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> mean = std::nullopt,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> rstd = std::nullopt,
    const tt_metal::MemoryConfig &output_mem_config = tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

namespace tt_metal {

Tensor moreh_layernorm(
    const Tensor &input,
    uint32_t normalized_dims,
    float eps,
    std::optional<std::reference_wrapper<const Tensor>> gamma = std::nullopt,
    std::optional<std::reference_wrapper<const Tensor>> beta = std::nullopt,
    std::optional<std::reference_wrapper<const Tensor>> mean = std::nullopt,
    std::optional<std::reference_wrapper<const Tensor>> rstd = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal

}  // namespace tt
