// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

enum class TilizeOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct Tilize {
    const MemoryConfig output_mem_config;
    const DataType output_dtype;
    const bool use_multicore;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    TilizeOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("output_mem_config", "output_dtype", "use_multicore");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->output_mem_config), std::cref(this->output_dtype), std::cref(this->use_multicore));
    }
};


enum class TilizeWithValPaddingOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct TilizeWithValPadding {
    const Shape output_tensor_shape;
    const Shape input_tensor_start;
    const float pad_value;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    TilizeWithValPaddingOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("output_tensor_shape", "input_tensor_start", "pad_value", "output_mem_config", "output_dtype");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->output_tensor_shape),
            std::cref(this->input_tensor_start),
            std::cref(this->pad_value),
            std::cref(this->output_mem_config),
            std::cref(this->output_dtype));
    }
};

operation::ProgramWithCallbacks tilize_multi_core(const Tensor &a, Tensor& output);
operation::ProgramWithCallbacks tilize_single_core(const Tensor &a, Tensor& output);
operation::ProgramWithCallbacks tilize_with_val_padding_multi_core(const Tensor &a, Tensor& output, const Shape &output_tensor_shape, const Shape &input_tensor_start, const float pad_value);
operation::ProgramWithCallbacks tilize_with_val_padding_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_shape, const Shape &input_tensor_start, const float pad_value);

Tensor tilize (const Tensor &a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt, bool use_multicore = false);
Tensor tilize_with_zero_padding (const Tensor &a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);
Tensor tilize_with_val_padding (const Tensor &a, const Shape &output_tensor_shape, const Shape &input_tensor_start, const float pad_value, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype=std::nullopt);

}  // namespace tt_metal

}  // namespace tt
