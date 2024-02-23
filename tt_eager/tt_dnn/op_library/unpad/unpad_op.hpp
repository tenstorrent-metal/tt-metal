// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/common/constants.hpp"

namespace tt {

namespace tt_metal {

enum class UnpadOpParallelizationStrategy { MULTI_CORE = 0, SINGLE_CORE = 1 };

uint32_t get_tiled_start_offset(const Tensor &input_tensor, const Shape &output_tensor_start);
uint32_t get_rm_start_offset(const Tensor &output_tensor, const Shape &output_tensor_start);

operation::ProgramWithCallbacks unpad_single_core(
    const Tensor &a, Tensor &output, const Shape &output_tensor_start, const Shape &output_tensor_end);

namespace unpad_impl::multi_core {

operation::ProgramWithCallbacks get_program(
    const Tensor &a, Tensor &output, const Shape &output_tensor_start, const Shape &output_tensor_end);

}
struct Unpad {
    const Shape output_tensor_start;
    const Shape output_tensor_end;
    const MemoryConfig output_mem_config;

    const Shape output_shape;
    const Shape input_shape;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    UnpadOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

Tensor unpad(
    const Tensor &input_tensor_a,
    const Shape &output_tensor_start,
    const Shape &output_tensor_end,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

struct UnpadOnHost {
    const Shape output_tensor_start;
    const Shape output_tensor_end;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> compute_output_tensors(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

Tensor unpad_on_host(
    const Tensor &input_tensor,
    const Shape &output_tensor_start,
    const Shape &output_tensor_end,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal

}  // namespace tt
