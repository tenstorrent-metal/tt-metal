// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::tt_metal;
using DistributedTensor = std::vector<Tensor>;
namespace tt {

namespace tt_metal {

struct AllGather {
    const uint32_t dim;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramsWithCallbacks create_programs(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

operation::ProgramsWithCallbacks all_gather_multi_core(const DistributedTensor & input_tensors, std::vector<Tensor> &output_tensors, uint32_t dim);

inline std::vector<Tensor> all_gather(const DistributedTensor &input_tensors, uint32_t dim, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return operation::run(AllGather{dim, output_mem_config}, input_tensors);
}

}  // namespace tt_metal

}  // namespace tt
