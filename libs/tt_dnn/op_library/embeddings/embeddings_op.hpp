#pragma once

#include <optional>

#include "libs/tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::constants;


namespace tt {

namespace tt_metal {

struct Embeddings {

    const uint32_t num_embeddings;
    const uint32_t embedding_dim;
    const MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    operation::Hash compute_program_hash(
        const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;

};

inline Tensor embeddings( const uint32_t & num_embeddings, const uint32_t & embedding_dim, const Tensor &input_tensor, const Tensor &weights, const MemoryConfig& mem_config){
    return operation::run_without_autoformat(Embeddings{.num_embeddings=num_embeddings, .embedding_dim=embedding_dim, .output_mem_config=mem_config}, {input_tensor, weights}).at(0);

}


}
} // namespace tt::tt_metal
