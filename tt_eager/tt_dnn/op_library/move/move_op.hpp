/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/detail/tt_metal.hpp"
using namespace tt::constants;

namespace move_op_utils {
using namespace tt::tt_metal;

bool can_deallocate(const Tensor &input_tensor);

} // namespace move_op_utils

namespace tt {

namespace tt_metal {

enum class MoveOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1, MULTI_CORE_OVERLAP = 2
};

struct Move {
    const MemoryConfig output_mem_config;
    const MoveOpParallelizationStrategy move_op_parallelization_strategy;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    MoveOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

operation::ProgramWithCallbacks move_multi_core(const Tensor &input, Tensor &output);
operation::ProgramWithCallbacks move_multi_core_with_overlap(const Tensor &input, Tensor &output);
operation::ProgramWithCallbacks move_single_core(const Tensor &input, Tensor &output);

inline Tensor move(Tensor& input_tensor, std::optional<MemoryConfig>& mem_config) {
    TT_ASSERT(input_tensor.is_allocated(), "Expected input tensor to be allocated");
    auto input_mem_config = input_tensor.memory_config();
    auto input_address = input_tensor.buffer()->address();
    auto output_mem_config = mem_config.value_or(input_mem_config);

    if (not move_op_utils::can_deallocate(input_tensor)) {
        // TODO: Should this throw error?
        return input_tensor;
    }

    DeallocateBuffer(*input_tensor.buffer());
    auto output_tensor = create_device_tensor(input_tensor.shape(), input_tensor.dtype(), input_tensor.layout(), input_tensor.device(), output_mem_config);

    bool tilized = input_tensor.layout() == Layout::TILE;

    // get_parallelization_strategy
    uint32_t num_units = tilized ? input_tensor.volume() / TILE_HW : input_tensor.volume() / input_tensor.shape()[-1];

    bool move_within_same_mem_space = input_mem_config.buffer_type == output_mem_config.buffer_type;
    // Input and output addresses won't overlap if they are in different memory substrates
    bool non_overlap = not move_within_same_mem_space;
    const auto num_banks = input_tensor.device().num_banks(output_tensor.buffer()->buffer_type());
    uint32_t size_per_bank = tt_metal::detail::SizeBytesPerBank(output_tensor.buffer()->size(), output_tensor.buffer()->page_size(), num_banks);

    // If input and output buffers overlap, input has to be copied into circular buffer before writing to output
    // Only compute with storage cores allow CBs to be created
    auto compute_with_storage_grid_size = input_tensor.device().compute_with_storage_grid_size();
    const auto num_l1_banks = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    uint32_t size_per_l1_bank = tt_metal::detail::SizeBytesPerBank(output_tensor.buffer()->size(), output_tensor.buffer()->page_size(), num_l1_banks);

    if (move_within_same_mem_space) {
        switch (input_mem_config.buffer_type) {
            // If DRAM, inverse logic because memory is allocated bottom up
            case tt_metal::BufferType::DRAM: {
                non_overlap = output_tensor.buffer()->address() + size_per_bank <= input_address;
            }
            break;
            case tt_metal::BufferType::L1: {
                non_overlap = input_address + size_per_bank <= output_tensor.buffer()->address();
            }
            break;
            default: break;
        }
    }

    bool fits_in_cb = (L1_UNRESERVED_BASE + size_per_l1_bank) <= (output_mem_config.buffer_type == tt_metal::BufferType::L1 ? output_tensor.buffer()->address() : output_tensor.device().l1_size_per_core());

    MoveOpParallelizationStrategy move_op_parallelization_strategy = MoveOpParallelizationStrategy::SINGLE_CORE;
    if (num_units > 1 and non_overlap) {
        move_op_parallelization_strategy = MoveOpParallelizationStrategy::MULTI_CORE;
    } else if (num_units > 1 and (not non_overlap) and fits_in_cb and compute_with_storage_grid_size.x > 1 and compute_with_storage_grid_size.y > 1) {
        move_op_parallelization_strategy = MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP;
    }

    auto output = operation::run(Move{output_mem_config, move_op_parallelization_strategy}, {input_tensor, output_tensor}).at(0);
    input_tensor.deallocate();
    return output;
}

}  // namespace tt_metal

}  // namespace tt
