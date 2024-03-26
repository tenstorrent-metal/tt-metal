// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt::tt_metal {

enum class ScanOpParallelizationStrategy { SHARDED_MULTI_CORE = 0 };

enum class ScanOpDirection { ROWS, COLS, ROWS_REVERSED, COLS_REVERSED };

struct Scan {
    ScanOpDirection direction = ScanOpDirection::COLS_REVERSED;
    uint32_t n_tile_columns;

    void validate(const std::vector<Tensor> &input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
        return {input_tensors.at(0).get_legacy_shape()};  // In-place
    }

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const {
        return {input_tensors.at(0)};  // In-place
    }

    ScanOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
        return ScanOpParallelizationStrategy::SHARDED_MULTI_CORE;
    }

    static constexpr auto attribute_names = std::make_tuple("direction", "n_tile_columns");

    const auto attribute_values() const { return std::make_tuple(direction, n_tile_columns); }
};

Tensor scan(const Tensor &a);

}  // namespace tt::tt_metal
