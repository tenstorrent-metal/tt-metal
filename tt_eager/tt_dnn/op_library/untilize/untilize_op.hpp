/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"

namespace tt {

namespace tt_metal {

enum class UntilizeOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct Untilize {
    const MemoryConfig output_mem_config;
    const bool use_multicore;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    UntilizeOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("output_mem_config", "use_multicore");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->output_mem_config), std::cref(this->use_multicore));
    }
};

enum class UntilizeWithUnpaddingOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct UntilizeWithUnpadding {
    const Shape output_tensor_start;
    const Shape output_tensor_end;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    UntilizeWithUnpaddingOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("output_tensor_start", "output_tensor_end", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->output_tensor_start), std::cref(this->output_tensor_end), std::cref(this->output_mem_config));
    }
};

operation::ProgramWithCallbacks untilize_multi_core(const Tensor &a, Tensor& output);
operation::ProgramWithCallbacks untilize_single_core(const Tensor &a, Tensor& output);
operation::ProgramWithCallbacks untilize_with_unpadding_multi_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end);
operation::ProgramWithCallbacks untilize_with_unpadding_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end);

Tensor untilize (const Tensor &a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, bool use_multicore = false);
Tensor untilize_with_unpadding(const Tensor &a, const Shape &output_tensor_start, const Shape &output_tensor_end, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// NOTE: UntilizeWithHalo is only for sharded input/output
struct UntilizeWithHalo {
    const uint32_t pad_val_;
    const uint32_t in_b;
    const uint32_t in_h;
    const uint32_t in_w;
    const int32_t max_out_nsticks_per_core_;
    const uint32_t stride_;
    const PoolConfig pc_;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "pad_val", "in_b", "in_h", "in_w", "out_shard_size_max_per_core", "stride", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->pad_val_),
            std::cref(this->in_b),
            std::cref(this->in_h),
            std::cref(this->in_w),
            std::cref(this->max_out_nsticks_per_core_),
            std::cref(this->stride_),
            std::cref(this->output_mem_config));
    }
};
Tensor untilize_with_halo(const Tensor &a, const uint32_t pad_val, const uint32_t &in_b, const uint32_t &in_h, const uint32_t &in_w, const uint32_t stride = 1, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

struct UntilizeWithHaloReaderConfigs{
    std::vector<uint16_t> local_data_start_size;
    uint16_t local_data_write_offset;
    std::vector<uint16_t> pad_start_size;
    std::vector<uint16_t> left_left_send_configs;
    std::vector<uint16_t> left_send_configs;
    std::vector<uint16_t> right_send_configs;
    std::vector<uint16_t> right_right_send_configs;
};
std::vector<UntilizeWithHaloReaderConfigs> get_untilize_with_halo_reader_configs(std::vector<uint16_t> data_indices,
                                                                                std::vector<uint16_t> data_start_size,
                                                                                std::vector<uint16_t> pad_start_size,
                                                                                const uint32_t &num_cores,

                                                                                const uint32_t &in_b,
                                                                                const uint32_t &in_h,
                                                                                const uint32_t &in_w,
                                                                                const uint32_t& pad_h,
                                                                                const uint32_t& pad_w,
                                                                                const uint32_t& window_h,
                                                                                const uint32_t& window_w,
                                                                                const uint32_t& stride_h,
                                                                                const uint32_t& stride_w);
namespace untilize_helpers {
uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks);
}

}  // namespace tt_metal

}  // namespace tt
