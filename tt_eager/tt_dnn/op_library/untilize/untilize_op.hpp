// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"

namespace tt {

namespace tt_metal {

#define MAX_PACK_UNTILIZE_WIDTH 8       // pack untilize currently does not support > 8 width

enum class UntilizeOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct Untilize {
    const MemoryConfig output_mem_config;
    const bool use_multicore;
    const bool use_pack_untilize;

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

operation::ProgramWithCallbacks untilize_multi_core(const Tensor &a, Tensor& output, bool use_pack_untilize = true);
operation::ProgramWithCallbacks untilize_single_core(const Tensor &a, Tensor& output, bool use_pack_untilize = true);
operation::ProgramWithCallbacks untilize_with_unpadding_multi_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end);
operation::ProgramWithCallbacks untilize_with_unpadding_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end);

Tensor untilize (const Tensor &a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, bool use_multicore = true, bool use_pack_untilize = true);
Tensor untilize_with_unpadding(const Tensor &a, const Shape &output_tensor_start, const Shape &output_tensor_end, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

struct UntilizeWithHaloV2 {
    const uint32_t pad_val_;
    const uint32_t ncores_nhw_;
    const uint32_t max_out_nsticks_per_core_;
    const std::vector<int32_t> local_pad_nsegments_per_core_;
    const std::vector<int32_t> ll_data_nsegments_per_core_;
    const std::vector<int32_t> l_data_nsegments_per_core_;
    const std::vector<int32_t> local_data_nsegments_per_core_;
    const std::vector<int32_t> r_data_nsegments_per_core_;
    const std::vector<int32_t> rr_data_nsegments_per_core_;
    const std::vector<int32_t> local_data_src_start_offsets_per_core_;
    const std::vector<int32_t> ll_data_src_start_offsets_per_core_;
    const std::vector<int32_t> l_data_src_start_offsets_per_core_;
    const std::vector<int32_t> r_data_src_start_offsets_per_core_;
    const std::vector<int32_t> rr_data_src_start_offsets_per_core_;
    const MemoryConfig out_mem_config_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "pad_val_",
        "ncores_nhw_",
        "max_out_nsticks_per_core_",
        "local_pad_nsegments_per_core_",
        "ll_data_nsegments_per_core_",
        "l_data_nsegments_per_core_",
        "local_data_nsegments_per_core_",
        "r_data_nsegments_per_core_",
        "rr_data_nsegments_per_core_",
        "out_mem_config_");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(pad_val_),
            std::cref(ncores_nhw_),
            std::cref(max_out_nsticks_per_core_),
            std::cref(local_pad_nsegments_per_core_),
            std::cref(ll_data_nsegments_per_core_),
            std::cref(l_data_nsegments_per_core_),
            std::cref(local_data_nsegments_per_core_),
            std::cref(r_data_nsegments_per_core_),
            std::cref(rr_data_nsegments_per_core_),
            std::cref(out_mem_config_));
    }
};
Tensor untilize_with_halo(const Tensor& input_tensor,
                             const Tensor& local_pad_start_and_size,
                             const Tensor& ll_data_start_and_size,
                             const Tensor& l_data_start_and_size,
                             const Tensor& local_data_start_and_size,
                             const Tensor& r_data_start_and_size,
                             const Tensor& rr_data_start_and_size,
                             const uint32_t pad_val,
                             const uint32_t ncores_nhw,
                             const uint32_t max_out_nsticks_per_core,
                             const std::vector<int32_t>& local_pad_nsegments_per_core,
                             const std::vector<int32_t>& ll_data_nsegments_per_core,
                             const std::vector<int32_t>& l_data_nsegments_per_core,
                             const std::vector<int32_t>& local_data_nsegments_per_core,
                             const std::vector<int32_t>& r_data_nsegments_per_core,
                             const std::vector<int32_t>& rr_data_nsegments_per_core,
                             const std::vector<int32_t>& local_data_src_start_offsets_per_core,
                             const std::vector<int32_t>& ll_data_src_start_offsets_per_core,
                             const std::vector<int32_t>& l_data_src_start_offsets_per_core,
                             const std::vector<int32_t>& r_data_src_start_offsets_per_core,
                             const std::vector<int32_t>& rr_data_src_start_offsets_per_core,
                             const MemoryConfig& mem_config);

namespace untilize_helpers {

uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks);

}
}  // namespace tt_metal
}  // namespace tt
