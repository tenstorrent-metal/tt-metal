// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
// #include <type_traits>
#include <vector>
#include <limits>

// #include "tt_dnn/op_library/ccl/ccl_common.hpp"

namespace ccl {

// TODO: let the kernel runtime args

enum ShardType : uint8_t { Width = 0, Height = 1, Block = 2 };

/*
 * Worker coordinate, used by EDM and some kernels to know which workers to signal
 */
struct WorkerXY {
    uint16_t x;
    uint16_t y;

    WorkerXY(uint16_t x, uint16_t y) : x(x), y(y) {}

    uint32_t to_uint32() const {
        return (y << 16) | x;
    }

    bool operator==(const WorkerXY& rhs) const {
        return x == rhs.x && y == rhs.y;
    }
    bool operator!=(const WorkerXY& rhs) const {
        return !(*this == rhs);
    }
};

template <bool T>
struct ArchDependentTypes;

template <>
struct ArchDependentTypes<true> {
    using workers_list_t = std::vector<ccl::WorkerXY>;
};

template <>
struct ArchDependentTypes<false> {
    using workers_list_t = ccl::WorkerXY*;
};


template <bool IS_HOST>
struct ShardAddrGenArgs final {
    static constexpr uint32_t UNINITIALIZED_VALUE_U32 = std::numeric_limits<uint32_t>::max();
    static constexpr uint16_t UNINITIALIZED_VALUE_U16 = std::numeric_limits<uint16_t>::max();

    uint32_t shards_start_address = UNINITIALIZED_VALUE_U32;
    uint32_t shard_size_in_bytes = UNINITIALIZED_VALUE_U32;
    uint16_t total_chunks_per_core = UNINITIALIZED_VALUE_U16;

    uint16_t starting_core_index = UNINITIALIZED_VALUE_U16;
    uint16_t starting_chunk_into_shard = UNINITIALIZED_VALUE_U16;

    uint16_t intra_core_stride_in_shards = UNINITIALIZED_VALUE_U16;
    uint16_t contiguous_chunks_before_stride = UNINITIALIZED_VALUE_U16;

    uint16_t num_dest_cores = UNINITIALIZED_VALUE_U16;

    typename ArchDependentTypes<IS_HOST>::workers_list_t dest_cores;
    bool is_clockwise = false;

    inline uint32_t get_expected_num_args() const {
        if constexpr (IS_HOST) {
            return 9 + dest_cores.size();
        } else {
            return 9 + this->num_dest_cores;
        }
    }
};

namespace all_gather {
inline void addr_gen_advance_width_sharded(
    uint16_t& curr_core_chunk_index,
    uint16_t& curr_worker_index,
    uint16_t& contiguous_chunk_count,
    // uint16_t& current_core_chunks_visited,
    const uint16_t& total_chunks_per_core,
    const uint16_t& num_dest_cores,
    const uint16_t& intra_core_stride_in_shards,
    const uint16_t& contiguous_chunks_before_stride,
    bool is_clockwise
) {
    if (is_clockwise) {
        bool do_stride = contiguous_chunk_count == contiguous_chunks_before_stride;
        bool stride_induced_chunk_wraparound = (do_stride && curr_core_chunk_index < intra_core_stride_in_shards);
        bool do_chunk_wrap = curr_core_chunk_index == 0 || stride_induced_chunk_wraparound;

        // current_core_chunks_visited++;
        if (do_chunk_wrap) {
            bool do_core_wrap = curr_worker_index == 0;
            curr_core_chunk_index = total_chunks_per_core - (intra_core_stride_in_shards - curr_core_chunk_index);
            contiguous_chunk_count = 1;
            if (do_core_wrap) {
                curr_worker_index = num_dest_cores - 1;
                // current_core_chunks_visited=0;
            } else {
                curr_worker_index--;
            }
        } else {

            if (do_stride) {
                contiguous_chunk_count = 1;
                // TT_ASSERT(curr_core_chunk_index >= intra_core_stride_in_shards);
                curr_core_chunk_index -= intra_core_stride_in_shards;
            } else {
                contiguous_chunk_count++;
                curr_core_chunk_index--;
            }
        }

    } else {
        // current_core_chunks_visited++;
        if (contiguous_chunk_count == contiguous_chunks_before_stride) {
            contiguous_chunk_count = 1;
            // TT_ASSERT(curr_core_chunk_index >= intra_core_stride_in_shards);
            curr_core_chunk_index += intra_core_stride_in_shards;
        } else {
            contiguous_chunk_count++;
            curr_core_chunk_index++;
        }

        bool do_chunk_wrap = curr_core_chunk_index >= total_chunks_per_core;
        if (do_chunk_wrap) {
            // current_core_chunks_visited = 0;
            curr_core_chunk_index = curr_core_chunk_index - total_chunks_per_core;
            curr_worker_index++;
            bool do_core_wrap = curr_worker_index == num_dest_cores;
            if (do_core_wrap) {
                curr_worker_index = 0;
            }
        }
    }
}

}; // namespace all_gather

}  // namespace ccl
