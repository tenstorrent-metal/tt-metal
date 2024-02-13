// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/eth_ring_gather_utils.hpp"

void kernel_main() {
    constexpr uint32_t num_transfers = get_compile_time_arg_val(0);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t rem_num_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_channels = get_compile_time_arg_val(4);
    constexpr uint32_t buffer_size = get_compile_time_arg_val(5);
    constexpr uint32_t local_l1_start_addr = get_compile_time_arg_val(6);
    constexpr uint32_t num_workers_per_buffer = get_compile_time_arg_val(7);
    constexpr uint32_t worker_semaphore_addr = get_compile_time_arg_val(8);

    uint64_t remote_worker_semaphore_addrs[num_channels][num_workers_per_buffer];
    for (uint32_t c = 0, i = 0; c < num_channels; ++c) {
        for (uint32_t w = 0; w < num_workers_per_buffer; ++w) {
            remote_worker_semaphore_addrs[c][w] = get_noc_addr(get_arg_val<uint32_t>(i), get_arg_val<uint32_t>(i+1), worker_semaphore_addr);
            i+=2;
        }
    }
    constexpr uint32_t semaphore_arg_start_idx = 2 * num_channels * num_workers_per_buffer;
    volatile tt_l1_ptr uint32_t* local_worker_ack_semaphore_ptrs[num_channels];
    // Initialize the semaphore ourselves in the kernel instead of from host
    for (uint32_t i = 0; i < num_channels; ++i) {
        local_worker_ack_semaphore_ptrs[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(semaphore_arg_start_idx + i));
    }

    uint8_t channel = 0;

    // We are storing 1 in worker_ack_semaphore_ptrs to send as an ack
    // This is also where the workers increment to show that they've finished reading
    // so we expect num_workers + 1 to be read
    for (uint32_t i = 0; i < num_transfers; ++i) {
        if constexpr (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                eth_wait_for_bytes(num_bytes, channel);
                eth_receiver_acknowledge(channel);
                // TODO: Debug why multicast didn't work
                for (uint32_t w = 0; w < num_workers_per_buffer; ++w) {
                    noc_semaphore_inc(remote_worker_semaphore_addrs[channel][w], 1);
                }
                eth_noc_semaphore_wait(local_worker_ack_semaphore_ptrs[channel], num_workers_per_buffer); // TODO: Poll other channels in the meantime to enable parallelism
                noc_semaphore_set(local_worker_ack_semaphore_ptrs[channel], 0);
                eth_receiver_done(channel);
                channel = get_next_buffer_channel_pointer<num_channels>(channel);
            }
        }
        if constexpr (rem_num_bytes > 0) {
            eth_wait_for_bytes(rem_num_bytes, channel);
            eth_receiver_acknowledge(channel);
            // TODO: Debug why multicast didn't work
            for (uint32_t w = 0; w < num_workers_per_buffer; ++w) {
                noc_semaphore_inc(remote_worker_semaphore_addrs[channel][w], 1);
            }
            eth_noc_semaphore_wait(local_worker_ack_semaphore_ptrs[channel], num_workers_per_buffer);
            noc_semaphore_set(local_worker_ack_semaphore_ptrs[channel], 0);
            eth_receiver_done(channel);
            channel = get_next_buffer_channel_pointer<num_channels>(channel);
        }
    }
}
