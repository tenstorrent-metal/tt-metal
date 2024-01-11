// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(1);
    const uint32_t sem_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_transfers = get_arg_val<uint32_t>(3);
    const uint32_t num_full_chunks = get_arg_val<uint32_t>(4);
    const uint32_t page_size = get_arg_val<uint32_t>(5);
    const uint32_t num_pages = get_arg_val<uint32_t>(6);
    const uint32_t num_bytes = get_arg_val<uint32_t>(7);
    const uint32_t rem_num_pages = get_arg_val<uint32_t>(8);
    const uint32_t rem_num_bytes = get_arg_val<uint32_t>(9);
    const uint32_t out_page_size = get_arg_val<uint32_t>(10);


    constexpr uint32_t sender_noc_x = get_compile_time_arg_val(0);
    constexpr uint32_t sender_noc_y = get_compile_time_arg_val(1);

    constexpr bool dst_is_dram = get_compile_time_arg_val(2) == 1;

    const InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr, .page_size = out_page_size};

    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    const uint64_t sender_semaphore_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sem_addr);
    volatile tt_l1_ptr uint32_t * start_idx_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_eth_l1_src_addr);

    const auto& get_and_sync_data = [&](const uint32_t num_pages, const uint32_t num_bytes) {
        for (uint32_t i = 0; i < num_transfers; ++i) {
            eth_wait_for_bytes(num_bytes);
            noc_semaphore_inc(sender_semaphore_noc_addr, 1);
            uint32_t start_idx = *start_idx_addr;
            uint32_t start_offset = *(start_idx_addr + 1);
            uint32_t local_eth_l1_curr_src_addr = local_eth_l1_src_addr + 32;
            for (uint32_t curr_idx = start_idx; curr_idx < start_idx + num_pages; ++curr_idx) {
                uint64_t dst_noc_addr = get_noc_addr(curr_idx, d, start_offset);
                noc_async_write(local_eth_l1_curr_src_addr, dst_noc_addr, page_size);
                local_eth_l1_curr_src_addr += page_size;
            }
            eth_noc_async_write_barrier();
            eth_noc_semaphore_wait(receiver_semaphore_addr_ptr, 1);
            noc_semaphore_set(receiver_semaphore_addr_ptr, 0);
            eth_receiver_done();
        }
    };

    for (uint32_t i = 0; i < num_full_chunks; ++i) {
        get_and_sync_data(num_pages, num_bytes);
    }
    if (rem_num_pages > 0) {
        get_and_sync_data(rem_num_pages, rem_num_bytes);
    }
}
