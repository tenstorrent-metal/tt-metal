// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(2);
    const uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(3);
    const uint32_t sem_addr = get_arg_val<uint32_t>(4);
    const uint32_t num_transfers = get_arg_val<uint32_t>(5);
    const uint32_t num_full_chunks = get_arg_val<uint32_t>(6);
    const uint32_t page_size = get_arg_val<uint32_t>(7);
    const uint32_t num_pages = get_arg_val<uint32_t>(8);
    const uint32_t num_bytes = get_arg_val<uint32_t>(9);
    const uint32_t rem_num_pages = get_arg_val<uint32_t>(10);
    const uint32_t rem_num_bytes = get_arg_val<uint32_t>(11);
    const uint32_t global_start_idx = get_arg_val<uint32_t>(12);
    const uint32_t offset = get_arg_val<uint32_t>(13);
    const uint32_t num_rows = get_arg_val<uint32_t>(14);
    const uint32_t num_cols = get_arg_val<uint32_t>(15);

    constexpr uint32_t receiver_noc_x = get_compile_time_arg_val(0);
    constexpr uint32_t receiver_noc_y = get_compile_time_arg_val(1);

    constexpr bool src_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr DataFormat df = static_cast<DataFormat>(get_compile_time_arg_val(4));

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = page_size,
        .data_format = df
    };

    const InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr,
        .page_size = page_size,
        .data_format = df
    };

    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    const uint64_t receiver_semaphore_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, sem_addr);
    const uint64_t receiver_data_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, remote_eth_l1_dst_addr);

    volatile tt_l1_ptr uint32_t * start_idx_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_eth_l1_src_addr);

    const auto& get_and_send_data = [&](const uint32_t page_idx, uint32_t& curr_idx, uint32_t& col_idx, const uint32_t num_pages, const uint32_t num_bytes, const uint32_t num_bytes_per_send, const uint32_t num_bytes_per_send_word_size) {
        uint32_t local_eth_l1_curr_src_addr = local_eth_l1_src_addr + 32;
        for (uint32_t p_idx = page_idx; p_idx < page_idx + num_pages; ++p_idx) {
            noc_async_read_tile(p_idx, s, local_eth_l1_curr_src_addr);
            local_eth_l1_curr_src_addr += page_size;
        }
        eth_noc_async_read_barrier();

        local_eth_l1_curr_src_addr = local_eth_l1_src_addr + 32;

        *(start_idx_addr) = curr_idx;
        *(start_idx_addr + 1) = col_idx;

        for (uint32_t i = 0; i < num_pages; ++i) {
            noc_async_write_tile(curr_idx, d, local_eth_l1_curr_src_addr);
            local_eth_l1_curr_src_addr += page_size;
            curr_idx++;
            col_idx++;
            if (col_idx == num_cols) {
                curr_idx += offset;
                col_idx = 0;
            }
        }

        eth_noc_async_write_barrier();
        // num_transfers = num_devices - 1
        for (uint32_t i = 0; i < num_transfers; ++i) {
            eth_send_bytes(local_eth_l1_src_addr, remote_eth_l1_dst_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);
            eth_wait_for_remote_receiver_done_and_get_local_receiver_data(
                sender_semaphore_addr_ptr,
                receiver_semaphore_noc_addr,
                receiver_data_noc_addr,
                local_eth_l1_src_addr,
                num_bytes
            );
        }

    };

    // TODO: Are these necessary?
    const uint32_t num_bytes_per_send = 16;
    const uint32_t num_bytes_per_send_word_size = num_bytes_per_send >> 4;

    uint32_t page_idx = 0;
    uint32_t curr_idx = global_start_idx;
    uint32_t col_idx = 0;

    // How many chunks we split our local device data into
    for (uint32_t i = 0; i < num_full_chunks; ++i) {
        get_and_send_data(page_idx, curr_idx, col_idx, num_pages, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);
        page_idx += num_pages;
    }

    if (rem_num_pages > 0) {
        // TODO: Are these necessary?
        const uint32_t rem_num_bytes_per_send = 16;
        const uint32_t rem_num_bytes_per_send_word_size = rem_num_bytes_per_send >> 4;
        get_and_send_data(page_idx, curr_idx, col_idx, rem_num_pages, rem_num_bytes, rem_num_bytes_per_send, rem_num_bytes_per_send_word_size);
    }
}
