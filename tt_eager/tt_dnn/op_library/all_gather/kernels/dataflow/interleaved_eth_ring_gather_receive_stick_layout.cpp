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

template<bool DRAM>
FORCE_INLINE void write_chunk(uint32_t& output_page_idx, uint32_t& row_idx, uint32_t local_eth_l1_curr_src_addr, const InterleavedAddrGen<DRAM>& d, const uint32_t num_rows, const uint32_t row_offset, const uint32_t num_pages, const uint32_t page_size) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(local_eth_l1_curr_src_addr, dst_noc_addr, page_size);
        local_eth_l1_curr_src_addr += page_size;
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
    }
    eth_noc_async_write_barrier();
}

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t buffer0 = get_arg_val<uint32_t>(1);
    const uint32_t buffer1 = get_arg_val<uint32_t>(2);
    const uint32_t sem_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t sender_noc_x = get_compile_time_arg_val(0);
    constexpr uint32_t sender_noc_y = get_compile_time_arg_val(1);

    constexpr bool dst_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t num_transfers = get_compile_time_arg_val(3);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_pages = get_compile_time_arg_val(6);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(8);
    constexpr uint32_t rem_num_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t output_start_page_idx = get_compile_time_arg_val(10);
    constexpr uint32_t output_start_addr_offset = get_compile_time_arg_val(11);
    constexpr uint32_t row_start_idx = get_compile_time_arg_val(12);
    constexpr uint32_t row_offset = get_compile_time_arg_val(13);
    constexpr uint32_t num_rows = get_compile_time_arg_val(14);
    constexpr uint32_t out_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t last_output_page_offset = get_compile_time_arg_val(16);
    constexpr uint32_t output_page_offset = get_compile_time_arg_val(17);
    constexpr uint32_t last_output_addr_offset = get_compile_time_arg_val(18);
    constexpr uint32_t output_addr_offset = get_compile_time_arg_val(19);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(20);

    InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr + output_start_addr_offset, .page_size = out_page_size};

    const uint64_t sender_semaphore_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sem_addr);

    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t output_base_page_idx = output_start_page_idx;
    uint32_t output_page_idx = output_base_page_idx;
    uint32_t row_idx = row_start_idx;

    volatile tt_l1_ptr uint32_t* curr_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer0);
    volatile tt_l1_ptr uint32_t* next_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer1);

    for (uint32_t i = 0; i < num_transfers; ++i) {
        if constexpr (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                uint32_t src_addr = uint32_t(curr_addr) + 32;
                eth_wait_for_bytes_v2(curr_addr, num_bytes);
                write_chunk(output_page_idx, row_idx, src_addr, d, num_rows, row_offset, num_pages, page_size);
                eth_receiver_done_v2(curr_addr);
                noc_semaphore_inc(sender_semaphore_noc_addr, 1);
                std::swap(curr_addr, next_addr);
            }
        }
        if constexpr (rem_num_pages > 0) {
            uint32_t src_addr = uint32_t(curr_addr) + 32;
            eth_wait_for_bytes_v2(curr_addr, rem_num_bytes);
            write_chunk(output_page_idx, row_idx, src_addr, d, num_rows, row_offset, rem_num_pages, page_size);
            eth_receiver_done_v2(curr_addr);
            noc_semaphore_inc(sender_semaphore_noc_addr, 1);
            std::swap(curr_addr, next_addr);
        }

        if (input_ring_idx == 0) {
            input_ring_idx = num_transfers;
            if constexpr(output_addr_offset != 0) {
                d.bank_base_address += last_output_addr_offset;
            }
            if constexpr(output_page_offset != 0) {
                output_base_page_idx += last_output_page_offset;
            }
        } else {
            input_ring_idx--;
            if constexpr(output_addr_offset != 0) {
                d.bank_base_address -= output_addr_offset;
            }
            if constexpr(output_page_offset != 0) {
                output_base_page_idx -= output_page_offset;
            }
        }
        output_page_idx = output_base_page_idx;
        row_idx = row_start_idx;
    }
}
