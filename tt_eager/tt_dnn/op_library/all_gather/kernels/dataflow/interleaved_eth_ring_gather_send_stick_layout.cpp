// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

template<bool DRAM>
FORCE_INLINE void read_chunk(uint32_t& input_page_idx, uint32_t local_eth_l1_curr_src_addr, const InterleavedAddrGen<DRAM>& s, const uint32_t num_pages, const uint32_t page_size) {
    const uint32_t end_read_idx = input_page_idx + num_pages;
    for (; input_page_idx < end_read_idx; ++input_page_idx) {
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_eth_l1_curr_src_addr, page_size);
        local_eth_l1_curr_src_addr += page_size;
    }
    eth_noc_async_read_barrier();
}

template<bool DRAM>
FORCE_INLINE void read_chunk(uint32_t& input_page_idx, uint32_t& row_idx, uint32_t local_eth_l1_curr_src_addr, const InterleavedAddrGen<DRAM>& s, const uint32_t num_rows, const uint32_t row_offset, const uint32_t num_pages, const uint32_t page_size) {
     for (uint32_t i = 0; i < num_pages; ++i) {
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_eth_l1_curr_src_addr, page_size);
        local_eth_l1_curr_src_addr += page_size;
        input_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            input_page_idx += row_offset;
        }
    }
    eth_noc_async_read_barrier();
}

template<bool DRAM>
FORCE_INLINE void write_chunk_non_blocking(uint32_t& output_page_idx, uint32_t& row_idx, uint32_t local_eth_l1_curr_src_addr, const InterleavedAddrGen<DRAM>& d, const uint32_t num_rows,  const uint32_t row_offset, const uint32_t num_pages, const uint32_t page_size) {
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
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t buffer0 = get_arg_val<uint32_t>(2);
    const uint32_t buffer1 = get_arg_val<uint32_t>(3);
    const uint32_t sem_addr = get_arg_val<uint32_t>(4);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t num_transfers = get_compile_time_arg_val(2);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_pages = get_compile_time_arg_val(5);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(7);
    constexpr uint32_t rem_num_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t input_start_page_idx = get_compile_time_arg_val(9);
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


    const InterleavedAddrGen<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = input_page_size};
    InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr + output_start_addr_offset, .page_size = out_page_size};

    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    constexpr uint32_t num_bytes_per_send = num_bytes;
    constexpr uint32_t num_bytes_per_send_word_size = num_bytes_per_send >> 4;
    constexpr uint32_t rem_num_bytes_per_send = rem_num_bytes;
    constexpr uint32_t rem_num_bytes_per_send_word_size = rem_num_bytes_per_send >> 4;

    int32_t input_ring_idx = input_start_ring_idx;
    uint32_t input_page_idx = input_start_page_idx;
    uint32_t output_base_page_idx = output_start_page_idx;
    uint32_t output_page_idx = output_base_page_idx;
    uint32_t row_idx = row_start_idx;

    volatile tt_l1_ptr uint32_t* curr_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer0);
    volatile tt_l1_ptr uint32_t* next_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer1);

    if constexpr(num_full_chunks > 0) {
        for (uint32_t c = 0; c < num_full_chunks; ++c) {
            // This function also increments input_page_idx
            uint32_t src_addr = uint32_t(curr_addr) + 32;
            eth_wait_for_receiver_done_v2(curr_addr);
            read_chunk<src_is_dram>(input_page_idx, src_addr, s, num_pages, input_page_size);
            write_chunk_non_blocking<dst_is_dram>(output_page_idx, row_idx, src_addr, d, num_rows, row_offset, num_pages, input_page_size);
            eth_send_bytes_v2(curr_addr, src_addr, src_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);
            eth_send_done_v2(curr_addr);
            std::swap(curr_addr, next_addr);
            eth_noc_async_write_barrier();
        }
    }
    if constexpr(rem_num_pages > 0) {
        uint32_t src_addr = uint32_t(curr_addr) + 32;
        if constexpr (input_start_ring_idx == 0) {
            for (volatile uint32_t i = 0; i < 1000000; ++i);
        }
        eth_wait_for_receiver_done_v2(curr_addr);
        read_chunk<src_is_dram>(input_page_idx, src_addr, s, rem_num_pages, input_page_size);
        write_chunk_non_blocking<dst_is_dram>(output_page_idx, row_idx, src_addr, d, num_rows, row_offset, rem_num_pages, input_page_size);
        eth_send_bytes_v2(curr_addr, src_addr, src_addr, rem_num_bytes, rem_num_bytes_per_send, rem_num_bytes_per_send_word_size);
        eth_send_done_v2(curr_addr);
        std::swap(curr_addr, next_addr);
        eth_noc_async_write_barrier();
    }

    uint32_t sem_idx = 1;

    // num_transfers = num_devices - 1
    for (uint32_t i = 1; i < num_transfers; ++i) {
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
        if constexpr(num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                eth_noc_semaphore_wait_v2(sender_semaphore_addr_ptr, sem_idx);
                sem_idx++;
                // This function also increments input_page_idx
                uint32_t src_addr = uint32_t(curr_addr) + 32;
                eth_wait_for_receiver_done_v2(curr_addr);
                read_chunk<dst_is_dram>(output_page_idx, row_idx, src_addr, d, num_rows, row_offset, num_pages, input_page_size);
                eth_send_bytes_v2(curr_addr, src_addr, src_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);
                eth_send_done_v2(curr_addr);
                std::swap(curr_addr, next_addr);
            }
        }
        if constexpr(rem_num_pages > 0) {
            eth_noc_semaphore_wait_v2(sender_semaphore_addr_ptr, sem_idx);
            sem_idx++;
            uint32_t src_addr = uint32_t(curr_addr) + 32;
            eth_wait_for_receiver_done_v2(curr_addr);
            read_chunk<dst_is_dram>(output_page_idx, row_idx, src_addr, d, num_rows, row_offset, rem_num_pages, input_page_size);
            eth_send_bytes_v2(curr_addr, src_addr, src_addr, rem_num_bytes, rem_num_bytes_per_send, rem_num_bytes_per_send_word_size);
            eth_send_done_v2(curr_addr);
            std::swap(curr_addr, next_addr);
        }
    }
    eth_wait_for_receiver_done_v2(curr_addr);
    eth_wait_for_receiver_done_v2(next_addr);
    noc_semaphore_set(sender_semaphore_addr_ptr, 0);
}
