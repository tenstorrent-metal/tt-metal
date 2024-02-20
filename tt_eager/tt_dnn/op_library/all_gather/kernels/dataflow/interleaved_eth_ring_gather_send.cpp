// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

template <uint8_t NUM_CHANNELS>
FORCE_INLINE uint8_t get_next_buffer_channel_pointer(uint8_t pointer) {
    if constexpr (NUM_CHANNELS % 2 == 0) {
        constexpr uint8_t CHANNEL_WRAP_MASK = NUM_CHANNELS - 1;
        return pointer = (pointer + 1) & CHANNEL_WRAP_MASK;
    } else {
        pointer = (pointer + 1);
        return pointer == NUM_CHANNELS ? 0 : pointer;
    }
}

template<bool DRAM>
FORCE_INLINE void read_chunk(uint32_t& input_page_idx, uint32_t local_eth_l1_curr_src_addr, const InterleavedAddrGenFast<DRAM>& s, const uint32_t num_pages, const uint32_t page_size) {
    const uint32_t end_read_idx = input_page_idx + num_pages;
    for (; input_page_idx < end_read_idx; ++input_page_idx) {
        noc_async_read_tile(input_page_idx, s, local_eth_l1_curr_src_addr);
        local_eth_l1_curr_src_addr += page_size;
    }
    eth_noc_async_read_barrier();
}

template<bool DRAM>
FORCE_INLINE void read_chunk(uint32_t& input_page_idx, uint32_t& col_idx, uint32_t& row_idx, uint32_t local_eth_l1_curr_src_addr, const InterleavedAddrGenFast<DRAM>& s, const uint32_t num_cols, const uint32_t num_rows, const uint32_t col_offset, const uint32_t row_offset, const uint32_t num_pages, const uint32_t page_size) {
     for (uint32_t i = 0; i < num_pages; ++i) {
        noc_async_read_tile(input_page_idx, s, local_eth_l1_curr_src_addr);
        local_eth_l1_curr_src_addr += page_size;
        input_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            input_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                input_page_idx += row_offset;
            }
        }
    }
    eth_noc_async_read_barrier();
}

template<bool DRAM>
FORCE_INLINE void write_chunk_non_blocking(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, uint32_t local_eth_l1_curr_src_addr, const InterleavedAddrGenFast<DRAM>& d, const uint32_t num_cols, const uint32_t num_rows, const uint32_t col_offset, const uint32_t row_offset, const uint32_t num_pages, const uint32_t page_size) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        noc_async_write_tile(output_page_idx, d, local_eth_l1_curr_src_addr);
        local_eth_l1_curr_src_addr += page_size;
        output_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
    }
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr DataFormat df = static_cast<DataFormat>(get_compile_time_arg_val(2));
    constexpr uint32_t num_transfers = get_compile_time_arg_val(3);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_pages = get_compile_time_arg_val(6);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(8);
    constexpr uint32_t rem_num_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t input_start_idx = get_compile_time_arg_val(10);
    constexpr uint32_t output_start_idx = get_compile_time_arg_val(11);
    constexpr uint32_t row_start_idx = get_compile_time_arg_val(12);
    constexpr uint32_t col_start_idx = get_compile_time_arg_val(13);
    constexpr uint32_t row_offset = get_compile_time_arg_val(14);
    constexpr uint32_t col_offset = get_compile_time_arg_val(15);
    constexpr uint32_t num_rows = get_compile_time_arg_val(16);
    constexpr uint32_t num_cols = get_compile_time_arg_val(17);
    constexpr uint32_t last_output_page_offset = get_compile_time_arg_val(18);
    constexpr uint32_t output_page_offset = get_compile_time_arg_val(19);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(20);
    constexpr uint32_t num_channels = get_compile_time_arg_val(21);
    constexpr uint32_t local_l1_start_addr = get_compile_time_arg_val(22);
    constexpr uint32_t sem_addr = get_compile_time_arg_val(23);

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

    constexpr uint32_t num_bytes_per_send = num_bytes;
    constexpr uint32_t num_bytes_per_send_word_size = num_bytes_per_send >> 4;
    constexpr uint32_t rem_num_bytes_per_send = rem_num_bytes;
    constexpr uint32_t rem_num_bytes_per_send_word_size = rem_num_bytes_per_send >> 4;

    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t input_page_idx = input_start_idx;
    uint32_t output_base_page_idx = output_start_idx;
    uint32_t output_page_idx = output_base_page_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;

    uint32_t local_l1_src_addrs[num_channels];
    for (uint32_t i = 0, curr_addr = local_l1_start_addr; i < num_channels; ++i) {
        local_l1_src_addrs[i] = curr_addr;
        curr_addr += num_bytes;
    }

    uint8_t channel = 0;
    if constexpr(num_full_chunks > 0) {
        for (uint32_t c = 0; c < num_full_chunks; ++c) {
            // This function also increments input_page_idx
            const uint32_t& src_addr = local_l1_src_addrs[channel];
            eth_wait_receiver_done(channel);
            read_chunk<src_is_dram>(input_page_idx, src_addr, s, num_pages, page_size);
            write_chunk_non_blocking<dst_is_dram>(output_page_idx, col_idx, row_idx, src_addr, d, num_cols, num_rows, col_offset, row_offset, num_pages, page_size);
            eth_send_bytes(src_addr, src_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size, channel);
            eth_send_done(channel);
            channel = get_next_buffer_channel_pointer<num_channels>(channel);
            eth_noc_async_write_barrier();
        }
    }
    if constexpr(rem_num_pages > 0) {
        const uint32_t& src_addr = local_l1_src_addrs[channel];
        eth_wait_receiver_done(channel);
        read_chunk<src_is_dram>(input_page_idx, src_addr, s, rem_num_pages, page_size);
        write_chunk_non_blocking<dst_is_dram>(output_page_idx, col_idx, row_idx, src_addr, d, num_cols, num_rows, col_offset, row_offset, rem_num_pages, page_size);
        eth_send_bytes(src_addr, src_addr, rem_num_bytes, rem_num_bytes_per_send, rem_num_bytes_per_send_word_size, channel);
        eth_send_done(channel);
        channel = get_next_buffer_channel_pointer<num_channels>(channel);
        eth_noc_async_write_barrier();
    }

    uint32_t sem_idx = 1;

    // num_transfers = num_devices - 1
    for (uint32_t i = 1; i < num_transfers; ++i) {
        if (input_ring_idx == 0) {
            input_ring_idx = num_transfers;
            output_base_page_idx += last_output_page_offset;
        } else {
            input_ring_idx--;
            output_base_page_idx -= output_page_offset;
        }
        output_page_idx = output_base_page_idx;
        col_idx = col_start_idx;
        row_idx = row_start_idx;
        if constexpr(num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                eth_noc_semaphore_wait_min(sender_semaphore_addr_ptr, sem_idx);
                sem_idx++;
                // This function also increments input_page_idx
                const uint32_t& src_addr = local_l1_src_addrs[channel];
                eth_wait_receiver_done(channel);
                read_chunk<dst_is_dram>(output_page_idx, col_idx, row_idx, src_addr, d, num_cols, num_rows, col_offset, row_offset, num_pages, page_size);
                eth_send_bytes(src_addr, src_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size, channel);
                eth_send_done(channel);
                channel = get_next_buffer_channel_pointer<num_channels>(channel);
            }
        }
        if constexpr(rem_num_pages > 0) {
            eth_noc_semaphore_wait_min(sender_semaphore_addr_ptr, sem_idx);
            sem_idx++;
            const uint32_t& src_addr = local_l1_src_addrs[channel];
            eth_wait_receiver_done(channel);
            read_chunk<dst_is_dram>(output_page_idx, col_idx, row_idx, src_addr, d, num_cols, num_rows, col_offset, row_offset, rem_num_pages, page_size);
            eth_send_bytes(src_addr, src_addr, rem_num_bytes, rem_num_bytes_per_send, rem_num_bytes_per_send_word_size, channel);
            eth_send_done(channel);
            channel = get_next_buffer_channel_pointer<num_channels>(channel);
        }
    }
    for (uint32_t i = 0; i < num_channels; ++i) {
        eth_wait_receiver_done(i);
    }
    noc_semaphore_set(sender_semaphore_addr_ptr, 0);

}
