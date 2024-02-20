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
FORCE_INLINE void write_chunk(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, uint32_t local_eth_l1_curr_src_addr, const InterleavedAddrGenFast<DRAM>& d, const uint32_t num_cols, const uint32_t num_rows, const uint32_t col_offset, const uint32_t row_offset, const uint32_t num_pages, const uint32_t page_size) {
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
    eth_noc_async_write_barrier();
}

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t sender_noc_x = get_compile_time_arg_val(0);
    constexpr uint32_t sender_noc_y = get_compile_time_arg_val(1);

    constexpr bool dst_is_dram = get_compile_time_arg_val(2) == 1;

    constexpr DataFormat df = static_cast<DataFormat>(get_compile_time_arg_val(3));
    constexpr uint32_t num_transfers = get_compile_time_arg_val(4);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t page_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_pages = get_compile_time_arg_val(7);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(9);
    constexpr uint32_t rem_num_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t output_start_page_idx = get_compile_time_arg_val(11);
    constexpr uint32_t row_start_idx = get_compile_time_arg_val(12);
    constexpr uint32_t col_start_idx = get_compile_time_arg_val(13);
    constexpr uint32_t row_offset = get_compile_time_arg_val(14);
    constexpr uint32_t col_offset = get_compile_time_arg_val(15);
    constexpr uint32_t num_rows = get_compile_time_arg_val(16);
    constexpr uint32_t num_cols = get_compile_time_arg_val(17);
    constexpr uint32_t last_output_page_shift = get_compile_time_arg_val(18);
    constexpr uint32_t output_page_shift = get_compile_time_arg_val(19);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(20);
    constexpr uint32_t num_channels = get_compile_time_arg_val(21);
    constexpr uint32_t local_l1_start_addr = get_compile_time_arg_val(22);
    constexpr uint32_t sem_addr = get_compile_time_arg_val(23);

    const InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr,
        .page_size = page_size,
        .data_format = df
    };

    const uint64_t sender_semaphore_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sem_addr);

    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t output_base_page_idx = output_start_page_idx;
    uint32_t output_page_idx = output_base_page_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;

    uint8_t channel = 0;

    uint32_t local_l1_src_addrs[num_channels];
    for (uint32_t i = 0, curr_addr = local_l1_start_addr; i < num_channels; ++i) {
        local_l1_src_addrs[i] = curr_addr;
        curr_addr += num_bytes;
    }

    for (uint32_t i = 0; i < num_transfers; ++i) {
        if constexpr (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                const uint32_t& src_addr = local_l1_src_addrs[channel];
                eth_wait_for_bytes(num_bytes, channel);
                write_chunk(output_page_idx, col_idx, row_idx, src_addr, d, num_cols, num_rows, col_offset, row_offset, num_pages, page_size);
                eth_receiver_done(channel);
                noc_semaphore_inc(sender_semaphore_noc_addr, 1);
                channel = get_next_buffer_channel_pointer<num_channels>(channel);
            }
        }
        if constexpr (rem_num_pages > 0) {
            const uint32_t& src_addr = local_l1_src_addrs[channel];
            eth_wait_for_bytes(rem_num_bytes, channel);
            write_chunk(output_page_idx, col_idx, row_idx, src_addr, d, num_cols, num_rows, col_offset, row_offset, rem_num_pages, page_size);
            eth_receiver_done(channel);
            noc_semaphore_inc(sender_semaphore_noc_addr, 1);
            channel = get_next_buffer_channel_pointer<num_channels>(channel);
        }

        if (input_ring_idx == 0) {
            input_ring_idx = num_transfers;
            output_base_page_idx += last_output_page_shift;
        } else {
            input_ring_idx--;
            output_base_page_idx -= output_page_shift;
        }
        output_page_idx = output_base_page_idx;
        col_idx = col_start_idx;
        row_idx = row_start_idx;
    }
}
