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

FORCE_INLINE void read_chunk(uint32_t& core_idx, uint32_t& input_shard_offset, uint32_t& rem_shard_size, uint32_t bytes_to_read, const uint64_t (&remote_noc_addrs)[], uint32_t local_l1_write_addr, const uint32_t shard_size, const uint32_t base_shard_offset) {
    while (bytes_to_read > 0) {
        uint64_t noc_read_addr = remote_noc_addrs[core_idx] | input_shard_offset;
        if (rem_shard_size > bytes_to_read) {
            noc_async_read(noc_read_addr, local_l1_write_addr, bytes_to_read);
            rem_shard_size -= bytes_to_read;
            input_shard_offset += bytes_to_read;
            bytes_to_read = 0;
        } else {
            noc_async_read(noc_read_addr, local_l1_write_addr, rem_shard_size);
            core_idx++;
            input_shard_offset = base_shard_offset;
            bytes_to_read -= rem_shard_size;
            local_l1_write_addr += rem_shard_size;
            rem_shard_size = shard_size;
        }
    }
    eth_noc_async_read_barrier();
}

template<bool WIDTH_SHARD>
FORCE_INLINE void read_chunk(uint32_t& core_idx, uint32_t& row_idx, uint32_t& input_shard_offset, uint32_t& rem_block_size, uint32_t bytes_to_read, const uint64_t (&remote_noc_addrs)[], uint32_t local_l1_write_addr, const uint32_t block_size, const uint32_t num_rows, const uint32_t row_offset, const uint32_t base_shard_offset) {
    while (bytes_to_read > 0) {
        uint64_t noc_read_addr = remote_noc_addrs[core_idx] | input_shard_offset;
        if (rem_block_size > bytes_to_read) {
            noc_async_read(noc_read_addr, local_l1_write_addr, bytes_to_read);
            rem_block_size -= bytes_to_read;
            input_shard_offset += bytes_to_read;
            bytes_to_read = 0;
        } else {
            noc_async_read(noc_read_addr, local_l1_write_addr, rem_block_size);
            if constexpr(WIDTH_SHARD) {
                row_idx++;
                if (row_idx == num_rows) {
                    row_idx = 0;
                    core_idx++;
                    input_shard_offset = base_shard_offset;
                } else {
                    input_shard_offset += rem_block_size + row_offset;
                }
            } else {
                core_idx++;
                input_shard_offset = base_shard_offset;
            }
            bytes_to_read -= rem_block_size;
            local_l1_write_addr += rem_block_size;
            rem_block_size = block_size;
        }
    }
    eth_noc_async_read_barrier();
}

template<bool WIDTH_SHARD>
FORCE_INLINE void write_chunk_non_blocking(uint32_t& core_idx, uint32_t& row_idx, uint32_t& output_shard_offset, uint32_t& rem_block_size, uint32_t bytes_to_write, const uint64_t (&remote_noc_addrs)[], uint32_t local_l1_read_addr, const uint32_t block_size, const uint32_t num_rows, const uint32_t row_offset, const uint32_t base_shard_offset) {
    while (bytes_to_write > 0) {
        uint64_t noc_write_addr = remote_noc_addrs[core_idx] | output_shard_offset;
        if (rem_block_size > bytes_to_write) {
            noc_async_write(local_l1_read_addr, noc_write_addr, bytes_to_write);
            rem_block_size -= bytes_to_write;
            output_shard_offset += bytes_to_write;
            bytes_to_write = 0;
        } else {
            noc_async_write(local_l1_read_addr, noc_write_addr, rem_block_size);
            if constexpr(WIDTH_SHARD) {
                row_idx++;
                if (row_idx == num_rows) {
                    row_idx = 0;
                    core_idx++;
                    output_shard_offset = base_shard_offset;
                } else {
                    output_shard_offset += rem_block_size + row_offset;
                }
            } else {
                core_idx++;
                output_shard_offset = base_shard_offset;
            }
            bytes_to_write -= rem_block_size;
            local_l1_read_addr += rem_block_size;
            rem_block_size = block_size;
        }
    }
}

void kernel_main() {
    constexpr uint32_t num_cores = get_compile_time_arg_val(0);
    constexpr uint32_t row_major = get_compile_time_arg_val(1);
    constexpr uint32_t num_x = get_compile_time_arg_val(2);
    constexpr uint32_t num_y = get_compile_time_arg_val(3);
    constexpr uint32_t core_start_idx = get_compile_time_arg_val(4);
    constexpr bool width_sharded = get_compile_time_arg_val(5) == 1;
    constexpr uint32_t num_transfers = get_compile_time_arg_val(6);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(7);
    constexpr uint32_t input_shard_size = get_compile_time_arg_val(8);
    constexpr uint32_t input_shard_start_offset = get_compile_time_arg_val(9);
    constexpr uint32_t output_shard_start_offset = get_compile_time_arg_val(10);
    constexpr uint32_t rem_block_size = get_compile_time_arg_val(11);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t rem_num_bytes = get_compile_time_arg_val(13);
    constexpr uint32_t output_start_addr_offset = get_compile_time_arg_val(14);
    constexpr uint32_t row_start_idx = get_compile_time_arg_val(15);
    constexpr uint32_t row_offset = get_compile_time_arg_val(16);
    constexpr uint32_t num_rows = get_compile_time_arg_val(17);
    constexpr uint32_t last_output_shard_offset = get_compile_time_arg_val(18);
    constexpr uint32_t output_block_size = get_compile_time_arg_val(19);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(20);
    constexpr uint32_t num_channels = get_compile_time_arg_val(21);
    constexpr uint32_t local_l1_start_addr = get_compile_time_arg_val(22);
    constexpr uint32_t sem_addr = get_compile_time_arg_val(23);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t * remote_noc_x          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(2));
    volatile tt_l1_ptr uint32_t * remote_noc_y          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(2 + num_x));


    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    constexpr uint32_t num_bytes_per_send = num_bytes;
    constexpr uint32_t num_bytes_per_send_word_size = num_bytes_per_send >> 4;
    constexpr uint32_t rem_num_bytes_per_send = rem_num_bytes;
    constexpr uint32_t rem_num_bytes_per_send_word_size = rem_num_bytes_per_send >> 4;

    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t row_idx = row_start_idx;

    uint32_t local_l1_src_addrs[num_channels];
    for (uint32_t i = 0, curr_addr = local_l1_start_addr; i < num_channels; ++i) {
        local_l1_src_addrs[i] = curr_addr;
        curr_addr += num_bytes;
    }

    // TODO: This currently generates all cores in the shard grid
    // This is good if we split each shard across links
    // but we are currently splitting full buffer size across links
    // so we only need a subset per link
    uint64_t remote_noc_addrs[num_cores];
    uint32_t x = 0, y = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        remote_noc_addrs[i] = get_noc_addr(remote_noc_x[x], remote_noc_y[y], 0);
        if constexpr(row_major) {
            ++y;
            if (y == num_y) {
                y = 0;
                ++x;
                if (x == num_x) {
                    x = 0;
                }
            }
        } else {
            ++x;
            if (x == num_x) {
                x = 0;
                ++y;
                if (y == num_y) {
                    y = 0;
                }
            }
        }
    }

    uint32_t core_idx = core_start_idx;
    uint32_t input_shard_curr_offset = src_addr + input_shard_start_offset;
    uint32_t input_rem_block_size = rem_block_size;
    uint32_t output_rem_block_size = rem_block_size;
    uint32_t output_base_shard_offset = dst_addr + output_start_addr_offset;
    uint32_t output_shard_curr_offset = output_base_shard_offset + output_shard_start_offset;
    uint8_t channel = 0;
    if constexpr(num_full_chunks > 0) {
        for (uint32_t c = 0; c < num_full_chunks; ++c) {
            // This function also increments input_page_idx
            const uint32_t& src_addr = local_l1_src_addrs[channel];
            eth_wait_receiver_done(channel);
            read_chunk(core_idx, input_shard_curr_offset, input_rem_block_size, num_bytes, remote_noc_addrs, src_addr, input_shard_size, src_addr);
            write_chunk_non_blocking<width_sharded>(core_idx, row_idx, output_shard_curr_offset, output_rem_block_size, num_bytes, remote_noc_addrs, src_addr, output_block_size, num_rows, row_offset, output_base_shard_offset);
            eth_send_bytes(src_addr, src_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size, channel);
            eth_send_done(channel);
            channel = get_next_buffer_channel_pointer<num_channels>(channel);
            eth_noc_async_write_barrier();
        }
    }
    if constexpr(rem_num_bytes > 0) {
        const uint32_t& src_addr = local_l1_src_addrs[channel];
        eth_wait_receiver_done(channel);
        read_chunk(core_idx, input_shard_curr_offset, input_rem_block_size, rem_num_bytes, remote_noc_addrs, src_addr, input_shard_size, src_addr);
        write_chunk_non_blocking<width_sharded>(core_idx, row_idx, output_shard_curr_offset, output_rem_block_size, rem_num_bytes, remote_noc_addrs, src_addr, output_block_size, num_rows, row_offset, output_base_shard_offset);
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
            output_base_shard_offset += last_output_shard_offset;
        } else {
            input_ring_idx--;
            output_base_shard_offset -= output_block_size;
        }
        output_rem_block_size = rem_block_size;
        output_shard_curr_offset = output_shard_start_offset + output_base_shard_offset;
        row_idx = row_start_idx;
        core_idx = core_start_idx;
        if constexpr(num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                eth_noc_semaphore_wait_min(sender_semaphore_addr_ptr, sem_idx);
                sem_idx++;
                // This function also increments input_page_idx
                const uint32_t& src_addr = local_l1_src_addrs[channel];
                eth_wait_receiver_done(channel);
                read_chunk<width_sharded>(core_idx, row_idx, output_shard_curr_offset, output_rem_block_size, num_bytes, remote_noc_addrs, src_addr, output_block_size, num_rows, row_offset, output_base_shard_offset);
                eth_send_bytes(src_addr, src_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size, channel);
                eth_send_done(channel);
                channel = get_next_buffer_channel_pointer<num_channels>(channel);
            }
        }
        if constexpr(rem_num_bytes > 0) {
            eth_noc_semaphore_wait_min(sender_semaphore_addr_ptr, sem_idx);
            sem_idx++;
            const uint32_t& src_addr = local_l1_src_addrs[channel];
            eth_wait_receiver_done(channel);
            read_chunk<width_sharded>(core_idx, row_idx, output_shard_curr_offset, output_rem_block_size, rem_num_bytes, remote_noc_addrs, src_addr, output_block_size, num_rows, row_offset, output_base_shard_offset);
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
