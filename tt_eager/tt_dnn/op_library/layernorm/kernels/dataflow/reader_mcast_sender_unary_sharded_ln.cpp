// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "debug/dprint.h"
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {

    constexpr uint32_t reduce_receiver_semaphore_addr = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_addr   = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks                     = get_compile_time_arg_val(2);
    constexpr uint32_t block_h                        = get_compile_time_arg_val(3);
    constexpr uint32_t block_h_size_bytes             = get_compile_time_arg_val(4);

    const uint32_t mcast_dest_noc_start_x               = get_arg_val<uint32_t>(0);
    const uint32_t mcast_dest_noc_start_y               = get_arg_val<uint32_t>(1);
    const uint32_t mcast_dest_noc_end_x                 = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_end_y                 = get_arg_val<uint32_t>(3);
    const uint32_t noc_same_coord                       = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t * noc_diff_coord        = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(5));

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0;
    constexpr uint32_t cb_ex = tt::CB::dataflow1;
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3;
    constexpr uint32_t cb_ex2 = tt::CB::dataflow4;
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);
    const DataFormat data_format = get_dataformat(cb_ex_partial);

    const uint64_t reduce_sender_semaphore_noc_addr = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        reduce_sender_semaphore_addr);

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        0);

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    *(reduce_sender_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);


    // global reduce
    uint32_t l1_write_addr_ex = get_write_ptr(cb_ex);
    noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_blocks-1);
    noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);
    cb_wait_front(cb_ex_partial, block_h);
    uint32_t l1_read_addr_ex_par = get_read_ptr(cb_ex_partial);
    for (uint32_t i = 0; i < block_h; i++) {
        cb_reserve_back(cb_ex_external, num_blocks);
        uint32_t l1_write_addr_external = get_write_ptr(cb_ex_external);
        for(uint32_t block = 0; block < num_blocks; block++) {
            uint64_t noc_addr_ex_par = get_noc_addr(noc_same_coord, noc_diff_coord[block], l1_read_addr_ex_par);
            noc_async_read(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
            l1_write_addr_external += single_tile_size_bytes;
        }
        l1_read_addr_ex_par += single_tile_size_bytes;
        noc_async_read_barrier();
        cb_push_back(cb_ex_external, num_blocks);
    }
    uint64_t multicast_data_addr = multicast_data_noc | l1_write_addr_ex;
    cb_wait_front(cb_ex, block_h);
    noc_async_write_multicast(l1_write_addr_ex, multicast_data_addr, block_h_size_bytes, num_blocks-1);
    noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks-1);
    cb_pop_front(cb_ex_partial, block_h);

    // global reduce
    uint32_t l1_write_addr_ex2 = get_write_ptr(cb_ex2);
    noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_blocks-1);
    noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);
    cb_wait_front(cb_ex_partial2, block_h);
    uint32_t l1_read_addr_ex_par2 = get_read_ptr(cb_ex_partial2);
    for (uint32_t i = 0; i < block_h; i++) {
        cb_reserve_back(cb_ex_external2, num_blocks);
        uint32_t l1_write_addr_external = get_write_ptr(cb_ex_external2);
        for(uint32_t block = 0; block < num_blocks; block++) {
            uint64_t noc_addr_ex_par = get_noc_addr(noc_same_coord, noc_diff_coord[block], l1_read_addr_ex_par2);
            noc_async_read(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
            l1_write_addr_external += single_tile_size_bytes;
        }
        l1_read_addr_ex_par2 += single_tile_size_bytes;
        noc_async_read_barrier();
        cb_push_back(cb_ex_external2, num_blocks);
    }
    uint64_t multicast_data_addr2 = multicast_data_noc | l1_write_addr_ex2;
    cb_wait_front(cb_ex2, block_h);
    noc_async_write_multicast(l1_write_addr_ex2, multicast_data_addr2, block_h_size_bytes, num_blocks-1);
    noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks-1);
    cb_pop_front(cb_ex_partial2, block_h);

}
