// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug_print.h"

void kernel_main() {

    constexpr uint32_t reduce_receiver_semaphore_addr  = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_addr    = get_compile_time_arg_val(1);
    constexpr uint32_t block_h                        = get_compile_time_arg_val(2);

    const uint32_t self_noc_x                          = get_arg_val<uint32_t>(0);
    const uint32_t self_noc_y                          = get_arg_val<uint32_t>(1);
    const uint32_t noc_same_coord                      = get_arg_val<uint32_t>(2);
    const uint32_t noc_diff_coord                      = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0; // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CB::dataflow1; // E[x] global reduce
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3; // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2 = tt::CB::dataflow4; // E[(x-E[x])^2] global reduce

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial); // tile size
    const DataFormat data_format = get_dataformat(cb_ex_partial); // data format

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    const uint64_t reduce_receiver_semaphore_noc_addr = get_noc_addr(noc_same_coord, noc_diff_coord, reduce_receiver_semaphore_addr);


    // global reduce
    noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
    cb_wait_front(cb_ex_partial, block_h);
    noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
    cb_reserve_back(cb_ex, block_h);
    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
    cb_push_back(cb_ex, block_h);
    cb_pop_front(cb_ex_partial, block_h);


    // global reduce
    noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
    cb_wait_front(cb_ex_partial2, block_h);
    noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
    cb_reserve_back(cb_ex2, block_h);
    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
    cb_push_back(cb_ex2, block_h);
    cb_pop_front(cb_ex_partial2, block_h);

}
