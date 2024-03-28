// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t i = 0;
    // buffers
    uint32_t src_addr = get_arg_val<uint32_t>(i++);
    uint32_t index0_addr = get_arg_val<uint32_t>(i++);
    uint32_t index1_addr = get_arg_val<uint32_t>(i++);
    uint32_t index2_addr = get_arg_val<uint32_t>(i++);
    uint32_t index3_addr = get_arg_val<uint32_t>(i++);

    // input
    uint32_t input_stick_idx_stride_n = get_arg_val<uint32_t>(i++);
    uint32_t input_stick_idx_stride_c = get_arg_val<uint32_t>(i++);
    uint32_t input_stick_idx_stride_h = get_arg_val<uint32_t>(i++);

    // index
    uint32_t index0_is_defined = get_arg_val<uint32_t>(i++);
    uint32_t index1_is_defined = get_arg_val<uint32_t>(i++);
    uint32_t index2_is_defined = get_arg_val<uint32_t>(i++);
    uint32_t index3_is_defined = get_arg_val<uint32_t>(i++);
    uint32_t index0_stick_size = get_arg_val<uint32_t>(i++);
    uint32_t index1_stick_size = get_arg_val<uint32_t>(i++);
    uint32_t index2_stick_size = get_arg_val<uint32_t>(i++);
    uint32_t index3_stick_size = get_arg_val<uint32_t>(i++);
    uint32_t index_size = get_arg_val<uint32_t>(i++);
    int32_t index_start_dim = static_cast<int32_t>(get_arg_val<uint32_t>(i++));
    int32_t index_end_dim = static_cast<int32_t>(get_arg_val<uint32_t>(i++));

    // output
    uint32_t output_size_n = get_arg_val<uint32_t>(i++);
    uint32_t output_size_c = get_arg_val<uint32_t>(i++);
    uint32_t output_size_h = get_arg_val<uint32_t>(i++);
    uint32_t output_size_w = get_arg_val<uint32_t>(i++);

    // etc
    uint32_t start_id = get_arg_val<uint32_t>(i++);
    uint32_t num_sticks = get_arg_val<uint32_t>(i++);
    uint32_t stick_size = get_arg_val<uint32_t>(i++);


    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_in2 = tt::CB::c_in2;
    constexpr auto cb_in3 = tt::CB::c_in3;
    constexpr auto cb_in4 = tt::CB::c_in4;

    constexpr bool in_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool index0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool index1_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool index2_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool index3_is_dram = get_compile_time_arg_val(4) == 1;

    const InterleavedAddrGen<in_is_dram> s0 = {.bank_base_address = src_addr, .page_size = stick_size};

    const InterleavedAddrGen<index0_is_dram> index0 = {
        .bank_base_address = index0_addr, .page_size = index0_stick_size};
    const InterleavedAddrGen<index1_is_dram> index1 = {
        .bank_base_address = index1_addr, .page_size = index1_stick_size};
    const InterleavedAddrGen<index2_is_dram> index2 = {
        .bank_base_address = index2_addr, .page_size = index2_stick_size};
    const InterleavedAddrGen<index3_is_dram> index3 = {
        .bank_base_address = index3_addr, .page_size = index3_stick_size};

    uint32_t index_is_defined[4] = {
        index0_is_defined,
        index1_is_defined,
        index2_is_defined,
        index3_is_defined,
    };

    tt::CB index_cbs[4] = {
        cb_in1,
        cb_in2,
        cb_in3,
        cb_in4,
    };

    uint32_t output_size_list[4] = {
        output_size_n,
        output_size_c,
        output_size_h,
        output_size_w,
    };

    uint32_t input_stick_idx_strides[3] = {
        input_stick_idx_stride_n,
        input_stick_idx_stride_c,
        input_stick_idx_stride_h,
    };

    uint32_t index_stick_sizes[4] = {
        index0_stick_size,
        index1_stick_size,
        index2_stick_size,
        index3_stick_size,
    };


    // case1. offset = 0, ok
    // expected result = [0, 1, 2, 3, 4, 5, 6, 7, ]
    // real     result = [0, 1, 2, 3, 4, 5, 6, 7, ]
    {
        cb_reserve_back(cb_in4, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in4);
        uint32_t noc_id = 0;
        uint32_t noc_offset = 0 * 4;
        uint64_t src_noc_addr = get_noc_addr(noc_id, index3, noc_offset);
        uint32_t read_size = 8 * 4;
        noc_async_read(src_noc_addr, l1_write_addr, read_size);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* index_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);

        DPRINT << "=================================\n";
        DPRINT << "noc_offset = " << noc_offset << "\n";
        DPRINT << "Input index_l1_ptr = [";
        for (uint32_t x = 0 ;x < 8; x ++) {
            DPRINT << index_l1_ptr[x] << ", ";
        }
        DPRINT << "]\n";
    }

    // case2. offset = 1, not ok
    // expected result = [1, 2, 3, 4, 5, 6, 7, 8, ]
    // real     result = [0, 1, 2, 3, 4, 5, 6, 7, ]
    {
        cb_reserve_back(cb_in4, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in4);
        uint32_t noc_id = 0;
        uint32_t noc_offset = 1 * 4;
        uint64_t src_noc_addr = get_noc_addr(noc_id, index3, noc_offset);
        uint32_t read_size = 8 * 4;
        noc_async_read(src_noc_addr, l1_write_addr, read_size);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* index_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);

        DPRINT << "=================================\n";
        DPRINT << "noc_offset = " << noc_offset << "\n";
        DPRINT << "Input index_l1_ptr = [";
        for (uint32_t x = 0 ;x < 8; x ++) {
            DPRINT << index_l1_ptr[x] << ", ";
        }
        DPRINT << "]\n";
    }

    // case3. offset = 4, not ok
    // expected result = [4, 5, 6, 7, 8, 9, 10, 11]
    // real     result = [0, 0, 0, 0, 4, 5,  6,  7]
    {
        cb_reserve_back(cb_in4, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in4);
        uint32_t noc_id = 0;
        uint32_t noc_offset = 4 * 4;
        uint64_t src_noc_addr = get_noc_addr(noc_id, index3, noc_offset);
        uint32_t read_size = 8 * 4;
        noc_async_read(src_noc_addr, l1_write_addr, read_size);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* index_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);

        DPRINT << "=================================\n";
        DPRINT << "noc_offset = " << noc_offset << "\n";
        DPRINT << "Input index_l1_ptr = [";
        for (uint32_t x = 0 ;x < 8; x ++) {
            DPRINT << index_l1_ptr[x] << ", ";
        }
        DPRINT << "]\n";
    }

    // case4. offset = 8, ok
    // expected result = [8, 9, 10, 11, 12, 13, 14, 15,]
    // real     result = [8, 9, 10, 11, 12, 13, 14, 15,]
    {
        cb_reserve_back(cb_in4, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in4);
        uint32_t noc_id = 0;
        uint32_t noc_offset = 8 * 4;
        uint64_t src_noc_addr = get_noc_addr(noc_id, index3, noc_offset);
        uint32_t read_size = 8 * 4;
        noc_async_read(src_noc_addr, l1_write_addr, read_size);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* index_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);

        DPRINT << "=================================\n";
        DPRINT << "noc_offset = " << noc_offset << "\n";
        DPRINT << "Input index_l1_ptr = [";
        for (uint32_t x = 0 ;x < 8; x ++) {
            DPRINT << index_l1_ptr[x] << ", ";
        }
        DPRINT << "]\n";
    }
}
