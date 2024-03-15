// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t num_offset              = get_arg_val<uint32_t>(1);
    uint32_t nums_batches              = get_arg_val<uint32_t>(2);
    uint32_t batch_size_in_elements              = get_arg_val<uint32_t>(3);
    uint32_t elements_in_last_batch              = get_arg_val<uint32_t>(4);
    uint32_t element_size                       = get_arg_val<uint32_t>(5);



    constexpr uint32_t local_cb_id_index = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb_id_index = get_compile_time_arg_val(1);
    constexpr uint32_t num_non_zero_indices_semaphore = get_compile_time_arg_val(2);
    constexpr uint32_t writer_core_x = get_compile_time_arg_val(3);
    constexpr uint32_t writer_core_y = get_compile_time_arg_val(4);
    constexpr uint32_t num_indices_addr = get_compile_time_arg_val(5);
    constexpr uint32_t indices_addr = get_compile_time_arg_val(6);
    constexpr bool src0_is_dram          = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t reader_complete_semaphore_addr = get_compile_time_arg_val(8);

    const InterleavedAddrGen<src0_is_dram> s0 = {
        .bank_base_address = input_addr,
        .page_size = batch_size_in_elements * element_size
    };

    uint32_t num_non_zero_indices = 0;
    uint32_t index


    uint32_t l1_write_addr = get_write_ptr(local_cb_id_index);

    uint32_t indices_cb_id_index_addr = get_write_ptr(indices_cb_id_index);
    volatile tt_l1_ptr int* indices_cb_id_index_addr_ptr = reinterpret_cast<volatile tt_l1_ptr int*>(indices_cb_id_index_addr);


    uint32_t local_index = 0;
    for (uint32_t i=0; i<nums_batches; i++) {
        uint64_t src_noc_addr = get_noc_addr(i, s0);
        uint32_t l1_write_addr = get_write_ptr(local_cb_id_index);
        noc_async_read(src_noc_addr, l1_write_addr, batch_size_in_elements * element_size);
        noc_async_read_barrier();
        volatile tt_l1_ptr int* addr_ptr = reinterpret_cast<volatile tt_l1_ptr int*>(l1_write_addr);
        uint32_t local_batch_size = (i == (num_batches-1) ) ? elements_in_last_batch : batch_size;
        for (uint32_t j = 0; j < local_batch_size; j++) {
            if(addr_ptr[j] != 0 ) {
                indices_cb_id_index_addr_ptr[num_non_zero_indices] = num_offset + local_index;
                num_non_zero_indices++;
            }
            local_index++;
        }
    }


    noc_semaphore_wait(num_non_zero_indices_semaphore, 1);
    //ATOMIC SECTION
    uint32_t l1_write_addr = get_write_ptr(local_cb_id_index);
    uint64_t src_noc_addr = get_noc_addr(writer_core_x, writer_core_y, num_indices_addr);
    noc_async_read(src_noc_addr, l1_write_addr, sizeof(uint32_t));
    volatile tt_l1_ptr uint32_t* addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
    uint32_t old_offset = addr_ptr[0];
    addr_ptr[0] = old_offset + num_non_zero_indices;

    uint32_t dst_noc_addr = get_noc_addr(writer_core_x, writer_core_y, indices_addr);
    noc_async_write(indices_cb_id_index_addr, dst_noc_addr + old_offset, num_zero_nsizeof(uint32_t));


    noc_semaphore_set(writer_semaphore_addr, 1);
    noc_semaphore_inc(reader_complete_semaphore_addr, 1);



}
