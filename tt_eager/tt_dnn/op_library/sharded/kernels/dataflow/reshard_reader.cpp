// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

 #include <stdint.h>

 void kernel_main() {
    constexpr uint32_t shard_cb = get_compile_time_arg_val(0);

    const uint32_t input_shard_addr  = get_arg_val<uint32_t>(0);
    const uint32_t num_output_pages = get_arg_val<uint32_t>(1);
    const uint32_t num_ranges = get_arg_val<uint32_t>(2);
    const uint32_t page_size = get_arg_val<uint32_t>(3);
    uint32_t arg_index = 4;

    cb_reserve_back(shard_cb, num_output_pages);
    uint32_t l1_write_addr = get_write_ptr(shard_cb);
    for(uint32_t range_id = 0; range_id <num_ranges; range_id++) {
        uint32_t core_id_x = get_arg_val<uint32_t>(arg_index++);
        uint32_t core_id_y = get_arg_val<uint32_t>(arg_index++);
        uint32_t start = get_arg_val<uint32_t>(arg_index++);
        uint32_t end = get_arg_val<uint32_t>(arg_index++);
        uint32_t stride = get_arg_val<uint32_t>(arg_index++);
        if(page_size == stride) {
            uint32_t size = end - start;
            uint64_t noc_address = get_noc_addr(core_id_x, core_id_y,
                                            input_shard_addr + start);
            noc_async_read(noc_address, l1_write_addr, size);
            l1_write_addr+=size;
        }
        else {
            for(uint32_t addr_offset = start; addr_offset < end; addr_offset+=stride) {
                uint64_t noc_address = get_noc_addr(core_id_x, core_id_y,
                                                input_shard_addr + addr_offset);
                noc_async_read(noc_address, l1_write_addr, page_size);
                l1_write_addr+=page_size;
            }
        }


    }
    noc_async_read_barrier();
    cb_push_back(shard_cb, num_output_pages);

}
