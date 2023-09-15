// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <array>

#include "dataflow_api.h"

//#define DEBUG

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t out_tensor_tile_id = get_arg_val<uint32_t>(0);
    uint32_t num_tensors = get_arg_val<uint32_t>(1);
    bool parallelize_last_dim = (bool)get_arg_val<uint32_t>(2);
    uint32_t tensor_idx = get_arg_val<uint32_t>(3);

    // COMPILE TIME ARGS
    // interleaved accessor args
    
    // WRITER COMPILE TIME ARGS
    //constexpr uint32_t out_num_tiles_per_tensor = get_compile_time_arg_val(2);
    constexpr uint32_t out_num_tiles_per_tensor_y = get_compile_time_arg_val(0);
    constexpr uint32_t out_num_tiles_per_tensor_x = get_compile_time_arg_val(1);
    constexpr uint32_t z = get_compile_time_arg_val(2);
    constexpr uint32_t z_stride = get_compile_time_arg_val(3);
    constexpr uint32_t y_stride = get_compile_time_arg_val(4);
    
    constexpr uint32_t cb_id_out0 = 0;  // same as cb_id_in0
    const DataFormat data_format = get_dataformat(cb_id_out0);


    uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);

    constexpr uint32_t onetile = 1;


    InterleavedAddrGenFast<false> l1_dst_addr_gens[num_tensors];
    InterleavedAddrGenFast<true> dram_dst_addr_gens[num_tensors];

    uint32_t bank_id = 0;
    uint32_t tile_id = 0;

    bool is_dram[num_tensors];
    uint32_t num_tiles_per_block[num_tensors];
    uint32_t tile_id_per_tensor[num_tensors];

    for (uint32_t i = 0; i < num_tensors; i++) {
         uint32_t src_addr  = get_arg_val<uint32_t>(5 + i);
         is_dram[i] = (bool)get_arg_val<uint32_t>(5 + num_tensors + i);
         dram_dst_addr_gens[i] = {
            .bank_base_address = src_addr,
            .page_size = single_tile_size_bytes,
            .data_format = data_format
         };

         l1_dst_addr_gens[i] = {
            .bank_base_address = src_addr,
            .page_size = single_tile_size_bytes,
            .data_format = data_format
         };
     }


    for (int i=0; i<num_tensors; i++) {
        if((!parallelize_last_dim) ||
            (parallelize_last_dim && tensor_idx == i)
        ){
            uint32_t z_stride_cum = 0;
            for (uint32_t k = 0; k < z; k++) {
                uint32_t y_stride_cum = 0;
                for (uint32_t j = 0; j < out_num_tiles_per_tensor_y; j++) {
                    for (uint32_t i = 0; i < out_num_tiles_per_tensor_x; i++) {
                        uint32_t tile_id = y_stride_cum + z_stride_cum + i;
                        cb_wait_front(cb_id_out0, onetile);
                        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                        if(is_dram[i]){
                            noc_async_write_tile(tile_id + out_tensor_tile_id, dram_dst_addr_gens[i], l1_read_addr);
                        }
                        else{
                            noc_async_write_tile(tile_id + out_tensor_tile_id, l1_dst_addr_gens[i], l1_read_addr);
                        }
                        noc_async_write_barrier();
                        cb_pop_front(cb_id_out0, onetile);
                    }
                    y_stride_cum += y_stride;
                }
                z_stride_cum += z_stride;
            }
        }
    }
    
    
    uint32_t num_cores_c = num_cores_y;
    uint32_t num_cores_r = num_cores_x * num_cores_z;
}
