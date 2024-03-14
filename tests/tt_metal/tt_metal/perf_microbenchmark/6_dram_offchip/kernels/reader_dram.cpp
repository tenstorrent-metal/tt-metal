// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

// #include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t rr_vc = get_compile_time_arg_val(0);

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t input_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t num_cb_tiles = get_arg_val<uint32_t>(3);
    uint32_t read_bytes = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id = 0;
    constexpr bool is_dram = true;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id);
    const DataFormat data_format = get_dataformat(cb_id);

    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = input_addr,
        .page_size = single_tile_size_bytes,
        .data_format = data_format,
    };

    uint32_t block_size = num_cb_tiles;
    cb_reserve_back(cb_id, num_cb_tiles);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        uint32_t cb_addr = get_write_ptr(cb_id);
        for (uint32_t i = 0; i < block_size; ++i) {


            #ifdef HYBRID_NOC
            uint32_t bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(input_start_tile_id);
            uint64_t noc_read_addr;

            if (bank_id == 0 or bank_id == 2) {
                noc_read_addr = s.get_noc_addr(1, input_start_tile_id, bank_id);
            } else {
                noc_read_addr = s.get_noc_addr(input_start_tile_id);
            }

            if constexpr(rr_vc) {
                if (bank_id == 0 or bank_id == 2) {
                    noc_async_read_one_packet(1, noc_read_addr, cb_addr, read_bytes, input_start_tile_id & NOC_UNICAST_READ_REQ_VC_RANGE_MASK);
                } else {
                    noc_async_read_one_packet(noc_read_addr, cb_addr, read_bytes, input_start_tile_id & NOC_UNICAST_READ_REQ_VC_RANGE_MASK);
                }
            } else {
                if (bank_id == 0 or bank_id == 2) {
                    noc_async_read_one_packet(1, noc_read_addr, cb_addr, read_bytes);
                } else {
                    noc_async_read_one_packet(noc_read_addr, cb_addr, read_bytes);
                }
            }
            #else

            uint64_t noc_read_addr = s.get_noc_addr(input_start_tile_id);

            if constexpr(rr_vc) {
                noc_async_read_one_packet(noc_read_addr, cb_addr, read_bytes, input_start_tile_id & NOC_UNICAST_READ_REQ_VC_RANGE_MASK);
            } else {
                noc_async_read_one_packet(noc_read_addr, cb_addr, read_bytes);
            }

            #endif

            cb_addr += read_bytes;
            input_start_tile_id++;
        }

        #ifdef HYBRID_NOC
        noc_async_read_barrier(1);
        #endif
        noc_async_read_barrier();
    }
}
