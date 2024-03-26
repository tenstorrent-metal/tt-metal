// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "debug/dprint.h"

// HW-bcast scale for fused scale-attn-softmax
FORCE_INLINE void generate_inv_sqrt_hw_bcast_tile() {
    constexpr auto cb_fused_scale = tt::CB::c_in2;
    uint32_t u = get_arg_val<uint32_t>(1);
    cb_reserve_back(cb_fused_scale, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_fused_scale));
    ptr[0] = u>>16;
    cb_push_back(cb_fused_scale, 1);
}

void kernel_main() {

    constexpr uint32_t cb_reduce_scaler = tt::CB::c_in1;
    const uint32_t reduce_scaler = get_arg_val<uint32_t>(0);

    #if FUSED_SCALE_MASK
    constexpr uint32_t block_wt = get_compile_time_arg_val(0);
    constexpr bool is_dram_mask = get_compile_time_arg_val(1) == 1;
    const uint32_t mask_addr  = get_arg_val<uint32_t>(2);
    const uint32_t mask_start_tile_id  = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_attn = tt::CB::c_in3;
    uint32_t mask_tile_bytes = get_tile_size(cb_attn);
    const DataFormat mask_data_format = get_dataformat(cb_attn);
    uint32_t mask_id = mask_start_tile_id;

    const InterleavedAddrGenFast<is_dram_mask> addr_mask = {
        .bank_base_address = mask_addr,
        .page_size = mask_tile_bytes,
        .data_format = mask_data_format
    };

    constexpr auto cb_fused_scale = tt::CB::c_in2;
    const uint32_t pre_scale = get_arg_val<uint32_t>(1);
    generate_bcast_unary_scalar(cb_fused_scale, pre_scale);


    // DPRINT << "Mask start tile id = " << mask_id << ENDL();
    // DPRINT << "Mask address: = " << mask_addr << ENDL();
    #if defined(CAUSAL_MASK) && !defined(SHARDED_CAUSAL_MASK)
        uint32_t fused_head = get_compile_time_arg_val(4);
        uint32_t fused_head_remainder = get_compile_time_arg_val(5);
        //DPRINT << "Fused Head = " << fused_head << " Fused Head Remainder = " << fused_head_remainder << ENDL();
        uint32_t block_ht = fused_head * block_wt + fused_head_remainder;
        uint32_t mask_num_tiles = get_arg_val<uint32_t>(4);
        for (uint32_t h = 0; h < block_ht; h++) {
            cb_reserve_back(cb_attn, block_wt);
            uint32_t l1_write_addr = get_write_ptr(cb_attn);
            for (uint32_t w = 0; w < block_wt; w++) {
                DPRINT << "Issuing read command for mask_tile_id: " << mask_id << ENDL();
                noc_async_read_tile(mask_id, addr_mask, l1_write_addr);
                l1_write_addr += mask_tile_bytes;
                ++mask_id;

                if (h == 0 && w == 0) {
                    generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);
                }
            }
            noc_async_read_barrier();

            cb_push_back(cb_attn, block_wt);
            if (mask_id == mask_num_tiles) {
                mask_id = 0;
            }
        }
        // for (uint32_t f = 0; f<fused_head; f++) {
        //     mask_id = mask_start_tile_id;
        //     for (uint32_t h = 0; h<block_wt; h++) {
        //         cb_reserve_back(cb_attn, block_wt);
        //         uint32_t l1_write_addr = get_write_ptr(cb_attn);
        //         for (uint32_t w = 0; w<block_wt; w++) {
        //             DPRINT << "Issuing read for mask_tile_id: " << mask_id << "From Address: " << addr_mask.get_noc_addr(mask_id) << ENDL();
        //             noc_async_read_tile(mask_id, addr_mask, l1_write_addr);
        //             l1_write_addr += mask_tile_bytes;
        //             ++mask_id;
        //         }
        //         noc_async_read_barrier();

        //         // DPRINT << "Writing the following number of tiles into cb_attn (block_wt): " << block_wt << ENDL();
        //         cb_push_back(cb_attn, block_wt);

        //         // This unhangs it!
        //         // if (h == 1 && block_wt == 2) {
        //         //     cb_push_back(cb_attn, block_wt);
        //         // }

        //         if (f == 0 && h == 0) {
        //             generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);
        //         }
        //     }
        // }
        // // Handle remainder
        // mask_id = mask_start_tile_id;
        // for (uint32_t f = 0; f < fused_head_remainder; f++) {
        //     cb_reserve_back(cb_attn, block_wt);
        //     uint32_t l1_write_addr = get_write_ptr(cb_attn);
        //     for (uint32_t w = 0; w<block_wt; w++) {
        //         noc_async_read_tile(mask_id, addr_mask, l1_write_addr);

        //         DPRINT << "Remainder: Issuing read for mask_tile_id: " << mask_id << "From Address: " << addr_mask.get_noc_addr(mask_id) << ENDL();
        //         l1_write_addr += mask_tile_bytes;
        //         ++mask_id;
        //     }
        //     noc_async_read_barrier();
        //     DPRINT << "Remainder: Writing the following number of tiles into cb_attn (block_wt): " << block_wt << ENDL();
        //     cb_push_back(cb_attn, block_wt);
        // }
    #elif defined(CAUSAL_MASK) && defined(SHARDED_CAUSAL_MASK)
        generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);
    #else
        cb_reserve_back(cb_attn, block_wt);
        uint32_t l1_write_addr = get_write_ptr(cb_attn);
        for (uint32_t w = 0; w<block_wt; w++) {
            noc_async_read_tile(mask_id, addr_mask, l1_write_addr);
            l1_write_addr += mask_tile_bytes;
            ++mask_id;
        }
        noc_async_read_barrier();
        cb_push_back(cb_attn, block_wt);

        generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);
    #endif

    #else
    generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);
    #endif
}
