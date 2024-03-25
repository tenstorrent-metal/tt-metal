// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/softmax.h"
#include "compute_kernel_api/reduce.h"
#include "debug/dprint.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t block_h                        = get_compile_time_arg_val(0);
    constexpr uint32_t block_w                        = get_compile_time_arg_val(1);
    constexpr uint32_t subblock_w                     = get_compile_time_arg_val(2);
    constexpr uint32_t num_subblocks_w                = get_compile_time_arg_val(3);

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in1, tt::CB::c_intermed0);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_bcast_scaler = tt::CB::c_in1;
    constexpr auto cb_fused_scale = tt::CB::c_in2;
    constexpr auto cb_fused_attn = tt::CB::c_in3;
    constexpr auto cb_exps = tt::CB::c_intermed0;
    constexpr auto cb_recipsumexps = tt::CB::c_intermed1;
    constexpr auto cb_scale_mask = tt::CB::c_intermed2;
    constexpr auto cb_out0 = tt::CB::c_out0;

    constexpr int dst0 = 0;
    int index_subblock_w_offset = 0;
    int index = 0;

    #ifdef SHARDED_CAUSAL_MASK
    DPRINT << "SHARDED_CAUSAL_MASK" << ENDL();
    #else
    // DPRINT << "NOT SHARDED_CASUAL_MASK" << ENDL();
    #endif

    for (uint32_t i = 0; i < block_h; i++) {
        DPRINT << "Running iteration: " << i << "out of " << block_h - 1 << " iterations." << ENDL();
        #if FUSED_SCALE_MASK
        // fused scale
        unpack_reconfig_data_format(cb_in0, cb_fused_scale);
        pack_reconfig_data_format(cb_scale_mask);
        cb_wait_front(cb_fused_scale, 1);
        // UNPACK(( DPRINT  << TSLICE(cb_fused_scale, 0, SliceRange::h0_w0_32()) << ENDL() ));
        mul_tiles_bcast_scalar_init_short();
        index_subblock_w_offset = 0;

        if (i == 4) {
            DPRINT << "Starting loop 1..." << ENDL();
        }
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            ACQ();
            cb_reserve_back(cb_scale_mask, subblock_w);
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                mul_tiles_bcast_scalar(cb_in0, cb_fused_scale, index, 0, w);
                pack_tile(w, cb_scale_mask);
            }
            cb_push_back(cb_scale_mask, subblock_w);
            REL();
            index_subblock_w_offset += subblock_w;
        }

        if (i == 4) {
            DPRINT << "Ending loop 1..." << ENDL();
        }
        cb_pop_front(cb_in0, block_w);
        unpack_reconfig_data_format(cb_scale_mask, cb_fused_attn);

        // fused attn
        cb_wait_front(cb_scale_mask, block_w);

        #ifndef SHARDED_CAUSAL_MASK
            if (i == 4) DPRINT << "Wait fused attn " << ENDL();
            cb_wait_front(cb_fused_attn, block_w);
            if (i == 4) DPRINT << "Wait fused attn done" << ENDL();
        #endif

        index_subblock_w_offset = 0;

        #ifdef CAUSAL_MASK
            add_tiles_init();
        #else
            add_bcast_rows_init_short();
        #endif

        exp_tile_init(EXP_APPROX);
        // if (i == 4) {
        //     DPRINT << "Starting loop 2..." << ENDL();
        // }
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            ACQ();
            #ifdef CAUSAL_MASK
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles(cb_scale_mask, cb_fused_attn, index, index, w);
                }
            #else
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles_bcast_rows(cb_scale_mask, cb_fused_attn, index, index, w);
                }
            #endif
            cb_reserve_back(cb_exps, subblock_w);
            for (uint32_t w = 0; w < subblock_w; w++) {
                exp_tile(w,EXP_APPROX);
                pack_tile(w, cb_exps);
            }
            cb_push_back(cb_exps, subblock_w);
            REL();
            index_subblock_w_offset += subblock_w;
        }

        // if (i == 4) {
        //     DPRINT << "Ending loop 2..." << ENDL();
        // }
        cb_pop_front(cb_scale_mask, block_w);

        #ifdef CAUSAL_MASK
            cb_pop_front(cb_fused_attn, block_w);
        #endif
        unpack_reconfig_data_format(cb_exps, cb_bcast_scaler);

        #else
        unpack_reconfig_data_format(cb_in0, cb_in0);
        pack_reconfig_data_format(cb_exps);
        // exp(x)
        index_subblock_w_offset = 0;
        copy_tile_to_dst_init_short();
        exp_tile_init(EXP_APPROX);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            ACQ();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                copy_tile(cb_in0, index, w);
            }
            cb_reserve_back(cb_exps, subblock_w);
            for (uint32_t w = 0; w < subblock_w; w++) {
                exp_tile(w, EXP_APPROX);
                pack_tile(w, cb_exps);
            }
            cb_push_back(cb_exps, subblock_w);
            REL();
            index_subblock_w_offset += subblock_w;
        }
        cb_pop_front(cb_in0, block_w);
        unpack_reconfig_data_format(cb_exps, cb_bcast_scaler);
        #endif // FUSED_SCALE_MASK

        // sum(exp(x))
        ACQ();
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        cb_wait_front(cb_exps, block_w);
        cb_wait_front(cb_bcast_scaler, 1);
        cb_reserve_back(cb_recipsumexps, 1);
        for (uint32_t w = 0; w < block_w; w++) {
            constexpr uint32_t bcast_scaler0 = 0;
            reduce_tile(REDUCE_OP, REDUCE_DIM, cb_exps, cb_bcast_scaler, w, bcast_scaler0, dst0);
        }
        reduce_revert_delta();
        recip_tile_init();
        recip_tile(dst0);
        pack_tile(dst0, cb_recipsumexps);
        cb_push_back(cb_recipsumexps, 1);
        REL();

        // exp(x) / (sum(exp(x)))
        pack_reconfig_data_format(cb_out0);
        cb_wait_front(cb_recipsumexps, 1);
        mul_bcast_cols_init_short();
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            ACQ();
            cb_reserve_back(cb_out0, subblock_w);
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_recipsumexps, index, 0, w);
                pack_tile(w, cb_out0);
            }
            cb_push_back(cb_out0, subblock_w);
            REL();
            index_subblock_w_offset += subblock_w;
        }
        cb_pop_front(cb_recipsumexps, 1);
        cb_pop_front(cb_exps, block_w);
    }

    DPRINT << "DONE" << ENDL();
}
}
