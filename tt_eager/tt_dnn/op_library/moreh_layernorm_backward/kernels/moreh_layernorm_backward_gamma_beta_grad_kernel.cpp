// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug_print.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_cols_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t NCHt = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);

    constexpr uint32_t gamma_grad_has_value = get_compile_time_arg_val(4);
    constexpr uint32_t beta_grad_has_value = get_compile_time_arg_val(5);
    constexpr uint32_t is_lastdim_layernorm = get_compile_time_arg_val(6);

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0);

    constexpr auto cb_dy = tt::CB::c_in0;      // output_grad(==dy)
    constexpr auto cb_x = tt::CB::c_in1;       // input(==x)
    constexpr auto cb_mean = tt::CB::c_in2;    // mean
    constexpr auto cb_rstd = tt::CB::c_in3;    // rstd
    constexpr auto cb_scaler = tt::CB::c_in4;  // scaler
    constexpr auto cb_mask_h = tt::CB::c_in5;  // mask_h

    // Sum[y * dy]
    constexpr auto cb_dgamma = tt::CB::c_out0;  // gamma_grad(==dgamma)
    // Sum[dy]
    constexpr auto cb_dbeta = tt::CB::c_out1;  // beta_grad(==dbeta)

    // y = (x - mean) / rstd
    constexpr auto cb_y = tt::CB::c_intermed0;           // output(==y)
    constexpr auto cb_ydy = tt::CB::c_intermed1;         // y * dy
    constexpr auto cb_dyadd = tt::CB::c_intermed2;       // Add[dy]
    constexpr auto cb_ydyadd = tt::CB::c_intermed3;      // Add[y * dy]
    constexpr auto cb_xmm = tt::CB::c_intermed4;         // x - mean
    constexpr auto cb_recip_rstd = tt::CB::c_intermed5;  // 1.0 / rstd
    constexpr auto cb_dycopy = tt::CB::c_intermed6;      // dycopy

    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_scaler, onetile);  // comes from the reader

    constexpr uint32_t TILE_H = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0;
    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;

    if (do_mask_h) {
        cb_wait_front(cb_mask_h, onetile);
    }

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    for (uint32_t w_idx = 0; w_idx < num_cols_per_core; w_idx++) {
        for (uint32_t h_idx = 0; h_idx < NCHt; h_idx++) {
            // Compute cb_dycopy
            // deepcopy and mask(optional)
            ACQ();
            cb_wait_front(cb_dy, onetile);  // comes from the reader
            cb_reserve_back(cb_dycopy, onetile);

            copy_tile_init();
            copy_tile(cb_dy, 0, dst0);

            if (do_mask_h && ((h_idx + 1) % origin_Ht == 0)) {
                copy_tile_init();
                copy_tile(cb_mask_h, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }

            pack_tile(dst0, cb_dycopy);

            cb_pop_front(cb_dy, onetile);
            cb_push_back(cb_dycopy, onetile);
            REL();

            // Compute cb_dyadd
            cb_wait_front(cb_dycopy, onetile);
            if (beta_grad_has_value) {
                if (h_idx == 0) {
                    ACQ();
                    cb_reserve_back(cb_dyadd, onetile);

                    copy_tile_init();
                    copy_tile(cb_dycopy, 0, dst0);

                    pack_tile(dst0, cb_dyadd);

                    cb_push_back(cb_dyadd, onetile);
                    REL();
                } else {
                    ACQ();
                    cb_wait_front(cb_dyadd, onetile);
                    cb_reserve_back(cb_dyadd, onetile);

                    add_tiles_init();
                    add_tiles(cb_dyadd, cb_dycopy, 0, 0, dst0);

                    pack_tile(dst0, cb_dyadd);

                    cb_pop_front(cb_dyadd, onetile);
                    cb_push_back(cb_dyadd, onetile);
                    REL();
                }
            }  // beta_grad_has_value
            // We don't pop cb_dycopy here.

            if (gamma_grad_has_value) {
                // Compute cb_xmm
                // x - mean and mask(optional)
                ACQ();
                cb_wait_front(cb_x, onetile);     // comes from the reader
                cb_wait_front(cb_mean, onetile);  // comes from the reader
                cb_reserve_back(cb_xmm, onetile);

                sub_bcast_cols_init_short();
                sub_tiles_bcast_cols(cb_x, cb_mean, 0, 0, dst0);

                if (do_mask_h && ((h_idx + 1) % origin_Ht == 0)) {
                    copy_tile_init();
                    copy_tile(cb_mask_h, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                pack_tile(dst0, cb_xmm);

                cb_pop_front(cb_x, onetile);
                cb_pop_front(cb_mean, onetile);
                cb_push_back(cb_xmm, onetile);
                REL();

                // Compute cb_recip_rstd
                // 1.0 / rstd
                ACQ();
                cb_wait_front(cb_rstd, onetile);  // comes from the reader
                cb_reserve_back(cb_recip_rstd, onetile);

                copy_tile_init();
                copy_tile(cb_rstd, 0, dst0);

                recip_tile_init();
                recip_tile(dst0);

                pack_tile(dst0, cb_recip_rstd);

                cb_pop_front(cb_rstd, onetile);
                cb_push_back(cb_recip_rstd, onetile);
                REL();

                // Compute cb_y
                // (x - mean) / rstd
                ACQ();
                cb_wait_front(cb_xmm, onetile);
                cb_wait_front(cb_recip_rstd, onetile);
                cb_reserve_back(cb_y, onetile);

                mul_bcast_cols_init_short();
                mul_tiles_bcast_cols(cb_xmm, cb_recip_rstd, 0, 0, dst0);

                pack_tile(dst0, cb_y);

                cb_pop_front(cb_xmm, onetile);
                cb_pop_front(cb_recip_rstd, onetile);
                cb_push_back(cb_y, onetile);
                REL();

                // Compute cb_ydy
                ACQ();
                cb_wait_front(cb_y, onetile);
                cb_reserve_back(cb_ydy, onetile);

                mul_tiles_init();
                mul_tiles(cb_y, cb_dycopy, 0, 0, dst0);

                pack_tile(dst0, cb_ydy);

                cb_pop_front(cb_y, onetile);
                cb_push_back(cb_ydy, onetile);
                REL();

                // Compute cb_ydyadd
                if (h_idx == 0) {
                    ACQ();
                    cb_wait_front(cb_ydy, onetile);
                    cb_reserve_back(cb_ydyadd, onetile);

                    copy_tile_init();
                    copy_tile(cb_ydy, 0, dst0);

                    pack_tile(dst0, cb_ydyadd);

                    cb_pop_front(cb_ydy, onetile);
                    cb_push_back(cb_ydyadd, onetile);
                    REL();
                } else {
                    ACQ();
                    cb_wait_front(cb_ydy, onetile);
                    cb_wait_front(cb_ydyadd, onetile);
                    cb_reserve_back(cb_ydyadd, onetile);

                    add_tiles_init();
                    add_tiles(cb_ydyadd, cb_ydy, 0, 0, dst0);

                    pack_tile(dst0, cb_ydyadd);

                    cb_pop_front(cb_ydy, onetile);
                    cb_pop_front(cb_ydyadd, onetile);
                    cb_push_back(cb_ydyadd, onetile);
                    REL();
                }
            }  // gamma_grad_has_value

            cb_pop_front(cb_dycopy, onetile);
        }  // h_idx loop

        if (gamma_grad_has_value) {
            // Compute cb_dgamma
            // Sum[y * dy]
            ACQ();
            cb_wait_front(cb_ydyadd, onetile);
            cb_reserve_back(cb_dgamma, onetile);

            reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
            reduce_tile(REDUCE_OP, REDUCE_DIM, cb_ydyadd, cb_scaler, 0, 0, dst0);

            pack_tile(dst0, cb_dgamma);

            reduce_revert_delta();
            cb_pop_front(cb_ydyadd, onetile);
            cb_push_back(cb_dgamma, onetile);
            REL();
        }  // gamma_grad_has_value

        if (beta_grad_has_value) {
            // Compute cb_dbeta
            // Sum[dy]
            ACQ();
            cb_wait_front(cb_dyadd, onetile);
            cb_reserve_back(cb_dbeta, onetile);

            reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
            reduce_tile(REDUCE_OP, REDUCE_DIM, cb_dyadd, cb_scaler, 0, 0, dst0);

            pack_tile(dst0, cb_dbeta);

            reduce_revert_delta();
            cb_pop_front(cb_dyadd, onetile);
            cb_push_back(cb_dbeta, onetile);
            REL();
        }  // beta_grad_has_value

    }  // w_idx loop
    cb_pop_front(cb_scaler, onetile);

    if (do_mask_h) {
        cb_pop_front(cb_mask_h, onetile);
    }
}  // void MAIN
}  // namespace NAMESPACE
