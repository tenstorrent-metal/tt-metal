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

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

inline bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr bool gamma_has_value = get_compile_time_arg_val(4) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(5) == 1;

    binary_op_init_common(tt::CB::c_in1, tt::CB::c_in2);

    constexpr auto cb_dy = tt::CB::c_in0;        // output_grad(==dy)
    constexpr auto cb_x = tt::CB::c_in1;         // input(==x)
    constexpr auto cb_mean = tt::CB::c_in2;      // mean
    constexpr auto cb_rstd = tt::CB::c_in3;      // rstd
    constexpr auto cb_scaler = tt::CB::c_in4;    // scaler
    constexpr auto cb_numel = tt::CB::c_in5;     // normalized_numel(==n)
    constexpr auto cb_gamma = tt::CB::c_in6;     // gamma
    constexpr auto cb_mask_h_w = tt::CB::c_in7;  // mask_h_w

    // dx = ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (1.0 / (n * rstd))
    constexpr auto cb_dx = tt::CB::c_out0;  // input_grad(==dx)

    // y = (x - mean) / rstd
    constexpr auto cb_dycopy = tt::CB::c_intermed0;       // copy output_grad(==dycopy)
    constexpr auto cb_y = tt::CB::c_intermed1;            // output(==y)
    constexpr auto cb_dysum = tt::CB::c_intermed2;        // Sum[dy]
    constexpr auto cb_ydysum = tt::CB::c_intermed3;       // Sum[y * dy]
    constexpr auto cb_recip_nrstd = tt::CB::c_intermed4;  // 1.0 / (n * rstd)

    constexpr auto cb_tmp1 = tt::CB::c_intermed5;  // tmp1
    constexpr auto cb_tmp2 = tt::CB::c_intermed6;  // tmp2
    constexpr auto cb_tmp3 = tt::CB::c_intermed7;  // tmp3

    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_scaler, onetile);  // comes from the reader
    cb_wait_front(cb_numel, onetile);   // comes from the reader

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0 && !is_lastdim_layernorm;
    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;

    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

    if (do_mask_h || do_mask_w) {
        cb_wait_front(cb_mask_h_w, 2);  // comes from the reader
    }

    constexpr uint32_t NCHt = num_rows_per_core;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        cb_wait_front(cb_mean, onetile);  // comes from the reader
        cb_wait_front(cb_rstd, onetile);  // comes from the reader

        // Compute cb_recip_nrstd
        // 1.0 / (n * rstd)
        ACQ();
        cb_reserve_back(cb_recip_nrstd, onetile);

        if (is_lastdim_layernorm) {
            mul_bcast_cols_init_short();
            mul_tiles_bcast_cols(cb_numel, cb_rstd, 0, 0, dst0);
        } else {
            mul_tiles_bcast_scalar_init_short();
            mul_tiles_bcast_scalar(cb_numel, cb_rstd, 0, 0, dst0);
        }

        recip_tile_init();
        recip_tile(dst0);

        pack_tile(dst0, cb_recip_nrstd);

        cb_push_back(cb_recip_nrstd, onetile);
        REL();

        // Compute cb_recip_rstd
        // 1.0 / rstd
        constexpr auto cb_recip_rstd = cb_tmp1;
        ACQ();
        cb_wait_front(cb_recip_nrstd, onetile);
        cb_reserve_back(cb_recip_rstd, onetile);

        mul_tiles_init();
        mul_tiles(cb_numel, cb_recip_nrstd, 0, 0, dst0);

        pack_tile(dst0, cb_recip_rstd);

        cb_push_back(cb_recip_rstd, onetile);
        REL();
        // We don't pop cb_recip_nrstd here.

        // y = (x - mean) / rstd
        cb_wait_front(cb_recip_rstd, onetile);
        cb_reserve_back(cb_y, Wt);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Compute cb_xmm
            // x - mean and mask(optional)
            constexpr auto cb_xmm = cb_tmp2;
            ACQ();
            cb_wait_front(cb_x, onetile);  // comes from the reader
            cb_reserve_back(cb_xmm, onetile);

            if (is_lastdim_layernorm) {
                sub_bcast_cols_init_short();
                sub_tiles_bcast_cols(cb_x, cb_mean, 0, 0, dst0);
            } else {
                sub_tiles_bcast_scalar_init_short();
                sub_tiles_bcast_scalar(cb_x, cb_mean, 0, 0, dst0);
            }

            if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                copy_tile_init();
                copy_tile(cb_mask_h_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }

            if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                copy_tile_init();
                copy_tile(cb_mask_h_w, 1, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }

            pack_tile(dst0, cb_xmm);

            cb_pop_front(cb_x, onetile);
            cb_push_back(cb_xmm, onetile);
            REL();

            // Compute cb_y
            // (x - mean) / rstd
            ACQ();
            cb_wait_front(cb_xmm, onetile);

            mul_tiles_init();
            mul_tiles(cb_xmm, cb_recip_rstd, 0, 0, dst0);

            pack_tile(dst0, cb_y);

            cb_pop_front(cb_xmm, onetile);
            cb_push_back(cb_y, onetile);
            REL();
        }  // Wt loop
        cb_pop_front(cb_recip_rstd, onetile);

        // Copy cb_dy to cb_dycopy
        cb_reserve_back(cb_dycopy, Wt);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            if (gamma_has_value) {
                // Compute cb_dycopy
                // dycopy = dy * gamma and mask(optional)
                ACQ();
                cb_wait_front(cb_dy, onetile);     // comes from the reader
                cb_wait_front(cb_gamma, onetile);  // comes from the reader

                if (is_lastdim_layernorm) {
                    mul_bcast_rows_init_short();
                    mul_tiles_bcast_rows(cb_dy, cb_gamma, 0, 0, dst0);
                } else {
                    mul_tiles_init();
                    mul_tiles(cb_dy, cb_gamma, 0, 0, dst0);
                }

                if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                    copy_tile_init();
                    copy_tile(cb_mask_h_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                    copy_tile_init();
                    copy_tile(cb_mask_h_w, 1, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                pack_tile(dst0, cb_dycopy);

                cb_pop_front(cb_dy, onetile);
                cb_pop_front(cb_gamma, onetile);
                cb_push_back(cb_dycopy, onetile);
                REL();
            } else {
                // Compute cb_dycopy
                // dycopy = dy and mask(optional)
                ACQ();
                cb_wait_front(cb_dy, onetile);  // comes from the reader

                copy_tile_init();
                copy_tile(cb_dy, 0, dst0);

                if (do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt)) {
                    copy_tile_init();
                    copy_tile(cb_mask_h_w, 0, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((wt + 1) % origin_Wt == 0)) {
                    copy_tile_init();
                    copy_tile(cb_mask_h_w, 1, dst1);

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                pack_tile(dst0, cb_dycopy);

                cb_pop_front(cb_dy, onetile);
                cb_push_back(cb_dycopy, onetile);
                REL();
            }
        }  // Wt loop

        // Compute cb_dyadd
        constexpr auto cb_dyadd = cb_tmp1;
        cb_wait_front(cb_dycopy, Wt);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            if (wt == 0) {
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
                add_tiles(cb_dyadd, cb_dycopy, 0, wt, dst0);

                pack_tile(dst0, cb_dyadd);

                cb_pop_front(cb_dyadd, onetile);
                cb_push_back(cb_dyadd, onetile);
                REL();
            }
        }  // Wt loop
        // We don't pop cb_dycopy here.

        // Compute cb_dysum
        // Sum[dy]
        ACQ();
        cb_wait_front(cb_dyadd, onetile);
        cb_reserve_back(cb_dysum, onetile);

        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        reduce_tile(REDUCE_OP, REDUCE_DIM, cb_dyadd, cb_scaler, 0, 0, dst0);
        reduce_revert_delta();

        pack_tile(dst0, cb_dysum);

        cb_pop_front(cb_dyadd, onetile);
        cb_push_back(cb_dysum, onetile);
        REL();

        // Compute cb_ydy and cb_ydyadd
        constexpr auto cb_ydy = cb_tmp2;
        constexpr auto cb_ydyadd = cb_tmp3;
        cb_wait_front(cb_y, Wt);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Compute cb_ydy
            ACQ();
            cb_reserve_back(cb_ydy, onetile);

            mul_tiles_init();
            mul_tiles(cb_y, cb_dycopy, wt, wt, dst0);

            pack_tile(dst0, cb_ydy);

            cb_push_back(cb_ydy, onetile);
            REL();

            // Compute cb_ydyadd
            if (wt == 0) {
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
        }  // Wt loop
        // We don't pop cb_y here.

        // Compute cb_ydysum
        // Sum[y * dy]
        ACQ();
        cb_wait_front(cb_ydyadd, onetile);
        cb_reserve_back(cb_ydysum, onetile);

        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        reduce_tile(REDUCE_OP, REDUCE_DIM, cb_ydyadd, cb_scaler, 0, 0, dst0);
        reduce_revert_delta();

        pack_tile(dst0, cb_ydysum);

        cb_pop_front(cb_ydyadd, onetile);
        cb_push_back(cb_ydysum, onetile);
        REL();

        // Compute cb_dx
        // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (1.0 / (n * rstd))
        cb_wait_front(cb_dysum, onetile);
        cb_wait_front(cb_ydysum, onetile);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Compute cb_ndy
            // n * dy
            constexpr auto cb_ndy = cb_tmp1;
            ACQ();
            cb_reserve_back(cb_ndy, onetile);

            mul_tiles_init();
            mul_tiles(cb_numel, cb_dycopy, 0, wt, dst0);

            pack_tile(dst0, cb_ndy);

            cb_push_back(cb_ndy, onetile);
            REL();

            // cb_ndymdysum
            // n * dy - Sum[dy]
            constexpr auto cb_ndymdysum = cb_tmp2;
            ACQ();
            cb_wait_front(cb_ndy, onetile);
            cb_reserve_back(cb_ndymdysum, onetile);

            if (is_lastdim_layernorm) {
                sub_bcast_cols_init_short();
                sub_tiles_bcast_cols(cb_ndy, cb_dysum, 0, 0, dst0);
            } else {
                sub_tiles_bcast_scalar_init_short();
                sub_tiles_bcast_scalar(cb_ndy, cb_dysum, 0, 0, dst0);
            }

            pack_tile(dst0, cb_ndymdysum);

            cb_pop_front(cb_ndy, onetile);
            cb_push_back(cb_ndymdysum, onetile);
            REL();

            // Compute cb_yydysum
            // y * Sum[y * dy]
            constexpr auto cb_yydysum = cb_tmp3;
            ACQ();
            cb_reserve_back(cb_yydysum, onetile);

            if (is_lastdim_layernorm) {
                mul_bcast_cols_init_short();
                mul_tiles_bcast_cols(cb_y, cb_ydysum, wt, 0, dst0);
            } else {
                mul_tiles_bcast_scalar_init_short();
                mul_tiles_bcast_scalar(cb_y, cb_ydysum, wt, 0, dst0);
            }

            pack_tile(dst0, cb_yydysum);

            cb_push_back(cb_yydysum, onetile);
            REL();

            // Compute cb_tmp1
            // (n * dy - Sum[dy]) - (y * Sum[y * dy])
            ACQ();
            cb_wait_front(cb_ndymdysum, onetile);
            cb_wait_front(cb_yydysum, onetile);
            cb_reserve_back(cb_tmp1, onetile);

            sub_tiles_init();
            sub_tiles(cb_ndymdysum, cb_yydysum, 0, 0, dst0);

            pack_tile(dst0, cb_tmp1);

            cb_pop_front(cb_ndymdysum, onetile);
            cb_pop_front(cb_yydysum, onetile);
            cb_push_back(cb_tmp1, onetile);
            REL();

            // Compute cb_dx
            // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (1.0 / (n * rstd))
            ACQ();
            cb_wait_front(cb_tmp1, onetile);
            cb_reserve_back(cb_dx, onetile);

            mul_tiles_init();
            mul_tiles(cb_tmp1, cb_recip_nrstd, 0, 0, dst0);

            pack_tile(dst0, cb_dx);

            cb_pop_front(cb_tmp1, onetile);
            cb_push_back(cb_dx, onetile);
            REL();
        }  // Wt loop
        cb_pop_front(cb_dycopy, Wt);
        cb_pop_front(cb_y, Wt);

        cb_pop_front(cb_recip_nrstd, onetile);
        cb_pop_front(cb_dysum, onetile);
        cb_pop_front(cb_ydysum, onetile);

        cb_pop_front(cb_mean, onetile);
        cb_pop_front(cb_rstd, onetile);
    }  // NCHt loop
    cb_pop_front(cb_scaler, onetile);
    cb_pop_front(cb_numel, onetile);

    if (do_mask_h || do_mask_w) {
        cb_wait_front(cb_mask_h_w, 2);
    }
}  // void MAIN
}  // namespace NAMESPACE
