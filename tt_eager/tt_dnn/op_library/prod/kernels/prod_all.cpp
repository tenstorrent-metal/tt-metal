// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "tt_metal/include/compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/negative.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }
namespace NAMESPACE {
void MAIN {

    uint32_t num_tiles = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_intermed0, tt::CB::c_out0);
    bool last_tile = false;
    bool once = true;
    for (uint32_t t = 0; t < num_tiles; t++) {
        if ( t == (num_tiles - 1))
        {
            last_tile = true;
        }
        cb_reserve_back(tt::CB::c_out0, 1);
        for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            ACQ();
            cb_wait_front(tt::CB::c_in0, 1);
            if (once)
            {
                cb_reserve_back(tt::CB::c_intermed0, 1);
                copy_tile_to_dst_init_short();
                copy_tile(tt::CB::c_in0, 0, 0); // copy from c_in[0] to DST[0]
                if (num_tiles == 1)
                    pack_tile(0, tt::CB::c_out0);
                else
                {
                    pack_tile(0, tt::CB::c_intermed0);
                    cb_push_back(tt::CB::c_intermed0, 1);
                }
            }
            REL();
            ACQ();
            mul_tiles_init();
            if (!once)
            {
                mul_tiles(tt::CB::c_in0, tt::CB::c_intermed0, 0, 0, 0);
                if (last_tile)
                {
                    pack_tile(0, tt::CB::c_out0);
                }
                else
                {
                    cb_pop_front(tt::CB::c_intermed0, 1);
                    cb_reserve_back(tt::CB::c_intermed0, 1);
                    pack_tile(0, tt::CB::c_intermed0);
                    cb_push_back(tt::CB::c_intermed0, 1);
                }
            }
            once = false;
            cb_pop_front(tt::CB::c_in0, 1);
            REL();
        }
        cb_push_back(tt::CB::c_out0, 1);
}
}
}
