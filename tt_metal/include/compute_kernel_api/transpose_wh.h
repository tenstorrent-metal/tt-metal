// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_datacopy.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A.h"
#endif


namespace ckernel {


/**
 * Paired Init function for transpose_wh. For general information on init functions refer to any_init.
 *
 *
 * | Argument       | Description                                                 | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | icb            | The identifier of the circular buffer (CB) containing input | uint32_t | 0 to 31                                        | True     |
 */
ALWI void transpose_wh_init(uint32_t icb)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, true>() ));
    #else
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE>(true, true, icb) ));
    #endif

    MATH(( llk_math_pack_sync_init<SyncHalf>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    UNPACK(( llk_setup_operands() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, true, false>() ));
    UNPACK(( llk_unpack_A_hw_configure_disaggregated<BroadcastType::NONE, true, true, false>(0) ));
    #else
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE>(true, true)  ));
    UNPACK(( llk_unpack_A_hw_configure_disaggregated<>(0, true) ));
    #endif
}

ALWI void transpose_wh_init_short(uint32_t icb)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, true>() ));
    #else
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE>(true, true, icb) ));
    #endif

    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, true, false>() ));
    #else
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE>(true, true)  ));
    #endif
}

/**
 * Performs a 32x32 transpose operation *B[w,h] = A[h,w]* on a tile in the CB
 * at a given index and writes the result to the DST register at index
 * dst_tile_index. The DST register buffer must be in acquired state via
 * *acquire_dst* call.
 *
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                             | Type     | Valid Range                                    | Required |
 * |----------------|---------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the circular buffer (CB) containing A | uint32_t | 0 to 31                                        | True     |
 * | in_tile_index  | The index of tile A within the first CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result B       | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
ALWI void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t idst)
{
    UNPACK((
        #ifdef ARCH_GRAYSKULL
        llk_unpack_A<BroadcastType::NONE, true>(icb, itile)
        #else
        llk_unpack_A<BroadcastType::NONE, false>(icb, itile)
        #endif
    ));

    MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(idst) ));
}



} // namespace ckernel
