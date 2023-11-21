/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

ALWI void mask_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_mask_init<true>() )); // TODO(AP): move out init
}

/**
 * Performs element-wise computation of mask on each element of a tile
 * in data and mask DST register. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 *
 * TODO: fix idst2_mask.
 * currently idst2_mask is not used and (idst_data + 1) is used for mask.
 * because don't know how to use 2 dst register with sfpu.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | dst_data_index | The index of the tile in DST REG for the data and result                   | uint32_t | Must be less than the acquired size of DST REG        | True     |
 * | dst_mask_index | The index of the tile in DST REG for the mask                              | uint32_t | Must be less than the acquired size of DST REG        | True     |
 */
ALWI void mask_tile(uint32_t idst_data, uint32_t idst2_mask) {
    MATH(( llk_math_eltwise_unary_sfpu_mask<true, SyncHalf>(idst_data) ));
}

} // namespace ckernel
