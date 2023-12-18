// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {


template <bool APPROXIMATION_MODE, int ITERATIONS=4>
inline void calculate_mask()
{
    const bool exponent_size_8 = true;
    const int mask_val_idx = 16;
    #pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat mask = dst_reg[mask_val_idx];
        v_if(_sfpu_is_fp16_zero_(mask, exponent_size_8)) {
            dst_reg[0] = 0;
        }
        v_endif;
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
