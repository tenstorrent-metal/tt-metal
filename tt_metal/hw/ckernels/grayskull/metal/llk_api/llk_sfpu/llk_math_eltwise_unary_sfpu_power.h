// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "ckernel_sfpu_power_iterative.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_power_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_power(uint dst_index, int pow = 0, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE>
                                (ckernel::sfpu::calculate_power_iterative<APPROXIMATE>,
                                ckernel::sfpu::calculate_power_iterative<APPROXIMATE>,
                                dst_index, vector_mode, pow);
}

}
