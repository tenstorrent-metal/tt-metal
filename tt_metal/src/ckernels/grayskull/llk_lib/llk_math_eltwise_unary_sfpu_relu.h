/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "ckernel_sfpu_relu.h"

namespace ckernel {

// New LLK SFPU APIs

// PRELU
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_prelu_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(sfpu::prelu_init<APPROXIMATE>);
}
template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_prelu(uint dst_index, uint param0) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::prelu<APPROXIMATE,4>,
                                ckernel::sfpu::prelu<APPROXIMATE,4>,
				 dst_index, Dim::RC, param0);
}

// RELU MAX
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_relu_max_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(sfpu::relu_max_init<APPROXIMATE>);
}
template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_relu_max(uint dst_index, uint param0) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::relu_max<APPROXIMATE,4>,
                                ckernel::sfpu::relu_max<APPROXIMATE,4>,
                                dst_index, Dim::RC, param0);
}

// RELU MIN
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_relu_min_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(sfpu::relu_min_init<APPROXIMATE>);
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_relu_min(uint dst_index, uint param0 = 0) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::relu_min<APPROXIMATE,4>,
                                ckernel::sfpu::relu_min<APPROXIMATE,4>,
                                dst_index, Dim::RC, param0);
}

// RELU
//RELU - implemented by relu-min
//relu = relu_min @ threshold = 0
template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_relu(uint dst_index) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::relu_min<APPROXIMATE,4>,
                                ckernel::sfpu::relu_min<APPROXIMATE,4>,
                                dst_index, Dim::RC, 0);

}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_relu_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(sfpu::relu_min_init<APPROXIMATE>);
}


// LEAKY RELU
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_lrelu_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(sfpu::lrelu_init<APPROXIMATE>);
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_lrelu(uint dst_index, int param0 = 0) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::calculate_lrelu<APPROXIMATE,4>,
				 ckernel::sfpu::calculate_lrelu<APPROXIMATE,4>,
				 dst_index, Dim::RC, param0);
}

}
