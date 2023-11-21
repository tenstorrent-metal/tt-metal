// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_sfpu.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_format_conversions.h"
#include "llk_math_common.h"
#include "llk_param_structs.h"


namespace ckernel {

template <SfpuType sfpu_type>
void static_assert_sfpu_type_dependent() {
    static_assert(sfpu_type == SfpuType::unused, "sfpu_type exception");
}
// local function declarations
inline void eltwise_unary_sfpu_configure_addrmod();
inline void eltwise_unary_sfpu_configure_mop();

template <SfpuType sfpu_op, bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu(
    uint dst_index,
    int vector_mode = Dim::RC,
    uint param0 = 0,
    uint param1 = 0,
    uint param2 = 0,
    uint param3 = 0,
    uint param4 = 0,
    uint param5 = 0) {
    if constexpr ((Dst == DstSync::SyncTile16) || (Dst == DstSync::SyncTile2)) {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(math_sync_tile_dst_index);
    } else {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    }
    if (vector_mode == Dim::R) {
        // Do a row vector, Face0 + Face1 -- first iteration
        const int ITERATIONS = 1;
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            sfpu::calculate_sfpu<sfpu_op, APPROXIMATE, 0, ITERATIONS>(param0, param1, param2, param3, param4, param5);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
        // Skip the next 2 faces
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    } else if (vector_mode == Dim::C) {
        // Do a column vector, Face0 + Face2 -- full face
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            sfpu::calculate_sfpu<sfpu_op, APPROXIMATE>(param0, param1, param2, param3, param4, param5);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    } else {
#pragma GCC unroll 0
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
        for (int face = 0; face < 4; face++) {
            sfpu::calculate_sfpu<sfpu_op, APPROXIMATE>(param0, param1, param2, param3, param4, param5);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }
    math::clear_dst_reg_addr();
}

template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init(
    uint param0 = 0, uint param1 = 0, uint param2 = 0, uint param3 = 0, uint param4 = 0, uint param5 = 0) {
    if constexpr (sfpu_op == SfpuType::dropout) {
        sfpu::sfpu_init<APPROXIMATE>(sfpu_op, param2);
    } else {
        sfpu::sfpu_init<APPROXIMATE>(sfpu_op);
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// New LLK SFPU APIs


template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_rsqrt(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::rsqrt, APPROXIMATE, dst_sync>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rsqrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_log(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::log, APPROXIMATE, dst_sync>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log, APPROXIMATE>();
}

//abs
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_abs(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::abs, APPROXIMATE, dst_sync>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::abs, APPROXIMATE>();
}

//sign
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_sign(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::sign, APPROXIMATE, dst_sync>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sign_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sign, APPROXIMATE>();
}

//log with base
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_log_with_base(uint dst_index, uint base, int vector_mode = Dim::RC) {
  llk_math_eltwise_unary_sfpu<SfpuType::log_with_base, APPROXIMATE, dst_sync>(dst_index, vector_mode, base);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log_with_base_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log_with_base, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_tanh(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::tanh, APPROXIMATE, dst_sync>(dst_index, vector_mode);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_signbit(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::signbit, APPROXIMATE, dst_sync>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_signbit_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::signbit, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_sigmoid(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::sigmoid, APPROXIMATE, dst_sync>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid, APPROXIMATE>();
}

template <DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_dropout(
    uint dst_index, int vector_mode, int integer_dropout, int scale_factor) {
    constexpr bool dont_care = false;
    llk_math_eltwise_unary_sfpu<SfpuType::dropout, dont_care, dst_sync>(
        dst_index, vector_mode, integer_dropout, scale_factor);
}

inline void llk_math_eltwise_unary_sfpu_dropout_init(uint seed = 0) {
    constexpr bool dont_care = false;
    constexpr uint dont_care_param = 0;

    llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, dont_care>(dont_care_param, dont_care_param, seed);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_max(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::max, APPROXIMATE, dst_sync>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_max_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::max, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_square(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::square, APPROXIMATE, dst_sync>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_square_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::square, APPROXIMATE>();
}

// Power : y = x^(const param0)
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_power(uint dst_index, uint pow = 0, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::power, APPROXIMATE, dst_sync>(dst_index, vector_mode, pow);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_power_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::power, APPROXIMATE>();
}


//EQZ
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_eqz(uint dst_index,int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::equal_zero, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_eqz_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::equal_zero, APPROXIMATE>();
}

//NEZ
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_nez(uint dst_index,int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::not_equal_zero, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_nez_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::not_equal_zero, APPROXIMATE>();
}

//LTZ
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull,int vector_mode = Dim::RC>
inline void llk_math_eltwise_unary_sfpu_ltz(uint dst_index) {
    llk_math_eltwise_unary_sfpu<SfpuType::less_than_zero, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_ltz_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::less_than_zero, APPROXIMATE>();
}

//GTZ
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_gtz(uint dst_index,int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::greater_than_zero, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gtz_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::greater_than_zero, APPROXIMATE>();
}

//LEZ
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_lez(uint dst_index,int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::less_than_equal_zero, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_lez_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::less_than_equal_zero, APPROXIMATE>();
}

//GEZ
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_gez(uint dst_index,int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::greater_than_equal_zero, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gez_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::greater_than_equal_zero, APPROXIMATE>();
}

//ELU - implemented in ckernel_sfpu_elu.h


//EXP2
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_exp2(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::exp2, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_exp2_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::exp2, APPROXIMATE>();
}

//heaviside
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_heaviside(uint dst_index,uint param0, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::heaviside, APPROXIMATE, dst_sync>(dst_index,vector_mode,param0);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_heaviside_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::heaviside, APPROXIMATE>();
}

//EXPM1
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_expm1(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::expm1, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_expm1_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::expm1, APPROXIMATE>();
}

//Asin
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_asin(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::asin, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_asin_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::asin, APPROXIMATE>();
}

//Atan
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_atan(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::atan, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_atan_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::atan, APPROXIMATE>();
}

//Acos
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_acos(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::acos, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_acos_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::acos, APPROXIMATE>();
}

//Mask
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_mask(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::mask, APPROXIMATE, dst_sync>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mask_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::mask, APPROXIMATE>();
}
