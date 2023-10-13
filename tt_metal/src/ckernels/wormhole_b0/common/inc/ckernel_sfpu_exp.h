/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

#include "ckernel_sfpu_recip.h"
using namespace sfpi;

namespace ckernel
{
namespace sfpu
{



sfpi_inline vFloat sfpu_exp(vFloat _val)
{
    // If exponent is > -1 extract it and replace with -1
    vFloat val = _val;
    vInt exp = exexp(val);
    v_if (exp >= 0) {
        val = setexp(val, 126);
    }
    v_endif;

    // Run series in Horner form
    vFloat tmp = val * vConst0p8373 + s2vFloat16b(0.863281);
    val = val * tmp + vConst1;

    v_if (exp >= 0) {
        val = val * val;
        for (int s_iter = 0; s_iter < 7; s_iter++) {
            exp = exp - 1;
            // Narrow predication on each loop
            v_and(exp >= 0);
            val = val * val;
        }
    }
    v_endif;

    //: ZERO_NEGATIVE
    v_if( _val < -87.0f) {
        val = 0.0f;
    } v_endif;

    return val;
}


template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE, bool SCALE_EN=false, int ITERATIONS=8>
void calculate_exponential(uint exp_base_scale_factor = 0)
{
    // Unroll 8 best for approx, unroll 0 for precise, compiler figures this out
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        if constexpr(SCALE_EN){
            val = val * s2vFloat16a(exp_base_scale_factor);
        }

        if constexpr (APPROXIMATION_MODE)
        {


            v_if( val < -87.0f ) {
	      dst_reg[0] = 0.0f;
	    } v_else {

            // * by 1/ln2 and add convert to 7.3 FxP format
            vFloat vConstLn2Recip = vConstFloatPrgm0;
            vFloat c23_73 = vConstFloatPrgm1;
            vInt adj_exp = vConstIntPrgm2;
            val = val * vConstLn2Recip + c23_73;

            // Remove Exponent of 7 and bias the Mantissa to 127.
            vInt val_short = adj_exp + reinterpret<vInt>(val);

            // SHL to move integer bits to exponent
            val_short <<= 10 - p_exp::FRAC_BITS;
            dst_reg[0] = reinterpret<vFloat>(val_short);

            // Needed for fused kernels such as math_row_softmax_tables which call calculate_exponential()
            // without using Relu in Packer to clamp -ve Infinity to 0.
            if constexpr (ZERO_NEGATIVE)
            {
                v_if (val_short < 0) {
                    dst_reg[0] = vConst0;
                }
                v_endif;
            }
	    }  v_endif;
        }
        else
        {
            // Force sign to 0 (make number positive)
            vFloat result = sfpu_exp(setsgn(val, 0));

            v_if (val < 0) {
                result = sfpu_reciprocal(result);
            }
            v_endif;

	    dst_reg[0] = result;
        }


	dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_exponential_body(vFloat in)
{
    vFloat out;
    vFloat _val = in;
    if constexpr (APPROXIMATION_MODE)
    {
        constexpr int FRAC_BITS = 3;
        constexpr uint SP_BIAS = 127 << FRAC_BITS;

        // * by 1/ln2 and add convert to 7.3 FxP format
        vFloat vConstLn2Recip = vConstFloatPrgm0;
        vFloat conv = in * vConstLn2Recip;

        // Clear exp bits
        vInt c23_73 = p_exp::C23_73;
        vInt tmp = reinterpret<vInt>(conv) - c23_73;

        // Add bias
        tmp += SP_BIAS;

        // SHL to move integer bits to exponent
        out = reinterpret<vFloat>(tmp << (10 - FRAC_BITS));
    }
    else
    {
        // Force sign to 0 (make number positive)
        out = sfpu_exp(setsgn(in, 0));

        v_if (in < 0) {
            out = sfpu_reciprocal(out);
        }
        v_endif;
    }

    //: ZERO_NEGATIVE
    v_if( _val < -87.0f) {
        out = 0.0f;
    } v_endif;

    return out;
}


template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE>
sfpi_inline vFloat calculate_exponential_body_improved(vFloat _val)
{
    vFloat val = _val;
    vFloat out;
    if constexpr (APPROXIMATION_MODE)
    {
        // * by 1/ln2 and add convert to 7.3 FxP format
        vFloat vConstLn2Recip = vConstFloatPrgm0;
        vFloat c23_73 = vConstFloatPrgm1;
        vInt adj_exp = vConstIntPrgm2;
        val = val * vConstLn2Recip + c23_73;

        // Remove Exponent of 7 and bias the Mantissa to 127.
        vInt val_short = adj_exp + reinterpret<vInt>(val);

        // SHL to move integer bits to exponent
        val_short <<= 10 - p_exp::FRAC_BITS;
        out = reinterpret<vFloat>(val_short);

        // Needed for fused kernels such as math_row_softmax_tables which call calculate_exponential()
        // without using Relu in Packer to clamp -ve Infinity to 0.
        if constexpr (ZERO_NEGATIVE)
        {
            v_if (val_short < 0) {
                out = vConst0;
            }
            v_endif;
        }
    }
    else
    {
        // Force sign to 0 (make number positive)
        out = sfpu_exp(setsgn(val, 0));
        v_if (val < 0) {
            out = sfpu_reciprocal(out);
        }
        v_endif;
    }

    //: ZERO_NEGATIVE
    v_if( _val < -87.0f) {
        out = 0.0f;
    } v_endif;

    return out;
}

template <bool APPROXIMATION_MODE>
void exp_init() {

    if constexpr (APPROXIMATION_MODE) {
        vConstFloatPrgm0 = 1.442695f; // ln2_recip
        vConstFloatPrgm1 = s2vFloat16b(p_exp::C23_73);
        vConstFloatPrgm2 = s2vFloat16b(p_exp::ADJ_EXP);
    }
    else{
        vConstFloatPrgm0 = 1.442695f; // ln2_recip
        vConstFloatPrgm1 = 2.0f;
        vConstFloatPrgm2 = 0.863281f;
    }
}

} // namespace sfpu
} // namespace ckernel
