#pragma once


#include "compute_kernel_api/common_globals.h"
#include "compute_kernel_api/pack.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_relu.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

//RELU MAX-MIN ops
ALWI void relu_max_tile(uint32_t idst,uint32_t param0) {
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_unary_sfpu_relu_max<APPROX, SyncHalf>(idst,param0) ));
    #else
    MATH(( llk_math_eltwise_unary_sfpu_relu_max<APPROX, SyncHalf>(idst, Dim::RC, param0) ));
    #endif
}

ALWI void relu_max_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_relu_max_init<APPROX>() ));
}

ALWI void relu_min_tile(uint32_t idst,uint32_t param0) {
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_unary_sfpu_relu_min<APPROX, SyncHalf>(idst,param0) ));
    #else
    MATH(( llk_math_eltwise_unary_sfpu_relu_min<APPROX, SyncHalf>(idst, Dim::RC, param0) ));
    #endif
}

ALWI void relu_min_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_relu_min_init<APPROX>() ));
}

//Leaky Relu : y = relu(x) + slope*-relu(-x)
ALWI void leaky_relu_tile(uint32_t idst,uint32_t param0) {
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_unary_sfpu_leaky_relu<APPROX, SyncHalf>(idst,param0) ));
    #else
    MATH(( llk_math_eltwise_unary_sfpu_lrelu<APPROX, SyncHalf>(idst, Dim::RC, param0) ));
    #endif
}

ALWI void leaky_relu_tile_init() {
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_unary_sfpu_leaky_relu_init<APPROX>() ));
    #else
    MATH(( llk_math_eltwise_unary_sfpu_lrelu_init<APPROX>() ));
    #endif
}

// relu is implemented via unpack with llk_pack_relu_config(0) enabled
ALWI void pack_relu_tile_to_stream(uint32_t idst, uint32_t cbid) {
    PACK(( llk_pack<false, SYNC, false >(idst, cbid) ));
}

ALWI void pack_relu_config(uint32_t enable) {
    PACK(( llk_pack_relu_config(enable) ));
}

} // namespace ckernel
