/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_relu.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {


/**
 * Performs element-wise computation of PReLU = max(0,x) + weight*min(0,x) each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | weight         | scaling of positive portion                                                | uint32_t | Greater than 0                                        | True     |
 */
ALWI void prelu_tile(uint32_t idst,uint32_t param0) {
  MATH(( llk_math_eltwise_unary_sfpu_prelu<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void prelu_tile_init() {
  MATH(( llk_math_eltwise_unary_sfpu_prelu_init<APPROX>() ));
}

/**
 * Performs element-wise computation of relu max (relu(max(x, upper_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | upper_limit    | Lowe limit of relu_max                                                    | uint32_t | Greater than 0                                        | True     |
 */
ALWI void relu_max_tile(uint32_t idst,uint32_t param0) {
  MATH(( llk_math_eltwise_unary_sfpu_relu_max<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void relu_max_tile_init() {
  MATH(( llk_math_eltwise_unary_sfpu_relu_max_init<APPROX>() ));
}

/**
 * Performs element-wise computation of relu min (relu(min(x, lower_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | lower_limit    | Upper limit of relu_min                                                    | uint32_t | Greater than 0                                        | True     |
 */
ALWI void relu_min_tile(uint32_t idst,uint32_t param0) {
  MATH(( llk_math_eltwise_unary_sfpu_relu_min<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void relu_min_tile_init() {
  MATH(( llk_math_eltwise_unary_sfpu_relu_min_init<APPROX>() ));
}

/**
 * Performs element-wise computation of relu (0 if negative else 1) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void relu_tile(uint32_t idst) {
  MATH(( llk_math_eltwise_unary_sfpu_relu<APPROX, SyncHalf>(idst) ));
}

ALWI void relu_tile_init() {
  MATH(( llk_math_eltwise_unary_sfpu_relu_init<APPROX>() ));
}

/**
 * Performs element-wise computation of leaky relu (relu(x) + slope*-relu(-x)) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | slope          | slope used in leaky relu calculation                                       | uint32_t | Greater than 0                                        | True     |
 */
ALWI void leaky_relu_tile(uint32_t idst,uint32_t param0) {
  MATH(( llk_math_eltwise_unary_sfpu_lrelu<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void leaky_relu_tile_init() {
  MATH(( llk_math_eltwise_unary_sfpu_lrelu_init<APPROX>() ));
}

} // namespace ckernel
