// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary.h"
#include "llk_math_matmul.h"
#include "llk_math_common.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB.h"
#include "llk_unpack_A.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack.h"
#include "llk_pack_common.h"
#endif


namespace ckernel {

/**
 * Shorthand template instantiation of sub_tiles_bcast.
 */
ALWI void sub_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWSUB, BroadcastType::COL, SyncHalf, MATH_FIDELITY, false>(idst) ));
    #else
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWSUB, BroadcastType::COL, SyncHalf, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst) ));
    #endif
    UNPACK(( llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1) ));
}

/**
 * Shorthand template instantiation of sub_tiles_bcast.
 */
ALWI void sub_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWSUB, BroadcastType::SCALAR, SyncHalf, MATH_FIDELITY, false>(idst) ));
    #else
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWSUB, BroadcastType::SCALAR, SyncHalf, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst) ));
    #endif
    UNPACK(( llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1) ));
}

/**
 * Shorthand template instantiation of mul_tiles_bcast.
 */
ALWI void mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::COL, SyncHalf, MATH_FIDELITY, false>(idst) ));
    #else
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::COL, SyncHalf, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst) ));
    #endif
    UNPACK(( llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1) ));
}

/**
 * Shorthand template instantiation of mul_tiles_bcast.
 */
ALWI void mul_tiles_bcast_rows(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, SyncHalf, MATH_FIDELITY, false>(idst) ));
    #else
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, SyncHalf, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst) ));
    #endif
    UNPACK(( llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1) ));
}

/**
 * Please refer to documentation for sub_tiles_bcast
 */
ALWI void add_tiles_bcast_rows(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWADD, BroadcastType::ROW, SyncHalf, MATH_FIDELITY, false>(idst) ));
    #else
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWADD, BroadcastType::ROW, SyncHalf, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst) ));
    #endif
    UNPACK(( llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1) ));
}

/**
 * Please refer to documentation for sub_tiles_bcast
 */
ALWI void add_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWADD, BroadcastType::COL, SyncHalf, MATH_FIDELITY, false>(idst) ));
    #else
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWADD, BroadcastType::COL, SyncHalf, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst) ));
    #endif
    UNPACK(( llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1) ));
}



/**
 * Associated init function that must be called before calling a bcast op.
 *
 * Return value: None
 *
 *
 * | Argument       | Description                                                   | Type          | Valid Range                                    | Required |
 * |----------------|---------------------------------------------------------------|---------------|------------------------------------------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t      | 0 to 31                                        | True     |
 * | icb1           | The indentifier of the circular buffer (CB) containing B      | uint32_t      | 0 to 31                                        | True     |
 * | ocb            | The indentifier of the circular buffer (CB) containing output | uint32_t      | 0 to 31                                        | False    |
 */
template<EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>
void init_bcast(uint32_t icb0, uint32_t icb1, uint32_t ocb = 16)
{
    if constexpr (tBcastOp == ELWMUL)
        MATH(( llk_math_eltwise_binary_init<tBcastOp, tBcastDim, MATH_FIDELITY>() ));
    else
        MATH(( llk_math_eltwise_binary_init<tBcastOp, tBcastDim>() ));

    UNPACK(( llk_setup_operands() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<tBcastDim>() )); // TODO(AP): move out init
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<tBcastDim>(icb0, icb1) ));
    #else
    UNPACK(( llk_unpack_AB_init<tBcastDim>(icb0, icb1) ));
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated(icb0, icb1) ));
    #endif
    // TODO(AP): running this specific init after common AB init causes a hang

    // clone of general init for AB TODO(AP): commonize
    //UNPACK(( llk_setup_operands() ));
    //UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
    //UNPACK(( llk_unpack_AB_hw_configure_disaggregated<BroadcastType::NONE>(icb0, icb1) ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(ocb) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    MATH(( llk_math_pack_sync_init<SyncHalf>() ));
}


/*
Internal helper function for all broadcast ops
*/
template<EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>
ALWI void any_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_binary<tBcastOp, tBcastDim, SyncHalf, MATH_FIDELITY, false>(idst) ));
    #else
    MATH(( llk_math_eltwise_binary<tBcastOp, tBcastDim, SyncHalf, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst) ));
    #endif

    UNPACK(( llk_unpack_AB<tBcastDim>(icb0, icb1, itile0, itile1) ));
}

/**
 * This documentation applies to either one of the 3 broadcast operation variants -
 * *add_tiles_bcast*, *sub_tiles_bcast* and *mul_tiles_bcast*.
 *
 * The description below describes *add_tiles_bcast*, the other 2 operations
 * use the same definition with the corresponding substitution of the math
 * operator.
 *
 * Performs a broadcast-operation *C=A+B* of tiles in two CBs at given indices
 * and writes the result to the DST register at index dst_tile_index. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call
 * is blocking and is only available on the compute engine.
 *
 * Broadcasting semantics are defined as follows:
 *
 * For *dim==BroadcastType::COL*, the input in *B* is expected to be a single tile with a
 * filled 0-column and zeros elsewhere.  The result is *C[h, w] = A[h,w] + B[w]*
 *
 * For *dim==Dim::C*, the input in *B* is expected to be a single tile with a
 * filled 0-row, and zeros elsewhere.  The result is *C[h, w] = A[h,w] + B[h]*
 *
 * For *dim==Dim::RC*, the input in *B* is expected to be a single tile with a
 * filled single value at location [0,0], and zeros elsewhere.  The result is
 * *C[h, w] = A[h,w] + B[0,0]*
 *
 * Return value: None
 *
 * DOX-TODO(AP): verify that the bcast tile is actually required to be filled
 * with zeros.
 *
 * | Argument       | Description                                              | Type          | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|---------------|------------------------------------------------|----------|
 * | tBcastDim      | Broadcast dimension                                      | BroadcastType | One of Dim::R, Dim::C, Dim::RC.                | True     |
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t      | 0 to 31                                        | True     |
 * | in1_cb_id      | The indentifier of the circular buffer (CB) containing B | uint32_t      | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t      | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t      | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t      | Must be less than the acquired size of DST REG | True     |
 */
template<BroadcastType tBcastDim>
ALWI void add_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast<EltwiseBinaryType::ELWADD, tBcastDim>(icb0, icb1, itile0, itile1, idst); }

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
template<BroadcastType tBcastDim>
ALWI void sub_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast<EltwiseBinaryType::ELWSUB, tBcastDim>(icb0, icb1, itile0, itile1, idst); }

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
template<BroadcastType tBcastDim>
ALWI void mul_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast<EltwiseBinaryType::ELWMUL, tBcastDim>(icb0, icb1, itile0, itile1, idst); }

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_rows to be executed correctly.
 * Required to be called before add_tiles_bcast if using column as broadcast type
 */
ALWI void add_bcast_rows_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWADD, BroadcastType::ROW>() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>() ));
    #else
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>(0, 1) ));
    #endif
}

ALWI void add_bcast_rows_init_short_post_matmul()
{
    // math
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_matmul_init<MATH_FIDELITY, DstTileFaceLayout::RowMajor>() ));
    MATH(( llk_math_eltwise_binary_init<ELWADD, BroadcastType::ROW>() ));
    MATH(( llk_math_pack_sync_init<SYNC>()  ));
    #else
    MATH(( llk_math_matmul_init<MATH_FIDELITY, DstTileFaceLayout::RowMajor>(0, 1)  ));
    MATH(( llk_math_eltwise_binary_init<ELWADD, BroadcastType::ROW, MATH_FIDELITY>() ));
    MATH(( llk_math_pack_sync_init<SYNC>()  ));
    #endif
    // unpacker
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_A_init_cm<BroadcastType::NONE, false, 0>(0, 255) ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>() ));
    #else
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>()  ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>(0, 1) ));
    #endif
    // packer
    // #ifdef ARCH_GRAYSKULL
    PACK(( llk_pack_init<false, false, DstTileFaceLayout::RowMajor>()  ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>()  ));
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>()  ));
    // #endif
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_cols to be executed correctly.
 * Required to be called before add_tiles_bcast if using column as broadcast type
 */
ALWI void add_bcast_cols_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWADD, BroadcastType::COL>() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>() ));
    #else
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>(0, 1) ));
    #endif
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for mul_bcast_cols to be executed correctly.
 */
ALWI void mul_tiles_bcast_scalar_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWMUL, BroadcastType::SCALAR, MATH_FIDELITY>() )); // TODO(AP)
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<BroadcastType::SCALAR>() ));
    #else
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<BroadcastType::SCALAR>(0, 1) ));
    #endif
}

/**
 * Performs a broadcast-multiply of a tile from icb0[itile0] with a scalar encoded as a tile from icb1[itile1].
 */
ALWI void mul_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_binary<ELWMUL, BroadcastType::SCALAR, SyncHalf, MATH_FIDELITY, false>(idst) ));
    #else
    MATH(( llk_math_eltwise_binary<ELWMUL, BroadcastType::SCALAR, SyncHalf, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst) ));
    #endif
    UNPACK(( llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1) ));
}


/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for mul_bcast_cols to be executed correctly.
 */
ALWI void mul_bcast_cols_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWMUL, BroadcastType::COL, MATH_FIDELITY>() )); // TODO(AP)
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>() ));
    #else
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>(0, 1) ));
    #endif
}

/**
 * Performs a switch-from-another-op tile hw reconfiguration step needed for mul_bcast_rows to be executed correctly.
 */
ALWI void mul_bcast_rows_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWMUL, BroadcastType::ROW, MATH_FIDELITY>() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>() ));
    #else
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>(0, 1) ));
    #endif
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for sub_bcast_cols to be executed correctly.
 */
ALWI void sub_bcast_cols_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWSUB, BroadcastType::COL, MATH_FIDELITY>() )); // TODO(AP)
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>() ));
    #else
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>(0, 1) ));
    #endif
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for sub_tiles_bcast_scalar to be executed correctly.
 */
ALWI void sub_tiles_bcast_scalar_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWSUB, BroadcastType::SCALAR, MATH_FIDELITY>() )); // TODO(AP)
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<BroadcastType::SCALAR>() ));
    #else
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<BroadcastType::SCALAR>(0, 1) ));
    #endif
}

} // namespace ckernel
