// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "tt_eager/tt_dnn/kernels/compute/moreh_common.hpp"


namespace NAMESPACE {
void MAIN {
    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_exps = tt::CB::c_intermed0;
    constexpr auto cb_recipsumexps = tt::CB::c_intermed1;
    constexpr auto cb_add = tt::CB::c_intermed2;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_in0, cb_exps);

    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;

    // packer_l1_acc_test
    {
        for (uint32_t n = 0; n < N; ++n) {
            // step 1, compute exp(x)
            cb_reserve_back(cb_add, onetile); // this cb is float32 format
            for (uint32_t i = 0; i < dim_size; ++i) {
                ACQ();
                cb_wait_front(cb_in0, onetile);

                unpack_reconfig_data_format_srca(cb_in0);
                copy_tile_init();
                copy_tile(cb_in0, 0, dst0); // tensor value 0

                exp_tile_init();
                exp_tile(dst0); // exp(0) = 1

                if (i == 0) {
                    PACK(( llk_pack_reconfig_l1_acc(0) )); // acc off
                } else {
                    PACK(( llk_pack_reconfig_l1_acc(1) )); // acc on
                }
                PACK(( pack_reconfig_data_format(cb_add) ));
                pack_tile(dst0, cb_add);

                cb_push_back(cb_add, onetile);

                cb_pop_front(cb_in0, onetile);
                REL();
            }

            // expected cb_add value is 2
            cb_wait_front(cb_add, onetile);
            for (uint32_t i = 0; i < dim_size; ++i) {
                ACQ();
                cb_reserve_back(cb_out0, onetile);

                unpack_reconfig_data_format_srca(cb_add);
                copy_tile_init();
                copy_tile(cb_add, 0, dst0);

                PACK(( llk_pack_reconfig_l1_acc(0) )); // acc off
                PACK(( pack_reconfig_data_format(cb_out0) ));
                pack_tile(dst0, cb_out0);

                cb_push_back(cb_out0, onetile);
                REL();
            }
            cb_pop_front(cb_add, onetile);
        }
    }
}
}  // namespace NAMESPACE
