// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/kernels/dataflow/moreh_common.hpp"

inline uint32_t get_read_tile_id(uint32_t tile_id, uint32_t dim, uint32_t input_tile_offset, uint32_t HtWt) {
    if(dim == 0){
        return tile_id;
    }else {
        uint32_t a = 0 ;
        while (tile_id >= HtWt){
            tile_id-= HtWt;
            a = a + 1;
        }
        uint32_t b = 0;
        for (uint32_t i = 0; i < input_tile_offset; ++i) {
            b = b + a;
        }
        return b + tile_id;
    }
}

void kernel_main() {
    const auto input_addr = get_arg_val<uint32_t>(0);
    const auto num_input_tiles = get_arg_val<uint32_t>(1);
    const auto num_output_tiles = get_arg_val<uint32_t>(2);
    const auto input_tile_offset = get_arg_val<uint32_t>(3);
    const auto start_id = get_arg_val<uint32_t>(4);
    const auto input_is_dram = get_compile_time_arg_val(0) == 1;
    const auto HtWt = get_arg_val<uint32_t>(6);
    const auto CHtWt = get_arg_val<uint32_t>(7);
    const auto dim = get_arg_val<uint32_t>(8);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_in1, scaler.u);

    uint32_t l1_write_addr_in0;
    uint32_t input_tile_bytes = get_tile_size(cb_id_in0);
    const auto input_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<input_is_dram> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        auto read_tile_id = get_read_tile_id(i, dim, CHtWt, HtWt);
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(read_tile_id, dram_input_addrg, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            read_tile_id += input_tile_offset;
        }
    }

}
