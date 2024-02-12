// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */

void kernel_main() {
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t src_addr = get_arg_val<uint32_t>(1);
    std::uint32_t dst_addr = get_arg_val<uint32_t>(2);
    std::uint32_t num_pages = get_arg_val<uint32_t>(3);
    std::uint32_t page_size = get_arg_val<uint32_t>(4);


    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    const InterleavedAddrGen<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = page_size};
    const InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr, .page_size = page_size};

    uint32_t curr_eth_l1_addr = local_eth_l1_src_addr;
    for (uint32_t i = 0; i < num_pages; ++i) {
        uint64_t src_noc_addr = get_noc_addr(i, s);
        noc_async_read(src_noc_addr, curr_eth_l1_addr, page_size);
        curr_eth_l1_addr += page_size;
    }
    noc_async_read_barrier();

    curr_eth_l1_addr = local_eth_l1_src_addr;
    for (uint32_t i = 0; i < num_pages; ++i) {
        uint64_t dst_noc_addr = get_noc_addr(i, d);
        noc_async_write(curr_eth_l1_addr, dst_noc_addr, page_size);
        curr_eth_l1_addr += page_size;
    }
    noc_async_write_barrier();
}
