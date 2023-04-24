#include <stdint.h>
#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
// #include "debug_print.h"

void kernel_main() {
    for (int i = 0; i < 4; i++) {
    volatile uint32_t* copy_desc_info_addr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(0));

    uint32_t num_workers = *copy_desc_info_addr;
    copy_desc_info_addr++;

    noc_prepare_deassert_reset_flag(DEASSERT_RESET_SRC_L1_ADDR);
    noc_prepare_assert_reset_flag(ASSERT_RESET_SRC_L1_ADDR);
    kernel_profiler::mark_time(6);
    uint32_t num_reads = *copy_desc_info_addr;
    copy_desc_info_addr++;
    for (uint32_t read = 0; read < num_reads; read++) {
        uint64_t src_addr = *reinterpret_cast<volatile uint64_t*>(copy_desc_info_addr);
        copy_desc_info_addr += 2; // uint64_t embedded as two uint32_t's since NOC address embedded
        uint32_t dst_addr = *copy_desc_info_addr;
        copy_desc_info_addr++;
        uint32_t transfer_size = *copy_desc_info_addr;
        copy_desc_info_addr++;
        noc_async_read(src_addr, dst_addr, transfer_size);
    }
    uint32_t num_writes = *copy_desc_info_addr;
    copy_desc_info_addr++;

    noc_async_read_barrier();

    for (uint32_t write = 0; write < num_writes; write++) {
        uint32_t src_addr = *copy_desc_info_addr;
        copy_desc_info_addr++;
        uint64_t dst_addr = *reinterpret_cast<volatile uint64_t*>(copy_desc_info_addr);
        // DPRINT << dst_addr << ENDL();
        copy_desc_info_addr += 2; // uint64_t embedded as two uint32_t's since NOC address embedded
        uint32_t transfer_size = *copy_desc_info_addr;
        copy_desc_info_addr++;
        noc_async_write_multicast(src_addr, dst_addr, transfer_size, num_workers);
    }
    uint32_t num_resets = *copy_desc_info_addr;
    copy_desc_info_addr++;


    // Prepare noc coordinates to send to receiver
    *reinterpret_cast<volatile uint64_t*>(DISPATCH_MESSAGE_REMOTE_SENDER_ADDR) = get_noc_addr(my_x[0], my_y[0], DISPATCH_MESSAGE_ADDR);
    noc_async_write_barrier();
    // DPRINT << 3 << ENDL();


    if (num_resets > 0) {

        // Let the receiver know that I am the sender
        for (uint32_t notify = 0; notify < num_resets; notify++) {
            uint64_t dst_addr = *reinterpret_cast<volatile uint64_t*>(copy_desc_info_addr);
            copy_desc_info_addr += 2;
            // DPRINT << dst_addr << ENDL();
            noc_async_write_multicast(DISPATCH_MESSAGE_REMOTE_SENDER_ADDR, dst_addr, 8, num_workers);
        }

        // DPRINT << 'E' << ENDL();
        noc_async_write_barrier();
        // DPRINT << 4 << ENDL();

        volatile uint32_t* message_addr_ptr = reinterpret_cast<volatile uint32_t*>(DISPATCH_MESSAGE_ADDR);
        *message_addr_ptr = 0;

        constexpr uint32_t reset_transfer_size = sizeof(uint32_t);
        volatile uint32_t* reset_copy_desc_start = copy_desc_info_addr;
        for (uint32_t deassert = 0; deassert < num_resets; deassert++) {
            uint64_t dst_addr = *reinterpret_cast<volatile uint64_t*>(copy_desc_info_addr);
            // DPRINT << dst_addr << ENDL();
            copy_desc_info_addr += 2; // uint64_t embedded as two uint32_t's since NOC address embedded
            noc_async_write_multicast(DEASSERT_RESET_SRC_L1_ADDR, dst_addr, 4, num_workers);
        }
        // DPRINT << 5 << ENDL();

        copy_desc_info_addr = reset_copy_desc_start;
        kernel_profiler::mark_time(7);
        while (*message_addr_ptr != num_workers); // Could be deasserting through a multicast, in which num_workers > num_resets

        for (uint32_t reset = 0; reset < num_resets; reset++) {
            uint64_t dst_addr = *reinterpret_cast<volatile uint64_t*>(copy_desc_info_addr);
            copy_desc_info_addr += 2; // uint64_t embedded as two uint32_t's since NOC address embedded
            noc_async_write_multicast(ASSERT_RESET_SRC_L1_ADDR, dst_addr, 4, num_workers);
        }
        noc_async_write_barrier();
    }
    }
}
