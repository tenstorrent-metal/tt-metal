#include <stdint.h>

#include "dataflow_api.h"
#include "debug_print.h"
#include "frameworks/tt_dispatch/impl/command.hpp"


// Dispatch constants
u64 worker_cores_multicast_soft_reset_addr = get_noc_multicast_addr(1, 1, 12, 10, 0xFFB121B0);
u64 worker_cores_multicast_notify_addr = get_noc_multicast_addr(1, 1, 12, 10, DISPATCH_MESSAGE_REMOTE_SENDER_ADDR);
u32 num_worker_cores = 108;

template <typename T>
void write_buffer(
    T addr_gen,
    u32 src_addr,
    u32 src_noc,
    u32 dst_addr,

    u32 num_bursts,
    u32 burst_size,
    u32 num_pages_per_burst,
    u32 page_size,
    u32 remainder_burst_size,
    u32 num_pages_per_remainder_burst,
    u32 banking_enum) {
    // Base address of where we are writing to
    addr_gen.bank_base_address = dst_addr;
    addr_gen.page_size = page_size;

    u32 id = 0;
    for (u32 j = 0; j < num_bursts; j++) {
        u32 data_addr = UNRESERVED_BASE;
        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;

        noc_async_read(src_noc_addr, data_addr, burst_size);

        src_addr += burst_size;
        noc_async_read_barrier();

        for (u32 k = 0; k < num_pages_per_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(id++);

            noc_async_write(data_addr, addr, page_size);
            data_addr += page_size;
        }
        noc_async_write_barrier();
    }
    // In case where the final burst a different size than the others
    if (remainder_burst_size) {
        u32 data_addr = UNRESERVED_BASE;
        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;
        noc_async_read(src_noc_addr, data_addr, remainder_burst_size);
        noc_async_read_barrier();

        for (u32 k = 0; k < num_pages_per_remainder_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(id++);

            noc_async_write(data_addr, addr, page_size);
            data_addr += page_size;
        }
        noc_async_write_barrier();
    }
}

FORCE_INLINE void write_buffers(
    u32 num_buffer_writes,
    volatile u32*& command_ptr,
    const InterleavedAddrGen<true>& dram_addr_gen,
    const InterleavedAddrGen<false>& l1_addr_gen) {
    for (u32 i = 0; i < num_buffer_writes; i++) {
        u32 src_addr = command_ptr[0];
        u32 src_noc = command_ptr[1];
        u32 dst_addr = command_ptr[2];
        u32 dst_noc_start = command_ptr[3];
        u32 num_bursts = command_ptr[4];
        u32 burst_size = command_ptr[5];
        u32 num_pages_per_burst = command_ptr[6];
        u32 page_size = command_ptr[7];
        u32 remainder_burst_size = command_ptr[8];
        u32 num_pages_per_remainder_burst = command_ptr[9];
        u32 banking_enum = command_ptr[10];

        DPRINT << 'A' << ':' << ' ' << src_addr << ENDL();
        DPRINT << 'B' << ':' << ' ' << src_noc << ENDL();
        DPRINT << 'C' << ':' << ' ' << dst_addr << ENDL();
        DPRINT << 'D' << ':' << ' ' << dst_noc_start << ENDL();
        DPRINT << 'E' << ':' << ' ' << num_bursts << ENDL();
        DPRINT << 'F' << ':' << ' ' << burst_size << ENDL();
        DPRINT << 'G' << ':' << ' ' << num_pages_per_burst << ENDL();
        DPRINT << 'H' << ':' << ' ' << page_size << ENDL();
        DPRINT << 'I' << ':' << ' ' << remainder_burst_size << ENDL();
        DPRINT << 'J' << ':' << ' ' << num_pages_per_remainder_burst << ENDL();
        DPRINT << 'K' << ':' << ' ' << banking_enum << ENDL();

        return;

        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;
        switch (banking_enum) {
            case 0:  // DRAM
                write_buffer(
                    dram_addr_gen,
                    src_addr,
                    src_noc,
                    dst_addr,

                    num_bursts,
                    burst_size,
                    num_pages_per_burst,
                    page_size,
                    remainder_burst_size,
                    num_pages_per_remainder_burst,
                    banking_enum);
                break;
            case 1:  // L1
                write_buffer(
                    l1_addr_gen,
                    src_addr,
                    src_noc,
                    dst_addr,

                    num_bursts,
                    burst_size,
                    num_pages_per_burst,
                    page_size,
                    remainder_burst_size,
                    num_pages_per_remainder_burst,
                    banking_enum);
                break;
        }

        command_ptr += 11;
    }
}

template <typename T>
FORCE_INLINE void read_buffer(
    T addr_gen,
    u32 dst_addr,
    u32 dst_noc,
    u32 src_addr,

    u32 num_bursts,
    u32 burst_size,
    u32 num_pages_per_burst,
    u32 page_size,
    u32 remainder_burst_size,
    u32 num_pages_per_remainder_burst,
    u32 banking_enum) {
    // Base address of where we are reading from
    addr_gen.bank_base_address = src_addr;
    addr_gen.page_size = page_size;

    u32 id = 0;
    for (u32 j = 0; j < num_bursts; j++) {
        u32 data_addr = UNRESERVED_BASE;
        u64 dst_noc_addr = (u64(dst_noc) << 32) | dst_addr;

        for (u32 k = 0; k < num_pages_per_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(id++);

            noc_async_read(addr, data_addr, page_size);
            data_addr += page_size;
        }
        noc_async_read_barrier();

        noc_async_write(UNRESERVED_BASE, dst_noc_addr, burst_size);
        dst_addr += burst_size;
        noc_async_write_barrier();
    }

    if (remainder_burst_size) {
        u32 data_addr = UNRESERVED_BASE;
        u64 dst_noc_addr = (u64(dst_noc) << 32) | dst_addr;

        for (u32 k = 0; k < num_pages_per_remainder_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(id++);

            noc_async_read(addr, data_addr, page_size);
            data_addr += page_size;
        }
        noc_async_read_barrier();

        noc_async_write(UNRESERVED_BASE, dst_noc_addr, remainder_burst_size);
        noc_async_write_barrier();
    }
}

FORCE_INLINE void read_buffers(
    u32 num_buffer_reads,
    volatile u32*& command_ptr,
    const InterleavedAddrGen<true>& dram_addr_gen,
    const InterleavedAddrGen<false>& l1_addr_gen) {
    for (u32 i = 0; i < num_buffer_reads; i++) {
        u32 dst_addr = command_ptr[0];
        u32 dst_noc = command_ptr[1];
        u32 src_addr = command_ptr[2];
        u32 src_noc_start = command_ptr[3];
        u32 num_bursts = command_ptr[4];
        u32 burst_size = command_ptr[5];
        u32 num_pages_per_burst = command_ptr[6];
        u32 page_size = command_ptr[7];
        u32 remainder_burst_size = command_ptr[8];
        u32 num_pages_per_remainder_burst = command_ptr[9];
        u32 banking_enum = command_ptr[10];

        switch (banking_enum) {
            case 0:  // DRAM
                read_buffer(
                    dram_addr_gen,
                    dst_addr,
                    dst_noc,
                    src_addr,

                    num_bursts,
                    burst_size,
                    num_pages_per_burst,
                    page_size,
                    remainder_burst_size,
                    num_pages_per_remainder_burst,
                    banking_enum);
                break;
            case 1:  // L1
                read_buffer(
                    l1_addr_gen,
                    dst_addr,
                    dst_noc,
                    src_addr,

                    num_bursts,
                    burst_size,
                    num_pages_per_burst,
                    page_size,
                    remainder_burst_size,
                    num_pages_per_remainder_burst,
                    banking_enum);
                break;
        }

        command_ptr += 11;
    }
}

FORCE_INLINE void write_program_section(
    u32 src, u32 src_noc, u32 transfer_size, u32 num_writes, volatile u32*& command_ptr) {
    // Bring in a program section into L1
    noc_async_read((u64(src_noc) << 32 | src), UNRESERVED_BASE, transfer_size);

    DPRINT << src_noc << ENDL();
    return;
    noc_async_read_barrier();
    // Write different parts of that program section to different worker cores
    for (u32 write = 0; write < num_writes; write++) {
        u32 src = command_ptr[0];
        u32 dst = command_ptr[1];
        u32 dst_noc = command_ptr[2];
        u32 transfer_size = command_ptr[3];
        u32 num_receivers = command_ptr[4];
        command_ptr += 5;
        noc_async_write_multicast(src, u64(dst_noc) << 32 | dst, transfer_size, num_receivers);
    }
    noc_async_write_barrier();
}

FORCE_INLINE void write_program(u32 num_program_relays, volatile u32*& command_ptr) {
    // DPRINT << 'N' << 'P' << 'R' << ':' << ' ' << num_program_relays << ENDL();
    for (u32 relay = 0; relay < num_program_relays; relay++) {
        u32 src = command_ptr[0];
        u32 src_noc = command_ptr[1];
        u32 transfer_size = command_ptr[2];
        u32 num_writes = command_ptr[3];
        command_ptr += 4;
        // DPRINT << 'H' << 'E' << 'L' << 'L' << 'O' << ENDL();
        // DPRINT << src << ENDL();
        // DPRINT << src_noc << ENDL();
        // DPRINT << transfer_size << ENDL();
        // DPRINT << num_writes << ENDL();
        // write_program_section(src, src_noc, transfer_size, num_writes, command_ptr);
    }

    // If we had to relay any program data, then we need to deassert the worker cores
    // if (num_program_relays) {
    //     // Multicast sender address (so that the worker cores know who to notify)
    //     // Don't need a barrier, will just wait for kernels to finish
    //     noc_async_write(DISPATCH_MESSAGE_REMOTE_SENDER_ADDR, worker_cores_multicast_notify_addr, 8);

    //     // This is only required until PK checks in his changes to separate kernels from firmware
    //     noc_semaphore_set_remote(DEASSERT_RESET_SRC_L1_ADDR, worker_cores_multicast_soft_reset_addr);

    //     // Wait on worker cores to notify me that they have completed
    //     while (reinterpret_cast<volatile u32*>(DISPATCH_MESSAGE_REMOTE_SENDER_ADDR)[0] != num_worker_cores);

    //     // This is only required until PK checks in his changes to separate kernels from firmware
    //     noc_semaphore_set_remote(ASSERT_RESET_SRC_L1_ADDR, worker_cores_multicast_soft_reset_addr);
    // }
}

FORCE_INLINE void handle_finish() {
    volatile u32* finish_ptr = get_cq_finish_ptr();
    finish_ptr[0] = 1;
    uint64_t finish_noc_addr = get_noc_addr(NOC_X(0), NOC_Y(4), HOST_CQ_FINISH_PTR);
    noc_async_write(u32(finish_ptr), finish_noc_addr, 4);
    noc_async_write_barrier();
    finish_ptr[0] = 0;
}
