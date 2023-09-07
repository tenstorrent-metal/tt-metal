/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>

#include "dataflow_api.h"
#include "debug_print.h"
#include "debug_status.h"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/src/firmware/riscv/grayskull/dev_mem_map.h"

static constexpr u32 PROGRAM_CB_ID = 0;

inline u32 min(u32 a, u32 b) {
    return (a < b) ? a: b;
}

void write_buffer(
    Buffer& buffer,
    u32 src_addr,
    u32 src_noc,
    u32 dst_addr,

    u32 padded_buf_size,
    u32 burst_size,
    u32 page_size,
    u32 padded_page_size) {
    // Base address of where we are writing to
    buffer.bank_base_address = dst_addr;
    buffer.page_size = padded_page_size;

    u32 bank_id = 0;
    while (padded_buf_size > 0) {

        // Read in a big chunk of data
        u32 read_size = min(burst_size, padded_buf_size);
        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;
        noc_async_read(src_noc_addr, DEVICE_COMMAND_DATA_ADDR, read_size);
        padded_buf_size -= read_size;
        src_addr += read_size;
        u32 local_addr = DEVICE_COMMAND_DATA_ADDR;
        noc_async_read_barrier();

        // Send pages within the chunk to their destination
        for (u32 i = 0; i < read_size; i += padded_page_size) {
            u64 dst_addr = buffer.get_noc_addr(bank_id++);
            noc_async_write(local_addr, dst_addr, page_size);
            local_addr += padded_page_size;
        }
        noc_async_write_barrier();
    }
}

FORCE_INLINE void write_buffers(
    u32 num_buffer_writes,
    volatile tt_l1_ptr u32*& command_ptr,
    Buffer& buffer) {
    for (u32 i = 0; i < num_buffer_writes; i++) {
        u32 src_addr = command_ptr[0];
        u32 src_noc = command_ptr[1];
        u32 dst_addr = command_ptr[2];

        u32 padded_buf_size = command_ptr[3];
        u32 burst_size = command_ptr[4];
        u32 page_size = command_ptr[5];
        u32 padded_page_size = command_ptr[6];
        u32 buf_type = command_ptr[7];

#define write_buffer_args                                                                                      \
    src_addr, src_noc, dst_addr, padded_buf_size, burst_size, page_size, padded_page_size

        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;
        switch (buf_type) {
            case 0:  // DRAM
                buffer.set_type(BufferType::DRAM);
                break;
            case 1:  // L1
                buffer.set_type(BufferType::L1);
                break;
        }
        write_buffer(buffer, write_buffer_args);

        command_ptr += 8;
    }
}

FORCE_INLINE void read_buffer(
    Buffer& buffer,
    u32 dst_addr,
    u32 dst_noc,
    u32 src_addr,

    u32 padded_buf_size,
    u32 burst_size,
    u32 page_size,
    u32 padded_page_size) {
    // Base address of where we are reading from
    buffer.bank_base_address = src_addr;
    buffer.page_size = padded_page_size;

    u32 bank_id = 0;
    while (padded_buf_size > 0) {
        // Read in pages until we don't have anymore memory
        // available
        u32 write_size = min(burst_size, padded_buf_size);
        u32 local_addr = DEVICE_COMMAND_DATA_ADDR;
        u64 dst_noc_addr = (u64(dst_noc) << 32) | dst_addr;
        dst_addr += write_size;
        padded_buf_size -= write_size;

        for (u32 i = 0; i < write_size; i += padded_page_size) {
            u64 src_addr = buffer.get_noc_addr(bank_id++);
            noc_async_read(src_addr, local_addr, page_size);
            local_addr += padded_page_size;
        }
        noc_async_read_barrier();
        noc_async_write(DEVICE_COMMAND_DATA_ADDR, dst_noc_addr, write_size);
        noc_async_write_barrier();
    }
}

FORCE_INLINE void read_buffers(
    u32 num_buffer_reads,
    volatile tt_l1_ptr u32*& command_ptr,
    Buffer& buffer) {
    for (u32 i = 0; i < num_buffer_reads; i++) {
        u32 dst_addr = command_ptr[0];
        u32 dst_noc = command_ptr[1];
        u32 src_addr = command_ptr[2];

        u32 padded_buf_size = command_ptr[3];
        u32 burst_size = command_ptr[4];
        u32 page_size = command_ptr[5];
        u32 padded_page_size = command_ptr[6];
        u32 buf_type = command_ptr[7];

#define read_buffer_args                                                                                       \
    dst_addr, dst_noc, src_addr, padded_buf_size, burst_size, page_size, padded_page_size

        switch (buf_type) {
            case 0:  // DRAM
                buffer.set_type(BufferType::DRAM);
                break;
            case 1:  // L1
                buffer.set_type(BufferType::L1);
                break;
        }

        read_buffer(buffer, read_buffer_args);

        command_ptr += 8;
    }
}

FORCE_INLINE void init_program_cb() {
    constexpr u32 program_cb_size = (MEM_L1_SIZE - DEVICE_COMMAND_DATA_ADDR);
    constexpr u32 program_cb_num_pages = program_cb_size / PROGRAM_PAGE_SIZE;
    cb_interface[PROGRAM_CB_ID].fifo_limit = ((DEVICE_COMMAND_DATA_ADDR + program_cb_size) >> 4) - 1;
    cb_interface[PROGRAM_CB_ID].fifo_wr_ptr = (DEVICE_COMMAND_DATA_ADDR >> 4);
    cb_interface[PROGRAM_CB_ID].fifo_rd_ptr = (DEVICE_COMMAND_DATA_ADDR >> 4);
    cb_interface[PROGRAM_CB_ID].fifo_size = (program_cb_size >> 4);
    cb_interface[PROGRAM_CB_ID].fifo_num_pages = program_cb_num_pages;
    cb_interface[PROGRAM_CB_ID].fifo_page_size = (PROGRAM_PAGE_SIZE >> 4);
}

FORCE_INLINE void write_program_page(u32 page_addr, volatile u32*& command_ptr) {
    u32 num_transfers = command_ptr[0];
    command_ptr++;
    u32 src = page_addr;
    // DPRINT << "Num transfers: " << num_transfers << ENDL();
    for (u32 i = 0; i < num_transfers; i++) {
        u32 num_bytes = command_ptr[0];
        u32 dst = command_ptr[1];
        u32 dst_noc = command_ptr[2];
        u32 num_recv = command_ptr[3];

        // DPRINT << "num_bytes: " << num_bytes << ENDL();
        // DPRINT << "dst: " << dst << ENDL();
        // DPRINT << "dst_noc: " << dst_noc << ENDL();
        // DPRINT << "num_recv: " << num_recv << ENDL();
        // DPRINT << "src: " << src << ENDL();
        // DPRINT << ENDL();

        // DPRINT << "Sending" << ENDL();
        // for (u32 i = src; i < src + min(48, num_bytes); i += sizeof(u32)) {
        //     DPRINT << *reinterpret_cast<volatile u32*>(i) << ENDL();
        // }
        // DPRINT << ENDL();

        noc_async_write_multicast(src, (u64(dst_noc) << 32) | dst, num_bytes, num_recv);
        command_ptr += 4;
        src = align(src + num_bytes, 16);
    }

    // Future optimization: Don't barrier here, only barrier after all pages are written.
    noc_async_write_barrier();
}

FORCE_INLINE void write_program(u32 num_program_srcs, volatile u32*& command_ptr, Buffer& buffer) {

    for (u32 program_src = 0; program_src < num_program_srcs; program_src++) {
        init_program_cb();
        u32 buffer_type = command_ptr[0];
        u32 num_pages = command_ptr[1];
        u32 bank_base_address = command_ptr[2];
        command_ptr += 3;

        // DPRINT << "buffer_type: " << buffer_type << ENDL();
        // DPRINT << "num_pages: " << num_pages << ENDL();
        // DPRINT << "bank_base_address: " << bank_base_address << ENDL();

        switch (buffer_type) {
            case 0:
                buffer.set_type(BufferType::DRAM);
                break;
            case 2:
                buffer.set_type(BufferType::SYSTEM_MEMORY);
                break;
        }
        buffer.bank_base_address = bank_base_address;
        buffer.page_size = PROGRAM_PAGE_SIZE;

        for (u32 page_idx = 0; page_idx < num_pages; page_idx++) {
            cb_reserve_back(PROGRAM_CB_ID, 1);
            u32 page_write_ptr = get_write_ptr(PROGRAM_CB_ID);
            u32 page_read_ptr = get_read_ptr(PROGRAM_CB_ID);
            noc_async_read(buffer.get_noc_addr(page_idx), page_write_ptr, PROGRAM_PAGE_SIZE);
            noc_async_read_barrier();

            // DPRINT << "Transfer" << ENDL();
            // for (u32 i = 0; i < PROGRAM_PAGE_SIZE; i += sizeof(u32)) {
            //     DPRINT << *reinterpret_cast<volatile u32*>(page_read_ptr + i) << ENDL();
            // }
            // DPRINT << ENDL();

            cb_push_back(PROGRAM_CB_ID, 1);
            cb_wait_front(PROGRAM_CB_ID, 1);

            write_program_page(page_write_ptr, command_ptr);
            cb_pop_front(PROGRAM_CB_ID, 1);
        }
    }
}

FORCE_INLINE void launch_program(u32 num_workers, u32 num_multicast_messages, volatile tt_l1_ptr u32*& command_ptr) {
    if (not num_workers)
        return;

    volatile uint32_t* message_addr_ptr = reinterpret_cast<volatile uint32_t*>(DISPATCH_MESSAGE_ADDR);
    *message_addr_ptr = 0;
    for (u32 i = 0; i < num_multicast_messages * 2; i += 2) {
        u64 worker_core_noc_coord = u64(command_ptr[i]) << 32;
        u32 num_messages = command_ptr[i + 1];
        u64 deassert_addr = worker_core_noc_coord | TENSIX_SOFT_RESET_ADDR;
        noc_semaphore_set_multicast(DEASSERT_RESET_SRC_L1_ADDR, deassert_addr, num_messages);
    }

    noc_async_write_barrier();

    // Wait on worker cores to notify me that they have completed
    DEBUG_STATUS('Q', 'W');
    while (reinterpret_cast<volatile tt_l1_ptr u32*>(DISPATCH_MESSAGE_ADDR)[0] != num_workers) {
        // DPRINT << "DONE: " << reinterpret_cast<volatile tt_l1_ptr u32*>(DISPATCH_MESSAGE_ADDR)[0] << ENDL();
        // for (volatile int i = 0; i < 1000000; i++);
    }

    DEBUG_STATUS('Q', 'D');
    for (u32 i = 0; i < num_multicast_messages * 2; i += 2) {
        u64 worker_core_noc_coord = u64(command_ptr[i]) << 32;
        u32 num_messages = command_ptr[i + 1];
        u64 assert_addr = worker_core_noc_coord | TENSIX_SOFT_RESET_ADDR;

        noc_semaphore_set_multicast(ASSERT_RESET_SRC_L1_ADDR, assert_addr, num_messages);
    }
    noc_async_write_barrier();
}

FORCE_INLINE void finish_program(u32 finish) {
    if (not finish)
        return;

    volatile tt_l1_ptr u32* finish_ptr = get_cq_finish_ptr();
    finish_ptr[0] = 1;
    uint64_t finish_noc_addr = get_noc_addr(PCIE_NOC_X, PCIE_NOC_Y, HOST_CQ_FINISH_PTR);
    noc_async_write(u32(finish_ptr), finish_noc_addr, 4);
    noc_async_write_barrier();
    finish_ptr[0] = 0;
}
