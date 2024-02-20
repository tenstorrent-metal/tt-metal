// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "risc_attribs.h"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "debug/dprint.h"

CQReadInterface cq_read_interface;

inline __attribute__((always_inline)) volatile uint32_t* get_cq_issue_read_ptr() {
    return reinterpret_cast<volatile uint32_t*>(CQ_ISSUE_READ_PTR);
}

inline __attribute__((always_inline)) volatile uint32_t* get_cq_issue_write_ptr() {
    return reinterpret_cast<volatile uint32_t*>(CQ_ISSUE_WRITE_PTR);
}

// Only the read interface is set up on the device... the write interface
// belongs to host
FORCE_INLINE
void setup_issue_queue_read_interface(const uint32_t issue_region_rd_ptr, const uint32_t issue_region_size) {
    cq_read_interface.issue_fifo_rd_ptr = issue_region_rd_ptr >> 4;
    cq_read_interface.issue_fifo_size = issue_region_size >> 4;
    cq_read_interface.issue_fifo_limit = (issue_region_rd_ptr + issue_region_size) >> 4;
    cq_read_interface.issue_fifo_rd_toggle = 0;
}

template <uint32_t num_command_slots>
FORCE_INLINE void wait_consumer_idle(volatile tt_l1_ptr uint32_t* db_semaphore_addr) {
    while (*db_semaphore_addr != num_command_slots) {
        DPRINT << "SEM: "<< *db_semaphore_addr << ENDL();
        for (volatile int i = 0; i < 100000000; i++);
    }
}

FORCE_INLINE
void wait_consumer_space_available(volatile tt_l1_ptr uint32_t* db_semaphore_addr) {
    while (*db_semaphore_addr == 0);
}

FORCE_INLINE
void update_producer_consumer_sync_semaphores(
    uint64_t producer_noc_encoding,
    uint64_t consumer_noc_encoding,
    volatile tt_l1_ptr uint32_t* producer_db_semaphore_addr,
    uint32_t consumer_db_semaphore) {
    // Decrement the semaphore value
    noc_semaphore_inc(producer_noc_encoding | uint32_t(producer_db_semaphore_addr), -1);  // Two's complement addition

    // Notify the consumer
    noc_semaphore_inc(consumer_noc_encoding | consumer_db_semaphore, 1);
    noc_async_write_barrier();  // Barrier for now
}

FORCE_INLINE
void issue_queue_wait_front() {
    DEBUG_STATUS('N', 'Q', 'W');
    uint32_t issue_write_ptr_and_toggle;
    uint32_t issue_write_ptr;
    uint32_t issue_write_toggle;
    do {
        issue_write_ptr_and_toggle = *get_cq_issue_write_ptr();
        issue_write_ptr = issue_write_ptr_and_toggle & 0x7fffffff;
        issue_write_toggle = issue_write_ptr_and_toggle >> 31;
    } while (cq_read_interface.issue_fifo_rd_ptr == issue_write_ptr and cq_read_interface.issue_fifo_rd_toggle == issue_write_toggle);
    DEBUG_STATUS('N', 'Q', 'D');
}

template <uint32_t host_issue_queue_read_ptr_addr>
FORCE_INLINE
void notify_host_of_issue_queue_read_pointer() {
    // These are the PCIE core coordinates
    constexpr static uint64_t pcie_address = (uint64_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32) | host_issue_queue_read_ptr_addr;
    uint32_t issue_rd_ptr_and_toggle = cq_read_interface.issue_fifo_rd_ptr | (cq_read_interface.issue_fifo_rd_toggle << 31);
    volatile tt_l1_ptr uint32_t* issue_rd_ptr_addr = get_cq_issue_read_ptr();
    issue_rd_ptr_addr[0] = issue_rd_ptr_and_toggle;
    noc_async_write(CQ_ISSUE_READ_PTR, pcie_address, 4);
    noc_async_write_barrier();
}

template <uint32_t host_issue_queue_read_ptr_addr>
FORCE_INLINE
void issue_queue_pop_front(uint32_t cmd_size_B) {
    // First part of equation aligns to nearest multiple of 32, and then we shift to make it a 16B addr. Both
    // host and device are consistent in updating their pointers in this way, so they won't get out of sync. The
    // alignment is necessary because we can only read/write from/to 32B aligned addrs in host<->dev communication.
    uint32_t cmd_size_16B = align(cmd_size_B, 32) >> 4;
    cq_read_interface.issue_fifo_rd_ptr += cmd_size_16B;
    if (cq_read_interface.issue_fifo_rd_ptr >= cq_read_interface.issue_fifo_limit) {
        cq_read_interface.issue_fifo_rd_ptr -= cq_read_interface.issue_fifo_size;
        cq_read_interface.issue_fifo_rd_toggle = not cq_read_interface.issue_fifo_rd_toggle;
    }

    notify_host_of_issue_queue_read_pointer<host_issue_queue_read_ptr_addr>();
}

FORCE_INLINE
void program_local_cb(uint32_t data_section_addr, uint32_t num_pages, uint32_t page_size, uint32_t cb_size) {
    uint32_t cb_id = 0;
    uint32_t fifo_addr = data_section_addr >> 4;
    uint32_t fifo_limit = fifo_addr + (cb_size >> 4);
    cb_interface[cb_id].fifo_limit = fifo_limit;  // to check if we need to wrap
    cb_interface[cb_id].fifo_wr_ptr = fifo_addr;
    cb_interface[cb_id].fifo_rd_ptr = fifo_addr;
    cb_interface[cb_id].fifo_size = cb_size >> 4;
    cb_interface[cb_id].tiles_acked = 0;
    cb_interface[cb_id].tiles_received = 0;
    cb_interface[cb_id].fifo_num_pages = num_pages;
    cb_interface[cb_id].fifo_page_size = page_size >> 4;
    DPRINT << "LOCAL CB ADDRESS: " << data_section_addr << ENDL();
}

template <uint32_t producer_cmd_base_addr, uint32_t producer_data_buffer_size, uint32_t consumer_cmd_base_addr, uint32_t consumer_data_buffer_size>
FORCE_INLINE
void program_consumer_cb(
    volatile db_cb_config_t* db_cb_config,
    volatile db_cb_config_t* remote_db_cb_config,
    uint64_t consumer_noc_encoding,
    uint32_t num_pages,
    uint32_t page_size,
    uint32_t cb_size) {
    /*
        This API programs the double-buffered CB space of the consumer. This API should be called
        before notifying the consumer that data is available.
    */
    uint32_t cb_start_rd_addr = get_cb_start_address();
    uint32_t cb_start_wr_addr = cb_start_rd_addr;

    db_cb_config->ack = 0;
    db_cb_config->recv = 0;
    db_cb_config->num_pages = num_pages;
    db_cb_config->page_size_16B = page_size >> 4;
    db_cb_config->total_size_16B = cb_size >> 4;
    db_cb_config->rd_ptr_16B = cb_start_rd_addr >> 4;
    db_cb_config->wr_ptr_16B = cb_start_wr_addr >> 4;
    db_cb_config->fifo_limit_16B = (cb_start_rd_addr + cb_size) >> 4;

    noc_async_write(
        (uint32_t)(db_cb_config), consumer_noc_encoding | (uint32_t)(remote_db_cb_config), sizeof(db_cb_config_t));
    noc_async_write_barrier();  // barrier for now
}

FORCE_INLINE
bool cb_producer_space_available(int32_t num_pages) {
    uint32_t operand = 0;
    uint32_t pages_acked_ptr = (uint32_t) get_cb_tiles_acked_ptr(operand);

    // while the producer (write-side interface) is waiting for space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    uint32_t pages_received = get_cb_tiles_received_ptr(operand)[0];

    int32_t free_space_pages;
    DEBUG_STATUS('C', 'R', 'B', 'W');

    // uint16_t's here because Tensix updates the val at tiles_acked_ptr as uint16 in llk_pop_tiles
    // TODO: I think we could have TRISC update tiles_acked_ptr, and we wouldn't need uint16 here
    uint16_t pages_acked = (uint16_t)reg_read(pages_acked_ptr);
    uint16_t free_space_pages_wrap =
        cb_interface[operand].fifo_num_pages - (pages_received - pages_acked);
    free_space_pages = (int32_t)free_space_pages_wrap;
    return free_space_pages >= num_pages;
}

FORCE_INLINE
bool cb_consumer_space_available(db_cb_config_t* db_cb_config, int32_t num_pages) {
    // TODO: delete cb_consumer_space_available and use this one

    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t free_space_pages_wrap = db_cb_config->num_pages - (db_cb_config->recv - db_cb_config->ack);
    int32_t free_space_pages = (int32_t)free_space_pages_wrap;
    DEBUG_STATUS('C', 'R', 'B', 'D');

    return free_space_pages >= num_pages;
}

template <uint32_t cmd_base_addr, uint32_t consumer_cmd_base_addr, uint32_t consumer_data_buffer_size>
FORCE_INLINE void relay_command(bool db_buf_switch, uint64_t consumer_noc_encoding) {
    /*
        Relays the current command to the consumer.
    */

    uint64_t consumer_command_slot_addr = consumer_noc_encoding | get_command_slot_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch);
    noc_async_write(cmd_base_addr, consumer_command_slot_addr, DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);
    noc_async_write_barrier();
}

enum PullAndRelayType {
    BUFFER = 0,
    CIRCULAR_BUFFER = 1
};

struct PullAndRelayCircularBuffer {
    uint64_t remote_noc_encoding;
    volatile tt_l1_ptr db_cb_config_t* local_multicore_cb_cfg;
    volatile tt_l1_ptr db_cb_config_t* remote_multicore_cb_cfg;
};

struct PullAndRelayBuffer {
    uint32_t page_id;
    Buffer buffer;
};

struct PullAndRelayCfg {

    uint32_t num_pages,
             page_size;

    PullAndRelayCircularBuffer cb_buff_cfg;
    PullAndRelayBuffer buff_cfg;

    union {
        uint32_t num_pages_to_read, num_pages_to_write;
    };
};

template <PullAndRelayType src_type, PullAndRelayType dst_type>
void pull_and_relay(
    PullAndRelayCfg& src_pr_cfg,
    PullAndRelayCfg& dst_pr_cfg,
    uint32_t num_pages
) {
    static_assert(src_type == PullAndRelayType::CIRCULAR_BUFFER or src_type == PullAndRelayType::BUFFER);
    static_assert(dst_type == PullAndRelayType::CIRCULAR_BUFFER or dst_type == PullAndRelayType::BUFFER);
    uint32_t num_reads_issued, num_reads_completed, num_writes_completed;
    num_reads_issued = num_reads_completed = num_writes_completed = 0;

    uint32_t num_pages_to_read = min(num_pages, src_pr_cfg.num_pages_to_read);
    uint32_t num_pages_to_write = min(num_pages, dst_pr_cfg.num_pages_to_write);

    DPRINT << "NUM_PAGES_TO_READ: " << num_pages_to_read << ENDL();
    DPRINT << "NUM_PAGES_TO_WRITE: " << num_pages_to_write << ENDL();

    // if constexpr (dst_type == PullAndRelayType::CIRCULAR_BUFFER) {
    //     while(true);
    // }
    while (num_writes_completed != num_pages) {
        // DPRINT << "NOT DONE WRITING" << ENDL();
        if (cb_producer_space_available(num_pages_to_read) and num_reads_issued < num_pages) {
            if constexpr (src_type == PullAndRelayType::CIRCULAR_BUFFER) {
                /*
                    In this case, we are pulling from a circular buffer. We pull from
                    circular buffers typically when our src is an erisc core.
                */
                // static_assert(false); // For now, until I bring up this feature
            } else if constexpr (src_type == PullAndRelayType::BUFFER) {
                /*
                    In this case, we are pulling from a buffer. We pull from
                    buffers when our src is in system memory, or we are pulling in
                    data from local chip SRAM/DRAM.
                */
                // DPRINT << "FETCHING BUFFER" << ENDL();
                src_pr_cfg.buff_cfg.buffer.noc_async_read_buffer(get_write_ptr(0), src_pr_cfg.buff_cfg.page_id, num_pages_to_read);
                src_pr_cfg.buff_cfg.page_id += num_pages_to_read;
            }

            cb_push_back(0, num_pages_to_read);
            num_reads_issued += num_pages_to_read;
            num_pages_to_read = min(num_pages - num_reads_issued, src_pr_cfg.num_pages_to_read);
        }

        if (num_reads_issued > num_writes_completed) {
            if (num_writes_completed == num_reads_completed) {
                noc_async_read_barrier();
                num_reads_completed = num_reads_issued;
            }

            if constexpr (dst_type == PullAndRelayType::CIRCULAR_BUFFER) {
                /*
                    In this case, we are relaying data down to a downstream core, usually for
                    the purpose of further relay.
                */
                // DPRINT << "RELAYING TO CB" << ENDL();
                uint32_t dst_addr = dst_pr_cfg.cb_buff_cfg.local_multicore_cb_cfg->wr_ptr_16B << 4;
                // DPRINT << "GOT DST ADDR" << ENDL();
                uint64_t dst_noc_addr = dst_pr_cfg.cb_buff_cfg.remote_noc_encoding | dst_addr;
                // DPRINT << "Relaying to " << dst_addr << ENDL();
                DPRINT << "PUSH BACK: " << num_pages_to_write << ENDL();
                noc_async_write(get_read_ptr(0), dst_noc_addr, (dst_pr_cfg.cb_buff_cfg.remote_multicore_cb_cfg->page_size_16B << 4) * num_pages_to_write);
                multicore_cb_push_back(
                    dst_pr_cfg.cb_buff_cfg.local_multicore_cb_cfg,  dst_pr_cfg.cb_buff_cfg.remote_multicore_cb_cfg, dst_pr_cfg.cb_buff_cfg.remote_noc_encoding, dst_pr_cfg.cb_buff_cfg.local_multicore_cb_cfg->fifo_limit_16B, num_pages_to_write);
            } else if constexpr (dst_type == PullAndRelayType::BUFFER) {
                /*
                    In this case, we are writing data directly to a buffer.
                */
                // DPRINT << "RELAYING TO BUFFER" << ENDL();
                dst_pr_cfg.buff_cfg.buffer.noc_async_write_buffer(get_read_ptr(0), dst_pr_cfg.buff_cfg.page_id, num_pages_to_write);
                dst_pr_cfg.buff_cfg.page_id += num_pages_to_write;
            }
            noc_async_writes_flushed();
            // DPRINT << "POP FRONT" << ENDL();
            cb_pop_front(0, num_pages_to_write);
            num_writes_completed += num_pages_to_write;
            num_pages_to_write = min(num_pages - num_writes_completed, dst_pr_cfg.num_pages_to_write);
        }
    }
    noc_async_write_barrier();
    DPRINT << "DONE P&R" << ENDL();
}
