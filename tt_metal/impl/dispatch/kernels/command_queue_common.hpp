// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/impl/dispatch/device_command.hpp"

static constexpr uint32_t l1_db_cb_addr_offset = 7 * 16;

FORCE_INLINE
uint32_t get_db_cb_l1_base(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset);
}

FORCE_INLINE
uint32_t get_db_cb_ack_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset);
}

FORCE_INLINE
uint32_t get_db_cb_recv_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 16);
}

FORCE_INLINE
uint32_t get_db_cb_num_pages_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 32);
}

FORCE_INLINE
uint32_t get_db_cb_page_size_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 48);
}

FORCE_INLINE
uint32_t get_db_cb_total_size_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 64);
}

FORCE_INLINE
uint32_t get_db_cb_rd_ptr_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 80);

}

FORCE_INLINE
uint32_t get_db_cb_wr_ptr_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + CQ_START);
}


template <uint32_t cmd_base_address, uint32_t data_buffer_size>
FORCE_INLINE uint32_t get_command_slot_addr(bool db_buf_switch) {
    static constexpr uint32_t command0_start = cmd_base_address;
    static constexpr uint32_t command1_start = command0_start + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_buffer_size;
    return (db_buf_switch) ? command0_start : command1_start;
}

template <uint32_t cmd_base_address, uint32_t data_buffer_size>
FORCE_INLINE uint32_t get_db_buf_addr(bool db_buf_switch) {
    static constexpr uint32_t buf0_start = cmd_base_address + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    static constexpr uint32_t buf1_start = buf0_start + data_buffer_size + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    return (not db_buf_switch) ? buf0_start : buf1_start;
}


FORCE_INLINE
void db_acquire(volatile uint32_t* semaphore, uint64_t noc_encoding) {
    while (semaphore[0] == 0);
    noc_semaphore_inc(noc_encoding | uint32_t(semaphore), -1); // Two's complement addition
    noc_async_write_barrier();
}

FORCE_INLINE
void multicore_cb_push_back(uint64_t consumer_noc_encoding, uint32_t consumer_fifo_limit, uint32_t consumer_fifo_size, bool db_buf_switch, uint32_t page_size, uint32_t num_to_write) {
    // TODO(agrebenisan): Should create a multi-core CB interface... struct in L1
    volatile tt_l1_ptr uint32_t* CQ_CONSUMER_CB_RECV_PTR = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_recv_addr(db_buf_switch));
    volatile tt_l1_ptr uint32_t* CQ_CONSUMER_CB_WRITE_PTR = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_wr_ptr_addr(db_buf_switch));

    *CQ_CONSUMER_CB_RECV_PTR += num_to_write;
    *CQ_CONSUMER_CB_WRITE_PTR += (page_size * num_to_write) >> 4;

    if ((*CQ_CONSUMER_CB_WRITE_PTR << 4) >= consumer_fifo_limit) {
        *CQ_CONSUMER_CB_WRITE_PTR -= consumer_fifo_size >> 4;
    }

    uint32_t pages_recv_addr = get_db_cb_recv_addr(db_buf_switch);
    noc_semaphore_set_remote(uint32_t(CQ_CONSUMER_CB_RECV_PTR), consumer_noc_encoding | pages_recv_addr);
}

FORCE_INLINE
void multicore_cb_wait_front(bool db_buf_switch, int32_t num_pages) {
    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint32_t pages_acked = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_ack_addr(db_buf_switch));
    volatile tt_l1_ptr uint32_t* pages_received_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_recv_addr(db_buf_switch));

    uint16_t pages_received;
    do {
        pages_received = uint16_t(*pages_received_ptr) - pages_acked;
    } while (pages_received < num_pages);
    DEBUG_STATUS('C', 'R', 'B', 'D');
}

void multicore_cb_pop_front(
    uint64_t producer_noc_encoding,
    bool db_buf_switch,
    uint32_t fifo_limit,
    uint32_t fifo_size,
    uint32_t num_pages,
    uint32_t page_size) {
    volatile tt_l1_ptr uint32_t* CQ_CONSUMER_CB_ACK_PTR = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_ack_addr(db_buf_switch));
    volatile tt_l1_ptr uint32_t* CQ_CONSUMER_CB_READ_PTR =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_rd_ptr_addr(db_buf_switch));

    *CQ_CONSUMER_CB_ACK_PTR += num_pages;
    *CQ_CONSUMER_CB_READ_PTR += (page_size * num_pages) >> 4;

    if ((*CQ_CONSUMER_CB_READ_PTR << 4) > fifo_limit) {
        *CQ_CONSUMER_CB_READ_PTR -= fifo_size >> 4;
    }

    uint32_t pages_ack_addr = get_db_cb_ack_addr(db_buf_switch);
    noc_semaphore_set_remote(uint32_t(CQ_CONSUMER_CB_ACK_PTR), producer_noc_encoding | pages_ack_addr);
}
