// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"

CQWriteInterface cq_write_interface;

FORCE_INLINE
void noc_async_write_multicast_one_packet_no_path_reserve(
    uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests) {

    DEBUG_STATUS('N', 'W', 'P', 'W');
    DEBUG_SANITIZE_WORKER_ADDR(src_local_l1_addr, size);
    DEBUG_SANITIZE_NOC_ADDR(dst_noc_addr_multicast, size);
    while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_REG_CMD_BUF));
    DEBUG_STATUS('N', 'W', 'P', 'D');

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_DISPATCH_MULTICAST_WRITE_VC) |
                             NOC_CMD_BRCST_PACKET |
                             NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr_multicast);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_RET_ADDR_MID, dst_noc_addr_multicast >> 32);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_AT_LEN_BE, size);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_REG_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc_index] += 1;
    noc_nonposted_writes_acked[noc_index] += num_dests;
}

FORCE_INLINE
void multicore_cb_wait_front(db_cb_config_t* db_cb_config, int32_t num_pages) {
    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t pages_received;
    do {
        pages_received = uint16_t(db_cb_config->recv - db_cb_config->ack);
    } while (pages_received < num_pages);
    DEBUG_STATUS('C', 'R', 'B', 'D');
}

void multicore_cb_pop_front(
    db_cb_config_t* db_cb_config,
    const db_cb_config_t* remote_db_cb_config,
    uint64_t producer_noc_encoding,
    uint32_t consumer_fifo_limit_16B,
    uint32_t num_to_write,
    uint32_t page_size_16B) {
    db_cb_config->ack += num_to_write;
    db_cb_config->rd_ptr += page_size_16B * num_to_write;

    if (db_cb_config->rd_ptr >= consumer_fifo_limit_16B) {
        db_cb_config->rd_ptr -= db_cb_config->total_size;
    }

    uint32_t pages_ack_addr = (uint32_t)(&(remote_db_cb_config->ack));
    noc_semaphore_set_remote((uint32_t)(&(db_cb_config->ack)), producer_noc_encoding | pages_ack_addr);
}

FORCE_INLINE
uint32_t get_read_ptr(db_cb_config_t* db_cb_config) { return (db_cb_config->rd_ptr) << 4; }

inline __attribute__((always_inline)) volatile uint32_t* get_cq_completion_write_ptr() {
    return reinterpret_cast<volatile uint32_t*>(CQ_COMPLETION_WRITE_PTR);
}

inline __attribute__((always_inline)) volatile uint32_t* get_cq_completion_read_ptr() {
    return reinterpret_cast<volatile uint32_t*>(CQ_COMPLETION_READ_PTR);
}

FORCE_INLINE
void completion_queue_reserve_back(uint32_t data_size_B) {
    DEBUG_STATUS('N', 'Q', 'R', 'B', 'W');
    uint32_t data_size_16B = align(data_size_B, 32) >> 4;
    uint32_t completion_rd_ptr_and_toggle;
    uint32_t completion_rd_ptr;
    uint32_t completion_rd_toggle;
    do {
        completion_rd_ptr_and_toggle = *get_cq_completion_read_ptr();
        completion_rd_ptr = completion_rd_ptr_and_toggle & 0x7fffffff;
        completion_rd_toggle = completion_rd_ptr_and_toggle >> 31;
    } while (
        ((cq_write_interface.completion_fifo_wr_ptr < completion_rd_ptr) and (cq_write_interface.completion_fifo_wr_ptr + data_size_16B > completion_rd_ptr)) or
        (completion_rd_toggle != cq_write_interface.completion_fifo_wr_toggle) and (cq_write_interface.completion_fifo_wr_ptr == completion_rd_ptr)
    );

    DEBUG_STATUS('N', 'Q', 'R', 'B', 'D');
}

template <uint32_t host_completion_queue_write_ptr_addr>
FORCE_INLINE
void notify_host_of_completion_queue_write_pointer() {
    constexpr static uint64_t pcie_address = (uint64_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32) | host_completion_queue_write_ptr_addr;  // For now, we are writing to host hugepages at offset
    uint32_t completion_wr_ptr_and_toggle = cq_write_interface.completion_fifo_wr_ptr | (cq_write_interface.completion_fifo_wr_toggle << 31);
    volatile tt_l1_ptr uint32_t* completion_wr_ptr_addr = get_cq_completion_write_ptr();
    completion_wr_ptr_addr[0] = completion_wr_ptr_and_toggle;
    noc_async_write(CQ_COMPLETION_WRITE_PTR, pcie_address, 4);
    // Consider changing this to be flush instead of barrier
    // Barrier for now because host reads the completion queue write pointer to determine how many pages can be read
    noc_async_write_barrier();
}

template <uint32_t host_completion_queue_write_ptr_addr>
FORCE_INLINE
void completion_queue_push_back(const uint32_t completion_queue_start_addr, uint32_t push_size_B) {
    uint32_t push_size_16B = align(push_size_B, 32) >> 4;
    cq_write_interface.completion_fifo_wr_ptr += push_size_16B;
    if (cq_write_interface.completion_fifo_wr_ptr >= cq_write_interface.completion_fifo_limit) {
        cq_write_interface.completion_fifo_wr_ptr = completion_queue_start_addr >> 4;
        // Flip the toggle
        cq_write_interface.completion_fifo_wr_toggle = not cq_write_interface.completion_fifo_wr_toggle;
    }

    // Notify host of updated completion wr ptr
    notify_host_of_completion_queue_write_pointer<host_completion_queue_write_ptr_addr>();
}

template <uint32_t host_completion_queue_write_ptr_addr>
FORCE_INLINE void write_buffers(
    db_cb_config_t* db_cb_config,
    const db_cb_config_t* remote_db_cb_config,
    volatile tt_l1_ptr uint32_t* command_ptr,
    const uint32_t completion_queue_start_addr,
    uint32_t num_destinations,
    bool sharded,
    uint32_t sharded_buffer_num_cores,
    uint64_t producer_noc_encoding,
    uint32_t producer_consumer_transfer_num_pages) {

    for (uint32_t i = 0; i < num_destinations; i++) {
        const uint32_t bank_base_address = command_ptr[1];
        const uint32_t num_pages = command_ptr[2];
        const uint32_t page_size = command_ptr[3];
        const uint32_t dst_buf_type = command_ptr[5];
        const uint32_t dst_page_index = command_ptr[7];

        uint32_t num_to_write;
        uint32_t src_addr = get_read_ptr(db_cb_config);
        uint32_t l1_consumer_fifo_limit = src_addr + (db_cb_config->total_size << 4);

        BufferType buffer_type = (BufferType)dst_buf_type;
        Buffer buffer;
        if (buffer_type == BufferType::SYSTEM_MEMORY or not(sharded)) {
            buffer.init(buffer_type, bank_base_address, page_size);
        }
        else {
            buffer.init_sharded(page_size, sharded_buffer_num_cores, bank_base_address,
                            command_ptr + COMMAND_PTR_SHARD_IDX);
        }

        if (buffer_type == BufferType::SYSTEM_MEMORY) {
            completion_queue_reserve_back(num_pages * page_size);
        }
        uint32_t page_id = dst_page_index;
        uint32_t end_page_id = page_id + num_pages;
        while (page_id < end_page_id) {
            num_to_write = min(end_page_id - page_id, producer_consumer_transfer_num_pages);
            multicore_cb_wait_front(db_cb_config, num_to_write);
            uint32_t src_addr = get_read_ptr(db_cb_config);
            buffer.noc_async_write_buffer(src_addr, page_id, num_to_write);
            noc_async_writes_flushed();
            multicore_cb_pop_front(
                db_cb_config,
                remote_db_cb_config,
                producer_noc_encoding,
                (l1_consumer_fifo_limit >> 4),
                num_to_write,
                db_cb_config->page_size);
            page_id += num_to_write;
        }
        if (buffer_type == BufferType::SYSTEM_MEMORY) {
            completion_queue_push_back<host_completion_queue_write_ptr_addr>(completion_queue_start_addr, num_pages * page_size);
        }
    }
    noc_async_write_barrier();
}

template <bool multicast>
FORCE_INLINE void write_program_page(uint32_t page_addr, volatile tt_l1_ptr uint32_t*& command_ptr, bool last_page) {
    uint32_t num_transfers = command_ptr[0];
    command_ptr++;
    uint32_t src = page_addr;

    for (uint32_t i = 0; i < num_transfers; i++) {
        uint32_t num_bytes = command_ptr[0];
        uint32_t dst = command_ptr[1];
        uint32_t dst_noc = command_ptr[2];
        uint32_t num_recv = command_ptr[3];
        bool last_transfer_in_group = command_ptr[4];

        uint64_t dst_noc_addr = (uint64_t(dst_noc) << 32) | dst;

        if constexpr (multicast) {
            noc_async_write_multicast_one_packet_no_path_reserve(src, dst_noc_addr, num_bytes, num_recv);
        } else {
            noc_async_write_one_packet(src, dst_noc_addr, num_bytes);
        }

        command_ptr += 6;
        if (last_transfer_in_group) {
            src = align(src + num_bytes, 16);
        }
    }
}

template <bool multicast>
FORCE_INLINE void program_page_transfer(
    db_cb_config_t* db_cb_config,
    const db_cb_config_t* remote_db_cb_config,
    volatile tt_l1_ptr uint32_t*& command_ptr,
    uint64_t producer_noc_encoding,
    uint32_t producer_consumer_transfer_num_pages,
    uint32_t num_pages_in_transfer) {
    uint32_t l1_consumer_fifo_limit = get_read_ptr(db_cb_config) + (db_cb_config->total_size << 4);
    for (uint32_t page_idx = 0; page_idx < num_pages_in_transfer;) {
        uint32_t num_to_write = min(num_pages_in_transfer - page_idx, producer_consumer_transfer_num_pages);
        multicore_cb_wait_front(db_cb_config, num_to_write);
        uint32_t src_addr = get_read_ptr(db_cb_config);
        for (uint32_t i = 0; i < num_to_write; i++) {
            write_program_page<multicast>(src_addr, command_ptr, i == num_to_write - 1);
            src_addr += DeviceCommand::PROGRAM_PAGE_SIZE;
        }
        page_idx += num_to_write;
        noc_async_writes_flushed();
        multicore_cb_pop_front(
            db_cb_config,
            remote_db_cb_config,
            producer_noc_encoding,
            (l1_consumer_fifo_limit >> 4),
            num_to_write,
            (DeviceCommand::PROGRAM_PAGE_SIZE >> 4));
    }
}

FORCE_INLINE
void write_and_launch_program(
    db_cb_config_t* db_cb_config,
    const db_cb_config_t* remote_db_cb_config,
    uint32_t program_transfer_start_addr,
    uint32_t num_pages,
    volatile tt_l1_ptr uint32_t*& command_ptr,
    uint64_t producer_noc_encoding,
    uint32_t producer_consumer_transfer_num_pages) {
    if (not num_pages) {
        return;
    }

    // GO signals are just data within pages, so we need to set
    // our local 'recv' address value to 0 before we initiate
    // any transfers
    volatile tt_l1_ptr uint32_t* message_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(DISPATCH_MESSAGE_ADDR);
    *message_addr_ptr = 0;

    volatile tt_l1_ptr CommandHeader* header = (CommandHeader*)command_ptr;
    command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(program_transfer_start_addr);
    for (uint32_t transfer_type_idx = 0; transfer_type_idx < (uint32_t) DeviceCommand::TransferType::NUM_TRANSFER_TYPES; transfer_type_idx++) {
        uint32_t num_pages_in_transfer;
        bool multicast = true;
        switch (transfer_type_idx) {
            DeviceCommand::TransferType transfer_type;
            case (uint32_t) DeviceCommand::TransferType::PROGRAM_MULTICAST_PAGES:
                num_pages_in_transfer = header->num_program_multicast_pages;
                break;
            case (uint32_t) DeviceCommand::TransferType::PROGRAM_UNICAST_PAGES:
                multicast = false;
                num_pages_in_transfer = header->num_program_unicast_pages;
                break;
            case (uint32_t) DeviceCommand::TransferType::RUNTIME_ARGS:
                multicast = false;
                num_pages_in_transfer = header->num_runtime_arg_pages;
                break;
            case (uint32_t) DeviceCommand::TransferType::CB_CONFIGS:
                num_pages_in_transfer = header->num_cb_config_pages;
                break;
            case (uint32_t) DeviceCommand::TransferType::GO_SIGNALS_MULTICAST:
                num_pages_in_transfer = header->num_go_signal_multicast_pages;
                break;
            case (uint32_t) DeviceCommand::TransferType::GO_SIGNALS_UNICAST:
                multicast = false;
                num_pages_in_transfer = header->num_go_signal_unicast_pages;
                break;
        }

        if (multicast) {
            program_page_transfer<true>(
                db_cb_config,
                remote_db_cb_config,
                command_ptr,
                producer_noc_encoding,
                producer_consumer_transfer_num_pages,
                num_pages_in_transfer);
        } else {
            program_page_transfer<false>(
                db_cb_config,
                remote_db_cb_config,
                command_ptr,
                producer_noc_encoding,
                producer_consumer_transfer_num_pages,
                num_pages_in_transfer);
        }
    }
}

FORCE_INLINE void wait_for_program_completion(uint32_t num_workers) {
    if (not num_workers)
        return;

    // Wait on worker cores to notify me that they have completed
    DEBUG_STATUS('Q', 'W');

    volatile tt_l1_ptr uint32_t* message_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(DISPATCH_MESSAGE_ADDR);

    while (*message_addr_ptr != num_workers)
        ;


    DEBUG_STATUS('Q', 'D');
}

template <uint32_t host_finish_addr>
FORCE_INLINE void notify_host_complete() {
    volatile tt_l1_ptr uint32_t* finish_ptr = get_cq_finish_ptr();
    finish_ptr[0] = 1;
    constexpr static uint64_t pcie_core_noc_encoding = uint64_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32;
    constexpr static uint64_t finish_noc_addr = pcie_core_noc_encoding | host_finish_addr;
    noc_async_write(uint32_t(finish_ptr), finish_noc_addr, 4);
    noc_async_write_barrier();
    finish_ptr[0] = 0;
}
