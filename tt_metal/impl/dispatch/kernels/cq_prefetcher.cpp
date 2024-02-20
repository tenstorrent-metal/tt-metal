// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/cq_prefetcher.hpp"
#include "debug/dprint.h"

template <bool program_src_buffer, bool program_dst_buffer>
FORCE_INLINE
void program_dynamic_buffers(volatile tt_l1_ptr uint32_t* buffer_transfer_ptr, PullAndRelayCfg& src_pr_cfg, PullAndRelayCfg& dst_pr_cfg, uint32_t page_size, uint32_t num_pages_to_read, uint32_t num_pages_to_write) {
    // TODO(agrebenisan): Make buf transfer a struct
    uint32_t src_bank_base_address = buffer_transfer_ptr[0];
    uint32_t dst_bank_base_address = buffer_transfer_ptr[1];
    uint32_t num_pages = buffer_transfer_ptr[2];
    uint32_t src_buf_type = buffer_transfer_ptr[4];
    uint32_t dst_buf_type = buffer_transfer_ptr[5];
    uint32_t src_page_index = buffer_transfer_ptr[6];
    uint32_t dst_page_index = buffer_transfer_ptr[7];

    if constexpr (program_src_buffer) {
        src_pr_cfg.buff_cfg.buffer.init((BufferType)src_buf_type, src_bank_base_address, page_size);
        src_pr_cfg.buff_cfg.page_id = src_page_index;
        src_pr_cfg.num_pages_to_read = num_pages_to_read;
    }

    if constexpr (program_dst_buffer) {
        dst_pr_cfg.buff_cfg.buffer.init((BufferType)dst_buf_type, dst_bank_base_address, page_size);
        dst_pr_cfg.buff_cfg.page_id = dst_page_index;
        dst_pr_cfg.num_pages_to_write = num_pages_to_write;
    }
    // pull_and_relay<PullAndRelayType::BUFFER, PullAndRelayType::BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages);
}

FORCE_INLINE
void pull_and_relay_buffer_to_cb(volatile tt_l1_ptr uint32_t* buffer_transfer_ptr, PullAndRelayCfg& src_pr_cfg, PullAndRelayCfg& dst_pr_cfg, uint32_t page_size, uint32_t num_pages_to_read, uint32_t num_pages_to_write) {
    uint32_t src_bank_base_address = buffer_transfer_ptr[0];
    uint32_t dst_bank_base_address = buffer_transfer_ptr[1];
    uint32_t num_pages = buffer_transfer_ptr[2];
    uint32_t src_buf_type = buffer_transfer_ptr[4];
    uint32_t dst_buf_type = buffer_transfer_ptr[5];
    uint32_t src_page_index = buffer_transfer_ptr[6];
    uint32_t dst_page_index = buffer_transfer_ptr[7];

    src_pr_cfg.buff_cfg.buffer.init((BufferType)src_buf_type, src_bank_base_address, page_size);
    src_pr_cfg.buff_cfg.page_id = src_page_index;
}

void kernel_main() {
    bool db_buf_switch = false;
    constexpr uint32_t host_issue_queue_read_ptr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t issue_queue_start_addr = get_compile_time_arg_val(1);
    constexpr uint32_t issue_queue_size = get_compile_time_arg_val(2);
    constexpr uint32_t command_start_addr = get_compile_time_arg_val(3);
    constexpr uint32_t data_section_addr = get_compile_time_arg_val(4);
    constexpr uint32_t data_buffer_size = get_compile_time_arg_val(5);
    constexpr uint32_t consumer_cmd_base_addr = get_compile_time_arg_val(6);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(7);
    constexpr uint32_t src_in_host_memory = true;//get_compile_time_arg_val(8);

    setup_issue_queue_read_interface(issue_queue_start_addr, issue_queue_size);

    // Initialize the producer/consumer DB semaphore
    // This represents how many buffers the producer can write to.
    // At the beginning, it can write to two different buffers.
    uint64_t my_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;
    uint64_t remote_noc_encoding = uint64_t(NOC_XY_ENCODING(CONSUMER_NOC_X, CONSUMER_NOC_Y)) << 32;
    uint64_t pcie_core_noc_encoding = uint64_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32;

    volatile tt_l1_ptr uint32_t* db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));  // Should be initialized to 2 by host

    PullAndRelayCfg src_pr_cfg;
    PullAndRelayCfg dst_pr_cfg;

    volatile db_cb_config_t* local_multicore_cb_cfg = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);
    volatile db_cb_config_t* remote_multicore_cb_cfg = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);

    constexpr uint32_t remote_cb_num_pages = (MEM_L1_SIZE - L1_UNRESERVED_BASE - DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND * 2) / DeviceCommand::PROGRAM_PAGE_SIZE;
    constexpr uint32_t remote_cb_size = remote_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

    // Dispatch core has a constant CB
    program_consumer_cb<consumer_cmd_base_addr, consumer_data_buffer_size, consumer_cmd_base_addr, consumer_data_buffer_size>(
        local_multicore_cb_cfg,
        remote_multicore_cb_cfg,
        remote_noc_encoding,
        remote_cb_num_pages,
        DeviceCommand::PROGRAM_PAGE_SIZE,
        remote_cb_size);

    // Set up dispatch core CB
    dst_pr_cfg.cb_buff_cfg.local_multicore_cb_cfg = local_multicore_cb_cfg;
    dst_pr_cfg.cb_buff_cfg.remote_multicore_cb_cfg = remote_multicore_cb_cfg;
    dst_pr_cfg.cb_buff_cfg.remote_noc_encoding = remote_noc_encoding;
    DPRINT << "REMOTE NOC: " << remote_noc_encoding << ENDL();
    DPRINT << "LOCAL ADDRESS: " << uint32_t(local_multicore_cb_cfg) << ENDL();
    DPRINT << "REMOTE ADDRESS: " << uint32_t(remote_multicore_cb_cfg) << ENDL();
    while (true) {
        if constexpr (src_in_host_memory) {
            DPRINT << "WAIT FOR CMD" << ENDL();
            issue_queue_wait_front();
            DPRINT << "DONE WAIT FOR CMD" << ENDL();
            uint32_t rd_ptr = (cq_read_interface.issue_fifo_rd_ptr << 4);
            uint64_t src_noc_addr = pcie_core_noc_encoding | rd_ptr;
            // DPRINT << "READING FROM " << rd_ptr << ENDL();
            noc_async_read(src_noc_addr, command_start_addr, min(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, issue_queue_size - rd_ptr));
            noc_async_read_barrier();
            // DPRINT << "BARRIER" << ENDL();
        } else { // erisc, need some form of enum instead of just bool
            while(true);
        }

        // Producer information
        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        volatile tt_l1_ptr CommandHeader* header = (CommandHeader*)command_ptr;

        uint32_t issue_data_size = header->issue_data_size;
        uint32_t num_buffer_transfers = header->num_buffer_transfers;
        uint32_t stall = header->stall;
        uint32_t page_size = header->page_size;
        uint32_t producer_cb_size = header->producer_cb_size;
        uint32_t consumer_cb_size = header->consumer_cb_size;
        uint32_t producer_cb_num_pages = header->producer_cb_num_pages;
        uint32_t consumer_cb_num_pages = header->consumer_cb_num_pages;
        uint32_t num_pages = header->num_pages;
        uint32_t producer_consumer_transfer_num_pages = header->producer_consumer_transfer_num_pages;
        bool is_program = header->is_program_buffer;
        bool issue_wrap = (DeviceCommand::WrapRegion)header->wrap == DeviceCommand::WrapRegion::ISSUE;

        DPRINT << "IS PROGRAM: " << (uint32_t)is_program << ENDL();

        db_cb_config_t* db_cb_config = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);
        const db_cb_config_t* remote_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);
        if (issue_wrap) {
            // Basically popfront without the extra conditional
            cq_read_interface.issue_fifo_rd_ptr = cq_read_interface.issue_fifo_limit - cq_read_interface.issue_fifo_size;  // Head to beginning of command queue
            cq_read_interface.issue_fifo_rd_toggle = not cq_read_interface.issue_fifo_rd_toggle;
            notify_host_of_issue_queue_read_pointer<host_issue_queue_read_ptr_addr>();
            continue;
        }
        // DPRINT << "PROGRAM LOCAL CB" << ENDL();
        program_local_cb(data_section_addr, producer_cb_num_pages, page_size, producer_cb_size);
        // DPRINT << "WAIT CONSUMER SPACE" << ENDL();
        wait_consumer_space_available(db_semaphore_addr);
        relay_command<command_start_addr, consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch, remote_noc_encoding);
        if (stall) {
            // DPRINT << "WAIT CONSUMER IDLE" << ENDL();
            wait_consumer_idle<2>(db_semaphore_addr);
        }

        // This should be cleaned up, logic kind of awkward
        // DPRINT << "is_program: " << (uint32_t)is_program << ENDL();
        volatile tt_l1_ptr uint32_t* buffer_transfer_ptr = command_ptr + DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;

        constexpr bool program_src_buffer = true;
        constexpr bool program_dst_buffer = true;
        if (is_program) {
            // DPRINT << "UPDATE SEMS FOR ENQUEUE PROGRAM" << ENDL();
            update_producer_consumer_sync_semaphores(my_noc_encoding, remote_noc_encoding, db_semaphore_addr, get_semaphore(0));
            uint32_t num_pages_to_read = remote_cb_num_pages / 2;
            uint32_t num_pages_to_write = 4;
            // DPRINT << "PULL AND RELAY RT ARGS DBR" << ENDL();
            program_dynamic_buffers<program_src_buffer, not program_dst_buffer>(buffer_transfer_ptr, src_pr_cfg, dst_pr_cfg, page_size, num_pages_to_read, num_pages_to_write);
            // DPRINT << "PULL AND RELAY RT ARGS" << ENDL();
            pull_and_relay<PullAndRelayType::BUFFER, PullAndRelayType::CIRCULAR_BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages);
            buffer_transfer_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
            // DPRINT << "PULL AND RELAY BINARIES AND GO SIGNALS" << ENDL();
            // program_dynamic_buffers<program_src_buffer, not program_dst_buffer>(buffer_transfer_ptr, src_pr_cfg, dst_pr_cfg, page_size, num_pages_to_read, num_pages_to_write);
            // pull_and_relay<PullAndRelayType::BUFFER, PullAndRelayType::CIRCULAR_BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages);
        } else {
            uint32_t num_pages_to_read = producer_cb_num_pages / 2;
            uint32_t num_pages_to_write = producer_consumer_transfer_num_pages;
            constexpr bool program_dst_buffer = true;
            program_dynamic_buffers<program_src_buffer, program_dst_buffer>(buffer_transfer_ptr, src_pr_cfg, dst_pr_cfg, page_size, num_pages_to_read, num_pages_to_write);
            pull_and_relay<PullAndRelayType::BUFFER, PullAndRelayType::BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages);
            update_producer_consumer_sync_semaphores(my_noc_encoding, remote_noc_encoding, db_semaphore_addr, get_semaphore(0));
        }

        // Need some synch mechanism for dispatch core to update completion queue
        issue_queue_pop_front<host_issue_queue_read_ptr_addr>(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + issue_data_size);
        db_buf_switch = not db_buf_switch;
    }
}
