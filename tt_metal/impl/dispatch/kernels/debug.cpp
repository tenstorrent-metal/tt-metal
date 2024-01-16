// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_producer.hpp"
#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t l1_go_flag_addr = get_compile_time_arg_val(0);
    constexpr uint32_t cmd_start_addr = get_compile_time_arg_val(1);
    constexpr uint32_t data_buffer_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_section_addr = get_compile_time_arg_val(3);
    constexpr uint32_t consumer_cmd_base_addr = get_compile_time_arg_val(4);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(5);

    uint64_t producer_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;
    uint64_t consumer_noc_encoding = uint64_t(NOC_XY_ENCODING(CONSUMER_NOC_X, CONSUMER_NOC_Y)) << 32;

    volatile tt_l1_ptr uint32_t* db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));  // Should be initialized to 1 by host

    volatile tt_l1_ptr uint32_t* l1_go_flag_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_go_flag_addr);

    bool db_buf_switch = false;
    while (true) {
        DPRINT << "waiting for custom go signal from " << l1_go_flag_addr << ENDL();
        uint32_t l1_go_flag_val;
        do {
            l1_go_flag_val = l1_go_flag_ptr[0];
        } while (l1_go_flag_val != 1);

        uint32_t command_start_addr = get_command_slot_addr<cmd_start_addr, data_buffer_size>(db_buf_switch);

        // Producer information
        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        uint32_t data_size = command_ptr[DeviceCommand::data_size_idx];
        uint32_t num_buffer_transfers = command_ptr[DeviceCommand::num_buffer_transfers_idx];
        uint32_t page_size = command_ptr[DeviceCommand::page_size_idx];
        uint32_t producer_cb_size = command_ptr[DeviceCommand::producer_cb_size_idx];
        uint32_t consumer_cb_size = command_ptr[DeviceCommand::consumer_cb_size_idx];
        uint32_t producer_cb_num_pages = command_ptr[DeviceCommand::producer_cb_num_pages_idx];
        uint32_t consumer_cb_num_pages = command_ptr[DeviceCommand::consumer_cb_num_pages_idx];
        uint32_t num_pages = command_ptr[DeviceCommand::num_pages_idx];
        uint32_t wrap = command_ptr[DeviceCommand::wrap_idx];
        uint32_t producer_consumer_transfer_num_pages = command_ptr[DeviceCommand::producer_consumer_transfer_num_pages_idx];
        uint32_t sharded_buffer_num_cores = command_ptr[DeviceCommand::sharded_buffer_num_cores_idx];
        uint32_t finish = command_ptr[DeviceCommand::finish_idx];

        program_local_cb(data_section_addr, producer_cb_num_pages, page_size, producer_cb_size);
        while (db_semaphore_addr[0] == 0)
            ;  // Check that there is space in the consumer

        DPRINT << "debug kernel got:"
               << " " << data_size
               << " " << num_buffer_transfers
               << " " << page_size
               << " " << producer_cb_size
               << " " << consumer_cb_size
               << " " << producer_cb_num_pages
               << " " << consumer_cb_num_pages
               << " " << num_pages
               << " " << wrap
               << " " << producer_consumer_transfer_num_pages
               << " " << sharded_buffer_num_cores
               << " " << finish << ENDL();

        program_consumer_cb<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch, consumer_noc_encoding, consumer_cb_num_pages, page_size, consumer_cb_size);
        relay_command<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch, consumer_noc_encoding);

        // Decrement the semaphore value
        noc_semaphore_inc(producer_noc_encoding | uint32_t(db_semaphore_addr), -1);  // Two's complement addition
        noc_async_write_barrier();

        // Notify the consumer
        noc_semaphore_inc(consumer_noc_encoding | get_semaphore(0), 1);
        noc_async_write_barrier();  // Barrier for now

        // Fetch data and send to the consumer
        // command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
        // uint32_t l1_consumer_fifo_limit = get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch) + consumer_cb_size;

        // bool sharded = sharded_buffer_num_cores > 1;

        // for (uint32_t i = 0; i < num_buffer_transfers; i++) {
        //     const uint32_t bank_base_address = command_ptr[0];
        //     const uint32_t num_pages = command_ptr[2];
        //     const uint32_t page_size = command_ptr[3];
        //     const uint32_t src_buf_type = command_ptr[4];
        //     const uint32_t src_page_index = command_ptr[6];

        //     uint32_t fraction_of_producer_cb_num_pages = consumer_cb_num_pages / 2;

        //     uint32_t num_to_write = min(num_pages, producer_consumer_transfer_num_pages); // This must be a bigger number for perf.
        //     uint32_t num_writes_completed = 0;

        //     while (num_writes_completed != num_pages) {
        //         if (cb_consumer_space_available(db_buf_switch, num_to_write)) {
        //             uint32_t dst_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_wr_ptr_addr(db_buf_switch))[0] << 4;
        //             uint64_t dst_noc_addr = consumer_noc_encoding | dst_addr;
        //             uint32_t l1_read_ptr = get_read_ptr(0);
        //             noc_async_write(l1_read_ptr, dst_noc_addr, page_size * num_to_write);
        //             multicore_cb_push_back(consumer_noc_encoding, l1_consumer_fifo_limit, consumer_cb_size, db_buf_switch, page_size, num_to_write);
        //             noc_async_write_barrier();
        //             num_writes_completed += num_to_write;
        //             num_to_write = min(num_pages - num_writes_completed, producer_consumer_transfer_num_pages);
        //         }
        //     }
        //     command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
        // }

        // db_buf_switch = not db_buf_switch;
        break;
    }
}
