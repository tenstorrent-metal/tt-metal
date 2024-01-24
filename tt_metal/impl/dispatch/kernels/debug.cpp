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

    uint32_t bank_base_address = get_arg_val<uint32_t>(0);

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


        db_cb_config_t *db_cb_config = (db_cb_config_t *)(CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset));
        db_cb_config_t *processor_db_cb_config = (db_cb_config_t *)(CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset));

        program_consumer_cb<consumer_cmd_base_addr, consumer_data_buffer_size>(
            db_cb_config,
            processor_db_cb_config,
            db_buf_switch,
            consumer_noc_encoding,
            consumer_cb_num_pages,
            page_size,
            consumer_cb_size);
        relay_command<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch, consumer_noc_encoding);

        // Decrement the semaphore value
        noc_semaphore_inc(producer_noc_encoding | uint32_t(db_semaphore_addr), -1);  // Two's complement addition
        noc_async_write_barrier();

        // Notify the consumer
        noc_semaphore_inc(consumer_noc_encoding | get_semaphore(0), 1);
        noc_async_write_barrier();  // Barrier for now

        uint32_t tx_consumer_cb_size = (db_cb_config->total_size << 4);
        uint32_t tx_consumer_cb_num_pages = db_cb_config->num_pages;

        DPRINT << "consumer_cb_size " << consumer_cb_size
               << " tx_consumer_cb_size " << tx_consumer_cb_size
               << " consumer_cb_num_pages " << consumer_cb_num_pages
               << " tx_consumer_cb_num_pages " << tx_consumer_cb_num_pages << ENDL();

        uint32_t fraction_of_producer_cb_num_pages = tx_consumer_cb_num_pages / 2;

        uint32_t l1_consumer_fifo_limit_16B =
            (get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch) + tx_consumer_cb_size) >> 4;

        uint32_t num_to_read = min(num_pages, fraction_of_producer_cb_num_pages);
        uint32_t num_to_write = min(num_pages, producer_consumer_transfer_num_pages); // This must be a bigger number for perf.
        uint32_t num_reads_issued = 0;
        uint32_t num_reads_completed = 0;
        uint32_t num_writes_completed = 0;
        uint32_t src_page_id = 0;

        // Fetch data and send to the consumer
        Buffer buffer;
        buffer.init(BufferType::DRAM, bank_base_address, page_size);

        DPRINT << "num pages: " << num_pages << ENDL();
        DPRINT << "bank base addr " << bank_base_address << ENDL();

        while (num_writes_completed != num_pages) {
            // Context switch between reading in pages and sending them to the consumer.
            // These APIs are non-blocking to allow for context switching.
            if (cb_producer_space_available(num_to_read) and num_reads_issued < num_pages) {
                uint32_t l1_write_ptr = get_write_ptr(0);
                buffer.noc_async_read_buffer(l1_write_ptr, src_page_id, num_to_read);
                cb_push_back(0, num_to_read);
                num_reads_issued += num_to_read;
                src_page_id += num_to_read;

                uint32_t num_pages_left = num_pages - num_reads_issued;

                DPRINT << "read " << num_to_read
                       << " num pages left " << num_pages_left << ENDL();

                num_to_read = min(num_pages_left, fraction_of_producer_cb_num_pages);
            }

            if (num_reads_issued > num_writes_completed and cb_consumer_space_available(db_cb_config, num_to_write)) {
                if (num_writes_completed == num_reads_completed) {
                    noc_async_read_barrier();
                    num_reads_completed = num_reads_issued;
                }

                uint32_t dst_addr = (db_cb_config->wr_ptr << 4);
                uint64_t dst_noc_addr = consumer_noc_encoding | dst_addr;
                uint32_t l1_read_ptr = get_read_ptr(0);

                // volatile tt_l1_ptr uint32_t* l1_data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_ptr);
                // for (int i = 0; i < 512; i++) {
                //     // DPRINT << "data[" << i << "] = " << l1_data[i] << ENDL();
                //     // l1_data[i] = i;
                // }

                noc_async_write(l1_read_ptr, dst_noc_addr, page_size * num_to_write);
                multicore_cb_push_back(
                    db_cb_config, processor_db_cb_config, consumer_noc_encoding, l1_consumer_fifo_limit_16B, num_to_write);
                noc_async_write_barrier();
                DPRINT << " wrote " << num_to_write << " and signalled to processor core " << ENDL();
                cb_pop_front(0, num_to_write);
                num_writes_completed += num_to_write;
                num_to_write = min(num_pages - num_writes_completed, producer_consumer_transfer_num_pages);
            }
        }

        // db_buf_switch = not db_buf_switch;
        break;
    }
}
