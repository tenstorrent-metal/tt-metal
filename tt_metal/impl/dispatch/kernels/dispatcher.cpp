// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_consumer.hpp"
// #include "debug/dprint.h"

// Dispatches fast dispatch commands to worker cores. Currently only runs on remote devices
void kernel_main() {
    constexpr uint32_t host_completion_queue_write_ptr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t completion_queue_start_addr = get_compile_time_arg_val(1);
    constexpr uint32_t completion_queue_size = get_compile_time_arg_val(2);
    constexpr uint32_t cmd_base_address = get_compile_time_arg_val(4);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(5);

    volatile tt_l1_ptr uint32_t* db_rx_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));  // Should be initialized to 0 by host
    volatile tt_l1_ptr uint32_t* db_tx_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));  // Should be num command slots in the remote signaller

    uint64_t producer_noc_encoding = uint64_t(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y)) << 32;
    uint64_t dispatcher_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;
    uint64_t signaller_noc_encoding = uint64_t(NOC_XY_ENCODING(SIGNALLER_NOC_X, SIGNALLER_NOC_Y)) << 32;

    bool db_rx_buf_switch = false;
    while (true) {
        // Wait for producer to supply a command
        db_acquire(db_rx_semaphore_addr, dispatcher_noc_encoding);

        // For each instruction, we need to jump to the relevant part of the device command
        uint32_t command_start_addr = get_command_slot_addr<cmd_base_address, consumer_data_buffer_size>(db_rx_buf_switch);
        uint32_t buffer_transfer_start_addr = command_start_addr + (DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER * sizeof(uint32_t));
        uint32_t program_transfer_start_addr = buffer_transfer_start_addr + ((DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION * DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS) * sizeof(uint32_t));

        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        uint32_t finish = command_ptr[DeviceCommand::finish_idx];       // Whether to notify the host that we have finished
        uint32_t num_workers = command_ptr[DeviceCommand::num_workers_idx];  // If num_workers > 0, it means we are launching a program
        uint32_t num_buffer_transfers = command_ptr[DeviceCommand::num_buffer_transfers_idx];   // How many WriteBuffer commands we are running
        uint32_t is_program = command_ptr[DeviceCommand::is_program_buffer_idx];
        uint32_t page_size = command_ptr[DeviceCommand::page_size_idx];
        uint32_t consumer_cb_size = command_ptr[DeviceCommand::consumer_cb_size_idx];
        uint32_t consumer_cb_num_pages = command_ptr[DeviceCommand::consumer_cb_num_pages_idx];
        uint32_t num_pages = command_ptr[DeviceCommand::num_pages_idx];
        uint32_t producer_consumer_transfer_num_pages = command_ptr[DeviceCommand::producer_consumer_transfer_num_pages_idx];
        uint32_t sharded_buffer_num_cores = command_ptr[DeviceCommand::sharded_buffer_num_cores_idx];
        uint32_t wrap = command_ptr[DeviceCommand::wrap_idx];

        if ((DeviceCommand::WrapRegion)wrap == DeviceCommand::WrapRegion::COMPLETION) {
            // should this case make it to the R chip ... the completion queue interface is on the MMIO chip
            // instead could add a connection from remote issue queue interface to remote completion queue interface to signal completion queue wrap
        } else if (is_program) {
            write_and_launch_program(program_transfer_start_addr, num_pages, command_ptr, producer_noc_encoding, consumer_cb_size, consumer_cb_num_pages, producer_consumer_transfer_num_pages, db_buf_switch);
            wait_for_program_completion(num_workers);
        } else {
            command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_transfer_start_addr);
            // TODO: Uplift this!
            // if the dst buffer type is system memory then we need to write it to the remote signaller to send back
            // write_buffers(command_ptr, completion_queue_start_addr, num_buffer_transfers,  sharded_buffer_num_cores, consumer_cb_size, consumer_cb_num_pages, producer_noc_encoding, producer_consumer_transfer_num_pages, db_buf_switch);
        }

        if (finish) {
            // tell remote signaller
        }

        // notify producer that it has completed a command
        noc_semaphore_inc(producer_noc_encoding | get_semaphore(0), 1);
        db_rx_buf_switch = not db_rx_buf_switch;
        noc_async_write_barrier(); // Barrier for now
    }
}
