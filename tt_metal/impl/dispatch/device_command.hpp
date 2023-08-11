/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>

#include "tt_metal/hostdevcommon/common_values.hpp"

using std::array;

static constexpr u32 NUM_DISPATCH_CORES = 108;  // TODO(agrebenisan): Need to fix for wormhole

// The beginning of data section for dispatcher
static constexpr u32 DEVICE_COMMAND_DATA_ADDR = 150 * 1024;

static constexpr u32 DEVICE_COMMAND_NUM_ENTRIES = 5632; // 22KB device command
static constexpr u32 NUM_ENTRIES_PER_BUFFER_RELAY = 8;
static constexpr u32 CONTROL_SECTION_NUM_ENTRIES = 16;
static constexpr u32 NUM_DATA_MOVEMENT_INSTRUCTIONS = 4;
static constexpr u32 RELAY_BUFFER_NUM_ENTRIES = NUM_DATA_MOVEMENT_INSTRUCTIONS * NUM_ENTRIES_PER_BUFFER_RELAY;
static constexpr u32
    RELAY_PROGRAM_NUM_ENTRIES =  // Whatever is left of the available size, we allocate for relaying program data
    DEVICE_COMMAND_NUM_ENTRIES - CONTROL_SECTION_NUM_ENTRIES - RELAY_BUFFER_NUM_ENTRIES - NUM_DISPATCH_CORES;

static constexpr u32 HUGE_PAGE_SIZE = 1024 * 1024 * 1024;

static constexpr u32 PROGRAM_PAGE_SIZE = 1024;

// We need to ensure that the command size is divisible by 32
static_assert(DEVICE_COMMAND_NUM_ENTRIES * sizeof(u32) % 32 == 0);

// To stay consistent with the 16B addressing on grayskull, I created this constant
static constexpr u32 NUM_16B_WORDS_IN_DEVICE_COMMAND = (DEVICE_COMMAND_NUM_ENTRIES * sizeof(u32)) / 16;
class DeviceCommand {
   private:
    static constexpr u32 num_4B_words_in_relay_buffer_instruction = 8;
    static constexpr u32 num_possible_relay_buffer_instructions = 4;

    // Command header
    static constexpr u32 wrap_idx = 0;
    static constexpr u32 finish_idx = 1;
    static constexpr u32 num_workers_idx = 2;
    static constexpr u32 num_multicast_messages_idx = 3;
    static constexpr u32 data_size_in_bytes_idx = 4;
    static constexpr u32 num_relay_buffer_reads_idx = 5;
    static constexpr u32 num_relay_buffer_writes_idx = 6;
    static constexpr u32 num_program_srcs_idx = 7;

    static_assert(CONTROL_SECTION_NUM_ENTRIES == 16);
    u32 worker_launch_idx = CONTROL_SECTION_NUM_ENTRIES;  // So far, we unicast the de-assert until Almeet provides
                                                          // support for program.logical_cores() -> core range set

    // Relay instructions
    u32 relay_buffer_entry_idx = CONTROL_SECTION_NUM_ENTRIES +
                                 NUM_DISPATCH_CORES;  // Not const, keeps track of which index in the array we're at

    static_assert(CONTROL_SECTION_NUM_ENTRIES + NUM_DISPATCH_CORES + RELAY_BUFFER_NUM_ENTRIES == 156);
    u32 relay_program_entry_idx = CONTROL_SECTION_NUM_ENTRIES + NUM_DISPATCH_CORES + RELAY_BUFFER_NUM_ENTRIES;

    array<u32, DEVICE_COMMAND_NUM_ENTRIES> desc;

    // Creates a buffer read or write in which the first address is a single page and the second can be multiple pages.
    // Num bursts corresponds to how many bursts of data we need to pull into the dispatch core (essentially the number
    // of relays). We try to read in as much data per burst as possible, and if the data is not divisible by num bursts,
    // we have a remainder step in which we try to relay the last chunk, specified by remainder_burst_size.
    void add_buffer_instruction(
        u32 addr0,
        u32 addr0_noc,
        u32 addr1,

        u32 padded_buf_size,
        u32 burst_size,
        u32 page_size,
        u32 padded_page_size,
        u32 buf_type);

   public:
    DeviceCommand();
    static constexpr u32 size() { return DEVICE_COMMAND_NUM_ENTRIES; }
    static constexpr u32 size_in_bytes() { return DEVICE_COMMAND_NUM_ENTRIES * sizeof(u32); }

    void finish();  // Creates a finish command, in which the command queue is blocked until the device notifies host of
                    // completion.

    void set_num_workers(const u32 num_workers);
    void set_num_multicast_messages(const u32 num_multicast_messages);  // Specifies how many core ranges to deassert

    void set_num_program_srcs(const u32 num_srcs);

    void add_read_buffer_instruction(
        const u32 dst,
        const u32 dst_noc,
        const u32 src,

        const u32 padded_buf_size,
        const u32 burst_size,
        const u32 page_size,
        const u32 padded_page_size,
        const u32 buf_type);

    void add_write_buffer_instruction(
        const u32 src,
        const u32 src_noc,
        const u32 dst,

        const u32 padded_buf_size,
        const u32 burst_size,
        const u32 page_size,
        const u32 padded_page_size,
        const u32 buf_type);

    void write_program_entry(const u32 val);

    void add_write_page_partial_instruction(
        const u32 num_bytes, const u32 dst, const u32 dst_noc, const u32 num_receivers);

    void set_data_size_in_bytes(const u32 data_size_in_bytes);

    void set_multicast_message_noc_coord(const u32 core_coord, const u32 num_messages);

    u32 get_data_size_in_bytes() const;

    const array<u32, DEVICE_COMMAND_NUM_ENTRIES>& get_desc() const;
};
