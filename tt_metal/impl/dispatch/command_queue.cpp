// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/command_queue.hpp"

#include <algorithm>  // for copy() and assign()
#include <iterator>   // for back_inserter
#include <memory>
#include <string>

#include "allocator/allocator.hpp"
#include "debug_tools.hpp"
#include "dev_msgs.h"
#include "tt_metal/common/logger.hpp"
#include "noc/noc_parameters.h"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"

using std::map;
using std::pair;
using std::set;
using std::shared_ptr;
using std::unique_ptr;

std::mutex finish_mutex;
std::condition_variable finish_cv;

namespace tt::tt_metal {

uint32_t get_noc_unicast_encoding(CoreCoord coord) { return NOC_XY_ENCODING(NOC_X(coord.x), NOC_Y(coord.y)); }

// EnqueueReadBufferCommandSection
std::vector<uint32_t> EnqueueReadBufferCommand::commands;

EnqueueReadBufferCommand::EnqueueReadBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    Buffer& buffer,
    void* dst,
    bool stall,
    SystemMemoryManager& manager,
    uint32_t src_page_index,
    std::optional<uint32_t> pages_to_read) :
    command_queue_id(command_queue_id),
    dst(dst),
    stall(stall),
    manager(manager),
    buffer(buffer),
    src_page_index(src_page_index),
    pages_to_read(pages_to_read.has_value() ? pages_to_read.value() : buffer.num_pages()) {

    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to read an invalid buffer");

    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());

    // TODO: ADD CQ_PREFETCH_CMD_STALL, CQ_PREFETCH_CMD_RELAY_INLINE, AND CQ_DISPATCH_CMD_WAIT!!!
    // Create commands once, subsequent enqueue_read_buffer calls can just update dynamic fields
    if (this->commands.empty()) {
        CQPrefetchCmd no_flush;
        no_flush.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH;
        no_flush.relay_inline.length = sizeof(CQDispatchCmd);
        no_flush.relay_inline.stride = align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), HUGEPAGE_ALIGNMENT);

        uint32_t *no_flush_ptr = (uint32_t *)&no_flush;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*no_flush_ptr++);
        }

        CQDispatchCmd dev_to_host_cmd;
        dev_to_host_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_HOST;
        dev_to_host_cmd.write_host.length = 0;

        uint32_t *dev_to_host_cmd_ptr = (uint32_t *)&dev_to_host_cmd;
        for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*dev_to_host_cmd_ptr++);
        }

        uint32_t padding;
        if ((padding = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) % HUGEPAGE_ALIGNMENT) != 0) {
            for (int i = 0; i < padding / sizeof(uint32_t); i++) {
                this->commands.push_back(0);
            }
        }

        CQPrefetchCmd relay_buffer;
        relay_buffer.base.cmd_id = CQ_PREFETCH_CMD_RELAY_PAGED;

        relay_buffer.relay_paged.is_dram = 0;
        relay_buffer.relay_paged.start_page = 0;
        relay_buffer.relay_paged.base_addr = 0;
        relay_buffer.relay_paged.page_size = 0;
        relay_buffer.relay_paged.pages = 0;

        uint32_t *relay_buffer_ptr = (uint32_t *)&relay_buffer;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*relay_buffer_ptr++);
        }

        if ((padding = sizeof(CQPrefetchCmd) % HUGEPAGE_ALIGNMENT) != 0) {
            for (int i = 0; i < padding / sizeof(uint32_t); i++) {
                this->commands.push_back(0);
            }
        }
    }
}

/**
 * @brief Create set of commands that are processed in order to facilitate reading data from buffer on device.
 * Commands generated:
 *  Header: (Optional) CQ_PREFETCH_CMD_STALL to avoid RAW hazards
 *  Payload: (Optional) CQ_PREFETCH_CMD_RELAY_INLINE to relay the dispatch command below
 *  Payload: (Optional) CQ_DISPATCH_CMD_WAIT to avoid RAW hazards
 *  Header: CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH instructs prefetcher to relay payload to dispatcher after TRANSFER_PAGE_SIZE is read by prefetcher
 *  Paylod: CQ_DISPATCH_CMD_WRITE_HOST instructs dispatcher to write data to host completion queue
 *  Header: CQ_PREFETCH_CMD_RELAY_PAGED instructs prefetcher to relay data from some interleaved buffer to dispatcher
 *  Payload: Empty
 * @param dst_address
 * @return * const void
 */
const void EnqueueReadBufferCommand::assemble_device_commands(uint32_t dst_address) {
    uint32_t padded_page_size = align(this->buffer.page_size(), 32);

    if (this->stall) {
        TT_THROW("Stall before reading is unimplemented in FD2.0");
    }

    uint32_t dev_to_host_cmd_idx = sizeof(CQPrefetchCmd) / sizeof(uint32_t);
    CQDispatchCmd *dev_to_host_cmd = (CQDispatchCmd*)(this->commands.data() + dev_to_host_cmd_idx);
    dev_to_host_cmd->write_host.length = this->pages_to_read * padded_page_size;

    uint32_t relay_paged_cmd_offset = sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd);
    relay_paged_cmd_offset += relay_paged_cmd_offset % HUGEPAGE_ALIGNMENT;
    uint32_t relay_buffer_cmd_idx = relay_paged_cmd_offset / sizeof(uint32_t);

    CQPrefetchCmd *relay_buffer = (CQPrefetchCmd*)(this->commands.data() + relay_buffer_cmd_idx);
    relay_buffer->relay_paged.is_dram = (this->buffer.buffer_type() == BufferType::DRAM);
    relay_buffer->relay_paged.start_page = this->src_page_index;
    relay_buffer->relay_paged.base_addr = this->buffer.address();
    relay_buffer->relay_paged.page_size = padded_page_size;
    relay_buffer->relay_paged.pages = this->pages_to_read;
}

void EnqueueReadBufferCommand::process() {
    this->assemble_device_commands(0);

    uint32_t fetch_size_bytes = this->commands.size() * sizeof(uint32_t);

    // move this into the command queue interface
    TT_ASSERT(fetch_size_bytes <= MAX_PREFETCH_COMMAND_SIZE, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((fetch_size_bytes >> PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "FetchQ command too large to represent");

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    this->manager.cq_write(this->commands.data(), fetch_size_bytes, write_ptr);
    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);

    // chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    // uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    // std::vector<uint32_t> recv( fetch_size_bytes / sizeof(uint32_t), 0);
    // tt::Cluster::instance().read_sysmem(recv.data(), fetch_size_bytes, write_ptr, mmio_device_id, channel);
    // for (int i = 0; i < recv.size(); i++) {
    //     std::cout << "cq has " << recv[i] << std::endl;
    // }

    std::cout << "fetch size " << fetch_size_bytes << std::endl;
    uint16_t fetch_size_16B = fetch_size_bytes >> PREFETCH_Q_LOG_MINSIZE;
    this->manager.fetch_queue_push_back(fetch_size_16B, this->command_queue_id);
}

// EnqueueWriteBufferCommand section
std::vector<uint32_t> EnqueueWriteBufferCommand::commands;

EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    const Buffer& buffer,
    const void* src,
    SystemMemoryManager& manager,
    uint32_t dst_page_index,
    std::optional<uint32_t> pages_to_write) :
    command_queue_id(command_queue_id),
    manager(manager),
    src(src),
    buffer(buffer),
    dst_page_index(dst_page_index),
    pages_to_write(pages_to_write.has_value() ? pages_to_write.value() : buffer.num_pages()) {
    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to write to an invalid buffer");
    this->device = device;
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());

    // TODO: ADD CQ_DISPATCH_CMD_WAIT!!!
    if (this->commands.empty()) {
        CQPrefetchCmd relay_write;
        relay_write.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
        // relay_inline attributes set in assemble_device_commands
        relay_write.relay_inline.length = 0;
        relay_write.relay_inline.stride = 0;

        uint32_t *relay_write_ptr = (uint32_t *)&relay_write;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*relay_write_ptr++);
        }

        CQDispatchCmd write_paged_cmd;
        write_paged_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PAGED;
         // write_paged attributes set in assemble_device_commands
        write_paged_cmd.write_paged.is_dram = 0;
        write_paged_cmd.write_paged.start_page = 0;
        write_paged_cmd.write_paged.base_addr = 0;
        write_paged_cmd.write_paged.page_size = 0;
        write_paged_cmd.write_paged.pages = 0;

        uint32_t *write_paged_cmd_ptr = (uint32_t *)&write_paged_cmd;
        for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*write_paged_cmd_ptr++);
        }
    }
}

const void EnqueueWriteBufferCommand::assemble_device_commands(uint32_t) {
    uint32_t num_pages = this->pages_to_write;
    uint32_t padded_page_size = align(this->buffer.page_size(), 32);

    uint32_t relay_write_idx = 0; // TODO: Update when wait is added
    CQPrefetchCmd *relay_write_cmd = (CQPrefetchCmd*)(this->commands.data() + relay_write_idx);
    uint32_t payload_size_bytes = sizeof(CQDispatchCmd) + (this->pages_to_write * padded_page_size);
    relay_write_cmd->relay_inline.length = payload_size_bytes;
    relay_write_cmd->relay_inline.stride = align(sizeof(CQPrefetchCmd) + payload_size_bytes, HUGEPAGE_ALIGNMENT);

    uint32_t write_paged_cmd_idx = relay_write_idx + (sizeof(CQPrefetchCmd) / sizeof(uint32_t));
    CQDispatchCmd *write_paged_cmd = (CQDispatchCmd*)(this->commands.data() + write_paged_cmd_idx);
    write_paged_cmd->write_paged.is_dram = uint8_t(this->buffer.buffer_type() == BufferType::DRAM);
    write_paged_cmd->write_paged.start_page = this->dst_page_index;
    write_paged_cmd->write_paged.base_addr = this->buffer.address();
    write_paged_cmd->write_paged.page_size = padded_page_size;
    write_paged_cmd->write_paged.pages = this->pages_to_write;
}

void EnqueueWriteBufferCommand::process() {
    this->assemble_device_commands(0);

    uint32_t padded_page_size = align(this->buffer.page_size(), 32);
    uint32_t data_size_in_bytes = this->pages_to_write * padded_page_size;
    uint32_t fetch_size_bytes = (this->commands.size() * sizeof(uint32_t)) + data_size_in_bytes;

    // TODO: move this into the command queue interface
    TT_ASSERT(fetch_size_bytes <= MAX_PREFETCH_COMMAND_SIZE, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((fetch_size_bytes >> PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "FetchQ command too large to represent");

    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    this->manager.cq_write(this->commands.data(), sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), write_ptr);
    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);

    uint32_t data_write_ptr = write_ptr + (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd));

    uint32_t unpadded_src_offset = this->dst_page_index * this->buffer.page_size();
    if (this->buffer.page_size() % 32 != 0 and this->buffer.page_size() != this->buffer.size()) {
        // If page size is not 32B-aligned, we cannot do a contiguous write
        uint32_t src_address_offset = unpadded_src_offset;
        uint32_t padded_page_size = align(this->buffer.page_size(), 32);
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_in_bytes;
             sysmem_address_offset += padded_page_size) {
            this->manager.cq_write(
                (char*)this->src + src_address_offset,
                this->buffer.page_size(),
                data_write_ptr + sysmem_address_offset);
            src_address_offset += this->buffer.page_size();
        }
    } else {
        this->manager.cq_write((char*)this->src + unpadded_src_offset, data_size_in_bytes, data_write_ptr);
    }

    uint16_t fetch_size_16B = fetch_size_bytes >> PREFETCH_Q_LOG_MINSIZE;
    this->manager.fetch_queue_push_back(fetch_size_16B, this->command_queue_id);
}

EnqueueProgramCommand::EnqueueProgramCommand(
    uint32_t command_queue_id,
    Device* device,
    const Program& program,
    SystemMemoryManager& manager,
    bool stall,
    std::optional<std::reference_wrapper<Trace>> trace) :
    command_queue_id(command_queue_id),
    manager(manager),
    program(program),
    stall(stall) {
    this->device = device;
    this->trace = trace;
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

const void EnqueueProgramCommand::assemble_device_commands(uint32_t host_data_src) {
}

void EnqueueProgramCommand::process() {
}

std::vector<uint32_t> EnqueueRecordEventCommand::commands;

EnqueueRecordEventCommand::EnqueueRecordEventCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, uint32_t event_id):
    command_queue_id(command_queue_id), device(device), manager(manager), event_id(event_id) {

    // TODO: ADD A WAIT!!
    if (this->commands.empty()) {
        // uint32_t stride = align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), HUGEPAGE_ALIGNMENT);

        uint32_t dispatch_event_payload = sizeof(CQDispatchCmd) + EVENT_PADDED_SIZE;
        // Command to write event to L1
        CQPrefetchCmd relay_event_l1;
        relay_event_l1.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
        relay_event_l1.relay_inline.length = dispatch_event_payload;
        relay_event_l1.relay_inline.stride = align(sizeof(CQPrefetchCmd) + dispatch_event_payload, HUGEPAGE_ALIGNMENT);

        uint32_t *relay_event_l1_ptr = (uint32_t *)&relay_event_l1;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*relay_event_l1_ptr++);
        }

        uint8_t num_hw_cqs = this->device->num_hw_cqs();
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
        tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_core(this->device->id(), channel, this->command_queue_id);
        CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, CoreType::WORKER);

        CQDispatchCmd write_event_l1_cmd;
        write_event_l1_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE;
        write_event_l1_cmd.write.num_mcast_dests = 0;
        write_event_l1_cmd.write.noc_xy_addr = NOC_XY_ENCODING(dispatch_physical_core.x, dispatch_physical_core.y);
        write_event_l1_cmd.write.addr = CQ_COMPLETION_LAST_EVENT;
        write_event_l1_cmd.write.length = EVENT_PADDED_SIZE;

        uint32_t *write_event_l1_cmd_ptr = (uint32_t *)&write_event_l1_cmd;
        for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*write_event_l1_cmd_ptr++);
        }

        for (int i = 0; i < EVENT_PADDED_SIZE / sizeof(uint32_t); i++) {
            this->commands.push_back(0);
        }

        uint32_t padding;
        if ((padding = (sizeof(CQPrefetchCmd) + dispatch_event_payload) % HUGEPAGE_ALIGNMENT) != 0) {
            for (int i = 0; i < padding / sizeof(uint32_t); i++) {
                this->commands.push_back(0);
            }
        }

        // Command to write event to completion queue
        CQPrefetchCmd relay_event_host;
        relay_event_host.base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
        relay_event_host.relay_inline.length = dispatch_event_payload;
        relay_event_host.relay_inline.stride = align(sizeof(CQPrefetchCmd) + dispatch_event_payload, HUGEPAGE_ALIGNMENT);

        uint32_t *relay_event_host_ptr = (uint32_t *)&relay_event_host;
        for (int i = 0; i < sizeof(CQPrefetchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*relay_event_host_ptr++);
        }

        CQDispatchCmd write_event_host_cmd;
        write_event_host_cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_HOST;
        write_event_host_cmd.write_host.length = EVENT_PADDED_SIZE;

        uint32_t *write_event_host_cmd_ptr = (uint32_t *)&write_event_host_cmd;
        for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
            this->commands.push_back(*write_event_host_cmd_ptr++);
        }

        for (int i = 0; i < EVENT_PADDED_SIZE / sizeof(uint32_t); i++) {
            this->commands.push_back(0);
        }

        if ((padding = (sizeof(CQPrefetchCmd) + dispatch_event_payload) % HUGEPAGE_ALIGNMENT) != 0) {
            for (int i = 0; i < padding / sizeof(uint32_t); i++) {
                this->commands.push_back(0);
            }
        }
    }
}

const void EnqueueRecordEventCommand::assemble_device_commands(uint32_t) {
    uint32_t event_payload_offset = sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd);
    uint32_t write_event_l1_idx = event_payload_offset / sizeof(uint32_t);

    uint32_t *event_l1_location = (uint32_t*)(this->commands.data() + write_event_l1_idx);
    *event_l1_location = this->event_id;

    uint32_t write_host_cmd_offset = align(event_payload_offset + EVENT_PADDED_SIZE, HUGEPAGE_ALIGNMENT);

    uint32_t write_event_host_idx = (write_host_cmd_offset + event_payload_offset) / sizeof(uint32_t);

    uint32_t *event_host_location = (uint32_t*)(this->commands.data() + write_event_host_idx);
    *event_host_location = this->event_id;
}

void EnqueueRecordEventCommand::process() {
    this->assemble_device_commands(0);

    uint32_t fetch_size_bytes = this->commands.size() * sizeof(uint32_t);

    // move this into the command queue interface
    TT_ASSERT(fetch_size_bytes <= MAX_PREFETCH_COMMAND_SIZE, "Generated prefetcher command exceeds max command size");
    TT_ASSERT((fetch_size_bytes >> PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "FetchQ command too large to represent");

    this->manager.fetch_queue_reserve_back(this->command_queue_id);

    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    this->manager.cq_write(this->commands.data(), fetch_size_bytes, write_ptr);
    this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);

    uint16_t fetch_size_16B = fetch_size_bytes >> PREFETCH_Q_LOG_MINSIZE;
    this->manager.fetch_queue_push_back(fetch_size_16B, this->command_queue_id);
}

EnqueueWaitForEventCommand::EnqueueWaitForEventCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, const Event& sync_event):
    command_queue_id(command_queue_id), device(device), manager(manager), sync_event(sync_event) {
        this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
        // Should not be encountered under normal circumstances (record, wait) unless user is modifying sync event ID.
        TT_ASSERT(command_queue_id != sync_event.cq_id,
            "EnqueueWaitForEventCommand cannot wait on it's own event id on the same CQ. CQ ID: {}", command_queue_id);
}

const void EnqueueWaitForEventCommand::assemble_device_commands(uint32_t) {
    // DeviceCommand command;
    // command.set_issue_data_size(0); // No extra data just CMD.
    // command.set_completion_data_size(align(EVENT_PADDED_SIZE, 32));
    // command.set_event(this->event);
    // if (tt::Cluster::instance().arch() == tt::ARCH::WORMHOLE_B0) {
    //     command.set_stall(); // Ensure ordering w/ programs in FD1.3+ prefetcher.
    // }
    // // #5529 - Cross chip sync needs to be implemented. Currently, we only support sync on the same chip.
    // TT_ASSERT(this->sync_event.device == this->device,
    //         "EnqueueWaitForEvent() cross-chip sync not yet supported. Sync event device: {} this device: {}",
    //         this->sync_event.device->id(), this->device->id());

    // auto &event_sync_hw_cq = this->sync_event.device->command_queue(this->sync_event.cq_id).hw_command_queue();
    // auto event_sync_core = this->sync_event.device->worker_core_from_logical_core(event_sync_hw_cq.completion_queue_writer_core);

    // // Let dispatcher know this is sync event, and what core/event_id to sync on.
    // command.set_is_event_sync(true);
    // command.set_event_sync_core_x(event_sync_core.x);
    // command.set_event_sync_core_y(event_sync_core.y);
    // command.set_event_sync_event_id(this->sync_event.event_id);
}

void EnqueueWaitForEventCommand::process() {
    // uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    // this->assemble_device_commands(0);
    // DeviceCommand cmd;
    // uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    // this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_id);
    // this->manager.cq_write(cmd.data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    // this->manager.issue_queue_push_back(cmd_size, detail::LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
}

// HWCommandQueue section
HWCommandQueue::HWCommandQueue(Device* device, uint32_t id) : manager(device->sysmem_manager()), completion_queue_thread{} {
    ZoneScopedN("CommandQueue_constructor");
    this->device = device;
    this->id = id;
    this->num_entries_in_completion_q = 0;
    this->num_completed_completion_q_reads = 0;

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    this->size_B = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / device->num_hw_cqs();

    tt_cxy_pair completion_q_writer_location =
        dispatch_core_manager::get(device->num_hw_cqs()).completion_queue_writer_core(device->id(), channel, this->id);

    this->completion_queue_writer_core = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);

    this->exit_condition = false;
    std::thread completion_queue_thread = std::thread(&HWCommandQueue::read_completion_queue, this);
    this->completion_queue_thread = std::move(completion_queue_thread);
    this->stall_before_read = false;
}

HWCommandQueue::~HWCommandQueue() {
    ZoneScopedN("HWCommandQueue_destructor");
    if (this->exit_condition) {
        this->completion_queue_thread.join();  // We errored out already prior
    } else {

        // TODO: SEND THE TERMINATE CMD?

        TT_ASSERT(
            this->issued_completion_q_reads.empty(),
            "There should be no reads in flight after closing our completion queue thread");
        TT_ASSERT(
            this->num_entries_in_completion_q == this->num_completed_completion_q_reads,
            "There shouldn't be any commands in flight after closing our completion queue thread. Num uncompleted commands: {}", this->num_entries_in_completion_q - this->num_completed_completion_q_reads);
        this->exit_condition = true;
        this->completion_queue_thread.join();
    }
}

template <typename T>
void HWCommandQueue::enqueue_command(T& command, bool blocking) {
    command.process();
    if (blocking) {
        this->finish();
    }

    // If this command has side-effects, then the next scheduled read needs
    // to stall before fetching. Else, it can pre-fetch
    this->stall_before_read = command.has_side_effects();
}

// TODO: Currently converting page ordering from interleaved to sharded and then doing contiguous read/write
//  Look into modifying command to do read/write of a page at a time to avoid doing copy
void convert_interleaved_to_sharded_on_host(const void* host, const Buffer& buffer, bool read = false) {
    const uint32_t num_pages = buffer.num_pages();
    const uint32_t page_size = buffer.page_size();

    const uint32_t size_in_bytes = num_pages * page_size;

    void* temp = malloc(size_in_bytes);
    memcpy(temp, host, size_in_bytes);

    const void* dst = host;
    std::set<uint32_t> pages_seen;
    for (uint32_t page_id = 0; page_id < num_pages; page_id++) {

        if (read) {
            auto host_page_id = page_id;
            auto dev_page_id = buffer.get_dev_to_host_mapped_page_id(host_page_id);
            TT_ASSERT(dev_page_id < num_pages and dev_page_id >= 0);
            memcpy((char*)dst + dev_page_id * page_size, (char*)temp + host_page_id * page_size, page_size);
        } else {
            auto dev_page_id = page_id;
            auto host_page_id = buffer.get_host_to_dev_mapped_page_id(dev_page_id);
            TT_ASSERT(host_page_id < num_pages and host_page_id >= 0);
            memcpy((char*)dst + host_page_id * page_size, (char*)temp + dev_page_id * page_size, page_size);
        }
    }
    free(temp);
}

void HWCommandQueue::enqueue_read_buffer(std::shared_ptr<Buffer> buffer, void* dst, bool blocking) {
    this->enqueue_read_buffer(*buffer, dst, blocking);
}

// Read buffer command is enqueued in the issue region and device writes requested buffer data into the completion region
void HWCommandQueue::enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking) {
    ZoneScopedN("HWCommandQueue_read_buffer");

    TT_ASSERT(not is_sharded(buffer.buffer_layout()), "Sharded buffer is not supported in FD 2.0");

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());

    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t pages_to_read = buffer.num_pages();
    uint32_t unpadded_dst_offset = 0;
    uint32_t src_page_index = 0;

    // this is a streaming command so we don't need to break down to multiple
    auto command = EnqueueReadInterleavedBufferCommand(
        this->id, this->device, buffer, dst, this->stall_before_read, this->manager, src_page_index, pages_to_read);

    this->issued_completion_q_reads.push(
        detail::ReadBufferDescriptor(buffer, padded_page_size, dst, unpadded_dst_offset, pages_to_read, src_page_index)
    );
    this->num_entries_in_completion_q++;

    this->enqueue_command(command, blocking);

    if (not blocking) { // should this be unconditional?
        std::shared_ptr<Event> event = std::make_shared<Event>();
        this->enqueue_record_event(event);
    }
}

void HWCommandQueue::enqueue_write_buffer(std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<const Buffer>> buffer, HostDataType src, bool blocking) {
    // Top level API to accept different variants for buffer and src
    // For shared pointer variants, object lifetime is guaranteed at least till the end of this function
    std::visit ([this, &buffer, &blocking](auto&& data) {
        using T = std::decay_t<decltype(data)>;
        std::visit ([this, &buffer, &blocking, &data](auto&& b) {
            using type_buf = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<T, const void*>) {
                if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                    this->enqueue_write_buffer(*b, data, blocking);
                } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                    this->enqueue_write_buffer(b.get(), data, blocking);
                }
            } else {
                if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                    this->enqueue_write_buffer(*b, data.get() -> data(), blocking);
                } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                    this->enqueue_write_buffer(b.get(), data.get() -> data(), blocking);
                }
            }
        }, buffer);
    }, src);
}

CoreType HWCommandQueue::get_dispatch_core_type() {
    return dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

void HWCommandQueue::enqueue_write_buffer(const Buffer& buffer, const void* src, bool blocking) {
    ZoneScopedN("HWCommandQueue_write_buffer");

    if (is_sharded(buffer.buffer_layout())) {
        TT_THROW("Sharded buffers are currently unsupported in FD2.0");
    }

    if (buffer.buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED or
        buffer.buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        convert_interleaved_to_sharded_on_host(src, buffer);
    }

    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t total_pages_to_write = buffer.num_pages();
    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->id);
    uint32_t dst_page_index = 0;
    while (total_pages_to_write > 0) {
        int32_t num_pages_available =
            (int32_t(command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id)) -
             int32_t(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd))) /
            int32_t(padded_page_size);

        uint32_t pages_to_write = std::min(total_pages_to_write, (uint32_t)num_pages_available);

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", this->id);

        auto command = EnqueueWriteInterleavedBufferCommand(
            this->id, this->device, buffer, src, this->manager, dst_page_index, pages_to_write);
        this->enqueue_command(command, false);

        total_pages_to_write -= pages_to_write;
        dst_page_index += pages_to_write;
    }

    // TODO: ADD enqueue_record_event

    if (blocking) {
        this->finish();
    }
}

void HWCommandQueue::enqueue_program(
    Program& program, std::optional<std::reference_wrapper<Trace>> trace, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_program");

    // Whether or not we should stall the producer from prefetching binary data. If the
    // data is cached, then we don't need to stall, otherwise we need to wait for the
    // data to land in DRAM first
    // bool stall;
    // if (not program.loaded_onto_device) {
    //     this->enqueue_write_buffer(*program.buffer, program.program_device_map.program_pages.data(), false);
    //     stall = true;
    //     program.loaded_onto_device = true;
    // } else {
    //     stall = false;
    // }

    // tt::log_debug(tt::LogDispatch, "EnqueueProgram for channel {}", this->id);
    // ProgramDeviceMap& program_device_map = program.program_device_map;
    // uint32_t host_data_num_pages = program_device_map.num_transfers_in_runtime_arg_pages.at(PageTransferType::MULTICAST).size() + program_device_map.num_transfers_in_cb_config_pages.at(PageTransferType::MULTICAST).size();
    // uint32_t host_data_and_device_command_size =
    //     DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + (host_data_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE);

    // if ((this->manager.get_issue_queue_write_ptr(this->id)) + host_data_and_device_command_size >=
    //     this->manager.get_issue_queue_limit(this->id)) {
    //     TT_FATAL(
    //         host_data_and_device_command_size <= this->manager.get_issue_queue_size(this->id) - CQ_START,
    //         "EnqueueProgram command size too large");
    //     // this->issue_wrap();
    // }

    // EnqueueProgramCommand command(
    //     this->id,
    //     this->device,
    //     program,
    //     this->manager,
    //     this->manager.get_next_event(this->id),
    //     stall,
    //     trace);
    // this->enqueue_command(command, blocking);
    // this->manager.next_completion_queue_push_back(align(EVENT_PADDED_SIZE, 32), this->id);
}

void HWCommandQueue::enqueue_record_event(std::shared_ptr<Event> event) {
    ZoneScopedN("HWCommandQueue_enqueue_record_event");

    // Populate event struct for caller. When async queues are enabled, this is in child thread, so consumers
    // of the event must wait for it to be ready (ie. populated) here. Set ready flag last. This couldn't be
    // in main thread otherwise event_id selection would get out of order due to main/worker thread timing.
    event->cq_id = this->id;
    event->event_id = this->manager.get_next_event(this->id);
    event->device = this->device;
    event->ready = true; // what does this mean???

    auto command = EnqueueRecordEventCommand(this->id, this->device, this->manager, event->event_id);
    this->enqueue_command(command, false);

    this->issued_completion_q_reads.push(detail::ReadEventDescriptor(event->event_id));
    this->num_entries_in_completion_q++;
}

void HWCommandQueue::enqueue_wait_for_event(std::shared_ptr<Event> event) {
    ZoneScopedN("HWCommandQueue_enqueue_wait_for_event");

    // uint32_t command_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    // if ((this->manager.get_issue_queue_write_ptr(this->id)) + command_size >= this->manager.get_issue_queue_limit(this->id)) {
    //     // this->issue_wrap();
    // }

    // auto command = EnqueueWaitForEventCommand(this->id, this->device, this->manager, this->manager.get_next_event(this->id), *event);
    // this->enqueue_command(command, false);
    // this->manager.next_completion_queue_push_back(align(EVENT_PADDED_SIZE, 32), this->id);
}


void HWCommandQueue::enqueue_trace() {
    ZoneScopedN("HWCommandQueue_enqueue_trace");
    TT_THROW("Not implemented");
}

void HWCommandQueue::copy_into_user_space(const detail::ReadBufferDescriptor &read_buffer_descriptor, uint32_t read_ptr, chip_id_t mmio_device_id, uint16_t channel) {
    const auto& [buffer_layout, page_size, padded_page_size, dev_page_to_host_page_mapping, dst, dst_offset, num_pages_read, cur_host_page_id] = read_buffer_descriptor;

    uint32_t read_data_ptr = read_ptr;
    if (dst_offset == 0) {
        // First piece of buffer data written to completion queue will be prepended with the dispatcher command
        //  because data is inlined with command coming into dispatcher

        // read_data_ptr += align(CQ_DISPATCH_CMD_SIZE, HUGEPAGE_ALIGNMENT);
    }

    std::cout << "Num pages read " << num_pages_read << " page size " << page_size << " padded page size " << padded_page_size << std::endl;

    uint32_t padded_num_bytes = (num_pages_read * padded_page_size) + sizeof(CQDispatchCmd);
    uint32_t contig_dst_offset = dst_offset;
    uint32_t remaining_bytes_to_read = padded_num_bytes;

    static std::vector<uint32_t> completion_q_data;

    while (remaining_bytes_to_read != 0) {
        this->manager.completion_queue_wait_front(this->id, this->exit_condition);

        if (this->exit_condition) {
            break;
        }

        uint32_t completion_q_write_ptr = this->manager.get_completion_queue_write_ptr(this->id);
        uint32_t bytes_xfered = std::min(padded_num_bytes, completion_q_write_ptr - read_data_ptr);
        bytes_xfered = std::min(bytes_xfered, remaining_bytes_to_read);
        uint32_t num_pages_xfered = (bytes_xfered + TRANSFER_PAGE_SIZE - 1) / TRANSFER_PAGE_SIZE;

        completion_q_data.resize(bytes_xfered / sizeof(uint32_t));

        std::cout << "remaining bytes to read: " << remaining_bytes_to_read
                  << " completion q write ptr: " << completion_q_write_ptr
                  << " bytes_xfered " << bytes_xfered
                  << " num_pages_xfered: " << num_pages_xfered
                  << " reading from " << read_data_ptr << std::endl;


        tt::Cluster::instance().read_sysmem(
            completion_q_data.data(), bytes_xfered, read_data_ptr, mmio_device_id, channel);


        bool first_iter = remaining_bytes_to_read == padded_num_bytes;

        this->manager.completion_queue_pop_front(num_pages_xfered, this->id);
        read_data_ptr += bytes_xfered;
        remaining_bytes_to_read -= bytes_xfered;

        if (buffer_layout == TensorMemoryLayout::INTERLEAVED or
            buffer_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            void* contiguous_dst = (void*)(uint64_t(dst) + contig_dst_offset);
            std::cout << "Writing to dst offset: " << contig_dst_offset << std::endl;
            uint32_t offset_in_completion_q_data = (contig_dst_offset == 0) ? (sizeof(CQDispatchCmd) / sizeof(uint32_t)) : 0;

            if (not first_iter) {
                offset_in_completion_q_data = 0;
            }

            std::cout << "Offset in completion q data " << offset_in_completion_q_data << std::endl;
            if ((page_size % 32) == 0) {
                uint32_t data_bytes_xfered = (contig_dst_offset == 0) ? (bytes_xfered - sizeof(CQDispatchCmd)) : bytes_xfered;
                std::cout << "Data bytes xfered: " << data_bytes_xfered << std::endl;
                memcpy(contiguous_dst, completion_q_data.data() + offset_in_completion_q_data, data_bytes_xfered);
                contig_dst_offset += data_bytes_xfered;
            } else {
                uint32_t non_contig_dst_offset = 0;
                std::cout << "Comq data size " << completion_q_data.size() << std::endl;
                for (uint32_t offset = offset_in_completion_q_data; offset < completion_q_data.size(); offset += (padded_page_size / sizeof(uint32_t))) {
                    // std::cout << " non_contig_dst_offset " << non_contig_dst_offset << std::endl;
                    // if (not first_iter) {
                    //     std::cout << " offset in hugepage vec: " << offset << std::endl;
                    // }
                    memcpy(
                        (char*)(uint64_t(contiguous_dst)),
                        completion_q_data.data() + offset,
                        page_size
                    );
                    non_contig_dst_offset += page_size;
                }
                std::cout << "non_contig_dst_offset " << non_contig_dst_offset << std::endl;
                // contig_dst_offset += non_contig_dst_offset;
            }

        } else if (
            buffer_layout == TensorMemoryLayout::WIDTH_SHARDED or
            buffer_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_THROW("Reading width sharded or block sharded buffers is unsupported in FD2.0");
            uint32_t host_page_id = cur_host_page_id;
            uint32_t read_src = read_data_ptr;
            // for (uint32_t offset = 0; offset < data_bytes_xfered; offset += padded_page_size) {
            //     uint32_t device_page_id = dev_page_to_host_page_mapping[host_page_id];
            //     void* page_dst = (void*)(uint64_t(dst) + device_page_id * page_size);
            //     tt::Cluster::instance().read_sysmem(
            //         page_dst, page_size, read_src + offset, mmio_device_id, channel);
            //     host_page_id++;
            // }
        }
    }
    std::cout << "Done copying into user buffer" << std::endl;
}

void HWCommandQueue::read_completion_queue() {
    tracy::SetThreadName("COMPLETION QUEUE");
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    while (true) {
        if (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            uint32_t num_events_to_read = this->num_entries_in_completion_q - this->num_completed_completion_q_reads;
            for (uint32_t i = 0; i < num_events_to_read; i++) {

                std::variant<detail::ReadBufferDescriptor, detail::ReadEventDescriptor> read_descriptor = *(this->issued_completion_q_reads.pop());

                this->manager.completion_queue_wait_front(this->id, this->exit_condition); // CQ DISPATCHER IS NOT HANDSHAKING WITH HOST RN

                if (this->exit_condition) {  // Early exit
                    return;
                }

                uint32_t read_ptr = this->manager.get_completion_queue_read_ptr(this->id);

                std::visit(
                    [&](auto&& read_descriptor)
                    {
                        using T = std::decay_t<decltype(read_descriptor)>;
                        if constexpr (std::is_same_v<T, detail::ReadBufferDescriptor>) {
                            std::cout << "got read buffer descriptor" << std::endl;
                            this->copy_into_user_space(read_descriptor, read_ptr, mmio_device_id, channel);
                        }
                        else if constexpr (std::is_same_v<T, detail::ReadEventDescriptor>) {
                            std::cout << "got read event descriptor" << std::endl;
                            static std::vector<uint32_t> dispatch_cmd_and_event((sizeof(CQDispatchCmd) + EVENT_PADDED_SIZE) / sizeof(uint32_t));
                            tt::Cluster::instance().read_sysmem(
                                dispatch_cmd_and_event.data(), sizeof(CQDispatchCmd) + EVENT_PADDED_SIZE, read_ptr, mmio_device_id, channel);
                            uint32_t event_completed = dispatch_cmd_and_event.at(sizeof(CQDispatchCmd) / sizeof(uint32_t));
                            TT_ASSERT(event_completed == read_descriptor.event_id, "Event Order Issue: expected to read back completion signal for event {} but got {}!", read_descriptor.event_id, event_completed);
                            this->manager.completion_queue_pop_front(1, this->id);
                            this->manager.set_last_completed_event(this->id, event_completed);
                        }
                    },
                    read_descriptor
                );
            }
            this->num_completed_completion_q_reads += num_events_to_read;
        } else if (this->exit_condition) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void HWCommandQueue::finish() {
    ZoneScopedN("HWCommandQueue_finish");
    tt::log_debug(tt::LogDispatch, "Finish for command queue {}", this->id);
    std::shared_ptr<Event> event = std::make_shared<Event>();
    this->enqueue_record_event(event);

    if (tt::llrt::OptionsG.get_test_mode_enabled()) {
        while (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            if (DPrintServerHangDetected()) {
                // DPrint Server hang. Mark state and early exit. Assert in main thread.
                this->exit_condition = true;
                this->dprint_server_hang = true;
                return;
            } else if (tt::watcher_server_killed_due_to_error()) {
                // Illegal NOC txn killed watcher. Mark state and early exit. Assert in main thread.
                this->exit_condition = true;
                this->illegal_noc_txn_hang = true;
                return;
            }
        }
    } else {
        std::cout << " in completion q " << this->num_entries_in_completion_q
                  << " completed reads " << this->num_completed_completion_q_reads << std::endl;
        while (this->num_entries_in_completion_q > this->num_completed_completion_q_reads);
    }
}

volatile bool HWCommandQueue::is_dprint_server_hung() {
    return dprint_server_hang;
}

volatile bool HWCommandQueue::is_noc_hung() {
    return illegal_noc_txn_hang;
}

void EnqueueAddBufferToProgram(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program, bool blocking) {
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ADD_BUFFER_TO_PROGRAM,
        .blocking = blocking,
        .buffer = buffer,
        .program = program,
    });
}

void EnqueueAddBufferToProgramImpl(const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program) {
    std::visit([program] (auto&& b) {
        using buffer_type = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<buffer_type, std::shared_ptr<Buffer>>) {
            std::visit([&b] (auto&& p) {
                using program_type = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<program_type, std::reference_wrapper<Program>>) {
                    p.get().add_buffer(b);
                }
                else {
                    p->add_buffer(b);
                }
            }, program);
        }
    }, buffer);
}

void EnqueueUpdateRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::vector<uint32_t> &update_idx, std::shared_ptr<RuntimeArgs> runtime_args_ptr, bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata {
            .core_coord = core_coord,
            .runtime_args_ptr = runtime_args_ptr,
            .kernel = kernel,
            .update_idx = update_idx,
    };
    cq.run_command( CommandInterface {
        .type = EnqueueCommandType::UPDATE_RUNTIME_ARGS,
        .blocking = blocking,
        .runtime_args_md = runtime_args_md,
    });
}

void EnqueueUpdateRuntimeArgsImpl (const RuntimeArgsMetadata& runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto& arg : *(runtime_args_md.runtime_args_ptr)) {
        std::visit([&resolved_runtime_args] (auto&& a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, Buffer*>) {
                resolved_runtime_args.push_back(a -> address());
            } else {
                resolved_runtime_args.push_back(a);
            }
        }, arg);
    }
    auto& kernel_runtime_args = runtime_args_md.kernel->runtime_args(runtime_args_md.core_coord);
    for (const auto& idx : runtime_args_md.update_idx) {
        kernel_runtime_args[idx] = resolved_runtime_args[idx];
    }
}

void EnqueueSetRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::shared_ptr<RuntimeArgs> runtime_args_ptr, bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata {
            .core_coord = core_coord,
            .runtime_args_ptr = runtime_args_ptr,
            .kernel = kernel,
    };
    cq.run_command( CommandInterface {
        .type = EnqueueCommandType::SET_RUNTIME_ARGS,
        .blocking = blocking,
        .runtime_args_md = runtime_args_md,
    });
}

void EnqueueSetRuntimeArgsImpl(const RuntimeArgsMetadata& runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto& arg : *(runtime_args_md.runtime_args_ptr)) {
        std::visit([&resolved_runtime_args] (auto&& a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, Buffer*>) {
                resolved_runtime_args.push_back(a -> address());
            } else {
                resolved_runtime_args.push_back(a);
            }
        }, arg);
    }
    runtime_args_md.kernel -> set_runtime_args(runtime_args_md.core_coord, resolved_runtime_args);
}

void EnqueueGetBufferAddr(CommandQueue& cq, uint32_t* dst_buf_addr, const Buffer* buffer, bool blocking) {
    cq.run_command( CommandInterface {
        .type = EnqueueCommandType::GET_BUF_ADDR,
        .blocking = blocking,
        .shadow_buffer = buffer,
        .dst = dst_buf_addr
    });
}

void EnqueueGetBufferAddrImpl(void* dst_buf_addr, const Buffer* buffer) {
    *(static_cast<uint32_t*>(dst_buf_addr)) = buffer -> address();
}
void EnqueueAllocateBuffer(CommandQueue& cq, Buffer* buffer, bool bottom_up, bool blocking) {
    auto alloc_md = AllocBufferMetadata {
        .buffer = buffer,
        .allocator = *(buffer->device()->allocator_),
        .bottom_up = bottom_up,
    };
    cq.run_command(CommandInterface {
        .type = EnqueueCommandType::ALLOCATE_BUFFER,
        .blocking = blocking,
        .alloc_md = alloc_md,
    });
}

void EnqueueAllocateBufferImpl(AllocBufferMetadata alloc_md) {
    Buffer* buffer = alloc_md.buffer;
    uint32_t allocated_addr;
    if(is_sharded(buffer->buffer_layout())) {
        allocated_addr = allocator::allocate_buffer(*(buffer->device()->allocator_), buffer->size(), buffer->page_size(), buffer->buffer_type(), alloc_md.bottom_up, buffer->num_cores());
    }
    else {
        allocated_addr = allocator::allocate_buffer(*(buffer->device()->allocator_), buffer->size(), buffer->page_size(), buffer->buffer_type(), alloc_md.bottom_up, std::nullopt);
    }
    buffer->set_address(static_cast<uint64_t>(allocated_addr));
}

void EnqueueDeallocateBuffer(CommandQueue& cq, Allocator& allocator, uint32_t device_address, BufferType buffer_type, bool blocking) {
    // Need to explictly pass in relevant buffer attributes here, since the Buffer* ptr can be deallocated a this point
    auto alloc_md = AllocBufferMetadata {
        .allocator = allocator,
        .buffer_type = buffer_type,
        .device_address = device_address,
    };
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::DEALLOCATE_BUFFER,
        .blocking = blocking,
        .alloc_md = alloc_md,
    });
}

void EnqueueDeallocateBufferImpl(AllocBufferMetadata alloc_md) {
    allocator::deallocate_buffer(alloc_md.allocator, alloc_md.device_address, alloc_md.buffer_type);
}

void EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, vector<uint32_t>& dst, bool blocking){
    // TODO(agrebenisan): Move to deprecated
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    Buffer & b = std::holds_alternative<std::shared_ptr<Buffer>>(buffer) ? *(std::get< std::shared_ptr<Buffer> > ( buffer )) :
                                                                            std::get<std::reference_wrapper<Buffer>>(buffer).get();
    // Only resizing here to keep with the original implementation. Notice how in the void*
    // version of this API, I assume the user mallocs themselves
    std::visit ( [&dst](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>>) {
            dst.resize(b.get().page_size() * b.get().num_pages() / sizeof(uint32_t));
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Buffer>>) {
            dst.resize(b->page_size() * b->num_pages() / sizeof(uint32_t));
        }
    }, buffer);

    // TODO(agrebenisan): Move to deprecated
    EnqueueReadBuffer(cq, buffer, dst.data(), blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, vector<uint32_t>& src, bool blocking){
    // TODO(agrebenisan): Move to deprecated
    EnqueueWriteBuffer(cq, buffer, src.data(), blocking);
}

void EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void* dst, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_READ_BUFFER,
        .blocking = blocking,
        .buffer = buffer,
        .dst = dst
    });
}

void EnqueueReadBufferImpl(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void* dst, bool blocking) {
    std::visit ( [&cq, dst, blocking](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer> > ) {
            cq.hw_command_queue().enqueue_read_buffer(b, dst, blocking);
        }
    }, buffer);
}

void EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer,
                                          HostDataType src, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WRITE_BUFFER,
        .blocking = blocking,
        .buffer = buffer,
        .src = src
    });
}

void EnqueueWriteBufferImpl(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer,
                                          HostDataType src, bool blocking) {
    std::visit ( [&cq, src, blocking](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer>> ) {
            cq.hw_command_queue().enqueue_write_buffer(b, src, blocking);
        }
    }, buffer);
}

void EnqueueProgram(CommandQueue& cq, std::variant < std::reference_wrapper<Program>, std::shared_ptr<Program> > program, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_THROW("EnqueueProgram currently unsupported in FD2.0");
    if (cq.get_mode() != CommandQueue::CommandQueueMode::TRACE) {
        TT_FATAL(cq.id() == 0, "EnqueueProgram only supported on first command queue on device for time being.");
    }
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_PROGRAM,
        .blocking = blocking,
        .program = program
    });
}

void EnqueueProgramImpl(CommandQueue& cq, std::variant < std::reference_wrapper<Program>, std::shared_ptr<Program> > program, bool blocking) {
    ZoneScoped;
    std::visit ( [&cq, blocking](auto&& program) {
        ZoneScoped;
        using T = std::decay_t<decltype(program)>;
        Device * device = cq.device();
        std::optional<std::reference_wrapper<Trace>> trace;
        if (cq.trace()) {
            trace = std::optional<std::reference_wrapper<Trace>>(*cq.trace());
        }
        if constexpr (std::is_same_v<T, std::reference_wrapper<Program>>) {
            detail::CompileProgram(device, program);
            program.get().allocate_circular_buffers();
            detail::ValidateCircularBufferRegion(program, device);
            cq.hw_command_queue().enqueue_program(program, trace, blocking);
            // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem leaks on device.
            program.get().release_buffers();
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Program>>) {
            detail::CompileProgram(device, *program);
            program->allocate_circular_buffers();
            detail::ValidateCircularBufferRegion(*program, device);
            cq.hw_command_queue().enqueue_program(*program, trace, blocking);
            // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem leaks on device.
            program->release_buffers();
        }
    }, program);
}

void EnqueueRecordEvent(CommandQueue& cq, std::shared_ptr<Event> event) {
    TT_THROW("EnqueueRecordEvent currently unsupported in FD2.0");
    TT_ASSERT(event->device == nullptr, "EnqueueRecordEvent expected to be given an uninitialized event");
    TT_ASSERT(event->event_id == -1, "EnqueueRecordEvent expected to be given an uninitialized event");
    TT_ASSERT(event->cq_id == -1, "EnqueueRecordEvent expected to be given an uninitialized event");

    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_RECORD_EVENT,
        .blocking = false,
        .event = event,
    });
}

void EnqueueRecordEventImpl(CommandQueue& cq, std::shared_ptr<Event> event) {
    cq.hw_command_queue().enqueue_record_event(event);
}


void EnqueueWaitForEvent(CommandQueue& cq, std::shared_ptr<Event> event) {

    detail::DispatchStateCheck(true);
    TT_THROW("EnqueueWaitForEvent currently unsupported in FD2.0");
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT,
        .blocking = false,
        .event = event,
    });
}

void EnqueueWaitForEventImpl(CommandQueue& cq, std::shared_ptr<Event> event) {
    event->wait_until_ready(); // Block until event populated. Worker thread.
    log_trace(tt::LogMetal, "EnqueueWaitForEvent() issued on Event(device_id: {} cq_id: {} event_id: {}) from device_id: {} cq_id: {}",
        event->device->id(), event->cq_id, event->event_id, cq.device()->id(), cq.id());
    cq.hw_command_queue().enqueue_wait_for_event(event);
}


void EventSynchronize(std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    TT_THROW("EventSynchronize currently unsupported in FD2.0");
    event->wait_until_ready(); // Block until event populated. Parent thread.
    log_trace(tt::LogMetal, "Issuing host sync on Event(device_id: {} cq_id: {} event_id: {})", event->device->id(), event->cq_id, event->event_id);

    while (event->device->sysmem_manager().get_last_completed_event(event->cq_id) < event->event_id) {
        if (tt::llrt::OptionsG.get_test_mode_enabled() && tt::watcher_server_killed_due_to_error()) {
            TT_ASSERT(false, "Command Queue could not complete EventSynchronize. See {} for details.", tt::watcher_get_log_file_name());
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
}

void Finish(CommandQueue& cq) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::FINISH,
        .blocking = true
    });
    TT_ASSERT(!(cq.device() -> hw_command_queue(cq.id()).is_dprint_server_hung()),
              "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
    TT_ASSERT(!(cq.device() -> hw_command_queue(cq.id()).is_noc_hung()),
              "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.",
               tt::watcher_get_log_file_name());
}

void FinishImpl(CommandQueue& cq) {
    cq.hw_command_queue().finish();
}

CommandQueue& BeginTrace(Trace& trace) {
    TT_THROW("BeginTrace currently unsupported in FD2.0");
    TT_ASSERT(not trace.trace_complete, "Already completed this trace");
    TT_ASSERT(trace.queue().empty(), "Cannot begin trace on one that already captured commands");
    return trace.queue();
}

void EndTrace(Trace& trace) {
    TT_THROW("EndTrace currently unsupported in FD2.0");
    TT_ASSERT(not trace.trace_complete, "Already completed this trace");
    trace.trace_complete = true;
    trace.validate();
}

uint32_t InstantiateTrace(Trace& trace, CommandQueue& cq) {
    detail::DispatchStateCheck(true);
    TT_THROW("InstantiateTrace currently unsupported in FD2.0");
    TT_ASSERT(cq.trace() == nullptr, "Multiple traces on a CQ is not supported yet");
    uint32_t trace_id = trace.instantiate(cq);
    return trace_id;
}

void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_THROW("EnqueueTrace currently unsupported in FD2.0");
    TT_ASSERT(cq.trace(), "A trace has not been instantiated on this command queue yet!");
    if (cq.trace()->trace_instances.count(trace_id) == 0) {
        TT_THROW("Trace instance " + std::to_string(trace_id) + " does not exist");
    }
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_TRACE,
        .blocking = blocking
    });
}

void EnqueueTraceImpl(CommandQueue& cq) {
    // STUB: Run the trace in eager mode for now
    auto& tq = cq.trace()->queue();
    for (const auto& cmd : tq.worker_queue) {
        cq.run_command_impl(cmd);
    }
}

CommandQueue::CommandQueue(Device* device, uint32_t id, CommandQueueMode mode) :
    device_ptr(device),
    trace_ptr(nullptr),
    cq_id(id),
    mode(mode),
    worker_state(CommandQueueState::IDLE) {
    if (this->async_mode()) {
        num_async_cqs++;
        // The main program thread launches the Command Queue
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
    }
}

CommandQueue::CommandQueue(Trace* trace) :
    device_ptr(nullptr),
    parent_thread_id(0),
    trace_ptr(trace),
    cq_id(-1),
    mode(CommandQueueMode::TRACE),
    worker_state(CommandQueueState::IDLE) {
    TT_ASSERT(this->trace_ptr, "A valid trace must be provided for a trace mode queue");
}

CommandQueue::~CommandQueue() {
    if (this->async_mode()) {
        this->stop_worker();
    }
    if (this->trace_mode()) {
        TT_ASSERT(this->trace()->trace_complete, "Trace capture must be complete before desctruction");
    } else {
        TT_ASSERT(this->worker_queue.empty(), "CQ{} worker queue must be empty on destruction", this->cq_id);
    }
}

HWCommandQueue& CommandQueue::hw_command_queue() {
    return this->device()->hw_command_queue(this->cq_id);
}

void CommandQueue::wait_until_empty() {
    log_trace(LogDispatch, "CQ{} WFI start", this->cq_id);
    if (this->async_mode()) {
        // Insert a flush token to push all prior commands to completion
        // Necessary to avoid implementing a peek and pop on the lock-free queue
        this->worker_queue.push(CommandInterface{.type = EnqueueCommandType::FLUSH});
    }
    while (true) {
        if (this->worker_queue.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    log_trace(LogDispatch, "CQ{} WFI complete", this->cq_id);
}

void CommandQueue::set_mode(const CommandQueueMode& mode) {
    TT_ASSERT(not this->trace_mode(), "Cannot change mode of a trace command queue, copy to a non-trace command queue instead!");
    if (this->mode == mode) {
        // Do nothing if requested mode matches current CQ mode.
        return;
    }
    this->mode = mode;
    if (this->async_mode()) {
        num_async_cqs++;
        num_passthrough_cqs--;
        // Record parent thread-id and start worker.
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
        num_async_cqs--;
        // Wait for all cmds sent in async mode to complete and stop worker.
        this->wait_until_empty();
        this->stop_worker();
    }
}

void CommandQueue::start_worker() {
    if (this->worker_state == CommandQueueState::RUNNING) {
        return;  // worker already running, exit
    }
    this->worker_state = CommandQueueState::RUNNING;
    this->worker_thread = std::make_unique<std::thread>(std::thread(&CommandQueue::run_worker, this));
    tt::log_debug(tt::LogDispatch, "CQ{} started worker thread", this->cq_id);
}

void CommandQueue::stop_worker() {
    if (this->worker_state == CommandQueueState::IDLE) {
        return;  // worker already stopped, exit
    }
    this->worker_state = CommandQueueState::TERMINATE;
    this->worker_thread->join();
    this->worker_state = CommandQueueState::IDLE;
    tt::log_debug(tt::LogDispatch, "CQ{} stopped worker thread", this->cq_id);
}

void CommandQueue::run_worker() {
    // forever loop checking for commands in the worker queue
    // Track the worker thread id, for cases where a command calls a sub command.
    // This is to detect cases where commands may be nested.
    worker_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    while (true) {
        if (this->worker_queue.empty()) {
            if (this->worker_state == CommandQueueState::TERMINATE) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        } else {
            std::shared_ptr<CommandInterface> command(this->worker_queue.pop());
            run_command_impl(*command);
        }
    }
}

void CommandQueue::run_command(const CommandInterface& command) {
    log_trace(LogDispatch, "CQ{} received {} in {} mode", this->cq_id, command.type, this->mode);
    if (not this->passthrough_mode()) {
        if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == parent_thread_id or this->trace_mode()) {
            // Push to worker queue for trace or async mode. In trace mode, store the execution in the queue.
            // In async mode when parent pushes cmd, feed worker through queue.
            this->worker_queue.push(command);
            if (command.blocking.has_value() and *command.blocking == true) {
                TT_ASSERT(not this->trace_mode(), "Blocking commands cannot be traced!");
                this->wait_until_empty();
            }
        }
        else {
            // Handle case where worker pushes command to itself (passthrough)
            TT_ASSERT(std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_thread_id, "Only main thread or worker thread can run commands through the SW command queue");
            run_command_impl(command);
        }
    } else {
        this->run_command_impl(command);
    }
}

void CommandQueue::run_command_impl(const CommandInterface& command) {
    log_trace(LogDispatch, "CQ{} running {}", this->cq_id, command.type);
    switch (command.type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueReadBufferImpl(*this, command.buffer.value(), command.dst.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER:
            TT_ASSERT(command.src.has_value(), "Must provide a src!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueWriteBufferImpl(*this, command.buffer.value(), command.src.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ALLOCATE_BUFFER:
            TT_ASSERT(command.alloc_md.has_value(), "Must provide buffer allocation metdata!");
            EnqueueAllocateBufferImpl(command.alloc_md.value());
            break;
        case EnqueueCommandType::DEALLOCATE_BUFFER:
            TT_ASSERT(command.alloc_md.has_value(), "Must provide buffer allocation metdata!");
            EnqueueDeallocateBufferImpl(command.alloc_md.value());
            break;
        case EnqueueCommandType::GET_BUF_ADDR:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst address!");
            TT_ASSERT(command.shadow_buffer.has_value(), "Must provide a shadow buffer!");
            EnqueueGetBufferAddrImpl(command.dst.value(), command.shadow_buffer.value());
            break;
        case EnqueueCommandType::SET_RUNTIME_ARGS:
            TT_ASSERT(command.runtime_args_md.has_value(), "Must provide RuntimeArgs Metdata!");
            EnqueueSetRuntimeArgsImpl(command.runtime_args_md.value());
            break;
        case EnqueueCommandType::UPDATE_RUNTIME_ARGS:
            TT_ASSERT(command.runtime_args_md.has_value(), "Must provide RuntimeArgs Metdata!");
            EnqueueUpdateRuntimeArgsImpl(command.runtime_args_md.value());
            break;
        case EnqueueCommandType::ADD_BUFFER_TO_PROGRAM:
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.program.has_value(), "Must provide a program!");
            EnqueueAddBufferToProgramImpl(command.buffer.value(), command.program.value());
            break;
        case EnqueueCommandType::ENQUEUE_PROGRAM:
            TT_ASSERT(command.program.has_value(), "Must provide a program!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueProgramImpl(*this, command.program.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_TRACE:
            EnqueueTraceImpl(*this);
            break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueRecordEventImpl(*this, command.event.value());
            break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueWaitForEventImpl(*this, command.event.value());
            break;
        case EnqueueCommandType::FINISH:
            FinishImpl(*this);
            break;
        case EnqueueCommandType::FLUSH:
            // Used by CQ to push prior commands
            break;
        default:
            TT_THROW("Invalid command type");
    }
    log_trace(LogDispatch, "CQ{} running {} complete", this->cq_id, command.type);
}

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, EnqueueCommandType const& type) {
    switch (type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER: os << "ENQUEUE_READ_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: os << "ENQUEUE_WRITE_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_PROGRAM: os << "ENQUEUE_PROGRAM"; break;
        case EnqueueCommandType::ENQUEUE_TRACE: os << "ENQUEUE_TRACE"; break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT: os << "ENQUEUE_RECORD_EVENT"; break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT: os << "ENQUEUE_WAIT_FOR_EVENT"; break;
        case EnqueueCommandType::FINISH: os << "FINISH"; break;
        case EnqueueCommandType::FLUSH: os << "FLUSH"; break;
        default: TT_THROW("Invalid command type!");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, CommandQueue::CommandQueueMode const& type) {
    switch (type) {
        case CommandQueue::CommandQueueMode::PASSTHROUGH: os << "PASSTHROUGH"; break;
        case CommandQueue::CommandQueueMode::ASYNC: os << "ASYNC"; break;
        case CommandQueue::CommandQueueMode::TRACE: os << "TRACE"; break;
        default: TT_THROW("Invalid CommandQueueMode type!");
    }
    return os;
}
