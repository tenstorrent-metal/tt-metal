// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>
#include <fstream>

#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/impl/dispatch/lock_free_queue.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "common/env_lib.hpp"
#include "noc/noc_parameters.h"
#include "tt_metal/host_api.hpp"

namespace tt::tt_metal {

using std::pair;
using std::set;
using std::shared_ptr;
using std::weak_ptr;
using std::tuple;
using std::unique_ptr;

// Only contains the types of commands which are enqueued onto the device
enum class EnqueueCommandType {
    ENQUEUE_READ_BUFFER,
    ENQUEUE_WRITE_BUFFER,
    ALLOCATE_BUFFER,
    DEALLOCATE_BUFFER,
    GET_BUF_ADDR,
    ADD_BUFFER_TO_PROGRAM,
    SET_RUNTIME_ARGS,
    UPDATE_RUNTIME_ARGS,
    ENQUEUE_PROGRAM,
    ENQUEUE_TRACE,
    ENQUEUE_RECORD_EVENT,
    ENQUEUE_WAIT_FOR_EVENT,
    ENQUEUE_WRAP,
    FINISH,
    FLUSH,
    INVALID
};

string EnqueueCommandTypeToString(EnqueueCommandType ctype);

// TEMPORARY! TODO(agrebenisan): need to use proper macro based on loading noc
#define NOC_X(x) x
#define NOC_Y(y) y

uint32_t get_noc_unicast_encoding(CoreCoord coord);

class Trace;
class CommandQueue;
class CommandInterface;

using WorkerQueue = LockFreeQueue<CommandInterface>;

class Command {
    EnqueueCommandType type_ = EnqueueCommandType::INVALID;

   public:
    Command() {}
    virtual void process() {};
    virtual EnqueueCommandType type() = 0;
    virtual const void assemble_device_commands(uint32_t buffer_size) = 0;
};

class EnqueueReadBufferCommand : public Command {
   private:
    SystemMemoryManager& manager;
    void* dst;
    uint32_t pages_to_read;
    uint32_t command_queue_id;
    bool stall;
    static std::vector<uint32_t> commands;

   protected:
    Device* device;
    uint32_t src_page_index;
   public:
    Buffer& buffer;
    EnqueueReadBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        Buffer& buffer,
        void* dst,
        bool stall,
        SystemMemoryManager& manager,
        uint32_t src_page_index = 0,
        std::optional<uint32_t> pages_to_read = std::nullopt);

    const void assemble_device_commands(uint32_t dst);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_READ_BUFFER; };

    constexpr bool has_side_effects() { return false; }
};

class EnqueueReadInterleavedBufferCommand : public EnqueueReadBufferCommand {
   public:
    EnqueueReadInterleavedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        Buffer& buffer,
        void* dst,
        bool stall,
        SystemMemoryManager& manager,
        uint32_t src_page_index = 0,
        std::optional<uint32_t> pages_to_read = std::nullopt)
            :EnqueueReadBufferCommand(command_queue_id,
                                device,
                                buffer,
                                dst,
                                stall,
                                manager,
                                src_page_index,
                                pages_to_read) {}
};


class EnqueueReadShardedBufferCommand : public EnqueueReadBufferCommand {
   public:
    EnqueueReadShardedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        Buffer& buffer,
        void* dst,
        bool stall,
        SystemMemoryManager& manager,
        uint32_t src_page_index = 0,
        std::optional<uint32_t> pages_to_read = std::nullopt)
            :EnqueueReadBufferCommand(command_queue_id,
                                device,
                                buffer,
                                dst,
                                stall,
                                manager,
                                src_page_index,
                                pages_to_read) {}
};

class EnqueueWriteShardedBufferCommand;
class EnqueueWriteInterleavedBufferCommand;
class EnqueueWriteBufferCommand : public Command {
   private:

    SystemMemoryManager& manager;
    const void* src;
    uint32_t pages_to_write;
    uint32_t command_queue_id;

   protected:
    Device* device;
    const Buffer& buffer;
    uint32_t dst_page_index;
   public:
    EnqueueWriteBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        const Buffer& buffer,
        const void* src,
        SystemMemoryManager& manager,
        uint32_t dst_page_index = 0,
        std::optional<uint32_t> pages_to_write = std::nullopt);

    const void assemble_device_commands(uint32_t src_address);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_WRITE_BUFFER; }

    constexpr bool has_side_effects() { return true; }
};

class EnqueueWriteInterleavedBufferCommand : public EnqueueWriteBufferCommand {
   public:
    EnqueueWriteInterleavedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        const Buffer& buffer,
        const void* src,
        SystemMemoryManager& manager,
        uint32_t dst_page_index = 0,
        std::optional<uint32_t> pages_to_write = std::nullopt)
        : EnqueueWriteBufferCommand(
            command_queue_id,
            device,
            buffer,
            src,
            manager,
            dst_page_index,
            pages_to_write){;}


};


class EnqueueWriteShardedBufferCommand : public EnqueueWriteBufferCommand {
   public:
    EnqueueWriteShardedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        const Buffer& buffer,
        const void* src,
        SystemMemoryManager& manager,
        uint32_t dst_page_index = 0,
        std::optional<uint32_t> pages_to_write = std::nullopt)
        : EnqueueWriteBufferCommand(
            command_queue_id,
            device,
            buffer,
            src,
            manager,
            dst_page_index,
            pages_to_write){;}


};

class EnqueueProgramCommand : public Command {
   private:
    uint32_t command_queue_id;
    Device* device;
    const Program& program;
    SystemMemoryManager& manager;
    bool stall;
    std::optional<std::reference_wrapper<Trace>> trace = {};

   public:
    EnqueueProgramCommand(uint32_t command_queue_id, Device* device, const Program& program, SystemMemoryManager& manager, bool stall, std::optional<std::reference_wrapper<Trace>> trace);

    const void assemble_device_commands(uint32_t src_address);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_PROGRAM; }

    constexpr bool has_side_effects() { return true; }
};

class EnqueueRecordEventCommand : public Command {
   private:
    uint32_t command_queue_id;
    Device* device;
    SystemMemoryManager& manager;
    uint32_t event_id;
    static std::vector<uint32_t> commands;

   public:
    EnqueueRecordEventCommand(uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, uint32_t event_id);

    const void assemble_device_commands(uint32_t);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_RECORD_EVENT; }

    constexpr bool has_side_effects() { return false; }
};

class EnqueueWaitForEventCommand : public Command {
   private:
    uint32_t command_queue_id;
    Device* device;
    SystemMemoryManager& manager;
    const Event& sync_event;

   public:
    EnqueueWaitForEventCommand(uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, const Event& sync_event);

    const void assemble_device_commands(uint32_t);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT; }

    constexpr bool has_side_effects() { return false; }
};

namespace detail {
inline bool LAZY_COMMAND_QUEUE_MODE = false;

/*
 Used so the host knows how to properly copy data into user space from the completion queue (in hugepages)
*/
struct ReadBufferDescriptor {
    TensorMemoryLayout buffer_layout;
    uint32_t page_size;
    uint32_t padded_page_size;
    vector<uint32_t> dev_page_to_host_page_mapping;
    void* dst;
    uint32_t dst_offset;
    uint32_t num_pages_read;
    uint32_t cur_host_page_id;

    ReadBufferDescriptor(Buffer& buffer, uint32_t padded_page_size, void* dst, uint32_t dst_offset, uint32_t num_pages_read, uint32_t cur_host_page_id) {
        this->buffer_layout = buffer.buffer_layout();
        this->page_size = buffer.page_size();
        this->padded_page_size = padded_page_size;
        if (this->buffer_layout == TensorMemoryLayout::WIDTH_SHARDED or this->buffer_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            this->dev_page_to_host_page_mapping = buffer.get_dev_page_to_host_page_mapping();
        }
        this->dst = dst;
        this->dst_offset = dst_offset;
        this->num_pages_read = num_pages_read;
        this->cur_host_page_id = cur_host_page_id;
    }
};

/*
 Used so host knows data in completion queue is just an event ID
*/
struct ReadEventDescriptor {
    uint32_t event_id;

    ReadEventDescriptor(uint32_t event) {
        this->event_id = event;
    }
};

typedef LockFreeQueue<std::variant<ReadBufferDescriptor, ReadEventDescriptor>> CompletionReaderQueue;
}

struct AllocBufferMetadata {
    Buffer* buffer;
    std::reference_wrapper<Allocator> allocator;
    BufferType buffer_type;
    uint32_t device_address;
    bool bottom_up;
};

struct RuntimeArgsMetadata {
    CoreCoord core_coord;
    std::shared_ptr<RuntimeArgs> runtime_args_ptr;
    std::shared_ptr<Kernel> kernel;
    std::vector<uint32_t> update_idx;
};

class HWCommandQueue {
   public:
    HWCommandQueue(Device* device, uint32_t id);

    ~HWCommandQueue();

    CoreCoord completion_queue_writer_core;
    volatile bool is_dprint_server_hung();
    volatile bool is_noc_hung();

   private:
    uint32_t id;
    uint32_t size_B;
    std::optional<uint32_t> last_event_id;
    std::thread completion_queue_thread;
    SystemMemoryManager& manager;
    bool stall_before_read;

    volatile bool exit_condition;
    volatile bool dprint_server_hang = false;
    volatile bool illegal_noc_txn_hang = false;
    volatile uint32_t num_entries_in_completion_q;  // issue queue writer thread increments this when an issued command is expected back in the completion queue
    volatile uint32_t num_completed_completion_q_reads; // completion queue reader thread increments this after reading an entry out of the completion queue
    detail::CompletionReaderQueue issued_completion_q_reads;

    Device* device;


    void copy_into_user_space(const detail::ReadBufferDescriptor &read_buffer_descriptor, uint32_t read_ptr, chip_id_t mmio_device_id, uint16_t channel);
    void read_completion_queue();

    template <typename T>
    void enqueue_command(T& command, bool blocking);

    void enqueue_read_buffer(std::shared_ptr<Buffer> buffer, void* dst, bool blocking);
    void enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking);
    void enqueue_write_buffer(std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<const Buffer>> buffer, HostDataType src, bool blocking);
    void enqueue_write_buffer(const Buffer& buffer, const void* src, bool blocking);
    void enqueue_program(Program& program, std::optional<std::reference_wrapper<Trace>> trace, bool blocking);
    void enqueue_record_event(std::shared_ptr<Event> event);
    void populate_record_event(std::shared_ptr<Event> event);
    void enqueue_wait_for_event(std::shared_ptr<Event> event);
    void enqueue_trace();
    void finish();
    void launch(launch_msg_t& msg);
    friend void EnqueueTraceImpl(CommandQueue& cq);
    friend void EnqueueProgramImpl(CommandQueue& cq, std::variant < std::reference_wrapper<Program>, std::shared_ptr<Program> > program, bool blocking);
    friend void EnqueueReadBufferImpl(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void* dst, bool blocking);
    friend void EnqueueWriteBufferImpl(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, HostDataType src, bool blocking);
    friend void EnqueueAllocateBufferImpl(AllocBufferMetadata alloc_md);
    friend void EnqueueDeallocateBufferImpl(AllocBufferMetadata alloc_md);
    friend void EnqueueGetBufferAddrImpl(void* dst_buf_addr, const Buffer* buffer);
    friend void EnqueueRecordEventImpl(CommandQueue& cq, std::shared_ptr<Event> event);
    friend void EnqueueWaitForEventImpl(CommandQueue& cq, std::shared_ptr<Event> event);
    friend void FinishImpl(CommandQueue & cq);
    friend void EnqueueRecordEvent(CommandQueue& cq, std::shared_ptr<Event> event);
    friend class Trace;
};

// Common interface for all command queue types
struct CommandInterface {
    EnqueueCommandType type;
    std::optional<bool> blocking;
    std::optional<std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>> buffer;
    std::optional<std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>>> program;
    std::optional<AllocBufferMetadata> alloc_md;
    std::optional<RuntimeArgsMetadata> runtime_args_md;
    std::optional<const Buffer*> shadow_buffer;
    std::optional<HostDataType> src;
    std::optional<void*> dst;
    std::optional<std::shared_ptr<Event>> event;
};

class CommandQueue {
   friend class Device;
   public:
    enum class CommandQueueMode {
        PASSTHROUGH = 0,
        ASYNC = 1,
        TRACE = 2,
    };
    CommandQueue() = delete;
    ~CommandQueue();

    // Command queue constructor
    private:
    CommandQueue(Device* device, uint32_t id, CommandQueueMode mode = CommandQueue::default_mode());

    // Trace queue constructor
    public:
    CommandQueue(Trace* trace);

    // Getters for private members
    Device* device() const { return this->device_ptr; }
    Trace* trace() const { return this->trace_ptr; }
    uint32_t id() const { return this->cq_id; }

    // Blocking method to wait for all commands to drain from the queue
    // Optional if used in passthrough mode (async_mode = false)
    void wait_until_empty();

    // Schedule a command to be run on the device
    // Blocking if in passthrough mode. Non-blocking if in async mode
    void run_command(const CommandInterface& command);

    // API for setting/getting the mode of the command queue
    void set_mode(const CommandQueueMode& mode);
    CommandQueueMode get_mode() const { return this->mode; }

    // Reference to the underlying hardware command queue, non-const because side-effects are allowed
    HWCommandQueue& hw_command_queue();

    // The empty state of the worker queue
    bool empty() const { return this->worker_queue.empty(); }

    static CommandQueueMode default_mode() {
        // Envvar is used for bringup and debug only. Will be removed in the future and should not be relied on in production.
        static int value = parse_env<int>("TT_METAL_CQ_ASYNC_MODE", static_cast<int>(CommandQueueMode::PASSTHROUGH));
        return static_cast<CommandQueue::CommandQueueMode>(value);
    }
    // Determine if any CQ is using Async mode
    static bool async_mode_set() { return num_async_cqs > 0; }
   private:
    enum class CommandQueueState {
        IDLE = 0,
        RUNNING = 1,
        TERMINATE = 2,
    };

    friend class Trace;
    friend void EnqueueTraceImpl(CommandQueue& cq);
    friend uint32_t InstantiateTrace(Trace& trace, CommandQueue& cq);

    // Initialize Command Queue Mode based on the env-var. This will be default, unless the user excplictly sets the mode using set_mode.
    CommandQueueMode mode;
    CommandQueueState worker_state;
    std::unique_ptr<std::thread> worker_thread;
    WorkerQueue worker_queue;
    uint32_t cq_id;
    Device* device_ptr;
    Trace* trace_ptr;

    void start_worker();
    void stop_worker();
    void run_worker();
    void run_command_impl(const CommandInterface& command);

    bool async_mode() { return this->mode == CommandQueueMode::ASYNC; }
    bool trace_mode() { return this->mode == CommandQueueMode::TRACE; }
    bool passthrough_mode() { return this->mode == CommandQueueMode::PASSTHROUGH; }

    std::atomic<std::size_t> worker_thread_id = -1;
    std::atomic<std::size_t> parent_thread_id = -1;
    // Track the number of CQs using async vs pt mode
    inline static uint32_t num_async_cqs = 0;
    inline static uint32_t num_passthrough_cqs = 0;
};

// Primitives used to place host only operations on the SW Command Queue.
// These are used in functions exposed through tt_metal.hpp or host_api.hpp
void EnqueueAllocateBuffer(CommandQueue& cq, Buffer* buffer, bool bottom_up, bool blocking);
void EnqueueDeallocateBuffer(CommandQueue& cq, Allocator& allocator, uint32_t device_address, BufferType buffer_type, bool blocking);
void EnqueueGetBufferAddr(CommandQueue& cq, uint32_t* dst_buf_addr, const Buffer* buffer, bool blocking);
void EnqueueSetRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::shared_ptr<RuntimeArgs> runtime_args_ptr, bool blocking);
void EnqueueUpdateRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::vector<uint32_t> &update_idx, std::shared_ptr<RuntimeArgs> runtime_args_ptr, bool blocking);
void EnqueueAddBufferToProgram(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program, bool blocking);

} // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, EnqueueCommandType const& type);
std::ostream& operator<<(std::ostream& os, CommandQueue::CommandQueueMode const& type);
