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
#include "jit_build/build.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "noc/noc_parameters.h"

namespace tt::tt_metal {

struct transfer_info {
    uint32_t size_in_bytes;
    uint32_t dst;
    uint32_t dst_noc_encoding;
    uint32_t num_receivers;
    bool last_transfer_in_group;
    bool linked;
};

struct ProgramMap {
    uint32_t num_workers;
    vector<uint32_t> program_pages;
    vector<transfer_info> program_page_transfers;
    vector<transfer_info> runtime_arg_page_transfers;
    vector<transfer_info> cb_config_page_transfers;
    vector<transfer_info> go_signal_page_transfers;
    vector<uint32_t> num_transfers_in_program_pages;
    vector<uint32_t> num_transfers_in_runtime_arg_pages;
    vector<uint32_t> num_transfers_in_cb_config_pages;
    vector<uint32_t> num_transfers_in_go_signal_pages;
};

// Only contains the types of commands which are enqueued onto the device
enum class EnqueueCommandType { ENQUEUE_READ_BUFFER, ENQUEUE_WRITE_BUFFER, ENQUEUE_PROGRAM, FINISH, ENQUEUE_WRAP, ENQUEUE_RESTART, INVALID };

string EnqueueCommandTypeToString(EnqueueCommandType ctype);

// TEMPORARY! TODO(agrebenisan): need to use proper macro based on loading noc
#define NOC_X(x) x
#define NOC_Y(y) y

uint32_t get_noc_unicast_encoding(CoreCoord coord);

class Trace;
class CommandQueue;

class Command {
    EnqueueCommandType type_ = EnqueueCommandType::INVALID;

   public:
    Command() {}
    virtual void process(){};
    virtual EnqueueCommandType type() = 0;
    virtual const DeviceCommand assemble_device_command(uint32_t buffer_size) = 0;
};

class EnqueueRestartCommand : public Command {
   private:
    Device* device;
    SystemMemoryManager& manager;
    uint32_t event;
    uint32_t command_queue_id;
   public:
    EnqueueRestartCommand(
        uint32_t command_queue_id,
        Device* device,
        SystemMemoryManager& manager,
        uint32_t event
    );

    const DeviceCommand assemble_device_command(uint32_t dst);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_RESTART; };
};

class EnqueueReadBufferCommand : public Command {
   private:
    SystemMemoryManager& manager;
    void* dst;
    uint32_t pages_to_read;
    uint32_t command_queue_id;
    uint32_t event;

    virtual const DeviceCommand create_buffer_transfer_instruction(uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) = 0;
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
        SystemMemoryManager& manager,
        uint32_t event,
        uint32_t src_page_index = 0,
        std::optional<uint32_t> pages_to_read = std::nullopt);

    const DeviceCommand assemble_device_command(uint32_t dst);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_READ_BUFFER; };
};

class EnqueueReadInterleavedBufferCommand : public EnqueueReadBufferCommand {
   private:
    const DeviceCommand create_buffer_transfer_instruction(uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) override;

   public:
    EnqueueReadInterleavedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        Buffer& buffer,
        void* dst,
        SystemMemoryManager& manager,
        uint32_t event,
        uint32_t src_page_index = 0,
        std::optional<uint32_t> pages_to_read = std::nullopt)
            :EnqueueReadBufferCommand(command_queue_id,
                                device,
                                buffer,
                                dst,
                                manager,
                                event,
                                src_page_index,
                                pages_to_read) {}
};


class EnqueueReadShardedBufferCommand : public EnqueueReadBufferCommand {
   private:
    const DeviceCommand create_buffer_transfer_instruction(uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) override;

   public:
    EnqueueReadShardedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        Buffer& buffer,
        void* dst,
        SystemMemoryManager& manager,
        uint32_t event,
        uint32_t src_page_index = 0,
        std::optional<uint32_t> pages_to_read = std::nullopt)
            :EnqueueReadBufferCommand(command_queue_id,
                                device,
                                buffer,
                                dst,
                                manager,
                                event,
                                src_page_index,
                                pages_to_read) {}
};

class EnqueueWriteShardedBufferCommand;
class EnqueueWriteInterleavedBufferCommand;
class EnqueueWriteBufferCommand : public Command {
   private:

    SystemMemoryManager& manager;
    uint32_t event;
    const void* src;
    uint32_t pages_to_write;
    uint32_t command_queue_id;

    virtual const DeviceCommand create_buffer_transfer_instruction(uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) = 0;
   protected:
    Device* device;
    Buffer& buffer;
    uint32_t dst_page_index;
   public:
    EnqueueWriteBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        Buffer& buffer,
        const void* src,
        SystemMemoryManager& manager,
        uint32_t event,
        uint32_t dst_page_index = 0,
        std::optional<uint32_t> pages_to_write = std::nullopt);

    const DeviceCommand assemble_device_command(uint32_t src_address);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_WRITE_BUFFER; }
};

class EnqueueWriteInterleavedBufferCommand : public EnqueueWriteBufferCommand {
   private:
    const DeviceCommand create_buffer_transfer_instruction(uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) override;
   public:
    EnqueueWriteInterleavedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        Buffer& buffer,
        const void* src,
        SystemMemoryManager& manager,
        uint32_t event,
        uint32_t dst_page_index = 0,
        std::optional<uint32_t> pages_to_write = std::nullopt)
        : EnqueueWriteBufferCommand(
            command_queue_id,
            device,
            buffer,
            src,
            manager,
            event,
            dst_page_index,
            pages_to_write){;}


};


class EnqueueWriteShardedBufferCommand : public EnqueueWriteBufferCommand {
   private:
    const DeviceCommand create_buffer_transfer_instruction(uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) override;
   public:
    EnqueueWriteShardedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        Buffer& buffer,
        const void* src,
        SystemMemoryManager& manager,
        uint32_t event,
        uint32_t dst_page_index = 0,
        std::optional<uint32_t> pages_to_write = std::nullopt)
        : EnqueueWriteBufferCommand(
            command_queue_id,
            device,
            buffer,
            src,
            manager,
            event,
            dst_page_index,
            pages_to_write){;}


};

class EnqueueProgramCommand : public Command {
   private:
    uint32_t command_queue_id;
    Device* device;
    Buffer& buffer;
    ProgramMap& program_to_dev_map;
    const Program& program;
    SystemMemoryManager& manager;
    uint32_t event;
    bool stall;
    std::optional<std::reference_wrapper<Trace>> trace = {};

   public:
    EnqueueProgramCommand(uint32_t command_queue_id, Device*, Buffer&, ProgramMap&, SystemMemoryManager&, uint32_t event, const Program& program, bool stall, std::optional<std::reference_wrapper<Trace>> trace);

    const DeviceCommand assemble_device_command(uint32_t src_address);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_PROGRAM; }
};

class EnqueueWrapCommand : public Command {
   protected:
    Device* device;
    SystemMemoryManager& manager;
    uint32_t command_queue_id;

   public:
    EnqueueWrapCommand(uint32_t command_queue_id, Device* device, SystemMemoryManager& manager);

    virtual const DeviceCommand assemble_device_command(uint32_t) = 0;

    virtual void process() = 0;

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_WRAP; }
};

class EnqueueIssueWrapCommand : public EnqueueWrapCommand {
    public:
     EnqueueIssueWrapCommand(uint32_t command_queue_id, Device* device, SystemMemoryManager& manager);

     const DeviceCommand assemble_device_command(uint32_t);

     void process();
};

class EnqueueCompletionWrapCommand : public EnqueueWrapCommand {
    private:
     uint32_t event;

    public:
     EnqueueCompletionWrapCommand(uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, uint32_t event);

     const DeviceCommand assemble_device_command(uint32_t);

     void process();
};

namespace detail{
    CommandQueue &GetCommandQueue(Device *device);
}


class Trace {

    private:
      struct TraceNode {
          DeviceCommand command;
          const vector<uint32_t> data;
          EnqueueCommandType command_type;
          uint32_t num_data_bytes;
      };
      bool trace_complete;
      CommandQueue& command_queue;
      vector<TraceNode> history;
      uint32_t num_data_bytes;
      void create_replay();

    friend class EnqueueProgramCommand;
    friend Trace BeginTrace(CommandQueue& cq);
    friend void EndTrace(Trace& trace);
    friend void EnqueueTrace(Trace& trace, bool blocking);

    public:
      Trace(CommandQueue& command_queue);
      void record(const TraceNode& trace_node);
};

namespace detail {
void EnqueueRestart(CommandQueue& cq);

struct IssuedReadData {
    Buffer& buffer;
    uint32_t padded_page_size;
    uint32_t num_pages_read;
    void* dst;

    IssuedReadData(Buffer& buffer, uint32_t padded_page_size, void* dst, uint32_t num_pages_read): buffer(buffer) {
        this->padded_page_size = padded_page_size;
        this->dst = dst;
        this->num_pages_read = num_pages_read;
    }
};

inline std::mutex issued_read_mutex;
inline std::mutex completion_wrap_mutex;

template <typename T, std::mutex& my_mutex>
class thread_safe_map {
    private:
     map<uint32_t, T> issued_events;
    public:
     thread_safe_map<T, my_mutex>() {}

     const T& at(uint32_t event) const {
        std::lock_guard lock(my_mutex);
        return this->issued_events.at(event);
     }

     void emplace(uint32_t event, const T& issued_read_data) {
        std::lock_guard lock(my_mutex);
        this->issued_events.emplace(event, issued_read_data);
     }

     void erase(uint32_t event) {
        std::lock_guard lock(my_mutex);
        this->issued_events.erase(event);
     }

     size_t count(uint32_t event) const {
        std::lock_guard lock(my_mutex);
        return this->issued_events.count(event);
     }

     size_t size() const {
        std::lock_guard lock(my_mutex);
        return this->issued_events.size();
     }
};

typedef thread_safe_map<IssuedReadData, issued_read_mutex> IssuedReadMap;
}


class CommandQueue {
   public:

    CommandQueue(Device* device, uint32_t id);

    ~CommandQueue();
    CoreCoord issue_queue_reader_core;
    CoreCoord completion_queue_writer_core;

   private:
    uint32_t id;
    uint32_t size_B;
    std::optional<uint32_t> last_event_id;
    std::thread completion_queue_thread;

    // std::condition_variable cv;
    std::atomic<bool> exit_condition;
    volatile std::atomic<uint32_t> num_issued_commands;
    detail::IssuedReadMap issued_reads;
    detail::thread_safe_map<uint32_t, detail::completion_wrap_mutex> issued_completion_wraps;

    SystemMemoryManager& manager;

    Device* device;

    map<uint64_t, std::unique_ptr<Buffer>>& program_to_buffer(const chip_id_t chip_id) {
        static map<chip_id_t, map<uint64_t, std::unique_ptr<Buffer>>> chip_to_program_to_buffer;
        if (chip_to_program_to_buffer.count(chip_id)) {
            return chip_to_program_to_buffer[chip_id];
        }
        map<uint64_t, std::unique_ptr<Buffer>> dummy;
        chip_to_program_to_buffer.emplace(chip_id, std::move(dummy));
        return chip_to_program_to_buffer[chip_id];
    }

    map<uint64_t, ProgramMap>& program_to_dev_map(const chip_id_t chip_id) {
        static map<chip_id_t, map<uint64_t, ProgramMap>> chip_to_program_to_dev_map;
        if (chip_to_program_to_dev_map.count(chip_id)) {
            return chip_to_program_to_dev_map[chip_id];
        }
        map<uint64_t, ProgramMap> dummy;
        chip_to_program_to_dev_map.emplace(chip_id, std::move(dummy));
        return chip_to_program_to_dev_map[chip_id];
    };

    void copy_into_user_space(uint32_t event, uint32_t read_ptr, chip_id_t mmio_device_id, uint16_t channel);
    void read_completion_queue();
    void enqueue_command(Command& command, bool blocking);
    void enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking);
    void enqueue_write_buffer(Buffer& buffer, const void* src, bool blocking);
    void enqueue_program(Program& program, std::optional<std::reference_wrapper<Trace>> trace, bool blocking);
    void finish();
    void issue_wrap();
    void completion_wrap(uint32_t event);
    void restart();
    void launch(launch_msg_t& msg);

    friend void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<uint32_t>& dst, bool blocking);
    friend void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<uint32_t>& src, bool blocking);
    friend void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, void* dst, bool blocking);
    friend void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, const void* src, bool blocking);
    friend void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking, std::optional<std::reference_wrapper<Trace>> trace);
    friend void Finish(CommandQueue& cq);
    friend void detail::EnqueueRestart(CommandQueue& cq);
    friend void ClearProgramCache(CommandQueue& cq);
    friend CommandQueue &detail::GetCommandQueue(Device *device);
    // friend uint32_t EnqueueRecordEvent(CommandQueue& command_queue);

    // Trace APIs
    friend Trace BeginTrace(CommandQueue& command_queue);
    friend void EndTrace(Trace& trace);
    friend void EnqueueTrace(Trace& trace, bool blocking);
    friend class Device;
    friend class Trace;
};

inline bool LAZY_COMMAND_QUEUE_MODE = false;

} // namespace tt::tt_metal
