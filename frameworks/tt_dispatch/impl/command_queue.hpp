#include <memory>
#include <thread>

#include "frameworks/tt_dispatch/impl/sysmem_cb.hpp"
#include "frameworks/tt_dispatch/impl/thread_safe_queue.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;
using std::shared_ptr;
using std::thread;

// Only contains the types of commands which are enqueued onto the device
enum class EnqueueCommandType {
    ENQUEUE_READ_BUFFER,
    ENQUEUE_WRITE_BUFFER,
    ENQUEUE_LAUNCH,
};

void write_to_system_memory(Device* device, shared_ptr<SystemMemoryWriter> writer, DeviceCommand& command) {
    writer->cb_reserve_back(device);
    writer->noc_write(device, command);
    writer->cb_push_back(device);
}

class Command {
    EnqueueCommandType type;

   public:
    Command() {}
    virtual void handle(){};
};

class EnqueueReadBufferCommand : public Command {
   private:
    Device* device;
    Buffer& buffer;
    shared_ptr<SystemMemoryWriter> writer;

   public:
    static constexpr EnqueueCommandType type = EnqueueCommandType::ENQUEUE_READ_BUFFER;

    EnqueueReadBufferCommand(Device* device, Buffer& buffer, shared_ptr<SystemMemoryWriter> writer) :
        writer(writer), buffer(buffer) {
        this->device = device;
    }

    void handle() {
        DeviceCommand command;
        write_to_system_memory(this->device, this->writer, command);
    }
};

class EnqueueWriteBufferCommand : public Command {
   private:
    Device* device;
    Buffer& buffer;
    shared_ptr<SystemMemoryWriter> writer;

   public:
    static constexpr EnqueueCommandType type = EnqueueCommandType::ENQUEUE_WRITE_BUFFER;

    EnqueueWriteBufferCommand(Device* device, Buffer& buffer, shared_ptr<SystemMemoryWriter> writer) :
        writer(writer), buffer(buffer) {
        this->device = device;
    }

    void handle() {
        DeviceCommand command;
        // write_to_system_memory(this->device, this->writer, command);
    }
};

class EnqueueLaunchCommand : public Command {
   private:
    Device* device;
    Program* program;
    shared_ptr<SystemMemoryWriter> writer;

   public:
    static constexpr EnqueueCommandType type = EnqueueCommandType::ENQUEUE_LAUNCH;

    EnqueueLaunchCommand(Device* device, Program* program, shared_ptr<SystemMemoryWriter> writer) : writer(writer) {
        this->device = device;
        this->program = program;
    }

    void handle() {
        DeviceCommand command;
        write_to_system_memory(this->device, this->writer, command);
    }
};

class CommandQueue {
   public:
    CommandQueue(Device* device) {
        auto worker_logic = [this]() {
            while (true) {       // Worker thread keeps on flushing
                this->internal_queue.peek()
                    ->handle();  // Only responsible for ensuring that command enqueued onto device... needs to be
                                 // handled prior to popping for 'flush' semantics to work
                this->internal_queue.pop();
            }
        };

        thread(worker_logic).detach();  // Detaching as we don't need to keep track of this explicitly with a class
                                        // attribute, and we don't want the thread to be destroyed at end of scope

        SystemMemoryWriter writer;
        shared_ptr<SystemMemoryWriter> p = std::make_shared<SystemMemoryWriter>(std::move(writer));
        this->sysmem_writer = std::move(p);
    }

    ~CommandQueue() { this->finish(); }

   private:
    Device* device;
    shared_ptr<SystemMemoryWriter> sysmem_writer;
    TSQueue<shared_ptr<Command>> internal_queue;
    void enqueue_command(Command& command, bool blocking) {
        shared_ptr<Command> p = std::make_shared<Command>(std::move(command));

        this->internal_queue.push(std::move(p));

        if (blocking) {
            this->finish();
        }
    }

    void enqueue_read_buffer(Device* device, Buffer& buffer, void* dst, bool blocking) {
        EnqueueReadBufferCommand command(device, buffer, this->sysmem_writer);
        this->enqueue_command(command, blocking);
    }

    void enqueue_write_buffer(Device* device, Buffer& buffer, void* src, bool blocking) {
        EnqueueWriteBufferCommand command(device, buffer, this->sysmem_writer);
        this->enqueue_command(command, blocking);
    }

    void enqueue_launch(Device* device, Program* program, bool blocking) {
        EnqueueLaunchCommand command(device, program, this->sysmem_writer);
        this->enqueue_command(command, blocking);
    }

    void flush() {
        while (this->internal_queue.size() > 0)
            ;  // Wait until all commands have been enqueued on device
    }

    void finish() { TT_THROW("CommandQueue.finish not yet implemented"); }

    friend void EnqueueReadBuffer(Device* device, CommandQueue& cq, Buffer& buffer, void* dst, bool blocking);
    friend void EnqueueWriteBuffer(Device* device, CommandQueue& cq, Buffer& buffer, void* src, bool blocking);
    friend void Launch();
    friend void Flush(CommandQueue& cq);
    friend void Finish(CommandQueue& cq);
};

void EnqueueReadBuffer(Device* device, CommandQueue& cq, Buffer& buffer, void* dst, bool blocking) {
    cq.enqueue_read_buffer(device, buffer, dst, blocking);
}

void EnqueueWriteBuffer(Device* device, CommandQueue& cq, Buffer& buffer, void* src, bool blocking) {
    cq.enqueue_write_buffer(device, buffer, src, blocking);
}

void EnqueueLaunch() { TT_THROW("EnqueueLaunch not yet implemented"); }

void Flush(CommandQueue& cq) { cq.flush(); }

void Finish(CommandQueue& cq) { cq.finish(); }
