#include <memory>
#include <thread>

#include "frameworks/tt_dispatch/impl/thread_safe_queue.hpp"
#include "frameworks/tt_dispatch/impl/sysmem_cb.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;
using std::thread;
using std::shared_ptr;

enum class CommandType { ENQUEUE_READ_BUFFER, ENQUEUE_WRITE_BUFFER, ENQUEUE_LAUNCH, FLUSH, FINISH };

void write_to_system_memory(shared_ptr<SystemMemoryWriter> writer, DeviceCommand& command) {
    writer->cb_reserve_back();
    writer->noc_write(command);
    writer->cb_push_back();
}

class Command {
    CommandType type;

   public:
    Command();
    virtual void handle();
};

class EnqueueReadBufferCommand : public Command {
   private:
    Device* device;
    DramBuffer* buffer;
    shared_ptr<SystemMemoryWriter> writer;

   public:
    static constexpr CommandType type = CommandType::ENQUEUE_READ_BUFFER;

    EnqueueReadBufferCommand(Device* device, DramBuffer* buffer, shared_ptr<SystemMemoryWriter> writer): writer(writer) {
        this->device = device;
        this->buffer = buffer;
    }

    void handle() {
        DeviceCommand command;
        write_to_system_memory(this->writer, command);
    }
};

class EnqueueWriteBufferCommand : public Command {
   private:
    Device* device;
    DramBuffer* buffer;
    shared_ptr<SystemMemoryWriter> writer;

   public:
    static constexpr CommandType type = CommandType::ENQUEUE_WRITE_BUFFER;

    EnqueueWriteBufferCommand(Device* device, DramBuffer* buffer, shared_ptr<SystemMemoryWriter> writer): writer(writer) {
        this->device = device;
        this->buffer = buffer;
    }

    void handle() {
        DeviceCommand command;
        write_to_system_memory(this->writer, command);
    }
};

class EnqueueLaunchCommand : public Command {
   private:
    Device* device;
    Program* program;
    shared_ptr<SystemMemoryWriter> writer;

   public:
    static constexpr CommandType type = CommandType::ENQUEUE_LAUNCH;

    EnqueueLaunchCommand(Device* device, Program* program, shared_ptr<SystemMemoryWriter> writer): writer(writer) {
        this->device = device;
        this->program = program;
    }

    void handle() {
        DeviceCommand command;
        write_to_system_memory(this->writer, command);
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

        thread(worker_logic).detach();  // Detaching as we don't need to keep track of explicitly with a class attribute

        SystemMemoryWriter writer = SystemMemoryWriter(device);
        shared_ptr<SystemMemoryWriter> p = std::make_unique<SystemMemoryWriter>(&writer);
        this->sysmem_writer = std::move(p);
    }

    ~CommandQueue() { this->finish(); }

   private:
    shared_ptr<SystemMemoryWriter> sysmem_writer;
    TSQueue<shared_ptr<Command>> internal_queue;
    void enqueue_command(Command& command, bool blocking) {
        shared_ptr<Command> p = std::make_unique<Command>(&command);

        this->internal_queue.push(std::move(p));

        if (blocking) {
            this->finish();
        }
    }

    void enqueue_read_buffer(Device* device, DramBuffer* buffer, void* dst, bool blocking) {
        EnqueueReadBufferCommand command(device, buffer, this->sysmem_writer);
        this->enqueue_command(command, blocking);
    }

    void enqueue_write_buffer(Device* device, DramBuffer* buffer, void* src, bool blocking) {
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

    friend void EnqueueReadBuffer(Device* device, CommandQueue& cq, DramBuffer* buffer, void* dst, bool blocking);
    friend void EnqueueWriteBuffer();
    friend void Launch();
    friend void Flush(CommandQueue& cq);
    friend void Finish(CommandQueue& cq);
};

void EnqueueReadBuffer(Device* device, CommandQueue& cq, DramBuffer* buffer, void* dst, bool blocking) {
    cq.enqueue_read_buffer(device, buffer, dst, blocking);
}

void EnqueueWriteBuffer() { TT_THROW("EnqueueWriteBuffer not yet implemented"); }

void EnqueueLaunch() { TT_THROW("EnqueueLaunch not yet implemented"); }

void Flush(CommandQueue& cq) { cq.flush(); }

void Finish(CommandQueue& cq) { cq.finish(); }
