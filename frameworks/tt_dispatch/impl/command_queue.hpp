
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
enum class EnqueueCommandType { ENQUEUE_READ_BUFFER, ENQUEUE_WRITE_BUFFER, ENQUEUE_LAUNCH, INVALID };

string EnqueueCommandTypeToString(EnqueueCommandType ctype) {
    switch (ctype) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER: return "EnqueueReadBuffer";
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: return "EnqueueWriteBuffer";
        case EnqueueCommandType::ENQUEUE_LAUNCH: return "EnqueueLaunch";
        default: TT_THROW("Invalid command type");
    }
}

void write_to_system_memory(Device* device, shared_ptr<SystemMemoryWriter> writer, DeviceCommand& command) {
    writer->cb_reserve_back(device);
    writer->noc_write(device, command);
    writer->cb_push_back(device);
}

class Command {
    EnqueueCommandType type_ = EnqueueCommandType::INVALID;

   public:
    Command() {}
    virtual void handle(){};
    virtual EnqueueCommandType type() = 0;
};

class EnqueueReadBufferCommand : public Command {
   private:
    Device* device;
    Buffer& buffer;
    shared_ptr<SystemMemoryWriter> writer;
    void* dst;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_READ_BUFFER;

   public:
    EnqueueReadBufferCommand(Device* device, Buffer& buffer, void* dst, shared_ptr<SystemMemoryWriter> writer) :
        writer(writer), buffer(buffer) {
        this->device = device;
        this->dst = dst;
    }

    void handle() {
        DeviceCommand command;
        write_to_system_memory(this->device, this->writer, command);
    }

    EnqueueCommandType type() { return this->type_; }
};

class EnqueueWriteBufferCommand : public Command {
   private:
    Device* device;
    Buffer& buffer;
    shared_ptr<SystemMemoryWriter> writer;
    void* src;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_WRITE_BUFFER;

   public:
    EnqueueWriteBufferCommand(Device* device, Buffer& buffer, void* src, shared_ptr<SystemMemoryWriter> writer) :
        writer(writer), buffer(buffer) {
        this->device = device;
        this->src = src;
    }

    void handle() {
        // log_debug(tt::LogDispatch, "Trying to write");
        std::cout << "HANDLING" << std::endl;
        // Need to ensure the lifetime of this buffer long enough to finish
        // the transfer... TODO later once I get to multiple back-to-back transfers
        // Buffer host_buf(this->device, this->buffer.size(), 0, this->buffer.size(), BufferType::SYSTEM_MEMORY);

        // TODO(agrebenisan): PERF ISSUE! For now need to explicitly deep-copy to
        // keep the same API as OpenCL, but eventually need to update cluster to be
        // able to directly write a void pointer to memory
        // vector<uint> copy((uint*)src, (uint*)src + host_buf.size() / sizeof(uint));

        // log_debug(tt::LogDispatch, "Trying to write");
        // uint sysmem_addr = 0;
        // this->device->cluster()->write_sysmem_vec(copy, sysmem_addr, 0);

        // DeviceCommand command;
        // write_to_system_memory(this->device, this->writer, command);

        std::cout << "Handled" << std::endl;
        log_debug(tt::LogDispatch, "Handled");
    }

    EnqueueCommandType type() { return this->type_; }
};

class EnqueueLaunchCommand : public Command {
   private:
    Device* device;
    Program* program;
    shared_ptr<SystemMemoryWriter> writer;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_LAUNCH;

   public:
    EnqueueLaunchCommand(Device* device, Program* program, shared_ptr<SystemMemoryWriter> writer) : writer(writer) {
        this->device = device;
        this->program = program;
    }

    void handle() {
        DeviceCommand command;
        write_to_system_memory(this->device, this->writer, command);
    }

    EnqueueCommandType type() { return this->type_; }
};

class CommandQueue {
   public:
    CommandQueue(Device* device) {
        auto worker_logic = [this]() {
            tt::log_debug(tt::LogDispatch, "Initialized worker thread");
            while (true) {  // Worker thread keeps on flushing
                tt::log_debug(
                    tt::LogDispatch, "Handling {}", EnqueueCommandTypeToString(this->internal_queue.peek()->type()));
                this->internal_queue.peek()
                    ->handle();  // Only responsible for ensuring that command enqueued onto device... needs to be
                                 // handled prior to popping for 'flush' semantics to work
                tt::log_debug(tt::LogDispatch, "Popping, size {}", this->internal_queue.size());
                this->internal_queue.pop();
                tt::log_debug(tt::LogDispatch, "size {}", this->internal_queue.size());
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
    void enqueue_command(shared_ptr<Command> command, bool blocking) {
        std::cout << "Enqueued type: " << EnqueueCommandTypeToString(command->type()) << std::endl;

        this->internal_queue.push(std::move(command));

        if (blocking) {
            this->finish();
        }
    }

    void enqueue_read_buffer(Device* device, Buffer& buffer, void* dst, bool blocking) {
        EnqueueReadBufferCommand command(device, buffer, dst, this->sysmem_writer);
        shared_ptr<EnqueueReadBufferCommand> p = std::make_shared<EnqueueReadBufferCommand>(std::move(command));
        this->enqueue_command(p, blocking);
    }

    void enqueue_write_buffer(Device* device, Buffer& buffer, void* src, bool blocking) {
        EnqueueWriteBufferCommand command(device, buffer, src, this->sysmem_writer);
        shared_ptr<EnqueueWriteBufferCommand> p = std::make_shared<EnqueueWriteBufferCommand>(std::move(command));
        this->enqueue_command(p, blocking);
    }

    void enqueue_launch(Device* device, Program* program, bool blocking) {
        EnqueueLaunchCommand command(device, program, this->sysmem_writer);
        shared_ptr<EnqueueLaunchCommand> p = std::make_shared<EnqueueLaunchCommand>(std::move(command));
        this->enqueue_command(p, blocking);
    }

    void flush() {
        std::mutex m;
        std::unique_lock<std::mutex> lock(m);
        this->internal_queue.empty_condition.wait(
            lock, [this]() { return !this->internal_queue.q.empty(); });
    }

    void finish() { TT_THROW("CommandQueue.finish not yet implemented"); }

    friend void EnqueueReadBuffer(Device* device, CommandQueue& cq, Buffer& buffer, void* dst, bool blocking);
    friend void EnqueueWriteBuffer(Device* device, CommandQueue& cq, Buffer& buffer, void* src, bool blocking);
    friend void Launch();
    friend void Flush(CommandQueue& cq);
    friend void Finish(CommandQueue& cq);
};

void EnqueueReadBuffer(Device* device, CommandQueue& cq, Buffer& buffer, void* dst, bool blocking) {
    log_debug(tt::LogDispatch, "EnqueueReadBuffer");
    cq.enqueue_read_buffer(device, buffer, dst, blocking);
}

void EnqueueWriteBuffer(Device* device, CommandQueue& cq, Buffer& buffer, void* src, bool blocking) {
    log_debug(tt::LogDispatch, "EnqueueWriteBuffer");
    cq.enqueue_write_buffer(device, buffer, src, blocking);
}

void EnqueueLaunch() {
    log_debug(tt::LogDispatch, "EnqueueLaunch");
    TT_THROW("EnqueueLaunch not yet implemented");
}

void Flush(CommandQueue& cq) {
    log_debug(tt::LogDispatch, "Flush");
    cq.flush();
}

void Finish(CommandQueue& cq) {
    log_debug(tt::LogDispatch, "Finish");
    cq.finish();
}
