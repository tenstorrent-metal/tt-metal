
#include <memory>
#include <thread>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "frameworks/tt_dispatch/impl/sysmem_cb.hpp"
#include "frameworks/tt_dispatch/impl/thread_safe_queue.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/src/firmware/riscv/grayskull/noc/noc_parameters.h"
#include "llrt/tt_debug_print_server.hpp"


using namespace tt::tt_metal;
using std::shared_ptr;
using std::thread;
using std::unique_ptr;

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

void write_to_system_memory(Device* device, SystemMemoryWriter& writer, const DeviceCommand& command) {
    // writer.cb_reserve_back(device);
    writer.noc_write(device, command);
    while(true);
    // writer.cb_push_back(device);
}

// TEMPORARY! TODO(agrebenisan): need to use proper macro based on loading noc
#define NOC_X(x) x
#define NOC_Y(y) y

uint noc_coord_to_uint(tt_xy_pair coord) { return NOC_XY_ENCODING(NOC_X(coord.x), NOC_Y(coord.y)); }

class Command {
    EnqueueCommandType type_ = EnqueueCommandType::INVALID;

   public:
    Command() {}
    virtual void handle(){};
    virtual EnqueueCommandType type() = 0;
    virtual DeviceCommand device_command() = 0;
};

class EnqueueReadBufferCommand : public Command {
   private:
    Device* device;
    Buffer& buffer;
    SystemMemoryWriter& writer;
    void* dst;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_READ_BUFFER;

   public:
    EnqueueReadBufferCommand(Device* device, Buffer& buffer, void* dst, SystemMemoryWriter& writer) :
        writer(writer), buffer(buffer) {
        this->device = device;
        this->dst = dst;
    }

    DeviceCommand device_command() {
        DeviceCommand command;
        return command;
    }

    void handle() { TT_THROW("EnqueueReadBufferCommand.handle not implemented yet"); }

    EnqueueCommandType type() { return this->type_; }
};

class EnqueueWriteBufferCommand : public Command {
   private:
    Device* device;
    Buffer& buffer;

    unique_ptr<Buffer> system_mem_buffer;  // Need to store a temporary sysmem buffer and ensure it is not freed until
                                           // the command has completed on device

    SystemMemoryWriter writer;
    void* src;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_WRITE_BUFFER;

   public:
    EnqueueWriteBufferCommand(Device* device, Buffer& buffer, void* src, SystemMemoryWriter& writer) :
        writer(writer), buffer(buffer) {
        this->device = device;
        this->src = src;
    }

    DeviceCommand device_command() {
        DeviceCommand command;

        tt::log_debug(tt::LogDispatch, "Getting device command");

        switch (this->buffer.buffer_type()) {
            case BufferType::DRAM: {
                tt_xy_pair dram_noc_coordinates = this->buffer.noc_coordinates();
                command.add_interleaved_dram_write(
                    this->system_mem_buffer->address(),
                    this->buffer.address(),
                    noc_coord_to_uint(dram_noc_coordinates),
                    this->buffer.page_size(),
                    this->buffer.size() / this->buffer.page_size());
            } break;
            case BufferType::L1: {
                TT_THROW("L1 Buffer write not implemented yet");
            } break;
            default: TT_THROW("Invalid buffer type for EnqueueWriteBufferCommand");
        }

        return command;
    }

    void handle() {
        // Need to ensure the lifetime of this buffer long enough to finish
        // the transfer, otherwise buffer destroyed and allocator will cleanup... TODO later once I get to multiple
        // back-to-back transfers
        Buffer a(this->device, this->buffer.size(), 0, this->buffer.size(), BufferType::SYSTEM_MEMORY);
        this->system_mem_buffer = std::make_unique<Buffer>(std::move(a));

        // TODO(agrebenisan): PERF ISSUE! For now need to explicitly deep-copy to
        // keep the same API as OpenCL, but eventually need to update cluster to be
        // able to directly write a void pointer to memory
        vector<uint> copy((uint*)src, (uint*)src + this->system_mem_buffer->size() / sizeof(uint));

        this->device->cluster()->write_sysmem_vec(copy, this->system_mem_buffer->address(), 0);

        write_to_system_memory(this->device, this->writer, this->device_command());
    }

    EnqueueCommandType type() { return this->type_; }
};

class EnqueueLaunchCommand : public Command {
   private:
    Device* device;
    Program* program;
    SystemMemoryWriter& writer;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_LAUNCH;

   public:
    EnqueueLaunchCommand(Device* device, Program* program, SystemMemoryWriter& writer) : writer(writer) {
        this->device = device;
        this->program = program;
    }

    DeviceCommand device_command() {
        DeviceCommand command;
        return command;
    }

    void handle() { TT_THROW("EnqueueLaunchCommand.handle not implemented yet"); }

    EnqueueCommandType type() { return this->type_; }
};

void send_dispatch_kernel_to_device(Device* device) {
    // Ideally, this should be some separate API easily accessible in
    // TT-metal, don't like the fact that I'm writing this from scratch
    std::string root_dir = tt::utils::get_root_dir();
    std::string arch_name = tt::utils::get_env_arch_name();
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("unary", "command_queue");
    std::string out_dir_path = root_dir + "/built_kernels/" + build_kernel_for_riscv_options.name;

    build_kernel_for_riscv_options.fp32_dest_acc_en = false;

    // Hard-coding as BRISC for now, could potentially be NCRISC
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/dispatch/command_queue.cpp";
    // std::map<string, string> brisc_defines = {{"IS_DISPATCH_KERNEL", ""}, {"DEVICE_DISPATCH_MODE", ""}};
    // build_kernel_for_riscv_options.brisc_defines = brisc_defines;
    generate_binary_for_risc(RISCID::BR, &build_kernel_for_riscv_options, out_dir_path, arch_name);

    tt_xy_pair dispatch_core = {1, 11};
    tt::llrt::test_load_write_read_risc_binary(
        device->cluster(), "built_kernels/command_queue/brisc/brisc.hex", 0, dispatch_core, 0);

    // Deassert reset of dispatch core BRISC. TODO(agrebenisan): Refactor once Paul's changes in
    // device->cluster()->set_remote_tensix_risc_reset(tt_cxy_pair(0, dispatch_core), TENSIX_DEASSERT_SOFT_RESET);
    // tt_start_debug_print_server(device->cluster(), {0}, {dispatch_core});
    tt::llrt::deassert_brisc_reset_for_all_chips_all_cores(device->cluster(), false);
    tt::log_debug(tt::LogDispatch, "Deasserted dispatch core");
    while(true);
}

class CommandQueue {
   public:
    CommandQueue(Device* device) {
        send_dispatch_kernel_to_device(device);
        // this->num_cqs++;

        // switch (this->num_cqs) {
        //     case 1: {
        //         // Logic to write the dispatch kernel to dispatch core
        //     } break;
        //     case 2: TT_THROW("2 command queues not implemented yet"); break;
        //     case 3: TT_THROW("3 command queues not implemented yet"); break;
        //     case 4: TT_THROW("4 command queues not implemented yet"); break;
        //     default: TT_THROW("Only a maximum of 4 command queues can be created so far");
        // }

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
    }

    ~CommandQueue() {
        this->finish();
        // this->num_cqs--;
    }

   private:
    Device* device;
    SystemMemoryWriter sysmem_writer;
    TSQueue<shared_ptr<Command>> internal_queue;
    void enqueue_command(shared_ptr<Command> command, bool blocking) {
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
        this->internal_queue.empty_condition.wait(lock, [this]() { return this->internal_queue.q.empty(); });
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
