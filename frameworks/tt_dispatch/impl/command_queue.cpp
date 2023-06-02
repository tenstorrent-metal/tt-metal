#include "command_queue.hpp"

#include "tt_metal/llrt/tt_debug_print_server.hpp"

u64 get_noc_multicast_encoding(const CoreCoord& top_left, const CoreCoord& bottom_right) {
    return NOC_MULTICAST_ENCODING(top_left.x, top_left.y, bottom_right.x, bottom_right.y);
}

ProgramToDeviceMap ConstructProgramToDeviceMap(const Device* device, Program& program) {
    // This function retrieves all the required information to group program binaries into sections,
    // such that each section is the largest amount of data that can be read into the dispatch
    // core's L1 at a time. For each section, it also specifies the relay program information,
    // as described in command.hpp.
    // Comment from Andrew Grebenisan, the author: Please do not refactor this function into
    // multiple pieces. It's quite long, however this function does one thing, and I want
    // potential debug of this function's logic to all be in one spot.
    ProgramToDeviceMap program_to_device_map;
    vector<u32>& program_vector = program_to_device_map.bins;
    vector<ProgramSection>& sections = program_to_device_map.sections;

    // 'section' here refers to a piece of the program buffer
    // that we can read in one shot into dispatch core L1
    u32 current_section_idx = 0;
    auto initialize_section = [&sections]() {
        // The purpose of this function is to create a new 'section'
        // as described in the above comment.

        vector<transfer_info> init_vec;
        map<char, vector<transfer_info>> init_map;
        init_map.emplace('B', init_vec);
        init_map.emplace('N', init_vec);
        init_map.emplace('U', init_vec);
        init_map.emplace('M', init_vec);
        init_map.emplace('P', init_vec);
        ProgramSection section = {.section = init_map, .size_in_bytes = 0};
        sections.push_back(section);
    };

    // Initialize program_to_device_map with all possible keys
    initialize_section();

    u32 start_in_bytes = 0;
    u32 kernel_size_in_bytes = 0;
    auto write_to_program_device_map = [&](char riscv_type, const Kernel* kernel) {
        vector<char> riscv_types;
        switch (riscv_type) {
            case 'C': riscv_types = {'U', 'M', 'P'}; break;
            case 'N':
            case 'B': riscv_types = {riscv_type}; break;
            default: TT_THROW("Invalid riscv_type");
        }

        tt::log_debug(tt::LogDispatch, "Writing to program device map for {}", riscv_type);

        size_t i = 0;

        const vector<ll_api::memory>& kernel_bins = kernel->binaries();
        CoreRangeSet cr_set = kernel->core_range_set();

        for (char riscv_type : riscv_types) {
            const ll_api::memory& kernel_bin = kernel_bins.at(i);
            i++;

            u32 num_bytes_so_far = program_vector.size() * sizeof(u32);
            u32 num_new_bytes = kernel_bin.size() * sizeof(u32);

            if (num_bytes_so_far + num_new_bytes > 1024 * 1024 - UNRESERVED_BASE) {
                current_section_idx++;
                kernel_size_in_bytes = 0;
                initialize_section();
            }

            start_in_bytes = start_in_bytes + kernel_size_in_bytes;

            kernel_bin.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t _, uint32_t len) {
                program_vector.insert(program_vector.end(), mem_ptr, mem_ptr + len);
                mem_ptr += len;
            });

            kernel_size_in_bytes = kernel_bin.size() * sizeof(u32);

            for (const CoreRange& core_range : cr_set.ranges()) {
                CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
                CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);

                u32 start_x = physical_start.x;
                u32 start_y = physical_start.y;
                u32 end_x = physical_end.x;
                u32 end_y = physical_end.y;
                u32 noc_multicast_encoding = NOC_MULTICAST_ENCODING(start_x, start_y, end_x, end_y);

                sections.at(current_section_idx)
                    .at(riscv_type)
                    .push_back(std::make_tuple(
                        start_in_bytes, kernel_size_in_bytes, noc_multicast_encoding, core_range.size()));
            }
            sections.at(current_section_idx).size_in_bytes += kernel_bin.size() * sizeof(u32);
        }

        TT_ASSERT(current_section_idx == 0, "Testing for just one section so far");

    };

    // TODO(agrebenisan): Once Almeet gets rid of kernel polymorphism,
    // need to come back and clean this up. Ideally this should be as
    // simple as just getting the type from the kernel.
    for (Kernel* kernel : program.kernels()) {
        char riscv_type;
        switch (kernel->kernel_type()) {
            case (KernelType::DataMovement): {
                auto dm_kernel = dynamic_cast<DataMovementKernel*>(kernel);
                switch (dm_kernel->data_movement_processor()) {
                    case (DataMovementProcessor::RISCV_0): riscv_type = 'B'; break;
                    case (DataMovementProcessor::RISCV_1): riscv_type = 'N'; break;
                    default: TT_THROW("Invalid kernel type");
                }
            } break;
            case (KernelType::Compute): riscv_type = 'C'; break;
            default: TT_THROW("Invalid kernel type");
        }

        write_to_program_device_map(riscv_type, kernel);
    }

    return program_to_device_map;
}

string EnqueueCommandTypeToString(EnqueueCommandType ctype) {
    switch (ctype) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER: return "EnqueueReadBuffer";
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: return "EnqueueWriteBuffer";
        default: TT_THROW("Invalid command type");
    }
}

u32 noc_coord_to_u32(CoreCoord coord) { return NOC_XY_ENCODING(NOC_X(coord.x), NOC_Y(coord.y)); }

// EnqueueReadBufferCommandSection
EnqueueReadBufferCommand::EnqueueReadBufferCommand(
    Device* device, Buffer& buffer, vector<u32>& dst, SystemMemoryWriter& writer) :
    dst(dst), writer(writer), buffer(buffer) {
    this->device = device;
}

const DeviceCommand EnqueueReadBufferCommand::device_command(u32 dst) {
    DeviceCommand command;
    command.set_data_size_in_bytes(this->buffer.size());

    u32 available_l1 = 1024 * 1024 - UNRESERVED_BASE;
    u32 potential_burst_size = available_l1;
    u32 num_bursts = this->buffer.size() / (available_l1);
    u32 num_pages_per_burst = potential_burst_size / this->buffer.page_size();
    u32 burst_size = num_pages_per_burst * this->buffer.page_size();
    u32 remainder_burst_size = this->buffer.size() - (num_bursts * burst_size);
    u32 num_pages_per_remainder_burst = remainder_burst_size / this->buffer.page_size();

    // Need to make a PCIE coordinate variable
    command.add_read_buffer_relay(
        dst,
        NOC_XY_ENCODING(NOC_X(0), NOC_Y(4)),
        this->buffer.address(),
        noc_coord_to_u32(this->buffer.noc_coordinates()),
        num_bursts,
        burst_size,
        num_pages_per_burst,
        this->buffer.page_size(),
        remainder_burst_size,
        num_pages_per_remainder_burst,
        (u32)(this->buffer.buffer_type()));

    return command;
}

void EnqueueReadBufferCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    u32 system_memory_temporary_storage_address = write_ptr + DeviceCommand::size_in_bytes();
    this->read_buffer_addr = system_memory_temporary_storage_address;
    const auto command_desc = this->device_command(system_memory_temporary_storage_address).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());
    u32 cmd_size = DeviceCommand::size_in_bytes() + this->buffer.size();

    // Change noc write name
    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_write(this->device, this->dst, system_memory_temporary_storage_address);

    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType EnqueueReadBufferCommand::type() { return this->type_; }

// EnqueueWriteBufferCommand section
EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
    Device* device, Buffer& buffer, vector<u32>& src, SystemMemoryWriter& writer) :
    writer(writer), src(src), buffer(buffer) {
    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to write to an invalid buffer");
    this->device = device;
}

const DeviceCommand EnqueueWriteBufferCommand::device_command(u32 src_address) {
    DeviceCommand command;

    // TT_ASSERT(this->buffer.size() % 32 == 0);
    command.set_data_size_in_bytes(this->buffer.size());

    u32 available_l1 = 1024 * 1024 - UNRESERVED_BASE;
    u32 potential_burst_size = available_l1;
    u32 num_bursts = this->buffer.size() / (available_l1);
    u32 num_pages_per_burst = potential_burst_size / this->buffer.page_size();
    u32 burst_size = num_pages_per_burst * this->buffer.page_size();
    u32 remainder_burst_size = this->buffer.size() - (num_bursts * burst_size);
    u32 num_pages_per_remainder_burst = remainder_burst_size / this->buffer.page_size();

    // Need to make a PCIE coordinate variable
    command.add_write_buffer_relay(
        src_address,
        NOC_XY_ENCODING(NOC_X(0), NOC_Y(4)),
        this->buffer.address(),
        noc_coord_to_u32(this->buffer.noc_coordinates()),
        num_bursts,
        burst_size,
        num_pages_per_burst,
        this->buffer.page_size(),
        remainder_burst_size,
        num_pages_per_remainder_burst,
        (u32)(this->buffer.buffer_type()));

    return command;
}

void EnqueueWriteBufferCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    u32 system_memory_temporary_storage_address = write_ptr + DeviceCommand::size_in_bytes();
    const auto command_desc = this->device_command(system_memory_temporary_storage_address).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());
    u32 cmd_size = DeviceCommand::size_in_bytes() + this->buffer.size();

    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_write(this->device, this->src, system_memory_temporary_storage_address);
    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType EnqueueWriteBufferCommand::type() { return this->type_; }

EnqueueProgramCommand::EnqueueProgramCommand(
    Device* device, Buffer& buffer, ProgramToDeviceMap& program_to_dev_map, SystemMemoryWriter& writer) :
    buffer(buffer), program_to_dev_map(program_to_dev_map), writer(writer) {
    this->device = device;
}

const DeviceCommand EnqueueProgramCommand::device_command(u32) {
    // if (this->command_cache.count(&this->program)) {
    //     return this->command_cache.at(&this->program);
    // }
    // this->command_cache.emplace(&this->program, command);
    DeviceCommand command;

    command.launch();

    u32 src = this->buffer.address();
    u32 src_noc = noc_coord_to_u32(this->buffer.noc_coordinates());
    for (const ProgramSection& section : this->program_to_dev_map.sections) {
        u32 transfer_size = section.size_in_bytes;
        vector<TrailingWriteCommand> write_commands;
        u32 dst_code_location;
        for (const auto& [riscv_type, transfer_info_vector] : section.section) {
            switch (riscv_type) {
                case 'B': dst_code_location = MEM_BRISC_FIRMWARE_BASE; break;
                case 'N': dst_code_location = MEM_NCRISC_INIT_IRAM_L1_BASE; break;
                case 'U': dst_code_location = MEM_TRISC0_BASE; break;
                case 'M': dst_code_location = MEM_TRISC1_BASE; break;
                case 'P': dst_code_location = MEM_TRISC2_BASE; break;
                default: TT_THROW("Invalid riscv type");
            }

            for (const transfer_info& transfer : transfer_info_vector) {
                TrailingWriteCommand trailing_write = {
                    .src = std::get<0>(transfer),  // Refactor to use methods to get relevant data from tuple
                    .dst = dst_code_location,
                    .dst_noc = std::get<2>(transfer),
                    .transfer_size = std::get<1>(transfer),
                    .num_receivers = std::get<3>(transfer)};
                write_commands.push_back(trailing_write);
            }
        }

        command.add_write_program_relay(src, src_noc, transfer_size, write_commands);
        tt::log_debug(tt::LogDispatch, "Testing");
        // u32 i = 0;
        // for (u32 el: command.get_desc()) {
        //     tt::log_debug(tt::LogDispatch, "El idx {}, el {}", i, el);
        //     i++;
        // }
    }

    return command;
}

void EnqueueProgramCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    u32 system_memory_temporary_storage_address = write_ptr + DeviceCommand::size_in_bytes();
    const DeviceCommand command = this->device_command(0);
    const auto command_desc = this->device_command(0).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());
    u32 cmd_size = DeviceCommand::size_in_bytes();

    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType EnqueueProgramCommand::type() { return this->type_; }

// FinishCommand section
FinishCommand::FinishCommand(Device* device, SystemMemoryWriter& writer) : writer(writer) { this->device = device; }

const DeviceCommand FinishCommand::device_command(u32) {
    DeviceCommand command;
    command.finish();

    return command;
}

void FinishCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    const auto command_desc = this->device_command(0).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());

    u32 cmd_size = DeviceCommand::size_in_bytes();

    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);

    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType FinishCommand::type() { return this->type_; }

// Sending dispatch kernel. TODO(agrebenisan): Needs a refactor
void send_dispatch_kernel_to_device(Device* device) {
    // Ideally, this should be some separate API easily accessible in
    // TT-metal, don't like the fact that I'm writing this from scratch
    std::string arch_name = tt::get_string_lowercase(device->arch());
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("unary", "command_queue");

    build_kernel_for_riscv_options.fp32_dest_acc_en = false;

    // Hard-coding as BRISC for now, could potentially be NCRISC
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/dispatch/command_queue.cpp";
    std::map<string, string> brisc_defines = {{"IS_DISPATCH_KERNEL", ""}, {"DEVICE_DISPATCH_MODE", ""}};
    build_kernel_for_riscv_options.brisc_defines = brisc_defines;
    bool profile = false;

    GenerateBankToNocCoordHeaders(device, &build_kernel_for_riscv_options, "command_queue");
    generate_binary_for_risc(
        RISCID::BR, &build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, arch_name, 0, {}, profile);

    // Currently hard-coded. TODO(agrebenisan): Once we add support for multiple dispatch cores, this can be refactored,
    // but don't yet have a plan for where this variable should exist.
    CoreCoord dispatch_core = {1, 11};
    tt::llrt::test_load_write_read_risc_binary(device->cluster(), "command_queue/brisc/brisc.hex", 0, dispatch_core, 0);

    // Deassert reset of dispatch core BRISC. TODO(agrebenisan): Refactor once Paul's changes in
    tt::llrt::internal_::setup_riscs_on_specified_core(
        device->cluster(), 0, tt::llrt::TensixRiscsOptions::BRISC_ONLY, {dispatch_core});
    device->cluster()->set_remote_tensix_risc_reset(tt_cxy_pair(0, dispatch_core), TENSIX_DEASSERT_SOFT_RESET);
}

// CommandQueue section
CommandQueue::CommandQueue(Device* device) {
    tt_start_debug_print_server(device->cluster(), {0}, {{1, 11}});

    send_dispatch_kernel_to_device(device);
    this->device = device;
}

CommandQueue::~CommandQueue() {
    this->finish();

    // For time being, asserting reset of the whole board. Will need
    // to rethink once we get to multiple command queues
    tt::llrt::assert_reset_for_all_chips(this->device->cluster());
}

void CommandQueue::enqueue_command(shared_ptr<Command> command, bool blocking) {
    // For the time-being, doing the actual work of enqueing in
    // the main thread.
    // TODO(agrebenisan): Perform the following in a worker thread
    command->process();

    if (blocking) {
        this->finish();
    }
}

void CommandQueue::enqueue_read_buffer(Buffer& buffer, vector<u32>& dst, bool blocking) {
    shared_ptr<EnqueueReadBufferCommand> command =
        std::make_shared<EnqueueReadBufferCommand>(this->device, buffer, dst, this->sysmem_writer);

    // TODO(agrebenisan): Provide support so that we can achieve non-blocking
    // For now, make read buffer blocking since after the
    // device moves data into the buffer we want to read out
    // of, we then need to consume it into a vector. This
    // is easiest way to bring this up
    TT_ASSERT(blocking, "EnqueueReadBuffer only has support for blocking mode currently");
    this->enqueue_command(command, blocking);

    this->device->cluster()->read_sysmem_vec(dst, command->read_buffer_addr, command->buffer.size(), 0);
}

void CommandQueue::enqueue_write_buffer(Buffer& buffer, vector<u32>& src, bool blocking) {
    TT_ASSERT(not blocking, "EnqueueWriteBuffer only has support for non-blocking mode currently");
    shared_ptr<EnqueueWriteBufferCommand> command =
        std::make_shared<EnqueueWriteBufferCommand>(this->device, buffer, src, this->sysmem_writer);
    this->enqueue_command(command, blocking);
}

void CommandQueue::enqueue_program(Program& program, bool blocking) {
    TT_ASSERT(not blocking, "EnqueueProgram only has support for non-blocking mode currently");

    // Need to relay the program into DRAM if this is the first time
    // we are seeing it
    static int channel_id = 0;  // Are there issues with this being static?
    if (not this->program_to_buffer.count(&program)) {
        ProgramToDeviceMap program_to_device_map = ConstructProgramToDeviceMap(this->device, program);

        vector<u32>& program_data = program_to_device_map.bins;
        u32 program_data_size_in_bytes = program_data.size() * sizeof(u32);
        unique_ptr<Buffer> program_buffer = std::make_unique<Buffer>(
            this->device, program_data_size_in_bytes, channel_id, program_data_size_in_bytes, BufferType::DRAM);

        tt::log_debug(tt::LogDispatch, "Program buffer size in B {}", program_data_size_in_bytes);

        this->enqueue_write_buffer(*program_buffer, program_data, blocking);
        channel_id =
            (channel_id + 1) % this->device->cluster()
                                   ->get_soc_desc(0)
                                   .dram_cores.size();  // TODO(agrebenisan): Pull in num DRAM banks from SOC descriptor

        // We need to hold onto this buffer so that the program doesn't get de-allocated
        this->program_to_buffer.emplace(&program, std::move(program_buffer));
        this->program_to_dev_map.emplace(&program, std::move(program_to_device_map));
    }

    shared_ptr<EnqueueProgramCommand> command = std::make_shared<EnqueueProgramCommand>(
        this->device,
        *this->program_to_buffer.at(&program),
        this->program_to_dev_map.at(&program),
        this->sysmem_writer);

    this->enqueue_command(command, blocking);
}

void CommandQueue::finish() {
    FinishCommand command(this->device, this->sysmem_writer);
    shared_ptr<FinishCommand> p = std::make_shared<FinishCommand>(std::move(command));
    this->enqueue_command(p, false);

    // We then poll to check that we're done.
    vector<u32> finish_vec;
    do {
        this->device->cluster()->read_sysmem_vec(finish_vec, HOST_CQ_FINISH_PTR, 4, 0);
    } while (finish_vec.at(0) != 1);

    // Reset this value to 0 before moving on
    finish_vec.at(0) = 0;
    this->device->cluster()->write_sysmem_vec(finish_vec, HOST_CQ_FINISH_PTR, 0);
}

// OpenCL-like APIs
void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& dst, bool blocking) {
    tt::log_debug(tt::LogDispatch, "EnqueueReadBuffer");

    TT_ASSERT(blocking, "Non-blocking EnqueueReadBuffer not yet supported");
    cq.enqueue_read_buffer(buffer, dst, blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& src, bool blocking) {
    tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer");

    cq.enqueue_write_buffer(buffer, src, blocking);
}

void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking) {
    tt::log_debug(tt::LogDispatch, "EnqueueProgram");

    cq.enqueue_program(program, blocking);
}

void Finish(CommandQueue& cq) {
    tt::log_debug(tt::LogDispatch, "Finish");

    cq.finish();
}
