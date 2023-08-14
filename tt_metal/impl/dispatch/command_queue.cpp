// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug_tools.hpp"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/llrt/tt_debug_print_server.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace tt::tt_metal {

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

u64 get_noc_multicast_encoding(const CoreCoord& top_left, const CoreCoord& bottom_right) {
    return NOC_MULTICAST_ENCODING(top_left.x, top_left.y, bottom_right.x, bottom_right.y);
}

u32 align(u32 addr, u32 alignment) { return ((addr - 1) | (alignment - 1)) + 1; }

u32 noc_coord_to_u32(CoreCoord coord) { return NOC_XY_ENCODING(NOC_X(coord.x), NOC_Y(coord.y)); }

ProgramMap ConstructProgramMap(const Device* device, Program& program) {
    /*
        TODO(agrebenisan): Move this logic to compile program
    */
    vector<pair<u32, u32>> multicast_message_noc_coords;
    vector<u32> program_pages;
    vector<transfer_info> program_page_transfers;
    vector<transfer_info> runtime_arg_transfers;
    vector<u32> num_transfers_in_program_page;
    vector<u32> num_transfers_in_runtime_arg_page;
    u32 idx = 0;
    u32 num_transfers_in_page_counter = 0;

    auto update_program_pages = [&program_pages, &idx](
        u32 num_bytes, const vector<u32>::const_iterator& data) {

        u32 num_u32s = num_bytes / sizeof(u32);

        if (idx + num_u32s > program_pages.size()) {
            u32 num_bytes_to_reserve = align(num_bytes, PROGRAM_PAGE_SIZE);
            u32 old_size = program_pages.size();
            program_pages.resize(program_pages.size() + num_bytes_to_reserve / sizeof(u32));
        }

        // Need to ensure that the binaries are 16B aligned
        std::copy(data, data + num_u32s, program_pages.begin() + idx);

        // std::cout << "Span" << std::endl;
        // for (u32 i = idx; i < idx + num_u32s; i++) {
        //     std::cout << program_pages[i] << std::endl;
        // }
        const vector<u32> padding(align(num_u32s, 16 / sizeof(u32)) - num_u32s, 0);
        std::copy(padding.begin(), padding.end(), program_pages.begin() + idx + num_u32s);
    };

    auto advance_idx = [&idx](u32 num_bytes) {
        idx = align(idx + num_bytes / sizeof(u32), 16 / sizeof(u32));
    };

    auto update_program_page_transfers = [&num_transfers_in_page_counter, &idx, &advance_idx](
        u32 num_bytes, u32 dst, vector<transfer_info>& transfers, vector<u32>& num_transfers, const vector<pair<u32, u32>>& dst_noc_multicast_info, bool advance) {

        u32 transfer_info_dst = dst;
        while (num_bytes) {
            u32 cur_space_in_page = PROGRAM_PAGE_SIZE - ((idx * sizeof(u32)) % PROGRAM_PAGE_SIZE);
            u32 transfer_info_num_bytes = std::min(cur_space_in_page, num_bytes);

            if (advance) {
                advance_idx(transfer_info_num_bytes);
            }

            for (const auto& [dst_noc_multicast_encoding, num_receivers]: dst_noc_multicast_info) {
                transfers.push_back(std::make_tuple(transfer_info_num_bytes, transfer_info_dst, dst_noc_multicast_encoding, num_receivers));
                num_transfers_in_page_counter++;
            }

            transfer_info_dst += transfer_info_num_bytes;
            num_bytes -= transfer_info_num_bytes;

            if (((idx * sizeof(u32)) % PROGRAM_PAGE_SIZE) == 0) {
                num_transfers.push_back(num_transfers_in_page_counter);
                num_transfers_in_page_counter = 0;
            }
        }
    };


    auto extract_dst_noc_multicast_info = [&device](const set<CoreRange>& ranges) -> vector<pair<u32, u32>> {
        vector<pair<u32, u32>> dst_noc_multicast_info;
        for (const CoreRange& core_range: ranges) {
            CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
            CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);
            u32 dst_noc_multicast_encoding = get_noc_multicast_encoding(physical_start, physical_end);
            u32 num_receivers = core_range.size();
            dst_noc_multicast_info.push_back(std::make_pair(dst_noc_multicast_encoding, num_receivers));
        }
        return dst_noc_multicast_info;
    };

    // Step 1: Get the locations of the worker cores and how many worker cores there are in this program
    for (const CoreRange& core_range: program.get_worker_core_range_set().ranges()) {
        CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
        CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);
        multicast_message_noc_coords.push_back(
            std::make_pair(get_noc_multicast_encoding(physical_start, physical_end), core_range.size()));
    }

    static map<RISCV, u32> processor_to_local_mem_addr = {
        {RISCV::BRISC, MEM_BRISC_INIT_LOCAL_L1_BASE},
        {RISCV::NCRISC, MEM_NCRISC_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC0, MEM_TRISC0_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC1, MEM_TRISC1_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC2, MEM_TRISC2_INIT_LOCAL_L1_BASE}
    };

    // Step 2: Construct pages for kernel binaries
    for (KernelID kernel_id: program.kernel_ids()) {
        const Kernel* kernel = detail::GetKernel(program, kernel_id);
        vector<pair<u32, u32>> dst_noc_multicast_info = extract_dst_noc_multicast_info(kernel->core_range_set().ranges());
        for (const ll_api::memory& kernel_bin: kernel->binaries()) {
            kernel_bin.process_spans([&](vector<u32>::const_iterator mem_ptr, u64 dst, u32 len) {
                u32 num_bytes = len * sizeof(u32);
                update_program_pages(num_bytes, mem_ptr);

                if ((dst & MEM_LOCAL_BASE) == MEM_LOCAL_BASE) {
                    dst = (dst & ~MEM_LOCAL_BASE) + processor_to_local_mem_addr.at(kernel->processor());
                } else if ((dst & MEM_NCRISC_IRAM_BASE) == MEM_NCRISC_IRAM_BASE) {
                    dst = (dst & ~MEM_NCRISC_IRAM_BASE) + MEM_NCRISC_INIT_IRAM_L1_BASE;
                }

                update_program_page_transfers(num_bytes, dst, program_page_transfers, num_transfers_in_program_page, dst_noc_multicast_info, true);
            });
        }
    }

    // Step 3: Continue constructing pages for circular buffer configs
    for (const CircularBuffer& cb: program.circular_buffers()) {
        vector<u32> cb_vector = {cb.address() >> 4, cb.size() >> 4, cb.num_tiles(), (cb.size() / cb.num_tiles()) >> 4};
        vector<pair<u32, u32>> dst_noc_multicast_info = extract_dst_noc_multicast_info(cb.core_range_set().ranges());
        u32 num_bytes = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(u32);
        update_program_pages(num_bytes, cb_vector.begin());
        for (const auto buffer_index: cb.buffer_indices()) {
            update_program_page_transfers(
                num_bytes, CIRCULAR_BUFFER_CONFIG_BASE + buffer_index * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(u32), program_page_transfers, num_transfers_in_program_page, dst_noc_multicast_info, false);
        }
        advance_idx(num_bytes);
    }

    // Step 4: Continue constructing pages for semaphore configs
    for (const Semaphore& semaphore: program.semaphores()) {
        vector<u32> semaphore_vector = {semaphore.initial_value(), 0, 0, 0};
        vector<pair<u32, u32>> dst_noc_multicast_info = extract_dst_noc_multicast_info(semaphore.core_range_set().ranges());
        u32 num_bytes = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(u32);
        update_program_pages(num_bytes, semaphore_vector.begin());
        update_program_page_transfers(num_bytes, semaphore.address(), program_page_transfers, num_transfers_in_program_page, dst_noc_multicast_info, false);
        advance_idx(num_bytes);
    }

    if (num_transfers_in_page_counter) {
        num_transfers_in_program_page.push_back(num_transfers_in_page_counter);
        u32 cur_space_in_page = PROGRAM_PAGE_SIZE - ((idx * sizeof(u32)) % PROGRAM_PAGE_SIZE);
        // Runtime arguments begin in a new page
        advance_idx(cur_space_in_page / sizeof(u32));
    }

    // Step 5: Get transfer info for runtime args
    for (const auto kernel_id : program.kernel_ids()) {
        const Kernel* kernel = detail::GetKernel(program, kernel_id);
        u32 dst;
        switch (kernel->processor()) {
            case RISCV::NCRISC: {
                dst = NCRISC_L1_ARG_BASE;
            }; break;
            case RISCV::BRISC: {
                dst = BRISC_L1_ARG_BASE;
            } break;
            default: continue; // So far, only data movement kernels have runtime args
        }
        for (const auto& [core_coord, runtime_args]: kernel->runtime_args()) {
            u32 num_bytes = runtime_args.size() * sizeof(u32);
            u32 dst_noc = noc_coord_to_u32(core_coord);

            // Only one receiver per set of runtime arguments
            update_program_page_transfers(num_bytes, dst, runtime_arg_transfers, num_transfers_in_runtime_arg_page, {{dst_noc, 1}}, true);
        }
    }

    // for (u32 el: program_pages) {
    //     std::cout << el << std::endl;
    // }

    // u32 debug_idx = 0;
    // for (const auto& [num_bytes, dst, dst_noc, num_recv]: program_page_transfers) {
    //     u32 num_u32s = num_bytes / sizeof(u32);
    //     std::cout << "Transfer" << std::endl;
    //     for (u32 i = debug_idx; i < debug_idx + num_u32s; i++) {
    //         std::cout << program_pages[i] << std::endl;
    //     }
    //     debug_idx = align(debug_idx + num_u32s, 4);
    // }

    return {
        .num_workers = u32(program.logical_cores().size()),
        .multicast_message_noc_coords = std::move(multicast_message_noc_coords),
        .program_pages = std::move(program_pages),
        .program_page_transfers = std::move(program_page_transfers),
        .runtime_arg_transfers = std::move(runtime_arg_transfers),
        .num_transfers_in_program_page = std::move(num_transfers_in_program_page),
        .num_transfers_in_runtime_arg_page = std::move(num_transfers_in_runtime_arg_page),
    };
}

// EnqueueReadBufferCommandSection
EnqueueReadBufferCommand::EnqueueReadBufferCommand(
    Device* device, Buffer& buffer, vector<u32>& dst, SystemMemoryWriter& writer) :
    dst(dst), writer(writer), buffer(buffer) {
    this->device = device;
}

const DeviceCommand EnqueueReadBufferCommand::assemble_device_command(u32 dst_address) {
    DeviceCommand command;

    u32 num_pages = this->buffer.size() / this->buffer.page_size();
    u32 padded_page_size = align(this->buffer.page_size(), 32);
    u32 data_size_in_bytes = padded_page_size * num_pages;
    command.set_data_size_in_bytes(data_size_in_bytes);

    u32 available_l1 = MEM_L1_SIZE - DEVICE_COMMAND_DATA_ADDR;
    u32 burst_size = (available_l1 / padded_page_size) * padded_page_size;

    vector<CoreCoord> pcie_cores = this->device->cluster()->get_soc_desc(this->device->pcie_slot()).pcie_cores;
    TT_ASSERT(pcie_cores.size() == 1, "Should only have one pcie core");
    const CoreCoord& pcie_core = pcie_cores.at(0);

    command.add_read_buffer_instruction(
        dst_address,
        NOC_XY_ENCODING(pcie_core.x, pcie_core.y),
        this->buffer.address(),

        data_size_in_bytes,
        burst_size,
        this->buffer.page_size(),
        padded_page_size,
        (u32) this->buffer.buffer_type());

    return command;
}

void EnqueueReadBufferCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    u32 system_memory_temporary_storage_address = write_ptr + DeviceCommand::size_in_bytes();
    this->read_buffer_addr = system_memory_temporary_storage_address;
    const auto command_desc = this->assemble_device_command(system_memory_temporary_storage_address).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());

    u32 num_pages = this->buffer.size() / this->buffer.page_size();
    u32 padded_page_size = align(this->buffer.page_size(), 32);
    u32 data_size_in_bytes = padded_page_size * num_pages;
    u32 cmd_size = DeviceCommand::size_in_bytes() + data_size_in_bytes;

    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
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

const DeviceCommand EnqueueWriteBufferCommand::assemble_device_command(u32 src_address) {
    DeviceCommand command;

    u32 num_pages = this->buffer.size() / this->buffer.page_size();
    u32 padded_page_size = this->buffer.page_size();
    if (this->buffer.page_size() != this->buffer.size()) {
        padded_page_size = align(this->buffer.page_size(), 32);
    }
    u32 data_size_in_bytes = padded_page_size * num_pages;
    command.set_data_size_in_bytes(data_size_in_bytes);

    u32 available_l1 = MEM_L1_SIZE - DEVICE_COMMAND_DATA_ADDR;
    u32 burst_size = (available_l1 / padded_page_size) * padded_page_size;

    vector<CoreCoord> pcie_cores = this->device->cluster()->get_soc_desc(this->device->pcie_slot()).pcie_cores;
    TT_ASSERT(pcie_cores.size() == 1, "Should only have one pcie core");
    const CoreCoord& pcie_core = pcie_cores.at(0);

    command.add_write_buffer_instruction(
        src_address,
        NOC_XY_ENCODING(pcie_core.x, pcie_core.y),
        this->buffer.address(),

        data_size_in_bytes,
        burst_size,
        this->buffer.page_size(),
        padded_page_size,
        (u32)(this->buffer.buffer_type()));

    return command;
}

void EnqueueWriteBufferCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    u32 system_memory_temporary_storage_address = write_ptr + DeviceCommand::size_in_bytes();

    const auto command_desc = this->assemble_device_command(system_memory_temporary_storage_address).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());

    u32 num_pages = this->buffer.size() / this->buffer.page_size();

    u32 padded_page_size = this->buffer.page_size();
    if (this->buffer.page_size() != this->buffer.size()) {
        padded_page_size = align(this->buffer.page_size(), 32);
    }

    u32 data_size_in_bytes = padded_page_size * num_pages;

    u32 cmd_size = DeviceCommand::size_in_bytes() + data_size_in_bytes;
    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);

    // Need to deal with the edge case where our page
    // size is not 32B aligned
    if (this->buffer.page_size() % 32 != 0 and this->buffer.page_size() != this->buffer.size()) {
        vector<u32>::const_iterator src_iterator = this->src.begin();
        u32 num_u32s_in_page = this->buffer.page_size() / sizeof(u32);
        u32 num_pages = this->buffer.size() / this->buffer.page_size();
        u32 dst = system_memory_temporary_storage_address;
        for (u32 i = 0; i < num_pages; i++) {
            vector<u32> src_page(src_iterator, src_iterator + num_u32s_in_page);
            this->writer.cq_write(this->device, src_page, dst);
            src_iterator += num_u32s_in_page;
            dst = align(dst + this->buffer.page_size(), 32);
        }
    } else {
        this->writer.cq_write(this->device, this->src, system_memory_temporary_storage_address);
    }

    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType EnqueueWriteBufferCommand::type() { return this->type_; }

EnqueueProgramCommand::EnqueueProgramCommand(
    Device* device,
    Buffer& buffer,
    ProgramMap& program_to_dev_map,
    SystemMemoryWriter& writer,
    vector<u32>& runtime_args) :
    buffer(buffer), program_to_dev_map(program_to_dev_map), writer(writer), runtime_args(runtime_args) {
    this->device = device;
}

const DeviceCommand EnqueueProgramCommand::assemble_device_command(u32 runtime_args_src) {
    DeviceCommand command;
    command.set_num_workers(this->program_to_dev_map.num_workers);
    command.set_num_multicast_messages(this->program_to_dev_map.multicast_message_noc_coords.size());

    // Set the noc coords for all the worker cores
    for (const auto& [multicast_message_noc_coord, num_messages] : this->program_to_dev_map.multicast_message_noc_coords) {
        command.set_multicast_message_noc_coord(multicast_message_noc_coord, num_messages);
    }

    u32 program_src = this->buffer.address();
    u32 program_src_noc = noc_coord_to_u32(this->buffer.noc_coordinates());

    auto populate_program_data_transfer_instructions = [&command](
        u32 src_buffer_type, u32 base_address, const vector<u32>& num_transfers_in_pages, const vector<transfer_info>& transfers_in_pages) {
        command.write_program_entry(src_buffer_type);
        command.write_program_entry(num_transfers_in_pages.size());
        command.write_program_entry(base_address);
        u32 i = 0;
        for (u32 j = 0; j < num_transfers_in_pages.size(); j++) {
            u32 num_transfers_in_page = num_transfers_in_pages[j];
            command.write_program_entry(num_transfers_in_page);
            for (u32 k = 0; k < num_transfers_in_page; k++) {
                const auto [num_bytes, dst, dst_noc, num_receivers] = transfers_in_pages[i];
                command.add_write_page_partial_instruction(num_bytes, dst, dst_noc, num_receivers);
                i++;
            }
        }
    };

    /*
    the loop we are programming has the following structure
    for i in range(num_src_nocs):
        for j in range(num_pages)
             for k in range(num_transfers_in_page):
                 transfer instruction
    */
    const u32 num_program_srcs = 2; // One src noc for program binaries, one for runtime args in system memory
    command.set_num_program_srcs(num_program_srcs);
    populate_program_data_transfer_instructions(u32(BufferType::DRAM), this->buffer.address(), this->program_to_dev_map.num_transfers_in_program_page, this->program_to_dev_map.program_page_transfers);
    populate_program_data_transfer_instructions(u32(BufferType::SYSTEM_MEMORY), runtime_args_src, this->program_to_dev_map.num_transfers_in_runtime_arg_page, this->program_to_dev_map.runtime_arg_transfers);

    return command;
}

void EnqueueProgramCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    u32 system_memory_temporary_storage_address = write_ptr + DeviceCommand::size_in_bytes();

    const DeviceCommand cmd = this->assemble_device_command(system_memory_temporary_storage_address);
    const auto command_desc = cmd.get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());
    const u32 cmd_size = DeviceCommand::size_in_bytes() + cmd.get_data_size_in_bytes();

    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_write(this->device, this->runtime_args, system_memory_temporary_storage_address);
    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType EnqueueProgramCommand::type() { return this->type_; }

// FinishCommand section
FinishCommand::FinishCommand(Device* device, SystemMemoryWriter& writer) : writer(writer) { this->device = device; }

const DeviceCommand FinishCommand::assemble_device_command(u32) {
    DeviceCommand command;
    command.finish();
    return command;
}

void FinishCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    const auto command_desc = this->assemble_device_command(0).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());

    u32 cmd_size = DeviceCommand::size_in_bytes();

    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType FinishCommand::type() { return this->type_; }

// EnqueueWrapCommand section
EnqueueWrapCommand::EnqueueWrapCommand(Device* device, SystemMemoryWriter& writer) : writer(writer) {
    this->device = device;
}

const DeviceCommand EnqueueWrapCommand::assemble_device_command(u32) {
    DeviceCommand command;
    return command;
}

void EnqueueWrapCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;

    u32 space_left = HUGE_PAGE_SIZE - write_ptr;

    // Since all of the values will be 0, this will be equivalent to
    // a bunch of NOPs
    vector<u32> command_vector(space_left / sizeof(u32), 0);
    command_vector.at(0) = 1;  // wrap

    this->writer.cq_reserve_back(this->device, space_left);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_push_back(this->device, space_left);
}

EnqueueCommandType EnqueueWrapCommand::type() { return this->type_; }

// Sending dispatch kernel. TODO(agrebenisan): Needs a refactor
void send_dispatch_kernel_to_device(Device* device) {
    // Ideally, this should be some separate API easily accessible in
    // TT-metal, don't like the fact that I'm writing this from scratch

    Program dispatch_program = Program();
    CoreCoord dispatch_logical_core = {0, 9};
    vector<CoreCoord> pcie_cores = device->cluster()->get_soc_desc(device->pcie_slot()).pcie_cores;
    TT_ASSERT(pcie_cores.size() == 1, "Should only have one pcie core");
    std::map<string, string> dispatch_defines = {
        {"IS_DISPATCH_KERNEL", ""},
        {"PCIE_NOC_X", std::to_string(pcie_cores[0].x)},
        {"PCIE_NOC_Y", std::to_string(pcie_cores[0].y)},
    };
    auto dispatch_kernel = tt::tt_metal::CreateDataMovementKernel(
        dispatch_program,
        "tt_metal/impl/dispatch/kernels/command_queue.cpp",
        dispatch_logical_core,
        tt::tt_metal::DataMovementConfig{.processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default, .defines = dispatch_defines}
    );

    CompileProgram(device, dispatch_program);
    tt::tt_metal::ConfigureDeviceWithProgram(device, dispatch_program);

    u32 fifo_addr = (HOST_CQ_FINISH_PTR + 32) >> 4;
    vector<u32> fifo_addr_vector = {fifo_addr};
    tt::tt_metal::detail::WriteToDeviceL1(device, dispatch_logical_core, CQ_READ_PTR, fifo_addr_vector);
    tt::tt_metal::detail::WriteToDeviceL1(device, dispatch_logical_core, CQ_WRITE_PTR, fifo_addr_vector);

    // Initialize wr toggle
    vector<u32> toggle_start_vector = {0};
    tt::tt_metal::detail::WriteToDeviceL1(device, dispatch_logical_core, CQ_READ_TOGGLE, toggle_start_vector);
    tt::tt_metal::detail::WriteToDeviceL1(device, dispatch_logical_core, CQ_WRITE_TOGGLE, toggle_start_vector);

    tt::llrt::internal_::setup_riscs_on_specified_core(
        device->cluster(), 0, tt::llrt::TensixRiscsOptions::BRISC_ONLY, {device->worker_core_from_logical_core(dispatch_logical_core)});
    device->cluster()->set_remote_tensix_risc_reset(tt_cxy_pair(0, device->worker_core_from_logical_core(dispatch_logical_core)), TENSIX_DEASSERT_SOFT_RESET);
}

// CommandQueue section
CommandQueue::CommandQueue(Device* device) {
    vector<u32> pointers(CQ_START / sizeof(u32), 0);
    pointers[0] = CQ_START >> 4;  // rd ptr (96 >> 4 = 6)

    device->cluster()->write_sysmem_vec(pointers, 0, 0);

    tt_start_debug_print_server(device->cluster(), {0}, {{1, 11}, {1, 1}}, DPRINT_HART_BR);
    send_dispatch_kernel_to_device(device);
    this->device = device;
}

CommandQueue::~CommandQueue() {
    // if (this->device->cluster_is_initialized()) {
    //     this->finish();
    // }
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
    ZoneScopedN("CommandQueue_read_buffer");
    u32 read_buffer_command_size = DeviceCommand::size_in_bytes() + buffer.size();
    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + read_buffer_command_size >= HUGE_PAGE_SIZE) {
        tt::log_assert(read_buffer_command_size <= HUGE_PAGE_SIZE - 96, "EnqueueReadBuffer command is too large");
        this->wrap();
    }
    tt::log_debug(tt::LogDispatch, "EnqueueReadBuffer");

    shared_ptr<EnqueueReadBufferCommand> command =
        std::make_shared<EnqueueReadBufferCommand>(this->device, buffer, dst, this->sysmem_writer);

    // TODO(agrebenisan): Provide support so that we can achieve non-blocking
    // For now, make read buffer blocking since after the
    // device moves data into the buffer we want to read out
    // of, we then need to consume it into a vector. This
    // is easiest way to bring this up
    TT_ASSERT(blocking, "EnqueueReadBuffer only has support for blocking mode currently");
    this->enqueue_command(command, blocking);

    u32 num_pages = buffer.size() / buffer.page_size();
    u32 padded_page_size = align(buffer.page_size(), 32);
    u32 data_size_in_bytes = padded_page_size * num_pages;

    this->device->cluster()->read_sysmem_vec(dst, command->read_buffer_addr, data_size_in_bytes, 0);

    // This vector is potentially padded due to alignment constraints, so need to now remove the padding
    if ((buffer.page_size() % 32) != 0) {
        vector<u32> new_dst(buffer.size() / sizeof(u32), 0);
        u32 padded_page_size_in_u32s = align(buffer.page_size(), 32) / sizeof(u32);
        u32 new_dst_counter = 0;
        for (u32 i = 0; i < dst.size(); i += padded_page_size_in_u32s) {
            for (u32 j = 0; j < buffer.page_size() / sizeof(u32); j++) {
                new_dst[new_dst_counter] = dst[i + j];
                new_dst_counter++;
            }
        }
        dst = new_dst;
    }
}

void CommandQueue::enqueue_write_buffer(Buffer& buffer, vector<u32>& src, bool blocking) {
    ZoneScopedN("CommandQueue_write_buffer");
    TT_ASSERT(not blocking, "EnqueueWriteBuffer only has support for non-blocking mode currently");
    uint32_t src_size_bytes = src.size() * sizeof(uint32_t);
    TT_ASSERT(
        src_size_bytes <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer", src_size_bytes, buffer.size());
    TT_ASSERT(
        buffer.page_size() < MEM_L1_SIZE - DEVICE_COMMAND_DATA_ADDR,
        "Buffer pages must fit within the command queue data section");

    u32 write_buffer_command_size = DeviceCommand::size_in_bytes() + buffer.size();
    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + write_buffer_command_size >= HUGE_PAGE_SIZE) {
        tt::log_assert(
            write_buffer_command_size <= HUGE_PAGE_SIZE - 96,
            "EnqueueWriteBuffer command is too large: {}",
            write_buffer_command_size);
        this->wrap();
    }
    tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer");

    shared_ptr<EnqueueWriteBufferCommand> command =
        std::make_shared<EnqueueWriteBufferCommand>(this->device, buffer, src, this->sysmem_writer);
    this->enqueue_command(command, blocking);
}

void CommandQueue::enqueue_program(Program& program, bool blocking) {
    ZoneScopedN("CommandQueue_enqueue_program");
    TT_ASSERT(not blocking, "EnqueueProgram only has support for non-blocking mode currently");

    // Need to relay the program into DRAM if this is the first time
    // we are seeing it
    const u64 program_id = program.get_id();
    if (not this->program_to_buffer.count(program_id)) {
        ProgramMap program_to_device_map = ConstructProgramMap(this->device, program);

        vector<u32>& program_pages = program_to_device_map.program_pages;
        u32 program_data_size_in_bytes = program_pages.size() * sizeof(u32);

        u32 write_buffer_command_size = DeviceCommand::size_in_bytes() + program_data_size_in_bytes;

        this->program_to_buffer.emplace(
            program_id,
            std::make_unique<Buffer>(
                this->device, program_data_size_in_bytes, PROGRAM_PAGE_SIZE, BufferType::DRAM));

        // tt::log_info("PROGRAM DATA SIZE BYTES {}", program_data_size_in_bytes);

        this->enqueue_write_buffer(*this->program_to_buffer.at(program_id), program_pages, blocking);

        this->program_to_dev_map.emplace(program_id, std::move(program_to_device_map));
    }
    tt::log_debug(tt::LogDispatch, "EnqueueProgram");

    vector<u32> runtime_args; // TODO(agrebenisan): Should pre-reserve space I need
    for (const auto kernel_id: program.kernel_ids()) {
        const Kernel* kernel = detail::GetKernel(program, kernel_id);
        for (const auto& [_, core_runtime_args]: kernel->runtime_args()) {
            runtime_args.insert(runtime_args.end(), core_runtime_args.begin(), core_runtime_args.end());
            const vector<u32> padding(align(runtime_args.size(), 16 / sizeof(u32)), 0);
            runtime_args.insert(runtime_args.end(), padding.begin(), padding.end());
        }
    }

    u32 runtime_args_and_device_command_size = DeviceCommand::size_in_bytes() + (runtime_args.size() * sizeof(u32));
    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + runtime_args_and_device_command_size >=
        HUGE_PAGE_SIZE) {
        tt::log_assert(
            runtime_args_and_device_command_size <= HUGE_PAGE_SIZE - 96, "EnqueueProgram command size too large");
        this->wrap();
    }

    shared_ptr<EnqueueProgramCommand> command = std::make_shared<EnqueueProgramCommand>(
        this->device,
        *this->program_to_buffer.at(program_id),
        this->program_to_dev_map.at(program_id),
        this->sysmem_writer,
        runtime_args);

    this->enqueue_command(command, blocking);
}

void CommandQueue::finish() {
    ZoneScopedN("CommandQueue_finish");
    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + DeviceCommand::size_in_bytes() >= HUGE_PAGE_SIZE) {
        this->wrap();
    }
    tt::log_debug(tt::LogDispatch, "Finish");

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

void CommandQueue::wrap() {
    ZoneScopedN("CommandQueue_wrap");
    tt::log_debug(tt::LogDispatch, "EnqueueWrap");
    EnqueueWrapCommand command(this->device, this->sysmem_writer);
    shared_ptr<EnqueueWrapCommand> p = std::make_shared<EnqueueWrapCommand>(std::move(command));
    this->enqueue_command(p, false);
}

// OpenCL-like APIs
void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& dst, bool blocking) {
    TT_ASSERT(blocking, "Non-blocking EnqueueReadBuffer not yet supported");
    cq.enqueue_read_buffer(buffer, dst, blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& src, bool blocking) {
    cq.enqueue_write_buffer(buffer, src, blocking);
}

void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking) {
    detail::ValidateCircularBufferRegion(program, cq.device);
    cq.enqueue_program(program, blocking);
}

void Finish(CommandQueue& cq) { cq.finish(); }

} // namespace tt::tt_metal
