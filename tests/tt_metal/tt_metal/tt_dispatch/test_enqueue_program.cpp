#include "frameworks/tt_dispatch/impl/command_queue.hpp"
#include "tt_metal/host_api.hpp"

#include "tests/tt_metal/llrt/test_libs/debug_mailbox.hpp"

using namespace tt;

u32 NUM_TILES = 2048;

void zero_out_sysmem(Device *device) {
    // Prior to running anything, need to clear out system memory
    // to prevent anything being stale. Potentially make it a static
    // method on command queue
    vector<u32> zeros(1024 * 1024 * 1024 / sizeof(u32), 0);
    device->cluster()->write_sysmem_vec(zeros, 0, 0);
}

tt_metal::Program generate_eltwise_binary_program(Device *device) {
    // TODO(agrebenisan): This is directly copy and pasted from test_eltwise_binary.
    // We need to think of a better way to generate test data, so this section needs to be heavily refactored.

    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = NUM_TILES;
    uint32_t dram_buffer_size =
        single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    uint32_t dram_buffer_src0_addr = 0;
    int dram_src0_channel_id = 0;
    uint32_t dram_buffer_src1_addr = 256 * 1024 * 1024;
    int dram_src1_channel_id = 1;
    uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024;  // 512 MB (upper half)
    int dram_dst_channel_id = 0;

    uint32_t page_size = single_tile_size;
    auto src0_dram_buffer = tt_metal::Buffer(
        device, dram_buffer_size, dram_buffer_src0_addr, dram_src0_channel_id, page_size, tt_metal::BufferType::DRAM);
    auto src1_dram_buffer = tt_metal::Buffer(
        device, dram_buffer_size, dram_buffer_src1_addr, dram_src1_channel_id, page_size, tt_metal::BufferType::DRAM);
    auto dst_dram_buffer = tt_metal::Buffer(
        device, dram_buffer_size, dram_buffer_dst_addr, dram_dst_channel_id, page_size, tt_metal::BufferType::DRAM);

    auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
    auto dram_src1_noc_xy = src1_dram_buffer.noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src0_cb_addr,
        tt::DataFormat::Float16_b);

    uint32_t src1_cb_index = 1;
    uint32_t src1_cb_addr = 300 * 1024;
    auto cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src1_cb_addr,
        tt::DataFormat::Float16_b);

    uint32_t ouput_cb_index = 16;  // output operands start at index 16
    uint32_t output_cb_addr = 400 * 1024;
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        tt::DataFormat::Float16_b);


    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    auto binary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_dual_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    vector<uint32_t> compute_kernel_args = {
        NUM_TILES,  // per_core_block_cnt
        1,     // per_core_block_size
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode);
    eltwise_binary_kernel->add_define("ELTWISE_OP", "add_tiles");
    eltwise_binary_kernel->add_define("ELTWISE_OP_CODE", "0");


    tt_metal::CompileProgram(device, program);
    return program;
}

void test_program_to_device_map() {
    /*
        This black-box test ensures that the program device map generated on host
        is correct.
    */
    int pci_express_slot = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

    tt_metal::InitializeDevice(device);
    tt_metal::Program program = generate_eltwise_binary_program(device);
    ProgramToDeviceMap prog_to_device_map = ConstructProgramToDeviceMap(device, program);

    // Get kernel group path
    CoreCoord core_coord = {0, 0};
    string bin_path = program.kernels().at(0)->binary_path(core_coord);

    u32 brisc_size  = tt::llrt::get_risc_binary(bin_path + "/brisc/brisc.hex").size() * sizeof(u32);
    u32 ncrisc_size = tt::llrt::get_risc_binary(bin_path + "/ncrisc/ncrisc.hex").size() * sizeof(u32);
    u32 trisc0_size = tt::llrt::get_risc_binary(bin_path + "/tensix_thread0/tensix_thread0.hex").size() * sizeof(u32);
    u32 trisc1_size = tt::llrt::get_risc_binary(bin_path + "/tensix_thread1/tensix_thread1.hex").size() * sizeof(u32);
    u32 trisc2_size = tt::llrt::get_risc_binary(bin_path + "/tensix_thread2/tensix_thread2.hex").size() * sizeof(u32);

    auto section = prog_to_device_map.sections.at(0);
    // Test that the code sizes on disk match the sizes seen in the prog to device map
    TT_ASSERT(std::get<1>(section.at('B').at(0)) == brisc_size);
    TT_ASSERT(std::get<1>(section.at('N').at(0)) == ncrisc_size);
    TT_ASSERT(std::get<1>(section.at('U').at(0)) == trisc0_size);
    TT_ASSERT(std::get<1>(section.at('M').at(0)) == trisc1_size);
    TT_ASSERT(std::get<1>(section.at('P').at(0)) == trisc2_size);

    tt::log_debug(tt::LogDispatch, "BRISC size {}", brisc_size);
    tt::log_debug(tt::LogDispatch, "NCRISC {}", ncrisc_size);
    tt::log_debug(tt::LogDispatch, "TRISC0 {}", trisc0_size);
    tt::log_debug(tt::LogDispatch, "TRISC1 {}", trisc1_size);
    tt::log_debug(tt::LogDispatch, "TRISC2 size {}", trisc2_size);
    tt::log_debug(tt::LogDispatch, "Total bin size {}", brisc_size + ncrisc_size + trisc0_size + trisc1_size + trisc2_size);

    vector<pair<u32, u32>> src_and_sizes;
    set<u32> noc_encodings;
    set<u32> num_receivers;
    for (const auto &[key, val]: section.section) {
        transfer_info tinfo = val.at(0);
        src_and_sizes.push_back(std::make_pair(std::get<0>(tinfo), std::get<1>(tinfo)));
        noc_encodings.insert(std::get<2>(tinfo));
        num_receivers.insert(std::get<3>(tinfo));
    }

    // Ensure that the srcs are contiguous in L1
    sort(src_and_sizes.begin(), src_and_sizes.end());
    for (int i = 0; i < src_and_sizes.size() - 1; i++) {
        TT_ASSERT(src_and_sizes.at(i).first + src_and_sizes.at(i).second == src_and_sizes.at(i + 1).first);
    }

    // Ensure that all of the kernels in program have the exact same multicast NOC encoding and
    // num receivers
    TT_ASSERT(noc_encodings.size() == 1);
    TT_ASSERT(num_receivers.size() == 1);
}

void test_enqueue_eltwise_binary_program() {
    int pci_express_slot = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

    tt_metal::InitializeDevice(device);

    zero_out_sysmem(device);
    tt_metal::Program program = generate_eltwise_binary_program(device);
    {
        CommandQueue cq(device);

        // Enqueue program inputs
        vector<u32> inpa(NUM_TILES, 0x40000000); // 2 in float
        vector<u32> inpb(NUM_TILES, 0x40800000); // 4 in float
        Buffer bufa(device, NUM_TILES * 2048, 0, 2048, BufferType::DRAM);
        Buffer bufb(device, NUM_TILES * 2048, 0, 2048, BufferType::DRAM);
        // EnqueueWriteBuffer(cq, bufa, inpa, false);
        // EnqueueWriteBuffer(cq, bufb, inpb, false);
        EnqueueProgram(cq, program, false);

        Buffer out(device, NUM_TILES * 2048, 0, 2048, BufferType::DRAM);
        vector<u32> out_vec;

        read_trisc_debug_mailbox(device->cluster(), 0, {1, 11}, 0);

        Finish(cq);
        // EnqueueReadBuffer(cq, out, out_vec, true);
        vector<u32> golden_out(NUM_TILES, 0x40C00000); // 6 in float
        TT_ASSERT(out_vec == golden_out);
    }
}

int main() {
    // test_program_to_device_map();
    test_enqueue_eltwise_binary_program();
}
