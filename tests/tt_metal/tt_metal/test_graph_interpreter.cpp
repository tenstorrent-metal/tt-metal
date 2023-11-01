// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tests_common/sfpu_helper/sfpu_helper.hpp"
#include "llrt/tt_debug_print_server.hpp"
#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "common/utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

// Used for compiling blank kernel before setting environment variable
namespace blank {
struct hlk_args_t {
    std::int32_t dummy;
};
}
void run_compile_blank(tt_metal::Device *device) {

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options(device->id(), "blank_op");

    log_info(tt::LogBuildKernels, "Compiling OP: {}", build_kernel_for_riscv_options.name);

    void *hlk_args = new blank::hlk_args_t{
        .dummy = 0,
    };
    build_kernel_for_riscv_options.set_hlk_args_all_cores(hlk_args, sizeof(blank::hlk_args_t));
    build_kernel_for_riscv_options.set_hlk_file_name_all_cores("tt_metal/kernels/compute/blank.cpp");
    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";

    generate_binaries_params_t params;
    tt_metal::detail::GenerateDeviceHeaders(device, &build_kernel_for_riscv_options, build_kernel_for_riscv_options.name);
    generate_binaries_all_riscs(&build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, "grayskull", params);
}

const map<string, tt::OpCode> sfpu_name_to_op_code = {
    {"exponential", tt::OpCode::Exponential},
    {"reciprocal",  tt::OpCode::Reciprocal},
    {"gelu",        tt::OpCode::Gelu}
};

const map<string, tt::OpCode> binary_name_to_op_code = {
    {"add", tt::OpCode::Add},
    {"subtract", tt::OpCode::Subtract},
    {"multiply", tt::OpCode::Multiply}
};

float add(float x, float y) {
    return x + y;
}
float subtract(float x, float y) {
    return x - y;
}
float multiply(float x, float y) {
    return x * y;
}

vector<uint32_t> eltwise_binary(const vector<uint32_t> &srcA, const vector<uint32_t> srcB, std::function<float(float, float)> binary_func) {
    vector<uint32_t> dst;
    TT_ASSERT(srcA.size() == srcB.size(), "Vectors being added need to have the same size");

    for (int i = 0; i < srcA.size(); i++) {
        uint32_t elA = srcA.at(i);
        uint32_t topA = elA & 0xffff0000;
        uint32_t bottomA = elA << 16;
        float topA_ = *reinterpret_cast<float*>(&topA);
        float bottomA_ = *reinterpret_cast<float*>(&bottomA);

        uint32_t elB = srcB.at(i);
        uint32_t topB = elB & 0xffff0000;
        uint32_t bottomB = elB << 16;
        float topB_ = *reinterpret_cast<float*>(&topB);
        float bottomB_ = *reinterpret_cast<float*>(&bottomB);

        float top = binary_func(topA_, topB_);
        float bottom = binary_func(bottomA_, bottomB_);

        bfloat16 bfp16_top = bfloat16(top);
        bfloat16 bfp16_bottom = bfloat16(bottom);

        uint32_t new_val = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfp16_bottom, bfp16_top));
        dst.push_back(new_val);
    }
    return dst;
}

const map<string, std::function<float(float, float)>> binary_op_to_function = {
    {"add", add},
    {"subtract", subtract},
    {"multiply", multiply}
};



// This test just runs a chain of eltwise unary sfpu ops, and it's a good baseline test for the
// graph interpreter since there is no branching
bool run_chained_sfpu_test(int chain_length) {

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    tt::log_assert(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    TT_ASSERT(chain_length > 0 && chain_length <= 10, "Cannot have a graph of more than 10 ops in L1");

    bool pass = true;

    vector<string> sfpu_names = {"exponential", "reciprocal", "gelu"};
    vector<string> sfpu_chain;

    // Create random chain of sfpu ops
    for (int _ = 0; _ < chain_length; _++) {
        int idx = rand() % sfpu_names.size();
        string sfpu_name = sfpu_names.at(idx);
        sfpu_chain.push_back(sfpu_name);
    }

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);



        run_compile_blank(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        auto src_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferType::DRAM);
        uint32_t dram_buffer_src_addr = src_dram_buffer.address();
        auto dst_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferType::DRAM);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer.address();

        auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        uint32_t interm0_cb_index = 24;
        tt_metal::CircularBufferConfig cb_interm0_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{interm0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(interm0_cb_index, single_tile_size);
        auto cb_interm0 = tt_metal::CreateCircularBuffer(program, core, cb_interm0_config);

        std::vector<uint32_t> unary_reader_args = {
            dram_buffer_src_addr,
            (std::uint32_t)dram_src_noc_xy.x,
            (std::uint32_t)dram_src_noc_xy.y,
            num_tiles
        };

        std::vector<uint32_t> unary_writer_args = {
            dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tiles
        };

        auto unary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            uint(num_tiles),
            uint(chain_length)
        };

        auto graph_interpreter_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/graph_interpreter.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );


        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////
        //                      Write op info to L1
        ////////////////////////////////////////////////////////////////////////////
        for (int idx = 0; idx < chain_length; idx++) {


            OpCode op_code = sfpu_name_to_op_code.at(sfpu_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;

            if (idx == 0) {
                in0 = 0;
                in1 = 1;
            } else {
                in0 = 24;
                in1 = 25;
            }

            if (idx < chain_length - 1) {
                out = 24;
            } else {
                out = 16;
            }

            uint32_t pop_input = 1;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input,
                .pop1 = pop_input,
                .unary = 1
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, idx);
        }



        ////////////////////////////////////////////////////////////////////////////
        //                      Create input
        ////////////////////////////////////////////////////////////////////////////
        vector<uint32_t> src_vec = create_random_ones_and_twos_vector_of_bfloat16(
            dram_buffer_size,  std::chrono::system_clock::now().time_since_epoch().count());

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::WriteToBuffer(src_dram_buffer, src_vec);



        tt_metal::SetRuntimeArgs(program, unary_reader_kernel, core, unary_reader_args);
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, unary_writer_args);



        // TT_ASSERT(false);
        tt_metal::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        vector<uint32_t> golden = src_vec;
        for (string sfpu_name: sfpu_chain) {
            golden = sfpu(golden, sfpu_op_to_function.at(sfpu_name));
        }

        pass &= packed_uint32_t_vector_comparison(result_vec, golden, equal_within_absolute_tolerance_of_0p03);
        if (not pass) {
            std::cout << "GOLDEN" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles);

            std::cout << "RESULT" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles);
        }

        DeallocateBuffer(src_dram_buffer);
        DeallocateBuffer(dst_dram_buffer);

        tt_metal::CloseDevice(device);
        delete device;


    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_fatal(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

// This test just runs an add followed by gelu
bool run_binary_add_and_then_eltwise_gelu_test() {

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    tt::log_assert(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    uint32_t chain_length = 2;
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        auto src0_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferType::DRAM);
        auto src1_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferType::DRAM);
        auto dst_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferType::DRAM);

        auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer.noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = 1;
        tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        uint32_t interm0_cb_index = 24;
        tt_metal::CircularBufferConfig cb_interm0_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{interm0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(interm0_cb_index, single_tile_size);
        auto cb_interm0 = tt_metal::CreateCircularBuffer(program, core, cb_interm0_config);

        uint32_t interm1_cb_index = 25;
        tt_metal::CircularBufferConfig cb_interm1_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{interm1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(interm1_cb_index, single_tile_size);
        auto cb_interm1 = tt_metal::CreateCircularBuffer(program, core, cb_interm1_config);

        std::vector<uint32_t> binary_reader_args = {
            src0_dram_buffer.address(),
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            src1_dram_buffer.address(),
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            num_tiles
        };

        std::vector<uint32_t> unary_writer_args = {
            dst_dram_buffer.address(),
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tiles
        };

        auto binary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            uint(num_tiles),
            uint(chain_length)
        };

        auto graph_interpreter_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/graph_interpreter.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );


        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////
        //                      Write op info to L1
        ////////////////////////////////////////////////////////////////////////////

        // Add Op reads from src cb 0 and 1, and writes to interm cb 0
        {
            OpCode op_code = tt::OpCode::Add;

            uint32_t in0 = src0_cb_index;
            uint32_t in1 = src1_cb_index;
            uint32_t out = interm0_cb_index;

            uint32_t pop_input = 1;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input,
                .pop1 = pop_input,
                .unary = 0
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, 0);
        }

        // Gelu Op reads from interm cb 0, and writes to output cb
        {
            OpCode op_code = tt::OpCode::Gelu;

            uint32_t in0 = interm0_cb_index;
            uint32_t in1 = interm1_cb_index;
            uint32_t out = ouput_cb_index;

            uint32_t pop_input = 1;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input,
                .pop1 = pop_input,
                .unary = 1
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, 1);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Create input
        ////////////////////////////////////////////////////////////////////////////
        vector<uint32_t> src0_vec = create_random_ones_and_twos_vector_of_bfloat16(
            dram_buffer_size,  std::chrono::system_clock::now().time_since_epoch().count());

        vector<uint32_t> src1_vec = create_random_ones_and_twos_vector_of_bfloat16(
            dram_buffer_size,  std::chrono::system_clock::now().time_since_epoch().count());

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::WriteToBuffer(src0_dram_buffer, src0_vec);
        tt_metal::WriteToBuffer(src1_dram_buffer, src1_vec);



        tt_metal::SetRuntimeArgs(program, binary_reader_kernel, core, binary_reader_args);
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, unary_writer_args);



        // TT_ASSERT(false);
        tt_metal::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        vector<uint32_t> golden = sfpu(eltwise_binary(src0_vec, src1_vec, binary_op_to_function.at("add")), sfpu_op_to_function.at("gelu"));

        auto comparison_func = [](float a, float b){
            return equal_within_absolute_tolerance(a, b, 0.2);
        };
        pass &= packed_uint32_t_vector_comparison(result_vec, golden, comparison_func);
        if (not pass) {
            std::cout << "GOLDEN" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles);

            std::cout << "RESULT" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles);
        }

        tt_metal::CloseDevice(device);
        delete device;


    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_fatal(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}


// This test just runs a chain of eltwise binary ops, with branching
// This runs a specific hardcoded graph
bool run_forked_binary_test() {

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    tt::log_assert(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    int chain_length = 10;

    bool pass = true;

    vector<string> binary_names = {"add", "subtract", "multiply"};
    vector<string> binary_chain;

    // Create random chain of sfpu ops
    for (int _ = 0; _ < chain_length; _++) {
        int idx = rand() % binary_names.size();
        string binary_name = binary_names.at(idx);
        binary_chain.push_back(binary_name);
    }

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t num_dram_channels = 5;

        auto dst_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferType::DRAM);

        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        std::vector<tt_metal::Buffer> src_dram_buffers;

        std::vector<uint32_t> src_cb_buffer_indices;
        uint32_t src_cb_index = 0;
        for (uint32_t i = 0; i < num_dram_channels; i++){
            auto src_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferType::DRAM);
            src_dram_buffers.push_back(std::move(src_dram_buffer));
            tt_metal::CircularBufferConfig src_cb_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{src_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src_cb_index, single_tile_size);
            auto src_cb = tt_metal::CreateCircularBuffer(program, core, src_cb_config);
            src_cb_buffer_indices.push_back(src_cb_index);
            src_cb_index++;
        }


        uint32_t output_cb_index = 16; // output operands start at index 16
        tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_size);
        auto output_cb_buffer = tt_metal::CreateCircularBuffer(program, core, output_cb_config);

        std::vector<uint32_t> interm_cb_buffer_indices;
        uint32_t interm_cb_index = 24;
        for (uint32_t i = 0; i < 3; i++){
            tt_metal::CircularBufferConfig interim_cb_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{interm_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(interm_cb_index, single_tile_size);
            auto interm_cb = tt_metal::CreateCircularBuffer(program, core, interim_cb_config);
            interm_cb_buffer_indices.push_back(interm_cb_index);
            interm_cb_index++;
        }

        std::vector<uint32_t> nary_reader_args {num_dram_channels, num_tiles};
        for (uint32_t i = 0; i < num_dram_channels; i++){
            auto dram_src_noc_xy = src_dram_buffers[i].noc_coordinates();
            nary_reader_args.insert(nary_reader_args.end(),
            {
                src_dram_buffers[i].address(),
                (std::uint32_t)dram_src_noc_xy.x,
                (std::uint32_t)dram_src_noc_xy.y,
                i
            });
        }

        std::vector<uint32_t> unary_writer_args = {
            dst_dram_buffer.address(),
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tiles
        };


        auto nary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_nary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            uint(num_tiles),
            uint(chain_length)
        };

        auto graph_interpreter_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/graph_interpreter.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );


        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////
        //                      Write op info to L1
        ////////////////////////////////////////////////////////////////////////////

        // Op 0
        {
            uint32_t idx = 0;
            OpCode op_code = binary_name_to_op_code.at(binary_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;
            in0 = src_cb_buffer_indices[0];
            in1 = src_cb_buffer_indices[1];

            out = interm_cb_buffer_indices[0];
            uint32_t pop_input0 = 0;
            uint32_t pop_input1 = 1;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input0,
                .pop1 = pop_input1,
                .unary = 0
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, idx);
        }

        // Op 1
        {
            uint32_t idx = 1;
            OpCode op_code = binary_name_to_op_code.at(binary_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;
            in0 = src_cb_buffer_indices[2];
            in1 = src_cb_buffer_indices[3];

            out = interm_cb_buffer_indices[1];
            uint32_t pop_input0 = 1;
            uint32_t pop_input1 = 0;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input0,
                .pop1 = pop_input1,
                .unary = 0
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, idx);
        }

        // Op 2
        {
            uint32_t idx = 2;
            OpCode op_code = binary_name_to_op_code.at(binary_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;
            in0 = src_cb_buffer_indices[3];
            in1 = src_cb_buffer_indices[4];
            out = interm_cb_buffer_indices[2];
            uint32_t pop_input0 = 1;
            uint32_t pop_input1 = 1;
            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input0,
                .pop1 = pop_input1,
                .unary = 0
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, idx);
        }

        // Op 3
        {
            uint32_t idx = 3;
            OpCode op_code = binary_name_to_op_code.at(binary_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;
            in0 = interm_cb_buffer_indices[0];
            in1 = interm_cb_buffer_indices[1];

            out = interm_cb_buffer_indices[0];
            uint32_t pop_input0 = 1;
            uint32_t pop_input1 = 0;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input0,
                .pop1 = pop_input1,
                .unary = 0
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, idx);
        }

        // Op 4
        {
            uint32_t idx = 4;
            OpCode op_code = binary_name_to_op_code.at(binary_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;
            in0 = interm_cb_buffer_indices[1];
            in1 = interm_cb_buffer_indices[2];

            out = interm_cb_buffer_indices[2];
            uint32_t pop_input0 = 0;
            uint32_t pop_input1 = 1;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input0,
                .pop1 = pop_input1,
                .unary = 0
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, idx);
        }

        // Op 5
        {
            uint32_t idx = 5;
            OpCode op_code = binary_name_to_op_code.at(binary_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;
            in0 = interm_cb_buffer_indices[1];
            in1 = interm_cb_buffer_indices[2];

            out = interm_cb_buffer_indices[1];
            uint32_t pop_input0 = 1;
            uint32_t pop_input1 = 1;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input0,
                .pop1 = pop_input1,
                .unary = 0
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, idx);
        }

        // Op 6
        {
            uint32_t idx = 6;
            OpCode op_code = binary_name_to_op_code.at(binary_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;
            in0 = interm_cb_buffer_indices[0];
            in1 = interm_cb_buffer_indices[1];

            out = interm_cb_buffer_indices[0];
            uint32_t pop_input0 = 1;
            uint32_t pop_input1 = 1;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input0,
                .pop1 = pop_input1,
                .unary = 0
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, idx);
        }

        // Op 7
        {
            uint32_t idx = 7;
            OpCode op_code = binary_name_to_op_code.at(binary_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;
            in0 = interm_cb_buffer_indices[0];
            in1 = interm_cb_buffer_indices[0];

            out = interm_cb_buffer_indices[0];
            uint32_t pop_input0 = 1;
            uint32_t pop_input1 = 0;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input0,
                .pop1 = pop_input1,
                .unary = 0
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, idx);
        }

        // Op 8
        {
            uint32_t idx = 8;
            OpCode op_code = binary_name_to_op_code.at(binary_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;
            in0 = src_cb_buffer_indices[0];
            in1 = interm_cb_buffer_indices[0];

            out = interm_cb_buffer_indices[1];
            uint32_t pop_input0 = 1;
            uint32_t pop_input1 = 0;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input0,
                .pop1 = pop_input1,
                .unary = 0
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, idx);
        }

        // Op 9
        {
            uint32_t idx = 9;
            OpCode op_code = binary_name_to_op_code.at(binary_chain.at(idx));

            uint32_t in0;
            uint32_t in1;
            uint32_t out;
            in0 = interm_cb_buffer_indices[0];
            in1 = interm_cb_buffer_indices[1];

            out = output_cb_index;
            uint32_t pop_input0 = 1;
            uint32_t pop_input1 = 1;

            op_info_t op_info = {
                .op_code = (uint32_t) op_code,
                .cb_in0_id = in0,
                .cb_in1_id = in1,
                .cb_out_id = out,
                .pop0 = pop_input0,
                .pop1 = pop_input1,
                .unary = 0
            };

            tt_metal::detail::WriteToDeviceL1(device, core, op_info, idx);
        }


        ////////////////////////////////////////////////////////////////////////////
        //                      Create input
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::vector<uint32_t> > src_vecs;
        for(uint32_t i = 0; i < num_dram_channels; i++){
            vector<uint32_t> src_vec = create_random_ones_and_twos_vector_of_bfloat16(
                src_dram_buffers[i].size(),  std::chrono::system_clock::now().time_since_epoch().count());
            tt_metal::WriteToBuffer(src_dram_buffers[i], src_vec);
            src_vecs.push_back(src_vec);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////




        tt_metal::SetRuntimeArgs(program, nary_reader_kernel, core, nary_reader_args);
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, unary_writer_args);


        tt_metal::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        vector<uint32_t> interm0 = eltwise_binary(src_vecs[0], src_vecs[1], binary_op_to_function.at(binary_chain.at(0)));
        vector<uint32_t> interm1 = eltwise_binary(src_vecs[2], src_vecs[3], binary_op_to_function.at(binary_chain.at(1)));
        vector<uint32_t> interm2 = eltwise_binary(src_vecs[3], src_vecs[4], binary_op_to_function.at(binary_chain.at(2)));
        interm0 = eltwise_binary(interm0, interm1, binary_op_to_function.at(binary_chain.at(3)));
        interm2 = eltwise_binary(interm1, interm2, binary_op_to_function.at(binary_chain.at(4)));
        interm1 = eltwise_binary(interm1, interm2, binary_op_to_function.at(binary_chain.at(5)));
        interm0 = eltwise_binary(interm0, interm1, binary_op_to_function.at(binary_chain.at(6)));
        interm0 = eltwise_binary(interm0, interm0, binary_op_to_function.at(binary_chain.at(7)));
        interm1 = eltwise_binary(src_vecs[0], interm0, binary_op_to_function.at(binary_chain.at(8)));
        vector<uint32_t> golden = eltwise_binary(interm0, interm1, binary_op_to_function.at(binary_chain.at(9)));

        pass &= packed_uint32_t_vector_comparison(result_vec, golden, is_close_0p015);
        if (not pass) {
            std::cout << "GOLDEN" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles);

            std::cout << "RESULT" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles);
        }

        tt_metal::CloseDevice(device);
        delete device;


    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_fatal(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}


int main(int argc, char **argv) {

    // Trivial chain of sfpu ops
    char env[] = "HACK_FOR_GRAPH_INTERPRETER=1";
    putenv(env);

    bool pass = true;

    // Simple eltwise unary chain test
    pass &= run_chained_sfpu_test(10);

    // Binary add and then gelu on output
    pass &= run_binary_add_and_then_eltwise_gelu_test();

    // Binary forked graph
    pass &= run_forked_binary_test();

    TT_ASSERT(pass, "Graph interpreter test failed");
    return 0;
}
