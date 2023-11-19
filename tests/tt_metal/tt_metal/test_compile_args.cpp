// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include <filesystem>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "llrt/tt_debug_print_server.hpp"
#include "tt_metal/detail/tt_metal.hpp"
//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

bool test_compile_args(std::vector<uint32_t> compile_args_vec, int device_id) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    const tt_metal::Device& device =
        tt_metal::CreateDevice(device_id);



    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    tt_metal::KernelID unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_compile_args.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = compile_args_vec});


    tt_metal::KernelID unary_writer_kernel = tt_metal::CreateKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_args = {
        0 // dummy
    };

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program, "tt_metal/kernels/compute/blank.cpp",
        core, tt_metal::ComputeConfig{.compile_args = compute_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::detail::CompileProgram(device, program);

    CloseDevice(device);

    return true;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        int device_id = 0;
        static const std::string kernel_name = "test_compile_args";
        auto binary_path_str = get_kernel_compile_outpath(device_id) + kernel_name;
        std::filesystem::remove_all(binary_path_str);

        pass &= test_compile_args({0, 68, 0, 124}, device_id);
        pass &= test_compile_args({1, 5, 0, 124}, device_id);

        TT_FATAL(std::filesystem::exists(binary_path_str), "Expected kernel to be compiled!");

        std::filesystem::path binary_path{binary_path_str};
        auto num_built_kernels = std::distance(std::filesystem::directory_iterator(binary_path), std::filesystem::directory_iterator{});
        TT_FATAL(num_built_kernels == 2, "Expected compute kernel test_compile_args to be compiled twice!");

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}
