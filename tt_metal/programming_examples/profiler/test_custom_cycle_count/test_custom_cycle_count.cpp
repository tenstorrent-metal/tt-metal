// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"

using namespace tt;

bool RunCustomCycle(tt_metal::Device *device, int loop_count, string run_name = " ")
{
    bool pass = true;

    CoreCoord compute_with_storage_size = device->compute_with_storage_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
    CoreRange all_cores{.start=start_core, .end=end_core};

    tt_metal::Program program = tt_metal::Program();

    constexpr int loop_size = 50;
    constexpr bool profile_device = true;
    std::map<string, string> kernel_defines = {
        {"LOOP_COUNT", std::to_string(loop_count)},
        {"LOOP_SIZE", std::to_string(loop_size)}
    };

    tt_metal::KernelID brisc_kernel = tt_metal::CreateKernel(
        program, "tt_metal/programming_examples/profiler/test_custom_cycle_count/kernels/custom_cycle_count.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .defines = kernel_defines});

    tt_metal::KernelID ncrisc_kernel = tt_metal::CreateKernel(
        program, "tt_metal/programming_examples/profiler/test_custom_cycle_count/kernels/custom_cycle_count.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .defines = kernel_defines});

    vector<uint32_t> trisc_kernel_args = {};
    tt_metal::KernelID trisc_kernel = tt_metal::CreateKernel(
        program, "tt_metal/programming_examples/profiler/test_custom_cycle_count/kernels/custom_cycle_count_compute.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = trisc_kernel_args, .defines = kernel_defines}
    );



    tt_metal::LaunchProgram(device, program);

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    tt::log_assert(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);



        int loop_count = 2000;
        pass &= RunCustomCycle(device, loop_count);

        loop_count = 2001;
        pass &= RunCustomCycle(device, loop_count);

        tt_metal::CloseDevice(device);

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
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
