// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt::tt_metal;

void RunTest(Device *device) {
    // Set up program
    Program program = Program();

    std::set<CoreRange> core_ranges;
    //CoreCoord grid_size = device->logical_grid_size();
    CoreCoord grid_size = {1, 1};
    log_info(LogTest, "Creating core range...");
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord core(x, y);
            core_ranges.insert(CoreRange{.start=core, .end=core});
        }
    }

    // Kernels on brisc + ncrisc that just add two numbers
    log_info(LogTest, "Creating kernels...");
    KernelHandle brisc_kid = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        CoreRangeSet(core_ranges),
        tt_metal::DataMovementConfig {
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );
    KernelHandle ncrisc_kid = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        CoreRangeSet(core_ranges),
        tt_metal::DataMovementConfig {
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default
        }
    );

    // Write runtime args
    log_info(LogTest, "Writing args...");
    auto get_first_arg = [](Device *device, CoreCoord &core, uint32_t multiplier) {
        return (uint32_t) device->id() + (uint32_t) core.x * 10 * multiplier;
    };
    auto get_second_arg = [](Device *device, CoreCoord &core, uint32_t multiplier) {
        return (uint32_t) core.y * 100 * multiplier;
    };
    for (auto &core_range : core_ranges) {
        CoreCoord core = core_range.start;
        std::vector<uint32_t> brisc_rt_args = {
            get_first_arg(device, core, 1),
            get_second_arg(device, core, 1)
        };
        std::vector<uint32_t> ncrisc_rt_args = {
            get_first_arg(device, core, 2),
            get_second_arg(device, core, 2)
        };
        SetRuntimeArgs(program, brisc_kid, core, brisc_rt_args);
        SetRuntimeArgs(program, ncrisc_kid, core, ncrisc_rt_args);
    }

    auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    log_info(LogTest, "Running program...");
    try {
    if (slow_dispatch) {
        // Slow dispatch uses LaunchProgram
        tt::tt_metal::detail::LaunchProgram(device, program);
    } else {
        // Fast Dispatch uses the command queue
        CommandQueue& cq = tt::tt_metal::detail::GetCommandQueue(device);
        EnqueueProgram(cq, program, false);
        Finish(cq);
    }
    } catch (std::out_of_range& e) {
        const string error = string(e.what());
        log_info(tt::LogTest, "Caught exception: {}", error);
        EXPECT_TRUE(false);
        return;
    }

    // Check results
    log_info(LogTest, "Checking results...");
    for (auto &core_range : core_ranges) {
        CoreCoord core = core_range.start;
        std::vector<uint32_t> brisc_result;
        tt_metal::detail::ReadFromDeviceL1(
            device, core, BRISC_L1_RESULT_BASE, sizeof(uint32_t), brisc_result
        );
        std::vector<uint32_t> ncrisc_result;
        tt_metal::detail::ReadFromDeviceL1(
            device, core, NCRISC_L1_RESULT_BASE, sizeof(uint32_t), ncrisc_result
        );
        uint32_t expected_result = get_first_arg(device, core, 1) + get_second_arg(device, core, 1);
        if (expected_result != brisc_result[0])
            log_warning(
                LogTest,
                "Device {}, Core {}, BRISC result was incorrect. Expected {} but got {}",
                device->id(),
                core.str(),
                expected_result,
                brisc_result[0]
            );
        EXPECT_TRUE(expected_result == brisc_result[0]);
        expected_result = get_first_arg(device, core, 2) + get_second_arg(device, core, 2);
        if (expected_result != ncrisc_result[0])
            log_warning(
                LogTest,
                "Device {}, Core {}, NCRISC result was incorrect. Expected {} but got {}",
                device->id(),
                core.str(),
                expected_result,
                ncrisc_result[0]
            );
        EXPECT_TRUE(expected_result == ncrisc_result[0]);
    }
}

TEST(CommonMiscTests, AllCoresStressTest) {
    auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    // Skip fast dispatch until it's supported for remote device.
    if (!slow_dispatch)
        GTEST_SKIP();
    // Run 500 times to make sure that things work
    for (int idx = 0; idx < 500; idx++) {
        log_info(LogTest, "Running iteration #{}", idx);
        // Need to open/close the device each time in order to reproduce original issue.
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        vector<Device*> devices_;
        for (unsigned int id = 0; id < num_devices; id++) {
            auto* device = tt::tt_metal::CreateDevice(id);
            devices_.push_back(device);
        }

        // Run the test on each device
        for (Device *device : devices_) {
            RunTest(device);
        }

        // Close all devices
        log_info(LogTest, "Closing devices...");
        for (unsigned int id = 0; id < devices_.size(); id++) {
            tt::tt_metal::CloseDevice(devices_.at(id));
        }
        log_info(LogTest, "Running iteration #{}", idx);
    }

}
