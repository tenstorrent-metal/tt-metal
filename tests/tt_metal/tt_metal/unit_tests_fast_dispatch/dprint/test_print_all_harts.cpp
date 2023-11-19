// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <thread>
#include <chrono>

#include "command_queue_fixture.hpp"
#include "common/bfloat16.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A simple test for checking DPRINTs from all harts.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

TEST_F(CommandQueueWithDPrintFixture, TestPrintFromAllHarts) {
    bool pass = true;

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    // Set up program and command queue
    CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;
    Program program = Program();

    // Three different kernels to mirror typical usage and some previously
    // failing test cases, although all three kernels simply print.
    constexpr CoreCoord core = {0, 0}; // Print on first core only
    KernelID brisc_print_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/brisc_print.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    KernelID ncrisc_print_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/ncrisc_print.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    KernelID trisc_print_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/trisc_print.cpp",
        core,
        ComputeConfig{}
    );

    // Run the program
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Since the program takes almost no time to run, wait a bit for the print
    // server to catch up.
    std::this_thread::sleep_for (std::chrono::seconds(1));

    // Check that the expected print messages are in the log file
    vector<string> expected_prints({
        "Test Debug Print: Pack",
        "Test Debug Print: Unpack",
        "Test Debug Print: Math",
        "Test Debug Print: Data0",
        "Test Debug Print: Data1"
    });
    EXPECT_TRUE(
        FileContainsAllStrings(
            CommandQueueWithDPrintFixture::dprint_file_name,
            expected_prints
        )
    );
}
