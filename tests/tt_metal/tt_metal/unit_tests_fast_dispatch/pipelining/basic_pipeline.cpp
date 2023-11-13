// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "common/bfloat16.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::tt_metal;

namespace unit_tests::create_pipeline {

void create_and_run_row_pipeline(tt_metal::Device* device, u32 num_cores) {
    CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;

    tt_metal::Program program = tt_metal::Program();

    u32 num_tiles = 32;
    u32 block_size_tiles = 16;
    u32 num_blocks_in_CB = 2;
    u32 num_repetitions = 1;

    TT_ASSERT(num_cores >= 2 && num_cores <= 12);  // grayskull
    TT_ASSERT(num_tiles % block_size_tiles == 0);

    std::vector<CoreCoord> cores;
    for (u32 i = 0; i < num_cores; i++) {
        cores.push_back({i, 0});
    }

    log_info(LogTest, "num_cores: {}", num_cores);
    log_info(LogTest, "num_tiles: {}", num_tiles);
    log_info(LogTest, "block_size_tiles: {}", block_size_tiles);
    log_info(LogTest, "num_blocks_in_CB: {}", num_blocks_in_CB);
    log_info(LogTest, "num_repetitions: {}", num_repetitions);

    u32 single_tile_size = 2 * 1024;
    u32 block_size_bytes = block_size_tiles * single_tile_size;
    log_info(LogTest, "block_size_bytes: {}", block_size_bytes);
    log_info(LogTest, "CB size: {}", block_size_bytes * num_blocks_in_CB);

    // source and destination buffers
    u32 buffer_size = single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    u32 total_bytes_moved = buffer_size * num_repetitions;
    log_info(LogTest, "total_bytes_moved: {}", total_bytes_moved);

    // circular buffers in L1
    u32 cb_index = 8;
    u32 cb_size_tiles = num_blocks_in_CB * block_size_tiles;
    u32 cb_size_bytes = cb_size_tiles * single_tile_size;

    for (auto core : cores) {
        tt_metal::CircularBufferConfig cb_config = tt_metal::CircularBufferConfig(cb_size_bytes, {{cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index, single_tile_size);
        auto cb = tt_metal::CreateCircularBuffer(program, core, cb_config);
    }

    /// used only if IO data in DRAM
    tt_metal::Buffer src_buffer;
    tt_metal::Buffer dst_buffer;

    u32 src_address;
    CoreCoord src_noc_xy;
    u32 dst_address;
    CoreCoord dst_noc_xy;

    src_buffer = CreateBuffer(device, buffer_size, buffer_size, tt_metal::BufferStorage::DRAM);
    dst_buffer = CreateBuffer(device, buffer_size, buffer_size, tt_metal::BufferStorage::DRAM);

    src_address = src_buffer.address();
    src_noc_xy = src_buffer.noc_coordinates();
    dst_address = dst_buffer.address();
    dst_noc_xy = dst_buffer.noc_coordinates();

    // create kernels
    vector<tt_metal::KernelID> receiver_kernels;
    vector<tt_metal::KernelID> sender_kernels;
    for (int core_id = 0; core_id < num_cores; core_id++) {
        string receiver_kernel_name;
        if (core_id == 0) {
            receiver_kernel_name = "tt_metal/kernels/dataflow/reader_first_stage.cpp";
        } else {
            receiver_kernel_name = "tt_metal/kernels/dataflow/receiver_intermediate_stage.cpp";
        }

        std::vector<u32> receiver_kernel_compile_time_args = {cb_index, block_size_tiles};
        receiver_kernels.push_back(tt_metal::CreateDataMovementKernel(
            program,
            receiver_kernel_name,
            cores[core_id],
            DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = receiver_kernel_compile_time_args}));

        string sender_kernel_name;
        if (core_id == num_cores - 1) {
            sender_kernel_name = "tt_metal/kernels/dataflow/writer_last_stage.cpp";
        } else {
            sender_kernel_name = "tt_metal/kernels/dataflow/sender_intermediate_stage.cpp";
        }
        std::vector<u32> sender_kernel_compile_time_args = {cb_index, block_size_tiles};
        sender_kernels.push_back(tt_metal::CreateDataMovementKernel(
            program,
            sender_kernel_name,
            cores[core_id],
            DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = sender_kernel_compile_time_args}));

        // Add blank compute kernel
        tt_metal::CreateComputeKernel(program, "tt_metal/kernels/compute/blank.cpp", cores[core_id]);
    }

    // TODO(agrebenisan): Once semaphores are properly allocated at 16B-aligned addresses, then
    // will make proper sems. For now, using the original code.
    map<CoreCoord, vector<uint32_t>> sems;
    for (auto core : cores) {
        CoreRange cr = {.start = core, .end = core};

        auto sender_semaphore = tt_metal::CreateSemaphore(program, cr, INVALID);
        auto receiver_semaphore = tt_metal::CreateSemaphore(program, cr, INVALID);
        auto l1_valid_value_semaphore = tt_metal::CreateSemaphore(program, cr, VALID);

        tt::log_debug("SENDER SEM ADDR {}", sender_semaphore);
        tt::log_debug("RECEIVER SEM ADDR {}", receiver_semaphore);
        tt::log_debug("L1 VALID VALUE SEM ADDR {}", l1_valid_value_semaphore);

        vector<uint32_t> init_vec;
        sems.emplace(core, init_vec);
        sems.at(core).push_back(sender_semaphore);
        sems.at(core).push_back(receiver_semaphore);
        sems.at(core).push_back(l1_valid_value_semaphore);
    }

    for (int core_id = 0; core_id < num_cores; core_id++) {
        // TODO(agrebenisan):  Once semaphores are properly allocated at 16B-aligned addresses, then
        // will make proper sems. For now, using the original code.
        CoreCoord core = cores[core_id];
        auto sender_semaphore_addr = sems[core].at(0);
        auto receiver_semaphore_addr = sems[core].at(1);
        auto l1_valid_value_addr = sems[core].at(2);

        if (core_id == 0) {
            SetRuntimeArgs(
                program,
                receiver_kernels.at(core_id),
                core,
                {src_address, (u32)src_noc_xy.x, (u32)src_noc_xy.y, (u32)num_tiles, (u32)num_repetitions});
        } else {
            SetRuntimeArgs(
                program,
                receiver_kernels.at(core_id),
                core,
                {(u32)device->worker_core_from_logical_core(cores[core_id - 1]).x,
                 (u32)device->worker_core_from_logical_core(cores[core_id - 1]).y,
                 (u32)num_tiles,
                 (u32)sender_semaphore_addr,
                 (u32)receiver_semaphore_addr,
                 (u32)num_repetitions});
        }

        if (core_id == num_cores - 1) {
            SetRuntimeArgs(
                program,
                sender_kernels.at(core_id),
                core,
                {dst_address, (u32)dst_noc_xy.x, (u32)dst_noc_xy.y, (u32)num_tiles, (u32)num_repetitions});
        } else {
            SetRuntimeArgs(
                program,
                sender_kernels.at(core_id),
                core,
                {(u32)device->worker_core_from_logical_core(cores[core_id + 1]).x,
                 (u32)device->worker_core_from_logical_core(cores[core_id + 1]).y,
                 (u32)num_tiles,
                 (u32)sender_semaphore_addr,
                 (u32)receiver_semaphore_addr,
                 (u32)l1_valid_value_addr,
                 (u32)num_repetitions});
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    // send input data to the device
    std::vector<u32> src_vec =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    EnqueueWriteBuffer(cq, src_buffer, src_vec, false);

    EnqueueProgram(cq, program, false);
    Finish(cq);

    log_info(LogTest, "Kernels done.");

    log_info(LogTest, "Reading results from device...");
    std::vector<u32> result_vec;
    EnqueueReadBuffer(cq, dst_buffer, result_vec, true);

    ////////////////////////////////////////////////////////////////////////////
    //                      Validation & Teardown
    ////////////////////////////////////////////////////////////////////////////
    ASSERT_TRUE(src_vec == result_vec);
}

}  // namespace unit_tests::create_pipeline

TEST_F(CommandQueueFixture, TwoCorePipeline) {
    if (this->arch_ != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, 2);
}
TEST_F(CommandQueueFixture, TwelveCorePipeline) {
    if (this->arch_ != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, 12);
}
