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

struct PipelineRowConfig {
    size_t num_cores;
    size_t num_tiles;
    size_t block_size_tiles;
    size_t num_blocks_in_CB;
    size_t IO_data_in_dram;
    size_t num_repetitions;
};

void create_and_run_row_pipeline(tt_metal::Device* device, const PipelineRowConfig& test_config) {
    CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;

    tt_metal::Program program = tt_metal::Program();

    // uint32_t num_tiles = 32;
    // uint32_t block_size_tiles = 16;
    // uint32_t num_blocks_in_CB = 2;
    // uint32_t num_repetitions = 1;
    uint32_t num_cores = (uint32_t)test_config.num_cores;
    uint32_t num_tiles = (uint32_t)test_config.num_tiles;
    uint32_t block_size_tiles = (uint32_t)test_config.block_size_tiles;
    uint32_t num_blocks_in_CB = (uint32_t)test_config.num_blocks_in_CB;
    uint32_t num_repetitions = (uint32_t)test_config.num_repetitions;

    TT_ASSERT(num_cores >= 2 && num_cores <= 12);  // grayskull
    TT_ASSERT(num_tiles % block_size_tiles == 0);

    std::vector<CoreCoord> cores;
    for (uint32_t i = 0; i < num_cores; i++) {
        cores.push_back({i, 0});
    }

    log_info(LogTest, "num_cores: {}", num_cores);
    log_info(LogTest, "num_tiles: {}", num_tiles);
    log_info(LogTest, "block_size_tiles: {}", block_size_tiles);
    log_info(LogTest, "num_blocks_in_CB: {}", num_blocks_in_CB);
    log_info(LogTest, "num_repetitions: {}", num_repetitions);

    uint32_t single_tile_size = 2 * 1024;
    uint32_t block_size_bytes = block_size_tiles * single_tile_size;
    log_info(LogTest, "block_size_bytes: {}", block_size_bytes);
    log_info(LogTest, "CB size: {}", block_size_bytes * num_blocks_in_CB);

    // source and destination buffers
    uint32_t buffer_size = single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t total_bytes_moved = buffer_size * num_repetitions;
    log_info(LogTest, "total_bytes_moved: {}", total_bytes_moved);

    // circular buffers in L1
    uint32_t cb_index = 8;
    uint32_t cb_size_tiles = num_blocks_in_CB * block_size_tiles;
    uint32_t cb_size_bytes = cb_size_tiles * single_tile_size;

    for (auto core : cores) {
        tt_metal::CircularBufferConfig cb_config = tt_metal::CircularBufferConfig(cb_size_bytes, {{cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index, single_tile_size);
        auto cb = tt_metal::CreateCircularBuffer(program, core, cb_config);
    }

    /// used only if IO data in DRAM
    tt_metal::Buffer src_buffer;
    tt_metal::Buffer dst_buffer;

    uint32_t src_address;
    CoreCoord src_noc_xy;
    uint32_t dst_address;
    CoreCoord dst_noc_xy;

    src_buffer = CreateBuffer(device, buffer_size, buffer_size, test_config.IO_data_in_dram ? tt_metal::BufferType::DRAM : tt_metal::BufferType::L1);
    dst_buffer = CreateBuffer(device, buffer_size, buffer_size, test_config.IO_data_in_dram ? tt_metal::BufferType::DRAM : tt_metal::BufferType::L1);

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

        std::vector<uint32_t> receiver_kernel_compile_time_args = {cb_index, block_size_tiles};
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
        std::vector<uint32_t> sender_kernel_compile_time_args = {cb_index, block_size_tiles};
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
                {src_address, (uint32_t)src_noc_xy.x, (uint32_t)src_noc_xy.y, (uint32_t)num_tiles, (uint32_t)num_repetitions});
        } else {
            SetRuntimeArgs(
                program,
                receiver_kernels.at(core_id),
                core,
                {(uint32_t)device->worker_core_from_logical_core(cores[core_id - 1]).x,
                 (uint32_t)device->worker_core_from_logical_core(cores[core_id - 1]).y,
                 (uint32_t)num_tiles,
                 (uint32_t)sender_semaphore_addr,
                 (uint32_t)receiver_semaphore_addr,
                 (uint32_t)num_repetitions});
        }

        if (core_id == num_cores - 1) {
            SetRuntimeArgs(
                program,
                sender_kernels.at(core_id),
                core,
                {dst_address, (uint32_t)dst_noc_xy.x, (uint32_t)dst_noc_xy.y, (uint32_t)num_tiles, (uint32_t)num_repetitions});
        } else {
            SetRuntimeArgs(
                program,
                sender_kernels.at(core_id),
                core,
                {(uint32_t)device->worker_core_from_logical_core(cores[core_id + 1]).x,
                 (uint32_t)device->worker_core_from_logical_core(cores[core_id + 1]).y,
                 (uint32_t)num_tiles,
                 (uint32_t)sender_semaphore_addr,
                 (uint32_t)receiver_semaphore_addr,
                 (uint32_t)l1_valid_value_addr,
                 (uint32_t)num_repetitions});
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    // send input data to the device
    std::vector<uint32_t> src_vec =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());


    log_info(LogTest, "Writing to device buffer...");
    tt_metal::WriteToBuffer(src_buffer, src_vec);
    log_info(LogTest, "Writing to device buffer Done.");

    EnqueueProgram(cq, program, false);
    Finish(cq);

    log_info(LogTest, "Kernels done.");

    log_info(LogTest, "Reading results from device...");
    std::vector<uint32_t> result_vec;
    tt_metal::ReadFromBuffer(dst_buffer, result_vec);

    ////////////////////////////////////////////////////////////////////////////
    //                      Validation & Teardown
    ////////////////////////////////////////////////////////////////////////////
    ASSERT_TRUE(src_vec == result_vec);
}

}  // namespace unit_tests::create_pipeline

TEST_F(CommandQueueFixture, TestPipelineAcrossRows) {
    if (this->arch_ != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    unit_tests::create_pipeline::PipelineRowConfig test_config;

    // // saturate DRAM
    test_config.num_cores = 12;
    test_config.num_tiles = 64 * 1024;
    test_config.block_size_tiles = 16;
    test_config.num_blocks_in_CB = 2;
    test_config.IO_data_in_dram = true;
    test_config.num_repetitions = 1;
    unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, test_config);

    // saturate L1
    test_config.num_cores = 10;
    test_config.num_tiles = 64;
    test_config.block_size_tiles = 16;
    test_config.num_blocks_in_CB = 2;
    test_config.IO_data_in_dram = false;
    test_config.num_repetitions = 64;
    unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, test_config);

    // test #1
    test_config.num_cores = 12;
    test_config.num_tiles = 64;
    test_config.block_size_tiles = 1;
    test_config.num_blocks_in_CB = 16;
    test_config.IO_data_in_dram = false;
    test_config.num_repetitions = 128;
    unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, test_config);

    // test #2
    test_config.num_cores = 12;
    test_config.num_tiles = 64;
    test_config.block_size_tiles = 2;
    test_config.num_blocks_in_CB = 16;
    test_config.IO_data_in_dram = false;
    test_config.num_repetitions = 128;
    unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, test_config);

    // test #3
    test_config.num_cores = 12;
    test_config.num_tiles = 64;
    test_config.block_size_tiles = 4;
    test_config.num_blocks_in_CB = 16;
    test_config.IO_data_in_dram = false;
    test_config.num_repetitions = 128;
    unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, test_config);

    // test #4
    test_config.num_cores = 12;
    test_config.num_tiles = 64;
    test_config.block_size_tiles = 8;
    test_config.num_blocks_in_CB = 8;
    test_config.IO_data_in_dram = false;
    test_config.num_repetitions = 128;
    unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, test_config);

    // test #5
    test_config.num_cores = 12;
    test_config.num_tiles = 64;
    test_config.block_size_tiles = 16;
    test_config.num_blocks_in_CB = 4;
    test_config.IO_data_in_dram = false;
    test_config.num_repetitions = 128;
    unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, test_config);

    // test #6
    test_config.num_cores = 12;
    test_config.num_tiles = 64;
    test_config.block_size_tiles = 32;
    test_config.num_blocks_in_CB = 4;
    test_config.IO_data_in_dram = false;
    test_config.num_repetitions = 128;
    unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, test_config);

    // test #7
    test_config.num_cores = 12;
    test_config.num_tiles = 64;
    test_config.block_size_tiles = 64;
    test_config.num_blocks_in_CB = 4;
    test_config.IO_data_in_dram = false;
    test_config.num_repetitions = 128;
    unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, test_config);
}
