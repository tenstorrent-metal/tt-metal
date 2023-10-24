// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

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



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        auto src_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferType::DRAM);
        uint32_t dram_buffer_src_addr = src_dram_buffer.address();
        auto dst_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferType::DRAM);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer.address();

        auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        int num_cbs = 1; // works at the moment
        assert(num_tiles % num_cbs == 0);
        int num_tiles_per_cb = num_tiles / num_cbs;

        uint32_t cb0_index = 0;
        uint32_t num_cb_tiles = 8;
        tt_metal::CircularBufferConfig cb0_config = tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb0_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb0_index, single_tile_size);
        auto cb0 = tt_metal::CreateCircularBuffer(program, core, cb0_config);

        uint32_t cb1_index = 8;
        tt_metal::CircularBufferConfig cb1_config = tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb1_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb1_index, single_tile_size);
        auto cb1 = tt_metal::CreateCircularBuffer(program, core, cb1_config);

        uint32_t cb2_index = 16;
        tt_metal::CircularBufferConfig cb2_config = tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb2_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb2_index, single_tile_size);
        auto cb2 = tt_metal::CreateCircularBuffer(program, core, cb2_config);

        uint32_t cb3_index = 24;
        tt_metal::CircularBufferConfig cb3_config = tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb3_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb3_index, single_tile_size);
        auto cb3 = tt_metal::CreateCircularBuffer(program, core, cb3_config);

        std::vector<uint32_t> reader_cb_kernel_args = {8, 2};
        std::vector<uint32_t> writer_cb_kernel_args = {8, 4};

        auto reader_cb_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_cb_test.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_cb_kernel_args});

        auto writer_cb_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_cb_test.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_cb_kernel_args});

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        tt_metal::WriteToBuffer(src_dram_buffer, src_vec);



        tt_metal::SetRuntimeArgs(
            program,
            reader_cb_kernel,
            core,
            {dram_buffer_src_addr,
            (uint32_t)dram_src_noc_xy.x,
            (uint32_t)dram_src_noc_xy.y,
            (uint32_t)num_tiles_per_cb});

        tt_metal::SetRuntimeArgs(
            program,
            writer_cb_kernel,
            core,
            {dram_buffer_dst_addr,
            (uint32_t)dram_dst_noc_xy.x,
            (uint32_t)dram_dst_noc_xy.y,
            (uint32_t)num_tiles_per_cb});



        tt_metal::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= (src_vec == result_vec);

        pass &= tt_metal::CloseDevice(device);;

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
