// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <optional>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/detail/program.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

void check_program_is_mapped_to_correct_cores(const tt_metal::Program &program, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &compute_kernel_args) {
    // every kernel, semaphore and CB should be mapped to each core in the core ranges of core_range_set
    for (auto core_range : core_range_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                auto logical_core = CoreCoord{x, y};
                for (auto kernel_id : program.kernel_ids()) {
                    tt_metal::Kernel *kernel = tt_metal::detail::GetKernel(program, kernel_id);
                    TT_FATAL(kernel->is_on_logical_core(logical_core));
                    // Check that compute kernel compile time args are mapped to the correct cores
                    if (kernel->processor() == tt::RISCV::COMPUTE) {
                        auto kernel_compile_time_args = kernel->compile_time_args();
                        TT_FATAL(kernel_compile_time_args == compute_kernel_args);
                    }
                }
                for (auto cb : program.circular_buffers()) {
                    TT_FATAL(cb->is_on_logical_core(logical_core));
                }
                for (auto semaphore : program.semaphores() ){
                    TT_FATAL(semaphore.initialized_on_logical_core(logical_core));
                }
            }
        }
    }
}

void check_semaphores_are_initialized(tt_metal::Device *device, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &golden_sem_values) {
    for (auto core_range : core_range_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                auto logical_core = CoreCoord{x, y};
                std::vector<uint32_t> res;
                tt_metal::detail::ReadFromDeviceL1(device, logical_core, SEMAPHORE_BASE, SEMAPHORE_SIZE, res);
                std::vector<uint32_t> filtered_res;
                constexpr static uint32_t num_u32_to_skip = UINT32_WORDS_PER_SEMAPHORE * sizeof(uint32_t);
                for (int i = 0; i < res.size(); i+=num_u32_to_skip) {
                    filtered_res.push_back(res.at(i));
                }
                TT_FATAL(filtered_res == golden_sem_values);
            }
        }
    }
}

bool test_program_specified_with_core_range_set(tt_metal::Device *device, tt_metal::Program &program, const CoreRangeSet &core_range_set) {

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    bool pass = true;
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 4;
    uint32_t buffer_size = single_tile_size * num_tiles;

    auto src_dram_buffer = CreateBuffer(device, buffer_size, buffer_size, tt_metal::BufferType::DRAM);
    auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();

    std::map<CoreCoord, tt_metal::Buffer> core_to_l1_buffer;
    for (auto core_range : core_range_set.ranges()) {
        auto start = core_range.start;
        auto end = core_range.end;
        for (auto x = start.x; x <= end.x; x++) {
            for (auto y = start.y; y <= end.y; y++) {
                CoreCoord logical_core({.x=x, .y=y});
                auto dst_l1_buffer = CreateBuffer(device, buffer_size, buffer_size, tt_metal::BufferType::L1);
                core_to_l1_buffer.emplace(logical_core, dst_l1_buffer);
            }
        }
    }

    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
    // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 8;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core_range_set, cb_src0_config);

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 1;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(ouput_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core_range_set, cb_output_config);

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core_range_set,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core_range_set,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    // Each core range shares the same compute kernel args
    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles) // per_core_tile_cnt
    };

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core_range_set,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    auto size_per_semaphore = SEMAPHORE_SIZE / NUM_SEMAPHORES;
    std::vector<uint32_t> golden_sem_values;
    for (uint32_t i = 0; i < NUM_SEMAPHORES; i++) {
        uint32_t initial_value = i;
        auto semaphore_addr = tt_metal::CreateSemaphore(program, core_range_set, initial_value);
        golden_sem_values.push_back(initial_value);
        pass &= semaphore_addr == SEMAPHORE_BASE + (size_per_semaphore * i);
    }

    check_program_is_mapped_to_correct_cores(program, core_range_set, compute_kernel_args);

    tt_metal::detail::CompileProgram(device, program);

    pass &= tt_metal::detail::ConfigureDeviceWithProgram(device, program);

    check_semaphores_are_initialized(device, core_range_set, golden_sem_values);

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

    // Reader kernel on all cores reads from same location in DRAM
    std::vector<uint32_t> reader_rt_args = {
        src_dram_buffer.address(),
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,
        num_tiles
    };

    for (const auto &[core, dst_l1_buffer] : core_to_l1_buffer) {
        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel,
            core,
            reader_rt_args);

        auto l1_dst_noc_xy = dst_l1_buffer.noc_coordinates();
        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dst_l1_buffer.address(),
            (std::uint32_t)l1_dst_noc_xy.x,
            (std::uint32_t)l1_dst_noc_xy.y,
            num_tiles});
    }


    tt_metal::detail::LaunchProgram(device, program);

    for (const auto &[core, dst_l1_buffer] : core_to_l1_buffer) {
        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_l1_buffer, result_vec);
        bool copied_data_correctly = src_vec == result_vec;
        TT_FATAL(copied_data_correctly);
        pass &= copied_data_correctly;
    }

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;
    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);



        tt_metal::Program program = tt_metal::CreateProgram();
        CoreRange core_range_one = {.start={0, 0}, .end={1, 1}};
        CoreRange core_range_two = {.start={2, 2}, .end={3, 3}};
        CoreRangeSet core_ranges = CoreRangeSet({core_range_one, core_range_two});

        pass &= test_program_specified_with_core_range_set(device, program, core_ranges);

        ////////////////////////////////////////////////////////////////////////////
        //                              Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CloseDevice(device);

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

    TT_FATAL(pass);

    return 0;
}
