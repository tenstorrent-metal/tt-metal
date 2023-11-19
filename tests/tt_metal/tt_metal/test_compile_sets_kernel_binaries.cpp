// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/llrt/tt_memory.h"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

std::string get_latest_kernel_binary_path(int device_id, const tt_metal::Kernel *kernel) {
    auto root_dir = get_kernel_compile_outpath(device_id);
    TT_FATAL(kernel != nullptr);
    TT_FATAL(std::filesystem::exists(root_dir + kernel->name()));

    std::filesystem::path kernel_path{root_dir + kernel->name()};
    std::filesystem::file_time_type ftime = std::filesystem::last_write_time(*kernel_path.begin());
    std::string latest_hash;
    for (auto const& dir_entry : std::filesystem::directory_iterator{kernel_path}) {
        auto kbtime = std::filesystem::last_write_time(dir_entry.path());
        if (kbtime > ftime) {
            ftime = kbtime;
            latest_hash = dir_entry.path().filename().string();
        }
    }
    TT_FATAL(not latest_hash.empty());
    return kernel->name() + "/" + latest_hash;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        const tt_metal::Device& device =
            tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();

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

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 8;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t num_output_tiles = 1;
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        auto unary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            uint(num_tiles) // per_core_tile_cnt
        };

        auto eltwise_unary_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        // Check that binary memory objects in the kernel match the ones obtained from the persistent cache
        const KernelGroup *kernel_group = program.kernels_on_core(core);
        TT_FATAL(kernel_group != nullptr && kernel_group->compute_id.has_value() and kernel_group->riscv0_id.has_value() and kernel_group->riscv1_id.has_value());
        tt_metal::Kernel *compute_kernel = tt_metal::detail::GetKernel(program, kernel_group->compute_id.value());
        tt_metal::Kernel *riscv0_kernel = tt_metal::detail::GetKernel(program, kernel_group->riscv0_id.value());
        tt_metal::Kernel *riscv1_kernel = tt_metal::detail::GetKernel(program, kernel_group->riscv1_id.value());
        std::vector<string> kernel_names = {"reader_unary_push_4", "writer_unary", "eltwise_copy_3m"};
        for (auto kernel_name : kernel_names) {
            std::filesystem::remove_all(get_kernel_compile_outpath(device.id()) + kernel_name);
        }

        int num_compiles = 3;
        // kernel->binaries() returns 32B aligned binaries
        std::vector<ll_api::memory> compute_binaries;
        std::vector<ll_api::memory> brisc_binaries;
        std::vector<ll_api::memory> ncrisc_binaries;
        for (int i = 0; i < num_compiles; i++) {
            tt_metal::detail::CompileProgram(device, program);
            if (i == 0) {
                compute_binaries = compute_kernel->binaries(device.id());
                TT_FATAL(compute_binaries.size() == 3, "Expected 3 Compute binaries!");
                brisc_binaries = riscv0_kernel->binaries(device.id());
                TT_FATAL(brisc_binaries.size() == 1, "Expected 1 BRISC binary!");
                ncrisc_binaries = riscv1_kernel->binaries(device.id());
                TT_FATAL(ncrisc_binaries.size() == 1, "Expected 1 NCRISC binary!");
            } else {
                TT_FATAL(compute_kernel->binaries(device.id()) == compute_binaries);
                TT_FATAL(riscv0_kernel->binaries(device.id()) == brisc_binaries);
                TT_FATAL(riscv1_kernel->binaries(device.id()) == ncrisc_binaries);
            }
            std::string brisc_hex_path = get_latest_kernel_binary_path(device.id(), riscv0_kernel) + "/brisc/brisc.hex";
            ll_api::memory brisc_binary = llrt::get_risc_binary(brisc_hex_path, device.id(), false);
            TT_FATAL(brisc_binary == brisc_binaries.at(0), "Expected saved BRISC binary to be the same as binary in persistent cache");
            std::string ncrisc_hex_path = get_latest_kernel_binary_path(device.id(), riscv1_kernel) + "/ncrisc/ncrisc.hex";
            ll_api::memory ncrisc_binary = llrt::get_risc_binary(ncrisc_hex_path, device.id(), false);
            TT_FATAL(ncrisc_binary == ncrisc_binaries.at(0), "Expected saved NCRISC binary to be the same as binary in persistent cache");
            for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
                std::string trisc_id_str = std::to_string(trisc_id);
                std::string trisc_hex_path = get_latest_kernel_binary_path(device.id(), compute_kernel) + "/tensix_thread" + trisc_id_str + "/tensix_thread" + trisc_id_str + ".hex";
                ll_api::memory trisc_binary = llrt::get_risc_binary(trisc_hex_path, device.id(), false);
                TT_FATAL(trisc_binary == compute_binaries.at(trisc_id), "Expected saved TRISC binary for " + trisc_id_str + " to be the same as binary in persistent cache");
            }
        }

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
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}
