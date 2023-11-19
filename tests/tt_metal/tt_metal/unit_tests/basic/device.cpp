// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "basic_fixture.hpp"
#include "device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::basic::device {

/// @brief Ping number of bytes for specified grid_size
/// @param device
/// @param byte_size - size in bytes
/// @param l1_byte_address - l1 address to target for all cores
/// @param grid_size - grid size. will ping all cores from {0,0} to grid_size (non-inclusive)
/// @return
bool l1_ping(
    const tt_metal::Device& device, const size_t& byte_size, const size_t& l1_byte_address, const CoreCoord& grid_size) {
    bool pass = true;
    auto inputs = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
    for (int y = 0; y < grid_size.y; y++) {
        for (int x = 0; x < grid_size.x; x++) {
            CoreCoord dest_core({.x = static_cast<size_t>(x), .y = static_cast<size_t>(y)});
            tt_metal::detail::WriteToDeviceL1(device, dest_core, l1_byte_address, inputs);
        }
    }

    for (int y = 0; y < grid_size.y; y++) {
        for (int x = 0; x < grid_size.x; x++) {
            CoreCoord dest_core({.x = static_cast<size_t>(x), .y = static_cast<size_t>(y)});
            std::vector<uint32_t> dest_core_data;
            tt_metal::detail::ReadFromDeviceL1(device, dest_core, l1_byte_address, byte_size, dest_core_data);
            pass &= (dest_core_data == inputs);
            if (not pass) {
                log_error("Mismatch at Core: ={}", dest_core.str());
            }
        }
    }
    return pass;
}

/// @brief Ping number of bytes for specified channels
/// @param device
/// @param byte_size - size in bytes
/// @param l1_byte_address - l1 address to target for all cores
/// @param num_channels - num_channels. will ping all channels from {0} to num_channels (non-inclusive)
/// @return
bool dram_ping(
    const tt_metal::Device& device,
    const size_t& byte_size,
    const size_t& dram_byte_address,
    const unsigned int& num_channels) {
    bool pass = true;
    auto inputs = generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
    for (unsigned int channel = 0; channel < num_channels; channel++) {
        tt_metal::detail::WriteToDeviceDRAMChannel(device, channel, dram_byte_address, inputs);
    }

    for (unsigned int channel = 0; channel < num_channels; channel++) {
        std::vector<uint32_t> dest_channel_data;
        tt_metal::detail::ReadFromDeviceDRAMChannel(device, channel, dram_byte_address, byte_size, dest_channel_data);
        pass &= (dest_channel_data == inputs);
        if (not pass) {
            cout << "Mismatch at Channel: " << channel << std::endl;
        }
    }
    return pass;
}

/// @brief load_blank_kernels into all cores and will launch
/// @param device
/// @return
bool load_all_blank_kernels(const tt_metal::Device& device) {
    bool pass = true;
    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord compute_grid_size = device.compute_with_storage_grid_size();
    CoreRange all_cores = CoreRange(
        CoreCoord{.x = 0, .y = 0},
        CoreCoord{.x = compute_grid_size.x - 1, .y = compute_grid_size.y -1}
    );
    CreateKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CreateKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", all_cores, ComputeConfig{});

    tt_metal::detail::LaunchProgram(device, program);
    return pass;
}
}  // namespace unit_tests::basic::device


TEST_F(BasicFixture, MultiDeviceInitializeAndTeardown) {
    auto arch = tt::get_arch_from_string(get_env_arch_name());
    const size_t num_devices = tt::tt_metal::Device::detect_num_available_devices();
    if (is_multi_device_gs_machine(arch, num_devices)) {
        GTEST_SKIP();
    }
    ASSERT_TRUE(num_devices > 0);
    std::vector<std::reference_wrapper<const tt_metal::Device>> devices;

    try
    {
        for (unsigned int id = 0; id < num_devices; id++) {
            devices.emplace_back(tt::tt_metal::CreateDevice(id));
        }
    } catch (...) {}
    for (auto device : devices) {
        ASSERT_TRUE(tt::tt_metal::CloseDevice(device));
    }
}
TEST_F(BasicFixture, MultiDeviceLoadBlankKernels) {
    auto arch = tt::get_arch_from_string(get_env_arch_name());
    const size_t num_devices = tt::tt_metal::Device::detect_num_available_devices();
    if (is_multi_device_gs_machine(arch, num_devices)) {
        GTEST_SKIP();
    }
    ASSERT_TRUE(num_devices > 0);
    std::vector<std::reference_wrapper<const tt::tt_metal::Device>> devices;

    try
    {
        for (unsigned int id = 0; id < num_devices; id++) {
            devices.emplace_back(tt::tt_metal::CreateDevice(id));
        }
        for (unsigned int id = 0; id < num_devices; id++) {
            unit_tests::basic::device::load_all_blank_kernels(devices.at(id));
        }
    } catch (...) {}
    for (auto device: devices) {
        ASSERT_TRUE(tt::tt_metal::CloseDevice(device));
    }
}

TEST_F(BasicFixture, SingleDeviceInitializeAndTeardown) {
    auto arch = tt::get_arch_from_string(get_env_arch_name());
    const tt::tt_metal::Device& device  = tt::tt_metal::CreateDevice(0);
    ASSERT_TRUE(tt::tt_metal::CloseDevice(device));
}
TEST_F(BasicFixture, SingleDeviceHarvestingPrints) {
    auto arch = tt::get_arch_from_string(get_env_arch_name());
    const tt::tt_metal::Device& device = tt::tt_metal::CreateDevice(0);
    CoreCoord unharvested_logical_grid_size = {.x = 12, .y = 10};
    if (arch == tt::ARCH::WORMHOLE_B0) {
        unharvested_logical_grid_size = {.x = 8, .y = 10};
    }
    auto logical_grid_size = device.logical_grid_size();
    if (logical_grid_size == unharvested_logical_grid_size) {
        tt::log_info("Harvesting Disabled in SW");
    } else {
        tt::log_info("Harvesting Enabled in SW");
        tt::log_info("Number of Harvested Rows={}", unharvested_logical_grid_size.y - logical_grid_size.y);
    }

    tt::log_info("Logical -- Noc Coordinates Mapping");
    tt::log_info("[Logical <-> NOC0] Coordinates");
    for (int r = 0; r < logical_grid_size.y; r++) {
        string output_row = "";
        for (int c = 0; c < logical_grid_size.x; c++) {
            const CoreCoord logical_coord(c, r);
            const auto noc_coord = device.worker_core_from_logical_core(logical_coord);
            output_row += "{L[x" + std::to_string(c);
            output_row += "-y" + std::to_string(r);
            output_row += "]:N[x" + std::to_string(noc_coord.x);
            output_row += "-y" + std::to_string(noc_coord.y);
            output_row += "]}, ";
        }
        tt::log_info("{}", output_row);
    }
    ASSERT_TRUE(tt::tt_metal::CloseDevice(device));
}

TEST_F(BasicFixture, SingleDeviceLoadBlankKernels) {
    auto arch = tt::get_arch_from_string(get_env_arch_name());
    const unsigned int device_id = 0;
    const tt::tt_metal::Device& device = tt::tt_metal::CreateDevice(device_id);
    unit_tests::basic::device::load_all_blank_kernels(device);
    ASSERT_TRUE(tt::tt_metal::CloseDevice(device));
}
TEST_F(DeviceFixture, PingAllLegalDramChannels) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        {
            size_t start_byte_address = DRAM_UNRESERVED_BASE;
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 4, start_byte_address, devices_.at(id).get().num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 12, start_byte_address, devices_.at(id).get().num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 16, start_byte_address, devices_.at(id).get().num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 1024, start_byte_address, devices_.at(id).get().num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 2 * 1024, start_byte_address, devices_.at(id).get().num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 32 * 1024, start_byte_address, devices_.at(id).get().num_dram_channels()));
        }
        {
            size_t start_byte_address = devices_.at(id).get().dram_size_per_channel() - 32 * 1024;
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 4, start_byte_address, devices_.at(id).get().num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 12, start_byte_address, devices_.at(id).get().num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 16, start_byte_address, devices_.at(id).get().num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 1024, start_byte_address, devices_.at(id).get().num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 2 * 1024, start_byte_address, devices_.at(id).get().num_dram_channels()));
            ASSERT_TRUE(unit_tests::basic::device::dram_ping(
                devices_.at(id), 32 * 1024, start_byte_address, devices_.at(id).get().num_dram_channels()));
        }
    }
}
TEST_F(DeviceFixture, PingIllegalDramChannels) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto num_channels = devices_.at(id).get().num_dram_channels() + 1;
        size_t start_byte_address = DRAM_UNRESERVED_BASE;
        ASSERT_ANY_THROW(unit_tests::basic::device::dram_ping(devices_.at(id), 4, start_byte_address, num_channels));
    }
}

TEST_F(DeviceFixture, PingAllLegalL1Cores) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        {
            size_t start_byte_address = L1_UNRESERVED_BASE;  // FIXME: Should remove dependency on
                                                             // hostdevcommon/common_runtime_address_map.h header.
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 4, start_byte_address, devices_.at(id).get().logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 12, start_byte_address, devices_.at(id).get().logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 16, start_byte_address, devices_.at(id).get().logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 1024, start_byte_address, devices_.at(id).get().logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 2 * 1024, start_byte_address, devices_.at(id).get().logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 32 * 1024, start_byte_address, devices_.at(id).get().logical_grid_size()));
        }
        {
            size_t start_byte_address = devices_.at(id).get().l1_size_per_core() - 32 * 1024;
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 4, start_byte_address, devices_.at(id).get().logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 12, start_byte_address, devices_.at(id).get().logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 16, start_byte_address, devices_.at(id).get().logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 1024, start_byte_address, devices_.at(id).get().logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 2 * 1024, start_byte_address, devices_.at(id).get().logical_grid_size()));
            ASSERT_TRUE(unit_tests::basic::device::l1_ping(
                devices_.at(id), 32 * 1024, start_byte_address, devices_.at(id).get().logical_grid_size()));
        }
    }
}

TEST_F(DeviceFixture, PingIllegalL1Cores) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto grid_size = devices_.at(id).get().logical_grid_size();
        grid_size.x++;
        grid_size.y++;
        size_t start_byte_address = L1_UNRESERVED_BASE;  // FIXME: Should remove dependency on
                                                         // hostdevcommon/common_runtime_address_map.h header.
        ASSERT_ANY_THROW(unit_tests::basic::device::l1_ping(devices_.at(id), 4, start_byte_address, grid_size));
    }
}

// Harvesting tests

// Test methodology:
// 1. Host write single uint32_t value to each L1 bank
// 2. Launch a kernel to read and increment the value in each bank
// 3. Host validates that the value from step 1 has been incremented
// Purpose of this test is to ensure that L1 reader/writer APIs do not target harvested cores
TEST_F(DeviceFixture, ValidateKernelDoesNotTargetHarvestedCores) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        uint32_t num_l1_banks = this->devices_.at(id).get().num_banks(BufferType::L1);
        std::vector<uint32_t> host_input(1);
        std::map<uint32_t, uint32_t> bank_id_to_value;
        uint32_t l1_address = this->devices_.at(id).get().l1_size_per_core() - 2048;
        for (uint32_t bank_id = 0; bank_id < num_l1_banks; bank_id++) {
            host_input[0] = bank_id + 1;
            bank_id_to_value[bank_id] = host_input.at(0);
            CoreCoord logical_core = this->devices_.at(id).get().logical_core_from_bank_id(bank_id);
            uint32_t write_address = l1_address + this->devices_.at(id).get().l1_bank_offset_from_bank_id(bank_id);
            tt_metal::detail::WriteToDeviceL1(this->devices_.at(id), logical_core, write_address, host_input);
        }

        tt_metal::Program program = tt_metal::CreateProgram();
        string kernel_name = "tests/tt_metal/tt_metal/test_kernels/misc/ping_legal_l1s.cpp";
        CoreCoord logical_target_core = CoreCoord({.x = 0, .y = 0});
        uint32_t intermediate_l1_addr = L1_UNRESERVED_BASE;
        uint32_t size_bytes = host_input.size() * sizeof(uint32_t);
        tt_metal::KernelID kernel_id = tt_metal::CreateKernel(
            program,
            kernel_name,
            logical_target_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = {l1_address, intermediate_l1_addr, size_bytes}});

        tt_metal::detail::LaunchProgram(this->devices_.at(id), program);

        std::vector<uint32_t> output;
        for (uint32_t bank_id = 0; bank_id < num_l1_banks; bank_id++) {
            CoreCoord logical_core = this->devices_.at(id).get().logical_core_from_bank_id(bank_id);
            uint32_t read_address = l1_address + this->devices_.at(id).get().l1_bank_offset_from_bank_id(bank_id);
            tt_metal::detail::ReadFromDeviceL1(this->devices_.at(id), logical_core, read_address, size_bytes, output);
            ASSERT_TRUE(output.size() == host_input.size());
            uint32_t expected_value =
                bank_id_to_value.at(bank_id) + 1;  // ping_legal_l1s kernel increments each value it reads
            ASSERT_TRUE(output.at(0) == expected_value) << "Logical core " + logical_core.str() + " should have " +
                                                               std::to_string(expected_value) + " but got " +
                                                               std::to_string(output.at(0));
        }
    }
}
