/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "n300_device_fixture.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

constexpr std::int32_t WORD_SIZE = 16;  // 16 bytes per eth send packet
constexpr std::int32_t MAX_NUM_WORDS = eth_l1_mem::address_map::ERISC_APP_RESERVED_SIZE / WORD_SIZE;

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::erisc::direct_send {
// Tests ethernet direct send/receive from ERISC_APP_RESERVED_BASE
bool send_over_eth(
    const tt_metal::Device& sender_device,
    const tt_metal::Device& receiver_device,
    const CoreCoord& sender_core,
    const CoreCoord& receiver_core,
    const size_t& byte_size) {
    tt::log_debug(
        tt::LogTest,
        "Running direct send test with sender chip {} core {}, receiver chip {} core {}, sending {} bytes",
        sender_device.id(),
        sender_core.str(),
        receiver_device.id(),
        receiver_core.str(),
        byte_size);
    std::vector<CoreCoord> eth_cores = {
        {.x = 9, .y = 0},
        {.x = 1, .y = 0},
        {.x = 8, .y = 0},
        {.x = 2, .y = 0},
        {.x = 9, .y = 6},
        {.x = 1, .y = 6},
        {.x = 8, .y = 6},
        {.x = 2, .y = 6},
        {.x = 7, .y = 0},
        {.x = 3, .y = 0},
        {.x = 6, .y = 0},
        {.x = 4, .y = 0},
        {.x = 7, .y = 6},
        {.x = 3, .y = 6},
        {.x = 6, .y = 6},
        {.x = 4, .y = 6}};

    // Disable all eth core runtime app flags, zero out data write counter
    std::vector<uint32_t> run_test_app_flag = {0x0};
    for (const auto& eth_core : eth_cores) {
        llrt::write_hex_vec_to_core(
            sender_device.id(), eth_core, run_test_app_flag, eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
        llrt::write_hex_vec_to_core(
            receiver_device.id(), eth_core, run_test_app_flag, eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
        std::vector<uint32_t> zero = {0, 0};
        llrt::write_hex_vec_to_core(
            sender_device.id(), eth_core, zero, eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
        llrt::write_hex_vec_to_core(
            receiver_device.id(), eth_core, zero, eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
    }

    // TODO: is it possible that receiver core app is stil running when we push inputs here???
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    llrt::write_hex_vec_to_core(
        sender_device.id(), sender_core, inputs, eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE);

    // Zero out receiving address to ensure no stale data is causing tests to pass
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    llrt::write_hex_vec_to_core(
        receiver_device.id(), receiver_core, all_zeros, eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE);

    uint32_t arg_0 = byte_size;
    std::vector<uint32_t> args = {arg_0, 0};
    llrt::write_hex_vec_to_core(sender_device.id(), sender_core, args, eth_l1_mem::address_map::ERISC_L1_ARG_BASE);
    args = {arg_0, 1};
    llrt::write_hex_vec_to_core(receiver_device.id(), receiver_core, args, eth_l1_mem::address_map::ERISC_L1_ARG_BASE);

    // TODO: this should be updated to use kernel api
    ll_api::memory binary_mem_send = llrt::get_risc_binary("erisc/erisc_app.hex", sender_device.id(), true);
    ll_api::memory binary_mem_receive = llrt::get_risc_binary("erisc/erisc_app.hex", receiver_device.id(), true);

    for (const auto& eth_core : eth_cores) {
        llrt::write_hex_vec_to_core(
            sender_device.id(), eth_core, binary_mem_send.data(), eth_l1_mem::address_map::FIRMWARE_BASE);
        llrt::write_hex_vec_to_core(
            receiver_device.id(), eth_core, binary_mem_receive.data(), eth_l1_mem::address_map::FIRMWARE_BASE);
    }

    // Activate sender core runtime app
    run_test_app_flag = {0x1};
    // send remote first, otherwise eth core may be blocked, very ugly for now...
    if (receiver_device.id() == 1) {
        llrt::write_hex_vec_to_core(
            1, receiver_core, run_test_app_flag, eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
    } else {
        llrt::write_hex_vec_to_core(1, sender_core, run_test_app_flag, eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
    }
    if (sender_device.id() == 0) {
        llrt::write_hex_vec_to_core(0, sender_core, run_test_app_flag, eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
    } else {
        llrt::write_hex_vec_to_core(
            0, receiver_core, run_test_app_flag, eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
    }

    bool pass = true;
    auto readback_vec = llrt::read_hex_vec_from_core(
        receiver_device.id(), receiver_core, eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE, byte_size);
    pass &= (readback_vec == inputs);

    return pass;
}

}  // namespace unit_tests::erisc::direct_send

TEST_F(N300DeviceFixture, SingleEthCoreDirectSendChip0ToChip1) {
    ASSERT_TRUE(this->num_devices_ == 2);
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);
    CoreCoord sender_core_0 = {.x = 9, .y = 6};
    CoreCoord sender_core_1 = {.x = 1, .y = 6};

    CoreCoord receiver_core_0 = {.x = 9, .y = 0};
    CoreCoord receiver_core_1 = {.x = 1, .y = 0};

    ASSERT_TRUE(
        unit_tests::erisc::direct_send::send_over_eth(device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE));
    ASSERT_TRUE(
        unit_tests::erisc::direct_send::send_over_eth(device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * MAX_NUM_WORDS));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * MAX_NUM_WORDS));
}

TEST_F(N300DeviceFixture, SingleEthCoreDirectSendChip1ToChip0) {
    ASSERT_TRUE(this->num_devices_ == 2);
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);
    CoreCoord sender_core_0 = {.x = 9, .y = 0};
    CoreCoord sender_core_1 = {.x = 1, .y = 0};

    CoreCoord receiver_core_0 = {.x = 9, .y = 6};
    CoreCoord receiver_core_1 = {.x = 1, .y = 6};

    ASSERT_TRUE(
        unit_tests::erisc::direct_send::send_over_eth(device_1, device_0, sender_core_0, receiver_core_0, WORD_SIZE));
    ASSERT_TRUE(
        unit_tests::erisc::direct_send::send_over_eth(device_1, device_0, sender_core_1, receiver_core_1, WORD_SIZE));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, sender_core_0, receiver_core_0, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, sender_core_1, receiver_core_1, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, sender_core_0, receiver_core_0, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, sender_core_1, receiver_core_1, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, sender_core_0, receiver_core_0, WORD_SIZE * MAX_NUM_WORDS));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, sender_core_1, receiver_core_1, WORD_SIZE * MAX_NUM_WORDS));
}

TEST_F(N300DeviceFixture, BidirectionalEthCoreDirectSend) {
    ASSERT_TRUE(this->num_devices_ == 2);
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);
    CoreCoord sender_core_0 = {.x = 9, .y = 6};
    CoreCoord sender_core_1 = {.x = 1, .y = 6};

    CoreCoord receiver_core_0 = {.x = 9, .y = 0};
    CoreCoord receiver_core_1 = {.x = 1, .y = 0};

    ASSERT_TRUE(
        unit_tests::erisc::direct_send::send_over_eth(device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE));
    ASSERT_TRUE(
        unit_tests::erisc::direct_send::send_over_eth(device_1, device_0, receiver_core_0, sender_core_0, WORD_SIZE));
    ASSERT_TRUE(
        unit_tests::erisc::direct_send::send_over_eth(device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE));
    ASSERT_TRUE(
        unit_tests::erisc::direct_send::send_over_eth(device_1, device_0, receiver_core_1, sender_core_1, WORD_SIZE));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, receiver_core_0, sender_core_0, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, receiver_core_1, sender_core_1, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, receiver_core_0, sender_core_0, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, receiver_core_1, sender_core_1, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * MAX_NUM_WORDS));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, receiver_core_0, sender_core_0, WORD_SIZE * MAX_NUM_WORDS));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * MAX_NUM_WORDS));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        device_1, device_0, receiver_core_1, sender_core_1, WORD_SIZE * MAX_NUM_WORDS));
}

TEST_F(N300DeviceFixture, RandomDirectSendTests) {
    srand(0);
    ASSERT_TRUE(this->num_devices_ == 2);

    std::map<std::pair<int, CoreCoord>, std::pair<int, CoreCoord>> connectivity = {
        {{0, {.x = 9, .y = 6}}, {1, {.x = 9, .y = 0}}},
        {{1, {.x = 9, .y = 0}}, {0, {.x = 9, .y = 6}}},
        {{0, {.x = 1, .y = 6}}, {1, {.x = 1, .y = 0}}},
        {{1, {.x = 1, .y = 0}}, {0, {.x = 1, .y = 6}}}};
    for (int i = 0; i < 1000; i++) {
        auto it = connectivity.begin();
        std::advance(it, rand() % (connectivity.size()));

        const auto& send_chip = devices_.at(std::get<0>(it->first));
        CoreCoord sender_core = std::get<1>(it->first);
        const auto& receiver_chip = devices_.at(std::get<0>(it->second));
        CoreCoord receiver_core = std::get<1>(it->second);
        int num_words = rand() % MAX_NUM_WORDS + 1;

        ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
            send_chip, receiver_chip, sender_core, receiver_core, WORD_SIZE * num_words));
    }
}
