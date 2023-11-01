// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "basic_fixture.hpp"
#include "device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

// TODO: Uplift to DeviceFixture once it does not skip GS
TEST_F(BasicFixture, TestL1BuffersAllocatedTopDown) {
    // Make sure previous allocs/deallocs don't interfere with this test
    tt::concurrent::remove_shared_memory();
    tt::tt_metal::Device *device = tt::tt_metal::CreateDevice(0);

    std::vector<uint32_t> alloc_sizes = {32 * 1024, 64 * 1024, 128 * 1024};
    // for (unsigned int id = 0; id < num_devices_; id++) {
        size_t total_size_bytes = 0;
        size_t alloc_limit = device->bank_size(tt::tt_metal::BufferType::L1) - 64;
        std::vector<std::unique_ptr<Buffer>> buffers;
        int alloc_size_idx = 0;
        uint32_t total_buffer_size = 0;
        while (total_size_bytes < alloc_limit) {
            uint32_t buffer_size = alloc_sizes.at(alloc_size_idx);
            alloc_size_idx = (alloc_size_idx + 1) % alloc_sizes.size();
            if (total_buffer_size + buffer_size >= alloc_limit) {
                break;
            }
            std::unique_ptr<tt::tt_metal::Buffer> buffer = std::make_unique<tt::tt_metal::Buffer>(device, buffer_size, buffer_size, tt::tt_metal::BufferType::L1);
            buffers.emplace_back(std::move(buffer));
            total_buffer_size += buffer_size;
            EXPECT_EQ(buffers.back()->address(), device->l1_size_per_core() - total_buffer_size);
        }
        buffers.clear();
    // }

    tt::tt_metal::CloseDevice(device);
}

// TODO: Uplift to DeviceFixture once it does not skip GS
TEST_F(BasicFixture, TestL1BuffersDoNotGrowBeyondBankSize) {
    // Make sure previous allocs/deallocs don't interfere with this test
    tt::concurrent::remove_shared_memory();
    tt::tt_metal::Device *device = tt::tt_metal::CreateDevice(0);
    uint32_t alloc_limit = device->bank_size(tt::tt_metal::BufferType::L1);

    EXPECT_ANY_THROW(
        tt::tt_metal::Buffer buffer = tt::tt_metal::CreateBuffer(device, alloc_limit + 64, alloc_limit + 64, tt::tt_metal::BufferType::L1);
    );

    tt::tt_metal::CloseDevice(device);

}
