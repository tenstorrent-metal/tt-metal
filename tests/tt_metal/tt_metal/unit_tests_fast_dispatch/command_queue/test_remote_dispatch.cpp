// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"

using namespace tt::tt_metal;

namespace remote_tests {

TEST_F(CommandQueueMultiDeviceFixture, DummyTest) {
    std::cout << "here" << std::endl;
    auto device = devices_[1];
    CommandQueue &remote_cq = detail::GetCommandQueue(device);

    uint32_t one_gb = 1 << 30;
    std::vector<uint32_t> zero(one_gb/sizeof(uint32_t), 0);
    tt::Cluster::instance().write_sysmem(zero.data(), one_gb, 0, 0, 1);

    uint32_t num_pages = 1;
    uint32_t page_size = 2048;
    uint32_t buff_size = num_pages * page_size;
    Buffer bufa(device, buff_size, page_size, BufferType::DRAM);

    std::vector<uint32_t> src(buff_size / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < src.size(); i++) {
        src.at(i) = i;
    }

    EnqueueWriteBuffer(remote_cq, bufa, src.data(), false);

    std::vector<uint32_t> readback_data;
    readback_data.resize(DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER);
    tt::Cluster::instance().read_sysmem(readback_data.data(), DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER * sizeof(uint32_t), 96, 0, 1);
    tt::test_utils::print_vector_fixed_numel_per_row(readback_data, 32);
}

}   // namespace remote_tests
