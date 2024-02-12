// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "command_queue_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

constexpr std::int32_t WORD_SIZE = 16;  // 16 bytes per eth send packet
constexpr std::int32_t MAX_NUM_WORDS =
    (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE) / WORD_SIZE;
constexpr std::int32_t MAX_BUFFER_SIZE =
    (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE);

struct BankedConfig {
    size_t num_pages = 1;
    size_t size_bytes = 1 * 2 * 32 * 32;
    size_t page_size_bytes = 2 * 32 * 32;
    BufferType input_buffer_type = BufferType::L1;
    BufferType output_buffer_type = BufferType::L1;
    tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;
};

namespace fd_unit_tests::erisc::kernels {

bool eth_interleaved_readback_kernels(
    std::vector<tt::tt_metal::Device*> device_ring,
    const BankedConfig& cfg,
    const size_t& src_eth_l1_byte_address,
    const size_t& dst_eth_l1_byte_address,
    const size_t& sem_l1_byte_address,
    uint32_t num_bytes_per_send = 16) {
    bool pass = true;

    std::vector<CoreCoord> cores;
    auto input = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
                -1.0f, 1.0f, cfg.size_bytes / tt::test_utils::df::bfloat16::SIZEOF, 0);
    std::vector<uint32_t> all_zeros(cfg.size_bytes / sizeof(uint32_t), 0);
    std::vector<std::shared_ptr<Buffer>> input_buffers;
    std::vector<std::shared_ptr<Buffer>> output_buffers;
    std::vector<Program> programs;
    for (uint32_t i = 0; i < device_ring.size() - 1; ++i) {
        input_buffers.emplace_back(CreateBuffer(InterleavedBufferConfig{device_ring[i], cfg.size_bytes, cfg.page_size_bytes, cfg.input_buffer_type}));
        tt_metal::detail::WriteToBuffer(input_buffers[i], input);
        output_buffers.emplace_back(CreateBuffer(InterleavedBufferConfig{device_ring[i], cfg.size_bytes, cfg.page_size_bytes, cfg.output_buffer_type}));
        tt_metal::detail::WriteToBuffer(output_buffers[i], all_zeros);
        cores.emplace_back(device_ring[i]->get_ethernet_sockets(device_ring[i+1]->id())[0]);
        programs.emplace_back(Program{});
        auto eth_sender_kernel = tt_metal::CreateKernel(
            programs[i],
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_eth_read_write.cpp",
            cores[i],
            tt_metal::experimental::EthernetConfig{
                .eth_mode = tt_metal::Eth::SENDER,
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = {
                    uint32_t(input_buffers[i]->buffer_type() == BufferType::DRAM),
                    uint32_t(output_buffers[i]->buffer_type() == BufferType::DRAM)}});
            tt_metal::SetRuntimeArgs(
                programs[i],
                eth_sender_kernel,
                cores[i],
                {
                (uint32_t)(src_eth_l1_byte_address),
                (uint32_t)(input_buffers[i]->address()),
                (uint32_t)(output_buffers[i]->address()),
                (uint32_t)(cfg.num_pages),
                (uint32_t)cfg.page_size_bytes});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    std::vector<std::reference_wrapper<CommandQueue>> cqs;
    for (uint32_t i = 0; i < device_ring.size() - 1; ++i) {
        const auto& device = device_ring[i];
        tt::tt_metal::detail::CompileProgram(device, programs[i]);
        auto& cq = device->command_queue();

        EnqueueProgram(cq, programs[i], false);
        cqs.emplace_back(cq);
    }
    for (auto& cq : cqs) {
        Finish(cq);
    }

    for (uint32_t i = 0; i < device_ring.size() - 1; ++i) {
        const auto& device = device_ring[i];
        std::vector<uint32_t> readback_vec;
        tt_metal::detail::ReadFromBuffer(output_buffers[i], readback_vec);
        auto a = std::mismatch(input.begin(), input.end(), readback_vec.begin());
        bool p = (a.first == input.end());
        pass &= p;
        if (not p) {
            log_error(tt::LogTest, "Mismatch on Device {}", device->id());
            log_error(
                tt::LogTest, "Position: {} Expected: {} Read: {}", a.first - input.begin(), *a.first, *a.second);
        }
    }

    return pass;
}
}  // namespace fd_unit_tests::erisc::kernels


TEST_F(CommandQueuePCIDevicesFixture, EthKernelsInterleavedReadbackAllChips) {
    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 32;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 32;
    const size_t sem_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    BankedConfig test_config =
        BankedConfig{.num_pages = 125, .size_bytes = 125 * 512, .page_size_bytes = 512, .input_buffer_type=BufferType::DRAM, .output_buffer_type=BufferType::DRAM};
    std::vector<Device *> device_ring = {devices_[0], devices_[1], devices_[2], devices_[3], devices_[0]};
    if (device_ring.empty()) {
        GTEST_SKIP();
    }
    ASSERT_TRUE(fd_unit_tests::erisc::kernels::eth_interleaved_readback_kernels(
        device_ring, test_config, src_eth_l1_byte_address, dst_eth_l1_byte_address, sem_l1_byte_address));
}
