// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::tt_metal;
using DistributedTensor = std::vector<Tensor>;
namespace tt {

namespace tt_metal {

struct AllGather2 {
    const uint32_t dim;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const CoreCoord eth_sender_core;
    const CoreCoord eth_receiver_core;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

operation::ProgramWithCallbacks all_gather2_multi_core(
    const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dim, const uint32_t ring_size, const uint32_t ring_index, const CoreCoord eth_sender_core, const CoreCoord eth_receiver_core);

inline std::vector<std::tuple<Device*, Device*, CoreCoord, CoreCoord>> get_sender_receiver_cores(
    std::vector<tt::tt_metal::Device*> device_ring) {
    std::vector<std::tuple<Device*, Device*, CoreCoord, CoreCoord>> sender_receivers;
    sender_receivers.reserve(device_ring.size() - 1);

    // Special case for 2 devices to ensure core pairs are not the same for send and receive
    if (device_ring.size() - 1 == 2) {
        const auto& first_device = device_ring[0];
        const auto& second_device = device_ring[1];
        uint32_t i = 0;
        for (const auto& first_eth_core : first_device->get_active_ethernet_cores()) {
            auto [device_id, second_eth_core] = first_device->get_connected_ethernet_core(first_eth_core);
            if (second_device->id() == device_id) {
                Device *sender_device, *receiver_device;
                CoreCoord sender_eth_core, receiver_eth_core;
                if (i == 0) {
                    sender_device = first_device, receiver_device = second_device;
                    sender_eth_core = first_eth_core, receiver_eth_core = second_eth_core;
                } else {
                    sender_device = second_device, receiver_device = first_device;
                    sender_eth_core = second_eth_core, receiver_eth_core = first_eth_core;
                }
                sender_receivers.push_back({sender_device, receiver_device, sender_eth_core, receiver_eth_core});
                log_debug(
                    tt::LogOp,
                    "Sender: {} Receiver: {} Sender Eth: {} Receiver Eth: {}",
                    sender_device->id(),
                    receiver_device->id(),
                    sender_eth_core.str(),
                    receiver_eth_core.str());
                if (i > 0) {
                    break;
                }
                i++;
            }
        }
    } else {
        for (uint32_t i = 0; i < device_ring.size() - 1; ++i) {
            const auto& sender_device = device_ring[i];
            const auto& receiver_device = device_ring[i + 1];
            for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores()) {
                auto [device_id, receiver_eth_core] = sender_device->get_connected_ethernet_core(sender_eth_core);
                if (receiver_device->id() == device_id) {
                    sender_receivers.push_back({sender_device, receiver_device, sender_eth_core, receiver_eth_core});
                    log_debug(
                        tt::LogOp,
                        "Sender: {} Receiver: {} Sender Eth: {} Receiver Eth: {}",
                        sender_device->id(),
                        receiver_device->id(),
                        sender_eth_core.str(),
                        receiver_eth_core.str());
                    break;
                }
            }
        }
    }
    return sender_receivers;
}

inline std::vector<Tensor> all_gather2(const DistributedTensor &input_tensors, uint32_t dim, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {

    std::vector<Device *> device_ring;
    device_ring.reserve(input_tensors.size() + 1);
    for (const auto& i : input_tensors) {
        device_ring.push_back(i.device());
    }
    device_ring.push_back(input_tensors[0].device());

    const auto& sender_receivers = get_sender_receiver_cores(device_ring);
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());

    for (uint32_t i = 0; i < sender_receivers.size(); ++i) {
        auto& device = std::get<0>(sender_receivers[i]);
        const auto& eth_sender_core = std::get<2>(sender_receivers[i]);
        CoreCoord eth_receiver_core;
        bool found = false;
        for (uint32_t j = 0; j < sender_receivers.size(); ++j) {
            if (std::get<1>(sender_receivers[j])->id() == device->id()) {
                eth_receiver_core = std::get<3>(sender_receivers[j]);
                found = true;
                break;
            }
        }
        TT_FATAL(found, "No connection found");
        output_tensors.push_back(operation::run(AllGather2{dim, (uint32_t)input_tensors.size(), i, eth_sender_core, eth_receiver_core, output_mem_config}, {input_tensors.at(i)}).at(0));
    }
    return output_tensors;
}

}  // namespace tt_metal

}  // namespace tt
