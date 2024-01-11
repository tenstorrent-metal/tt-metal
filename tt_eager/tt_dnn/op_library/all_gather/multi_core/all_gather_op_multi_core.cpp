// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "eth_l1_address_map.h"
#include "tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

std::vector<std::tuple<Device*, Device*, CoreCoord, CoreCoord>> get_sender_receiver_cores(
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

namespace tt {

namespace tt_metal {

operation::ProgramsWithCallbacks all_gather_multi_core(
    const DistributedTensor& input_tensors, std::vector<Tensor>& output_tensors, uint32_t dim) {

    constexpr uint32_t header_size = 32;
    constexpr uint32_t MAX_BUFFER = eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE - header_size;
    constexpr size_t sem_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    constexpr size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 32;
    constexpr size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 32;

    std::map<Device*, tt_metal::Program> programs;
    std::vector<Device*> device_ring;
    device_ring.reserve(input_tensors.size() + 1);
    for (const auto& t : input_tensors) {
        device_ring.push_back(t.device());
    }
    device_ring.push_back(input_tensors[0].device());
    const auto& sender_receivers = get_sender_receiver_cores(device_ring);

    uint32_t bytes_per_chunk = 0, pages_per_chunk = 0, num_full_chunks = 0, rem_bytes = 0, rem_pages = 0;
    if (input_tensors[0].buffer()->size() > MAX_BUFFER) {
        bytes_per_chunk = round_down(MAX_BUFFER, input_tensors[0].buffer()->page_size());
        pages_per_chunk = bytes_per_chunk / input_tensors[0].buffer()->page_size();
        num_full_chunks = (uint32_t)(input_tensors[0].buffer()->size() / bytes_per_chunk);
        rem_bytes = (uint32_t)(input_tensors[0].buffer()->size() % bytes_per_chunk);
        rem_pages = rem_bytes / input_tensors[0].buffer()->page_size();
    } else {
        rem_bytes = input_tensors[0].buffer()->size();
        rem_pages = rem_bytes / input_tensors[0].buffer()->page_size();
    }
    std::cout<<bytes_per_chunk<<std::endl;
    std::cout<<pages_per_chunk<<std::endl;
    std::cout<<num_full_chunks<<std::endl;
    std::cout<<rem_bytes<<std::endl;
    std::cout<<rem_pages<<std::endl;

    bool rm = input_tensors[0].layout() == Layout::ROW_MAJOR;
    bool width = input_tensors[0].shape().rank() - 1 == dim;
    DataFormat df = tt_metal::datatype_to_dataformat_converter(input_tensors[0].dtype());
    uint32_t num_rows = 0, num_cols = 0, offset = 0, num_tiles = 0;

    if (rm) {
        num_cols = input_tensors[0].shape()[-1];
        num_rows = input_tensors[0].volume() / num_cols;
    } else {
        num_cols = input_tensors[0].shape()[-1] / TILE_WIDTH;
        num_rows = input_tensors[0].volume() / input_tensors[0].shape()[-1] / TILE_HEIGHT;
        offset = output_tensors[0].shape()[-1] / TILE_WIDTH - num_cols;
        num_tiles = input_tensors[0].volume() / TILE_HW;
    }
    std::cout<<num_rows<<std::endl;
    std::cout<<num_cols<<std::endl;
    std::cout<<num_tiles<<std::endl;
    std::cout<<offset<<std::endl;
    std::cout<<width<<std::endl;
    std::cout<<rm<<std::endl;

    uint32_t page_idx = 0;
    uint32_t page_offset = 0;

    for (uint32_t i = 0; i < sender_receivers.size(); ++i) {
        const auto& device = std::get<0>(sender_receivers[i]);
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

        auto& program = programs[device];
        const auto& input_buffer = input_tensors[i].buffer();
        const auto& output_buffer = output_tensors[i].buffer();

        bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
        bool output_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

        string sender_kernel, receiver_kernel;
        std::vector<uint32_t> sender_ct_args, sender_rt_args, receiver_ct_args, receiver_rt_args;
        if (rm) {
            sender_kernel = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/interleaved_eth_ring_gather_send_stick_layout.cpp";
            sender_ct_args = {
                    uint32_t(device->ethernet_core_from_logical_core(eth_receiver_core).x),
                    uint32_t(device->ethernet_core_from_logical_core(eth_receiver_core).y),
                    uint32_t(input_is_dram),
                    uint32_t(output_is_dram)};
            sender_rt_args = {(uint32_t)input_buffer->address(),
             (uint32_t)output_buffer->address(),
             (uint32_t)(src_eth_l1_byte_address),
             (uint32_t)dst_eth_l1_byte_address,
             (uint32_t)sem_l1_byte_address,
             (uint32_t)(sender_receivers.size() - 1),
             (uint32_t)num_full_chunks,
             (uint32_t)input_buffer->page_size(),
             (uint32_t)pages_per_chunk,
             (uint32_t)(bytes_per_chunk + header_size),
             (uint32_t)rem_pages,
             (uint32_t)(rem_bytes + header_size),
             (uint32_t)page_idx,
             (uint32_t)page_offset,
             (uint32_t)output_buffer->page_size()};

            receiver_kernel = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/interleaved_eth_ring_gather_receive_stick_layout.cpp";
            receiver_ct_args = {
                    uint32_t(device->ethernet_core_from_logical_core(eth_sender_core).x),
                    uint32_t(device->ethernet_core_from_logical_core(eth_sender_core).y),
                    uint32_t(output_is_dram)};
            receiver_rt_args = {
                (uint32_t)output_buffer->address(),
                (uint32_t)dst_eth_l1_byte_address,
                (uint32_t)sem_l1_byte_address,
                (uint32_t)(sender_receivers.size() - 1),
                (uint32_t)num_full_chunks,
                (uint32_t)input_buffer->page_size(),
                (uint32_t)pages_per_chunk,
                (uint32_t)(bytes_per_chunk + header_size),
                (uint32_t)rem_pages,
                (uint32_t)(rem_bytes + header_size),
                (uint32_t)output_buffer->page_size()
            };
        } else {
            sender_kernel = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/interleaved_eth_ring_gather_send.cpp";
            sender_ct_args = {
                    uint32_t(device->ethernet_core_from_logical_core(eth_receiver_core).x),
                    uint32_t(device->ethernet_core_from_logical_core(eth_receiver_core).y),
                    uint32_t(input_is_dram),
                    uint32_t(output_is_dram),
                    uint32_t(df)};
            sender_rt_args = {(uint32_t)input_buffer->address(),
             (uint32_t)output_buffer->address(),
             (uint32_t)(src_eth_l1_byte_address),
             (uint32_t)dst_eth_l1_byte_address,
             (uint32_t)sem_l1_byte_address,
             (uint32_t)(sender_receivers.size() - 1),
             (uint32_t)num_full_chunks,
             (uint32_t)input_buffer->page_size(),
             (uint32_t)pages_per_chunk,
             (uint32_t)(bytes_per_chunk + header_size),
             (uint32_t)rem_pages,
             (uint32_t)(rem_bytes + header_size),
             (uint32_t)page_idx,
             (uint32_t)offset,
             (uint32_t)num_rows,
             (uint32_t)num_cols};

            receiver_kernel = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/interleaved_eth_ring_gather_receive.cpp";
            receiver_ct_args = {
                    uint32_t(device->ethernet_core_from_logical_core(eth_sender_core).x),
                    uint32_t(device->ethernet_core_from_logical_core(eth_sender_core).y),
                    uint32_t(output_is_dram),
                    uint32_t(df)};
            receiver_rt_args = {
                (uint32_t)output_buffer->address(),
                (uint32_t)dst_eth_l1_byte_address,
                (uint32_t)sem_l1_byte_address,
                (uint32_t)(sender_receivers.size() - 1),
                (uint32_t)num_full_chunks,
                (uint32_t)input_buffer->page_size(),
                (uint32_t)pages_per_chunk,
                (uint32_t)(bytes_per_chunk + header_size),
                (uint32_t)rem_pages,
                (uint32_t)(rem_bytes + header_size),
                (uint32_t)offset,
                (uint32_t)num_rows,
                (uint32_t)num_cols
            };
        }

        auto eth_sender_kernel = tt_metal::CreateKernel(
            program,
            sender_kernel,
            eth_sender_core,
            tt_metal::experimental::SenderEthernetConfig{
                .compile_args = sender_ct_args});


        tt_metal::SetRuntimeArgs(
            program,
            eth_sender_kernel,
            eth_sender_core,
            sender_rt_args);

        // TODO: Sempahore support for eth cores
        llrt::write_hex_vec_to_core(
            device->id(), device->ethernet_core_from_logical_core(eth_sender_core), {INVALID}, sem_l1_byte_address);
        llrt::write_hex_vec_to_core(
            device->id(), device->ethernet_core_from_logical_core(eth_receiver_core), {INVALID}, sem_l1_byte_address);

        auto eth_receiver_kernel = tt_metal::CreateKernel(
            program,
            receiver_kernel,
            eth_receiver_core,
            tt_metal::experimental::ReceiverEthernetConfig{
                .compile_args = receiver_ct_args});

        tt_metal::SetRuntimeArgs(
            program,
            eth_receiver_kernel,
            eth_receiver_core,
            receiver_rt_args);

        std::cout<<page_idx<<std::endl;
        std::cout<<page_offset<<std::endl;
        if (rm) {
            if (width) {
                page_offset += input_buffer->page_size();
            } else {
                page_idx += input_buffer->size() / input_buffer->page_size();
            }
        } else {
            if (width) {
                page_idx += num_cols;
            } else {
                page_idx += num_tiles;
            }
        }
    }

    auto override_runtime_args_callback = [](const Program& program,
                                             const std::vector<Buffer*>& input_buffers,
                                             const std::vector<Buffer*>& output_buffers) {};

    return {std::move(programs), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
