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

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks all_gather_multi_core(const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dim, const uint32_t ring_size, const uint32_t ring_index, const chip_id_t receiver_device_id, const chip_id_t sender_device_id) {

    constexpr uint32_t header_size = 32;
    constexpr uint32_t semaphore_offset = 32;
    constexpr uint32_t MAX_BUFFER = eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE - header_size - semaphore_offset;
    constexpr size_t sem_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    constexpr size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + semaphore_offset;
    constexpr size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + semaphore_offset;

    tt_metal::Program program{};

    const auto& device = input_tensor.device();
    uint32_t sender_socket_idx = 0;
    uint32_t receiver_socket_idx = 0;
    if (receiver_device_id == sender_device_id) {
        if (ring_index == 0) {
            receiver_socket_idx = 1;
        } else {
            sender_socket_idx = 1;
        }
    }
    auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id)[sender_socket_idx];
    auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id)[receiver_socket_idx];
    uint32_t bytes_per_chunk = 0, pages_per_chunk = 0, num_full_chunks = 0, rem_bytes = 0, rem_pages = 0;
    if (input_tensor.buffer()->size() > MAX_BUFFER) {
        bytes_per_chunk = round_down(MAX_BUFFER, input_tensor.buffer()->page_size());
        pages_per_chunk = bytes_per_chunk / input_tensor.buffer()->page_size();
        num_full_chunks = (uint32_t)(input_tensor.buffer()->size() / bytes_per_chunk);
        rem_bytes = (uint32_t)(input_tensor.buffer()->size() % bytes_per_chunk);
        rem_pages = rem_bytes / input_tensor.buffer()->page_size();
    } else {
        rem_bytes = input_tensor.buffer()->size();
        rem_pages = rem_bytes / input_tensor.buffer()->page_size();
    }

    bool rm = input_tensor.layout() == Layout::ROW_MAJOR;
    bool width = input_tensor.shape().rank() - 1 == dim;
    DataFormat df = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t num_rows = 0, num_cols = 0, offset = 0, num_tiles = 0;

    if (rm) {
        num_cols = input_tensor.shape()[-1];
        num_rows = input_tensor.volume() / num_cols;
    } else {
        num_cols = input_tensor.shape()[-1] / TILE_WIDTH;
        num_rows = input_tensor.volume() / input_tensor.shape()[-1] / TILE_HEIGHT;
        offset = output_tensor.shape()[-1] / TILE_WIDTH - num_cols;
        num_tiles = input_tensor.volume() / TILE_HW;
    }


    const auto& input_buffer = input_tensor.buffer();
    const auto& output_buffer = output_tensor.buffer();

    bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    bool output_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    uint32_t page_idx = 0;
    uint32_t page_offset = 0;

    if (rm) {
        if (width) {
            page_offset = ring_index * input_buffer->page_size();
        } else {
            page_idx = ring_index * (input_buffer->size() / input_buffer->page_size());
        }
    } else {
        if (width) {
            page_idx = ring_index * num_cols;
        } else {
            page_idx = ring_index * num_tiles;
        }
    }

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
            (uint32_t)(ring_size - 1),
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
            (uint32_t)(ring_size - 1),
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
            (uint32_t)(ring_size - 1),
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
            (uint32_t)(ring_size - 1),
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

    auto override_runtime_arguments_callback = [eth_sender_kernel, eth_receiver_kernel, eth_sender_core, eth_receiver_core](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto& input = input_tensors[0];
        const auto& output = output_tensors[0];
        auto &sender_runtime_args = GetRuntimeArgs(program, eth_sender_kernel, eth_sender_core);
        sender_runtime_args[0] = input.buffer()->address();
        sender_runtime_args[1] = output.buffer()->address();

        auto &receiver_runtime_args = GetRuntimeArgs(program, eth_receiver_kernel, eth_receiver_core);
        receiver_runtime_args[0] = output.buffer()->address();
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
