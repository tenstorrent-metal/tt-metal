// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "eth_l1_address_map.h"
#include "tensor/tensor_impl.hpp"
#include "tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks all_gather_multi_core(const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dim, const uint32_t num_links, const uint32_t ring_size, const uint32_t ring_index, const chip_id_t receiver_device_id, const chip_id_t sender_device_id) {

    tt_metal::Program program{};

    const uint32_t max_buffer_per_chunk = round_down(all_gather_buffer_params::eth_buffer_size, input_tensor.buffer()->page_size());
    const uint32_t max_pages_per_chunk = max_buffer_per_chunk / input_tensor.buffer()->page_size();
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
    std::vector<CoreCoord> eth_sender_cores;
    eth_sender_cores.reserve(num_links);
    std::vector<CoreCoord> eth_receiver_cores;
    eth_receiver_cores.reserve(num_links);
    std::vector<KernelHandle> eth_sender_kernels;
    eth_sender_kernels.reserve(num_links);
    std::vector<KernelHandle> eth_receiver_kernels;
    eth_receiver_kernels.reserve(num_links);

    uint32_t num_input_pages = input_tensor.buffer()->size() / input_tensor.buffer()->page_size();
    uint32_t min_pages_per_link = num_input_pages / num_links;

    std::vector<uint32_t> pages_per_link(num_links, min_pages_per_link);
    for (uint32_t i = 0; i < num_input_pages % min_pages_per_link; ++i) {
        pages_per_link[i]++;
    }

    bool rm = input_tensor.layout() == Layout::ROW_MAJOR;
    bool width = input_tensor.shape().rank() - 1 == dim;
    DataFormat df = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t num_rows = 0, num_cols = 0, row_offset = 0, col_offset = 0, num_tiles = 0;

    if (rm) {
        num_cols = input_tensor.shape()[-1];
        auto input_shape = input_tensor.shape();
        auto output_shape = output_tensor.shape();
        num_rows = std::accumulate(input_shape.begin()+dim, input_shape.end() - 1, 1, std::multiplies<uint32_t>());
        row_offset = std::accumulate(output_shape.begin()+dim, output_shape.end() - 1, 1, std::multiplies<uint32_t>()) - num_rows;
    } else {
        num_cols = input_tensor.shape()[-1] / TILE_WIDTH;
        auto input_shape = input_tensor.shape();
        auto output_shape = output_tensor.shape();
        uint32_t num_output_cols = output_tensor.shape()[-1] / TILE_WIDTH;
        num_rows = std::accumulate(input_shape.begin()+dim, input_shape.end() - 1, 1, std::multiplies<uint32_t>()) / TILE_HEIGHT;
        row_offset = (std::accumulate(output_shape.begin()+dim, output_shape.end() - 1, 1, std::multiplies<uint32_t>()) / TILE_HEIGHT - num_rows) * num_output_cols;
        col_offset = num_output_cols - num_cols;
        num_tiles = num_rows * num_cols;
    }

    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();

    bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    bool output_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    uint32_t total_output_pages = output_buffer->size() / output_buffer->page_size();

    uint32_t input_start_page_idx = 0;
    uint32_t output_addr_offset = 0;
    uint32_t col_idx = 0;
    uint32_t row_idx = 0;
    uint32_t output_page_offset = 0;

    if (rm) {
        if (width) {
            output_addr_offset = input_buffer->page_size();
        } else {
            output_page_offset = num_rows;
        }
    } else {
        if (width) {
            output_page_offset = num_cols;
        } else {
            output_page_offset = num_tiles;
        }
    }
    uint32_t output_start_page_idx = ring_index * output_page_offset;
    uint32_t output_start_addr_offset = ring_index * output_addr_offset;
    uint32_t last_output_page_offset = (ring_size - 1) * output_page_offset;
    uint32_t last_output_addr_offset = (ring_size - 1) * output_addr_offset;
    uint32_t receiver_ring_index = ring_index == 0 ? ring_size - 1 : ring_index - 1;
    uint32_t receiver_output_start_addr_offset = receiver_ring_index * output_addr_offset;

    for (uint32_t i = 0; i < num_links; ++i) {
        auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id)[sender_socket_idx];
        eth_sender_cores.push_back(eth_sender_core);
        auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id)[receiver_socket_idx];
        eth_receiver_cores.push_back(eth_receiver_core);

        // TODO: Sempahore support for eth cores
        // llrt::write_hex_vec_to_core(
        //     device->id(), device->ethernet_core_from_logical_core(eth_sender_core), {INVALID}, sem_l1_byte_address);
        // llrt::write_hex_vec_to_core(
        //     device->id(), device->ethernet_core_from_logical_core(eth_receiver_core), {INVALID}, sem_l1_byte_address);

        uint32_t bytes_per_chunk = 0, pages_per_chunk = 0, num_full_chunks = 0, rem_bytes = 0, rem_pages = 0;
        uint32_t buffer_size = pages_per_link[i] * input_tensor.buffer()->page_size();
        if (pages_per_link[i] > max_pages_per_chunk) {
            bytes_per_chunk = max_buffer_per_chunk;
            pages_per_chunk = max_pages_per_chunk;
            num_full_chunks = buffer_size / bytes_per_chunk;
            rem_bytes = buffer_size % bytes_per_chunk;
            rem_pages = pages_per_link[i] % max_pages_per_chunk;
        } else {
            rem_bytes = buffer_size;
            rem_pages = pages_per_link[i];
        }
        uint32_t receiver_output_start_page_idx = output_start_page_idx;
        if (ring_index == 0) {
            receiver_output_start_page_idx += last_output_page_offset;
        } else {
            receiver_output_start_page_idx -= output_page_offset;
        }

        string sender_kernel, receiver_kernel;
        std::vector<uint32_t> sender_ct_args, sender_rt_args, receiver_ct_args, receiver_rt_args;
        if (rm) {
            sender_kernel = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/interleaved_eth_ring_gather_send_stick_layout.cpp";
            sender_ct_args = {
                static_cast<uint32_t>(input_is_dram),
                static_cast<uint32_t>(output_is_dram),
                static_cast<uint32_t>(ring_size - 1),
                static_cast<uint32_t>(num_full_chunks),
                static_cast<uint32_t>(input_buffer->page_size()),
                static_cast<uint32_t>(pages_per_chunk),
                static_cast<uint32_t>(bytes_per_chunk),
                static_cast<uint32_t>(rem_pages),
                static_cast<uint32_t>(rem_bytes),
                static_cast<uint32_t>(input_start_page_idx),
                static_cast<uint32_t>(output_start_page_idx),
                static_cast<uint32_t>(output_start_addr_offset),
                static_cast<uint32_t>(row_idx),
                static_cast<uint32_t>(row_offset),
                static_cast<uint32_t>(num_rows),
                static_cast<uint32_t>(output_buffer->page_size()),
                static_cast<uint32_t>(last_output_page_offset),
                static_cast<uint32_t>(output_page_offset),
                static_cast<uint32_t>(last_output_addr_offset),
                static_cast<uint32_t>(output_addr_offset),
                static_cast<uint32_t>(ring_index),
                static_cast<uint32_t>(all_gather_buffer_params::num_buffers),
                static_cast<uint32_t>(all_gather_buffer_params::eth_buffer_l1_byte_address),
                static_cast<uint32_t>(all_gather_buffer_params::eth_sem_l1_byte_address)};
            sender_rt_args = {static_cast<uint32_t>(input_buffer->address()),
                static_cast<uint32_t>(output_buffer->address())
                };

            receiver_kernel = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/interleaved_eth_ring_gather_receive_stick_layout.cpp";
            receiver_ct_args = {
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).x),
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).y),
                static_cast<uint32_t>(output_is_dram),
                static_cast<uint32_t>(ring_size - 1),
                static_cast<uint32_t>(num_full_chunks),
                static_cast<uint32_t>(input_buffer->page_size()),
                static_cast<uint32_t>(pages_per_chunk),
                static_cast<uint32_t>(bytes_per_chunk),
                static_cast<uint32_t>(rem_pages),
                static_cast<uint32_t>(rem_bytes),
                static_cast<uint32_t>(receiver_output_start_page_idx),
                static_cast<uint32_t>(receiver_output_start_addr_offset),
                static_cast<uint32_t>(row_idx),
                static_cast<uint32_t>(row_offset),
                static_cast<uint32_t>(num_rows),
                static_cast<uint32_t>(output_buffer->page_size()),
                static_cast<uint32_t>(last_output_page_offset),
                static_cast<uint32_t>(output_page_offset),
                static_cast<uint32_t>(last_output_addr_offset),
                static_cast<uint32_t>(output_addr_offset),
                static_cast<uint32_t>(receiver_ring_index),
                static_cast<uint32_t>(all_gather_buffer_params::num_buffers),
                static_cast<uint32_t>(all_gather_buffer_params::eth_buffer_l1_byte_address),
                static_cast<uint32_t>(all_gather_buffer_params::eth_sem_l1_byte_address)};
            receiver_rt_args = {
                static_cast<uint32_t>(output_buffer->address())
            };
        } else {
            sender_kernel = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/interleaved_eth_ring_gather_send.cpp";
            sender_ct_args = {
                static_cast<uint32_t>(input_is_dram),
                static_cast<uint32_t>(output_is_dram),
                static_cast<uint32_t>(df),
                static_cast<uint32_t>(ring_size - 1),
                static_cast<uint32_t>(num_full_chunks),
                static_cast<uint32_t>(input_buffer->page_size()),
                static_cast<uint32_t>(pages_per_chunk),
                static_cast<uint32_t>(bytes_per_chunk),
                static_cast<uint32_t>(rem_pages),
                static_cast<uint32_t>(rem_bytes),
                static_cast<uint32_t>(input_start_page_idx),
                static_cast<uint32_t>(output_start_page_idx),
                static_cast<uint32_t>(row_idx),
                static_cast<uint32_t>(col_idx),
                static_cast<uint32_t>(row_offset),
                static_cast<uint32_t>(col_offset),
                static_cast<uint32_t>(num_rows),
                static_cast<uint32_t>(num_cols),
                static_cast<uint32_t>(last_output_page_offset),
                static_cast<uint32_t>(output_page_offset),
                static_cast<uint32_t>(ring_index),
                static_cast<uint32_t>(all_gather_buffer_params::num_buffers),
                static_cast<uint32_t>(all_gather_buffer_params::eth_buffer_l1_byte_address),
                static_cast<uint32_t>(all_gather_buffer_params::eth_sem_l1_byte_address)};
            sender_rt_args = {static_cast<uint32_t>(input_buffer->address()),
                static_cast<uint32_t>(output_buffer->address())
                };

            receiver_kernel = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/interleaved_eth_ring_gather_receive.cpp";
            receiver_ct_args = {
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).x),
                static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).y),
                static_cast<uint32_t>(output_is_dram),
                static_cast<uint32_t>(df),
                static_cast<uint32_t>(ring_size - 1),
                static_cast<uint32_t>(num_full_chunks),
                static_cast<uint32_t>(input_buffer->page_size()),
                static_cast<uint32_t>(pages_per_chunk),
                static_cast<uint32_t>(bytes_per_chunk),
                static_cast<uint32_t>(rem_pages),
                static_cast<uint32_t>(rem_bytes),
                static_cast<uint32_t>(receiver_output_start_page_idx),
                static_cast<uint32_t>(row_idx),
                static_cast<uint32_t>(col_idx),
                static_cast<uint32_t>(row_offset),
                static_cast<uint32_t>(col_offset),
                static_cast<uint32_t>(num_rows),
                static_cast<uint32_t>(num_cols),
                static_cast<uint32_t>(last_output_page_offset),
                static_cast<uint32_t>(output_page_offset),
                static_cast<uint32_t>(receiver_ring_index),
                static_cast<uint32_t>(all_gather_buffer_params::num_buffers),
                static_cast<uint32_t>(all_gather_buffer_params::eth_buffer_l1_byte_address),
                static_cast<uint32_t>(all_gather_buffer_params::eth_sem_l1_byte_address)};
            receiver_rt_args = {
                static_cast<uint32_t>(output_buffer->address())
            };
        }
        auto eth_sender_kernel = tt_metal::CreateKernel(
            program, sender_kernel, eth_sender_core, tt_metal::experimental::SenderEthernetConfig(sender_ct_args));

        eth_sender_kernels.push_back(eth_sender_kernel);

        tt_metal::SetRuntimeArgs(
            program,
            eth_sender_kernel,
            eth_sender_core,
            sender_rt_args);

        auto eth_receiver_kernel = tt_metal::CreateKernel(
            program,
            receiver_kernel,
            eth_receiver_core,
            tt_metal::experimental::ReceiverEthernetConfig(receiver_ct_args));

        eth_receiver_kernels.push_back(eth_receiver_kernel);

        tt_metal::SetRuntimeArgs(
            program,
            eth_receiver_kernel,
            eth_receiver_core,
            receiver_rt_args);

        if (rm) {
            uint32_t num_rows_shifted = row_idx + pages_per_link[i];
            uint32_t num_blocks_shifted = width ? 0 : num_rows_shifted / num_rows;
            output_start_page_idx += pages_per_link[i] + num_blocks_shifted * row_offset;
            row_idx = width ? 0 : num_rows_shifted % num_rows;
        } else {
            uint32_t num_cols_shifted = col_idx + pages_per_link[i];
            uint32_t num_rows_shifted = num_cols_shifted / num_cols;
            uint32_t num_blocks_shifted = width ? 0 : num_rows_shifted / num_rows;
            output_start_page_idx += pages_per_link[i] + num_rows_shifted * col_offset + num_blocks_shifted * row_offset;
            col_idx = num_cols_shifted % num_cols;
            row_idx = width ? 0 : num_rows_shifted % num_rows;
        }
        input_start_page_idx += pages_per_link[i];
        if (receiver_device_id == sender_device_id) {
            receiver_socket_idx += 2;
            sender_socket_idx += 2;
        } else {
            receiver_socket_idx += 1;
            sender_socket_idx += 1;
        }
    }

    auto override_runtime_arguments_callback = [num_links, eth_sender_kernels, eth_receiver_kernels, eth_sender_cores, eth_receiver_cores] (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto& input = input_tensors[0];
        const auto& output = output_tensors[0];
        for (uint32_t i = 0; i < num_links; ++i) {
            auto &sender_runtime_args = GetRuntimeArgs(program, eth_sender_kernels[i], eth_sender_cores[i]);
            sender_runtime_args[0] = input.buffer()->address();
            sender_runtime_args[1] = output.buffer()->address();

            auto &receiver_runtime_args = GetRuntimeArgs(program, eth_receiver_kernels[i], eth_receiver_cores[i]);
            receiver_runtime_args[0] = output.buffer()->address();
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks all_gather_multi_core_sharded(const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dim, const uint32_t num_links, const uint32_t ring_size, const uint32_t ring_index, const chip_id_t receiver_device_id, const chip_id_t sender_device_id) {
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
    std::vector<CoreCoord> eth_sender_cores;
    eth_sender_cores.reserve(num_links);
    std::vector<CoreCoord> eth_receiver_cores;
    eth_receiver_cores.reserve(num_links);
    std::vector<KernelHandle> eth_sender_kernels;
    eth_sender_kernels.reserve(num_links);
    std::vector<KernelHandle> eth_receiver_kernels;
    eth_receiver_kernels.reserve(num_links);

    bool rm = input_tensor.layout() == Layout::ROW_MAJOR;
    ShardSpec input_shard_spec = input_tensor.shard_spec().value();
    ShardSpec output_shard_spec = input_tensor.shard_spec().value();
    bool width = input_tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
    DataFormat df = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    // Note that pages / page size here is considering the pages within a shard, as opposed to normally using 1 page = 1 shard

    uint32_t input_page_size;
    uint32_t input_pages_per_shard;

    if (rm) {
        input_pages_per_shard = input_shard_spec.shape[0];
        input_page_size = input_shard_spec.shape[1] * input_tensor.element_size();
    } else {
        input_pages_per_shard = input_shard_spec.shape[0] * input_shard_spec.shape[1] / TILE_HW;
        input_page_size = tt_metal::detail::TileSize(df);
    }
    uint32_t input_shard_size = input_pages_per_shard * input_page_size;

    const uint32_t max_buffer_per_chunk = round_down(all_gather_buffer_params::eth_buffer_size, input_page_size);
    const uint32_t max_pages_per_chunk = max_buffer_per_chunk / input_page_size;

    uint32_t num_input_pages = input_tensor.buffer()->size() / input_page_size;
    uint32_t min_pages_per_link = num_input_pages / num_links;
    std::vector<uint32_t> pages_per_link(num_links, min_pages_per_link);
    for (uint32_t i = 0; i < num_input_pages % min_pages_per_link; ++i) {
        pages_per_link[i]++;
    }

    uint32_t num_rows = 0, row_offset = 0, num_tiles = 0;
    if (width) {
        if (rm) {
            num_rows = input_shard_spec.shape[0];
            row_offset = (output_shard_spec.shape[1] - input_shard_spec.shape[1]) * input_tensor.element_size();
        } else {
            num_rows = input_shard_spec.shape[0] / TILE_HEIGHT;
            row_offset = (output_shard_spec.shape[1] - input_shard_spec.shape[1]) / TILE_WIDTH * tt_metal::detail::TileSize(df);
        }
    } else {
        num_rows = 1;
        row_offset = 0;
    }

    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();

    uint32_t block_size = 0;
    uint32_t row_idx = 0;
    uint32_t output_row_size = 0;
    if (width) {
        if (rm) {
            block_size = input_shard_spec.shape[1] * input_tensor.element_size();
            output_row_size = output_shard_spec.shape[1] * input_tensor.element_size();
        } else {
            block_size = input_shard_spec.shape[1] / TILE_WIDTH * tt_metal::detail::TileSize(df);
            output_row_size = output_shard_spec.shape[1] / TILE_WIDTH * tt_metal::detail::TileSize(df);
        }
    } else {
        block_size = input_shard_size;
    }

    uint32_t output_start_addr_offset = ring_index * block_size;
    uint32_t last_output_addr_offset = (ring_size - 1) * block_size;
    uint32_t receiver_ring_index = ring_index == 0 ? ring_size - 1 : ring_index - 1;
    uint32_t receiver_output_start_addr_offset = receiver_ring_index * block_size;

    uint32_t num_cores = input_shard_spec.num_cores();
    auto bbox = input_shard_spec.grid.bounding_box();
    uint32_t num_x = bbox.end.x - bbox.start.x + 1;
    uint32_t num_y = bbox.end.y - bbox.start.y + 1;
    uint32_t core_idx = 0;
    std::vector<uint32_t> remote_noc_x;
    std::vector<uint32_t> remote_noc_y;
    remote_noc_x.reserve(num_x);
    remote_noc_y.reserve(num_y);
    for (uint32_t x = bbox.start.x; x < bbox.end.x + 1; ++x) {
        remote_noc_x.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    for (uint32_t y = bbox.start.y; y < bbox.end.y + 1; ++y) {
        remote_noc_y.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    bool rm_orientation = input_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    uint32_t input_shard_start_offset = 0;
    uint32_t output_shard_start_offset = input_shard_start_offset;
    uint32_t rem_block_size = block_size;

    for (uint32_t i = 0; i < num_links; ++i) {
        auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id)[sender_socket_idx];
        eth_sender_cores.push_back(eth_sender_core);
        auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id)[receiver_socket_idx];
        eth_receiver_cores.push_back(eth_receiver_core);

        // TODO: Sempahore support for eth cores
        // llrt::write_hex_vec_to_core(
        //     device->id(), device->ethernet_core_from_logical_core(eth_sender_core), {INVALID}, sem_l1_byte_address);
        // llrt::write_hex_vec_to_core(
        //     device->id(), device->ethernet_core_from_logical_core(eth_receiver_core), {INVALID}, sem_l1_byte_address);

        uint32_t bytes_per_chunk = 0, pages_per_chunk = 0, num_full_chunks = 0, rem_bytes = 0, rem_pages = 0;
        uint32_t buffer_size = pages_per_link[i] * input_page_size;
        if (pages_per_link[i] > max_pages_per_chunk) {
            bytes_per_chunk = max_buffer_per_chunk;
            pages_per_chunk = max_pages_per_chunk;
            num_full_chunks = buffer_size / bytes_per_chunk;
            rem_bytes = buffer_size % bytes_per_chunk;
            rem_pages = pages_per_link[i] % max_pages_per_chunk;
        } else {
            rem_bytes = buffer_size;
            rem_pages = pages_per_link[i];
        }

        string sender_kernel, receiver_kernel;
        std::vector<uint32_t> sender_ct_args, sender_rt_args, receiver_ct_args, receiver_rt_args;

        sender_kernel = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/sharded_eth_ring_gather_send.cpp";
        sender_ct_args = {
            static_cast<uint32_t>(num_cores),
            static_cast<uint32_t>(rm_orientation),
            static_cast<uint32_t>(num_x),
            static_cast<uint32_t>(num_y),
            static_cast<uint32_t>(core_idx),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(ring_size - 1),
            static_cast<uint32_t>(num_full_chunks),
            static_cast<uint32_t>(input_shard_size),
            static_cast<uint32_t>(input_shard_start_offset),
            static_cast<uint32_t>(output_shard_start_offset),
            static_cast<uint32_t>(rem_block_size),
            static_cast<uint32_t>(bytes_per_chunk),
            static_cast<uint32_t>(rem_bytes),
            static_cast<uint32_t>(output_start_addr_offset),
            static_cast<uint32_t>(row_idx),
            static_cast<uint32_t>(row_offset),
            static_cast<uint32_t>(num_rows),
            static_cast<uint32_t>(last_output_addr_offset),
            static_cast<uint32_t>(block_size),
            static_cast<uint32_t>(ring_index),
            static_cast<uint32_t>(all_gather_buffer_params::num_buffers),
            static_cast<uint32_t>(all_gather_buffer_params::eth_buffer_l1_byte_address),
            static_cast<uint32_t>(all_gather_buffer_params::eth_sem_l1_byte_address)};
        sender_rt_args = {static_cast<uint32_t>(input_buffer->address()),
            static_cast<uint32_t>(output_buffer->address())
        };
        sender_rt_args.insert(sender_rt_args.end(), remote_noc_x.begin(), remote_noc_x.end());
        sender_rt_args.insert(sender_rt_args.end(), remote_noc_y.begin(), remote_noc_y.end());

        receiver_kernel = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/sharded_eth_ring_gather_receive.cpp";
        receiver_ct_args = {
            static_cast<uint32_t>(num_cores),
            static_cast<uint32_t>(rm_orientation),
            static_cast<uint32_t>(num_x),
            static_cast<uint32_t>(num_y),
            static_cast<uint32_t>(core_idx),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).x),
            static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).y),
            static_cast<uint32_t>(ring_size - 1),
            static_cast<uint32_t>(num_full_chunks),
            static_cast<uint32_t>(output_shard_start_offset),
            static_cast<uint32_t>(rem_block_size),
            static_cast<uint32_t>(bytes_per_chunk),
            static_cast<uint32_t>(rem_bytes),
            static_cast<uint32_t>(receiver_output_start_addr_offset),
            static_cast<uint32_t>(row_idx),
            static_cast<uint32_t>(row_offset),
            static_cast<uint32_t>(num_rows),
            static_cast<uint32_t>(last_output_addr_offset),
            static_cast<uint32_t>(block_size),
            static_cast<uint32_t>(receiver_ring_index),
            static_cast<uint32_t>(all_gather_buffer_params::num_buffers),
            static_cast<uint32_t>(all_gather_buffer_params::eth_buffer_l1_byte_address),
            static_cast<uint32_t>(all_gather_buffer_params::eth_sem_l1_byte_address)};
        receiver_rt_args = {
            static_cast<uint32_t>(output_buffer->address())
        };
        receiver_rt_args.insert(receiver_rt_args.end(), remote_noc_x.begin(), remote_noc_x.end());
        receiver_rt_args.insert(receiver_rt_args.end(), remote_noc_y.begin(), remote_noc_y.end());
        auto eth_sender_kernel = tt_metal::CreateKernel(
            program, sender_kernel, eth_sender_core, tt_metal::experimental::SenderEthernetConfig(sender_ct_args));

        eth_sender_kernels.push_back(eth_sender_kernel);

        tt_metal::SetRuntimeArgs(
            program,
            eth_sender_kernel,
            eth_sender_core,
            sender_rt_args);

        auto eth_receiver_kernel = tt_metal::CreateKernel(
            program,
            receiver_kernel,
            eth_receiver_core,
            tt_metal::experimental::ReceiverEthernetConfig(receiver_ct_args));

        eth_receiver_kernels.push_back(eth_receiver_kernel);

        tt_metal::SetRuntimeArgs(
            program,
            eth_receiver_kernel,
            eth_receiver_core,
            receiver_rt_args);

        core_idx += (input_shard_start_offset + buffer_size) / input_shard_size;
        input_shard_start_offset = (input_shard_start_offset + buffer_size) % input_shard_size;
        if (width) {
            row_idx = input_shard_start_offset / block_size;
        }
        output_shard_start_offset = row_idx * output_row_size + input_shard_start_offset % block_size;
        rem_block_size = block_size - input_shard_start_offset % block_size;
    }

    auto override_runtime_arguments_callback = [num_links, eth_sender_kernels, eth_receiver_kernels, eth_sender_cores, eth_receiver_cores] (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto& input = input_tensors[0];
        const auto& output = output_tensors[0];
        for (uint32_t i = 0; i < num_links; ++i) {
            auto &sender_runtime_args = GetRuntimeArgs(program, eth_sender_kernels[i], eth_sender_cores[i]);
            sender_runtime_args[0] = input.buffer()->address();
            sender_runtime_args[1] = output.buffer()->address();

            auto &receiver_runtime_args = GetRuntimeArgs(program, eth_receiver_kernels[i], eth_receiver_cores[i]);
            receiver_runtime_args[0] = output.buffer()->address();
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
