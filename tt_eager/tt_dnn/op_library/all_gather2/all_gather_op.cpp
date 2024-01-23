// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/all_gather2/all_gather_op.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include "eth_l1_address_map.h"

namespace tt {

namespace tt_metal {

void AllGather2::validate(const std::vector<Tensor> &input_tensors) const {
    constexpr uint32_t MAX_BUFFER =
        (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE) - 32;
    TT_FATAL(input_tensors.size() == 1);
    const auto& layout = input_tensors[0].layout();
    const auto& dtype = input_tensors[0].dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size <= MAX_BUFFER, "Page size too large");
    TT_FATAL(page_size % 32 == 0);

    // TODO: Validate ring
    TT_FATAL(input_tensors[0].storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensors[0].buffer() != nullptr , "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(input_tensors[0].memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
}

std::vector<Shape> AllGather2::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto shape = input_tensors[0].shape();
    shape[this->dim] *= this->ring_size;
    return std::vector<Shape>(input_tensors.size(), shape);
}

std::vector<Tensor> AllGather2::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    return operation::generic_create_output_tensors2(*this, input_tensors, input_tensor.dtype(), input_tensor.layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks AllGather2::create_program(const std::vector<Tensor> & input_tensors, std::vector<Tensor> &output_tensors) const {
    return all_gather2_multi_core(input_tensors.at(0), output_tensors.at(0), this->dim, this->ring_size, this->ring_index, this->eth_sender_core, this->eth_receiver_core);
}

tt::stl::reflection::Attributes AllGather2::attributes() const {
    return {
        {"dim", this->dim},
        {"ring_size", this->ring_size},
        {"ring_index", this->ring_index},
        {"eth_sender_core", this->eth_sender_core},
        {"eth_receiver_core", this->eth_receiver_core},
        {"output_mem_config", this->output_mem_config},
    };
}

}  // namespace tt_metal

}  // namespace tt
