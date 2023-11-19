// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <type_traits>

#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"

namespace tt {
namespace tt_metal {

Tensor fully_connected_(const Tensor& act, const Tensor& weights, std::optional<std::reference_wrapper<const Tensor>> bias, const MemoryConfig& output_mem_config) {
    Tensor mm_output = matmul(act, weights, output_mem_config);
    if (bias) {
        return bcast(mm_output, bias.value(), BcastOpMath::ADD, BcastOpDim::H, output_mem_config);
    }
    return mm_output;
}

Tensor fully_connected(const Tensor &act, const Tensor& weights, std::optional<std::reference_wrapper<const Tensor>> bias, const MemoryConfig& output_mem_config) {
    TT_ASSERT(act.storage_type() == StorageType::DEVICE && weights.storage_type() == StorageType::DEVICE, "Activation and weight tensors need to be on device");
    // Assuming padding is already included. Not adding padding here.
    // NOTE: Bias is never padded.
    const Device& device = act.device();
    return fully_connected_(act, weights, bias, output_mem_config);
}

}  // namespace tt_metal
}  // namespace tt
