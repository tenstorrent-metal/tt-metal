// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/prod/prod_nc_op.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         Prod
////////////////////////////////////////////////////////////////////////////
void Prod::validate(const std::vector<Tensor>& inputs) const {
    TT_ASSERT((dim >= 0 && dim <= 3), "dim should be 0 - 3");
    const auto& input = inputs.at(0);
    const auto& output = inputs.at(1);

    auto input_shape = input.shape();
    TT_ASSERT((input_shape.rank()), "rank should be 4");
    const auto& output_shape = output.shape();
    auto input_shape_wo_padding = input.shape().without_padding();
    const auto& output_shape_wo_padding = output.shape().without_padding();

    if (dim == 0 || dim == 1) {
        input_shape[dim] = 1;
        input_shape_wo_padding[dim] = 1;
    }

    for (int i = 0; i < input_shape.rank(); ++i) {
        TT_FATAL(input_shape[i] == output_shape[i]);
        // TT_FATAL(input_shape_wo_padding[i] == output_shape_wo_padding[i]);
    }
}

std::vector<Tensor> Prod::create_output_tensors(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

std::vector<Shape> Prod::compute_output_shapes(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};

}

operation::ProgramWithCallbacks Prod::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    auto& input = inputs.at(0);
    auto& output = inputs.at(1);

    return prod_nc_format(input, output, dim);
}

inline Shape compute_output_shape(const Shape& input_shape, const int64_t& dim) {
    auto output_shape = input_shape;
    auto padding = output_shape.padding();
    switch (dim) {
        case 0:
        case 1: output_shape[dim] = 1;
        break;
    }

    return {Shape(output_shape, padding)};
}

inline Tensor create_output_tensor(
    const Tensor& input_tensor, const Shape& output_shape, const MemoryConfig& mem_config) {
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE);
    return create_device_tensor(output_shape, input_tensor.dtype(), Layout::TILE, input_tensor.device(), mem_config);
}

// output as arg
Tensor prod_(const Tensor& input, const Tensor& output, const int64_t& dim) {
    operation::run(Prod{.dim = dim}, {input, output});
    return output;
}

// output creation inside
Tensor prod_(const Tensor& input, const int64_t& dim, const MemoryConfig& mem_config) {
    const auto& input_shape = input.shape();
    const auto& output_shape = compute_output_shape(input_shape, dim);
    auto output = create_output_tensor(input, output_shape, mem_config);

    const auto& output_shape_wo_padding = output.shape().without_padding();
    operation::run(Prod{.dim = dim}, {input, output});
    return output;
}

Tensor prod_nc(
    const Tensor& input,
    const Tensor& output,
    std::vector<int64_t>& dims,
    const MemoryConfig& mem_config) {
    // reduce for all dims
    if (dims.empty()) {
        dims = {0, 1, 2, 3};
    }

    std::vector<int64_t> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end());

    auto temp_input = input;
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        log_debug(LogTest, "{}:{} dim {}", __func__, __LINE__, sorted_dims[i]);
        auto temp_output = prod_(temp_input, sorted_dims[i], mem_config);
        temp_input = temp_output;
    }
    log_debug(LogTest, "{}:{} dim {}", __func__, __LINE__, sorted_dims.front());
    prod_(temp_input, output, sorted_dims.front());
    return output;
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
