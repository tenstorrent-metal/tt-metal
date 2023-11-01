// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/pool/average_pool.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_numpy/functions.hpp"

#include "tensor/tensor.hpp"
#include "common/constants.hpp"

using tt::tt_metal::Host;
using tt::tt_metal::Device;
using tt::tt_metal::Tensor;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::Shape;
using tt::tt_metal::AutoFormat;

Tensor run_avg_pool_2d_resnet(Shape& tensor_shape, Device* device) {
    auto input_tensor = tt::numpy::random::random(tensor_shape, DataType::BFLOAT16);
    auto padded_input_shape = AutoFormat::pad_to_tile_shape(tensor_shape, false, false);
    Tensor padded_input_tensor = input_tensor;
    if (!AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape)) {
        padded_input_tensor = AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, 0, Layout::TILE);    // pad with 0s
    }
    auto device_output = average_pool_2d(padded_input_tensor);
    return device_output.cpu();
};

int main () {
    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    Shape resnet18_shape = {1, 1, 7 * 7, 2048};
    auto result = run_avg_pool_2d_resnet(resnet18_shape, device);

    TT_ASSERT(result.shape() == Shape({1, 1, TILE_HEIGHT, 2048}));
    TT_ASSERT(result.shape().without_padding() == Shape({1, 1, 1, 2048}));

    tt::tt_metal::CloseDevice(device);
    return 0;
}
