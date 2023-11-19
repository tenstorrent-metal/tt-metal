// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_dnn/op_library/layout_conversion/layout_conversion_op.hpp"
#include "tt_dnn/op_library/data_transfer/data_transfer_op.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor AutoFormat::move_tensor_to_device(const Tensor &input, const Device& device, const MemoryConfig& mem_config) {
    if (input.storage_type() == StorageType::OWNED) {
        return data_transfer_to_device(input, device, mem_config);
    } else {
        return input;
    }
}

Tensor AutoFormat::format_input_tensor(const Tensor &input, const Device& device, const Shape& padded_shape, float pad_value, Layout target_layout, std::optional<MemoryConfig> target_mem_config) {
    bool pad_input = input.shape() != padded_shape;
    bool convert_layout = input.layout() != target_layout;

    if (!pad_input && !convert_layout) {
        return AutoFormat::move_tensor_to_device(input, device);
    }

    MemoryConfig mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    if (target_mem_config.has_value()) {
        mem_config = target_mem_config.value();
    } else if (input.storage_type() == StorageType::DEVICE) {
        mem_config = input.memory_config();
    }

    Tensor formatted_input = input;
    auto shape = formatted_input.shape();

    // TODO: Profile if it is faster to put host tensor to device and then pad/convert if possible
    // Device side conversions
    if (formatted_input.storage_type() == StorageType::DEVICE) {
        if (convert_layout && !pad_input) {
            if (target_layout == Layout::TILE && formatted_input.layout() == Layout::ROW_MAJOR) {
                return tilize(formatted_input, mem_config);
            } else if (target_layout == Layout::ROW_MAJOR && formatted_input.layout() == Layout::TILE) {
                return untilize(formatted_input, mem_config);
            }
        } else if (!convert_layout && pad_input) {
            if (formatted_input.layout() == Layout::ROW_MAJOR || formatted_input.layout() == Layout::TILE) {
                return pad(formatted_input, padded_shape, {0, 0, 0, 0}, pad_value, mem_config);
            }
        } else if (convert_layout && pad_input) {
            if (formatted_input.layout() == Layout::ROW_MAJOR && target_layout == Layout::TILE) {
                return tilize_with_val_padding(formatted_input, padded_shape, {0, 0, 0, 0}, pad_value, mem_config);
            }  else if (formatted_input.layout() == Layout::TILE && target_layout == Layout::ROW_MAJOR) {
                formatted_input = untilize(formatted_input, mem_config);
                return pad(formatted_input, padded_shape, {0, 0, 0, 0}, pad_value, mem_config);
            }
        }
        // Fall back to host conversions
        formatted_input = data_transfer_to_host(formatted_input);
    }

    // Host side conversions
    if (pad_input) {
        if (formatted_input.layout() != Layout::ROW_MAJOR) {
            formatted_input = layout_conversion_on_host(formatted_input, Layout::ROW_MAJOR);
            convert_layout = formatted_input.layout() != target_layout;
        }
        formatted_input = pad_on_host(formatted_input, padded_shape, {0, 0, 0, 0}, pad_value);
    }

    if(convert_layout) {
        formatted_input = layout_conversion_on_host(formatted_input, target_layout);
    }

    return AutoFormat::move_tensor_to_device(formatted_input, device, mem_config);
}


Tensor AutoFormat::format_output_tensor(const Tensor &output, const Shape& shape, const Device& device, Layout target_layout, std::optional<MemoryConfig> target_mem_config) {
    bool unpad_output = output.shape() != shape;
    bool convert_layout = output.layout() != target_layout;

    if (!unpad_output && !convert_layout) {
        return output;
    }
    MemoryConfig mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    if (target_mem_config.has_value()) {
        mem_config = target_mem_config.value();
    } else if (output.storage_type() == StorageType::DEVICE) {
        mem_config = output.memory_config();
    }

    Tensor formatted_output = output;
    // Device side conversions
    if (formatted_output.storage_type() == StorageType::DEVICE) {
        if (!unpad_output && convert_layout) {
            // If target layout is tile but shape does not support tile, we don't do any conversions
            if (target_layout == Layout::TILE && formatted_output.layout() == Layout::ROW_MAJOR) {
                if (AutoFormat::legal_tile_shape(formatted_output.shape())) {
                    formatted_output = tilize(formatted_output, mem_config);
                }
                return formatted_output;
            } else if (target_layout == Layout::ROW_MAJOR && formatted_output.layout() == Layout::TILE) {
                formatted_output = untilize(formatted_output, mem_config);
                return formatted_output;
            }

        } else if (unpad_output && !convert_layout) {
            // Output can be unpadded and layout supports the shape
            if ((formatted_output.layout() == Layout::TILE && AutoFormat::legal_tile_shape(shape)) ||
                (formatted_output.layout() == Layout::ROW_MAJOR && AutoFormat::legal_rm_shape(shape))) {
                formatted_output = unpad(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}, mem_config);
                return formatted_output;
            // Output is tile but shape cannot be tile. We leave in RM
            } else if (formatted_output.layout() == Layout::TILE && AutoFormat::legal_rm_shape(shape)) {
                formatted_output = untilize_with_unpadding(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}, mem_config);
                return formatted_output;
            }
        } else if (unpad_output && convert_layout) {
            if (formatted_output.layout() == Layout::TILE && target_layout == Layout::ROW_MAJOR && AutoFormat::legal_rm_shape(shape)) {
                formatted_output = untilize_with_unpadding(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}, mem_config);
                return formatted_output;
            } else if (formatted_output.layout() == Layout::ROW_MAJOR && target_layout == Layout::TILE && AutoFormat::legal_tile_shape(shape)) {
                formatted_output = unpad(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}, mem_config);
                formatted_output = tilize(formatted_output, mem_config);
                return formatted_output;
            }
        }
        // Fall back to host conversions
        formatted_output = data_transfer_to_host(formatted_output);
    }

    // Host side conversions
    if (unpad_output) {
        // Requires RM for unpad
        if (formatted_output.layout() != Layout::ROW_MAJOR) {
            formatted_output = layout_conversion_on_host(formatted_output, Layout::ROW_MAJOR);
            convert_layout = formatted_output.layout() != target_layout;
        }
        formatted_output = unpad_on_host(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
    }

    if (convert_layout) {
        // Default to RM layout if we can't match the formatted_input layout
        if (target_layout == Layout::TILE && !AutoFormat::legal_tile_shape(formatted_output.shape())) {
            if (formatted_output.layout() != Layout::ROW_MAJOR) {
                formatted_output = layout_conversion_on_host(formatted_output, Layout::ROW_MAJOR);
            }
        } else {
            formatted_output = layout_conversion_on_host(formatted_output, target_layout);
        }
    }

    // Send formatted_output to device if possible
    // Check that shape is supported on device
    if (formatted_output.storage_type() == StorageType::OWNED) {
        if (AutoFormat::legal_device_shape(formatted_output.shape(), formatted_output.layout())) {
            formatted_output = AutoFormat::move_tensor_to_device(formatted_output, device, mem_config);
        }
    }

    return formatted_output;
}

}
}
