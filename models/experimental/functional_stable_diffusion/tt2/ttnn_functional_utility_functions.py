# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tt_lib.fallback_ops import fallback_ops

import torch
from typing import Optional, Dict


def is_tile_dim_alligned(dim):
    return dim % 32 == 0


def pre_process_input_new(device, tensor):
    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
    batch_size = tensor.shape[0]
    input_channels = tensor.shape[1]
    input_height = tensor.shape[2]
    input_width = tensor.shape[3]
    tensor = fallback_ops.permute(
        tensor.value, (0, 2, 3, 1), output_layout=ttnn.ROW_MAJOR_LAYOUT, output_on_device=False
    )
    import math

    assert input_channels == tensor.shape()[3]
    padded_input_channels = math.ceil(input_channels / 16) * 16
    if padded_input_channels != input_channels:
        tensor = fallback_ops.pad(
            tensor,
            (0, padded_input_channels - input_channels, 0, 0, 0, 0),
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            output_on_device=False,
        )
    # Reshape 4d to 2d
    tensor = fallback_ops.reshape(
        tensor,
        1,
        1,
        batch_size * input_height * input_width,
        padded_input_channels,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_on_device=False,
    )
    tensor = ttnn.Tensor(tensor)
    tensor = ttnn.to_device(tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    return tensor


def fold_encoder_hidden_states(device, tensor, required_sequence_length):
    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
    assert tensor.shape[0] == 1
    batch_size = tensor.shape[1]
    sequence_length = tensor.shape[2]
    hidden_dim = tensor.shape[3]
    if (sequence_length * batch_size) < required_sequence_length:
        assert (required_sequence_length % batch_size) == 0
        sequence_length = required_sequence_length // batch_size
        tensor = ttnn.Tensor(
            fallback_ops.pad(
                tensor.value,
                (0, 0, 0, sequence_length - tensor.shape[2]),
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                output_on_device=False,
            )
        )
    # tensor = ttnn.Tensor(
    #     fallback_ops.reshape(
    #         tensor.value,
    #         1,
    #         1,
    #         batch_size * sequence_length,
    #         hidden_dim,
    #         output_layout=ttnn.ROW_MAJOR_LAYOUT,
    #         output_on_device=False,
    #     )
    # )
    tensor = ttnn.to_device(tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    return tensor


def pre_process_input(device, tensor):
    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
    batch_size = tensor.shape[0]
    input_channels = tensor.shape[1]
    input_height = tensor.shape[2]
    input_width = tensor.shape[3]
    tensor = fallback_ops.permute(
        tensor.value, (0, 2, 3, 1), output_layout=ttnn.ROW_MAJOR_LAYOUT, output_on_device=False
    )
    import math

    assert input_channels == tensor.shape()[3]
    padded_input_channels = math.ceil(input_channels / 16) * 16
    if padded_input_channels != input_channels:
        tensor = fallback_ops.pad(
            tensor,
            (0, padded_input_channels - input_channels, 0, 0, 0, 0),
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            output_on_device=False,
        )
    # Reshape 4d to 2d
    tensor = fallback_ops.reshape(
        tensor,
        1,
        1,
        batch_size * input_height * input_width,
        padded_input_channels,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_on_device=False,
    )
    tensor = ttnn.Tensor(tensor)
    tensor = ttnn.to_device(tensor, device)
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    return tensor


def post_process_output(device, tensor, batch_size, output_height, output_width, output_channels):
    tensor = ttnn.to_layout(
        tensor, ttnn.ROW_MAJOR_LAYOUT, use_multicore=ttnn.get_memory_config(tensor).shard_spec is not None
    )
    tensor = ttnn.from_device(tensor)
    assert output_channels == tensor.shape[3]
    tensor = fallback_ops.reshape(
        tensor.value,
        batch_size,
        output_height,
        output_width,
        output_channels,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        output_on_device=False,
    )
    tensor = fallback_ops.permute(tensor, (0, 3, 1, 2), output_layout=ttnn.ROW_MAJOR_LAYOUT, output_on_device=False)
    tensor = ttnn.Tensor(tensor)
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    tensor = ttnn.to_device(tensor, device)
    return tensor


def run_ttnn_conv_with_pre_and_post_tensor_formatting(
    device, ttnn_conv_op, tensor: ttnn.Tensor, batch_size, output_height, output_width, output_channels
) -> ttnn.Tensor:
    tensor = pre_process_input(device, tensor)
    # print("Running conv op")
    tensor = ttnn_conv_op(tensor)
    tensor = post_process_output(device, tensor, batch_size, output_height, output_width, output_channels)
    return tensor
