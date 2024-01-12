# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import tt_lib as ttl
from ttnn.core import reshape
from ttnn.decorators import decorate_operation
from ttnn.tensor import (
    Tensor,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
    TILE_LAYOUT,
    has_storage_type_of,
    DEVICE_STORAGE_TYPE,
)


def _torch_split_query_key_value_and_split_heads(input_tensor: Tensor, *, num_heads, **_):
    import ttnn
    import torch

    batch_size, sequence_size, three_times_hidden_size = input_tensor.shape
    hidden_size = three_times_hidden_size // 3
    head_size = hidden_size // num_heads

    tensor = torch.reshape(input_tensor, (batch_size, sequence_size, 3, num_heads, head_size))
    query_layer, key_layer, value_layer = (
        tensor[..., 0, :, :],
        tensor[..., 1, :, :],
        tensor[..., 2, :, :],
    )

    query_layer = torch.reshape(query_layer, (batch_size, sequence_size, num_heads, head_size))
    query_layer = torch.permute(query_layer, (0, 2, 1, 3)).contiguous().clone()

    key_layer = torch.reshape(key_layer, (batch_size, sequence_size, num_heads, head_size))
    key_layer = torch.permute(key_layer, (0, 2, 3, 1)).contiguous().clone()

    value_layer = torch.reshape(value_layer, (batch_size, sequence_size, num_heads, head_size))
    value_layer = torch.permute(value_layer, (0, 2, 1, 3)).contiguous().clone()

    return query_layer, key_layer, value_layer


def _fallback_split_query_key_value_and_split_heads(input_tensor: Tensor, *, num_heads, **_):
    import ttnn

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return _torch_split_query_key_value_and_split_heads(input_tensor, num_heads=num_heads)


@decorate_operation(torch_function=_fallback_split_query_key_value_and_split_heads)
def split_query_key_value_and_split_heads(
    input_tensor: Tensor,
    kv_input_tensor: Optional[Tensor] = None,
    *,
    num_heads: int,
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    split_query_key_value_and_split_heads(input_tensor: ttnn.Tensor, *, num_heads: int, core_grid: Tuple[int, int], memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tuple[Tensor, Tensor, Tensor]

    Splits tensor of shape [batch_size, sequence_size, 3 * hidden_size] into 3 tensors (Query, Key, Value) of shape [batch_size, sequence_size, hidden_size]. Then, reshapes and permutes them, to make them ready for computing attention scores. Equivalent pytorch code:

        batch_size, sequence_size, three_times_hidden_size = input_tensor.shape
        hidden_size = three_times_hidden_size // 3
        head_size = hidden_size // num_heads

        query_layer, key_layer, value_layer = torch.split(input_tensor, [hidden_size, hidden_size, hidden_size], dim=-1)

        query_layer = torch.reshape(query_layer, (batch_size, sequence_size, num_heads, head_size))
        query_layer = torch.permute(query_layer, (0, 2, 1, 3)).contiguous().clone()

        key_layer = torch.reshape(key_layer, (batch_size, sequence_size, num_heads, head_size))
        key_layer = torch.permute(key_layer, (0, 2, 3, 1)).contiguous().clone()

        value_layer = torch.reshape(value_layer, (batch_size, sequence_size, num_heads, head_size))
        value_layer = torch.permute(value_layer, (0, 2, 1, 3)).contiguous().clone()

    Args:
        * :attr:`input_tensor`: Input Tensor
        * :attr:`num_heads`: num heads to split into
        * :attr:`core_grid`: Compute and Storage Core Grid to use for the operation
        * :attr:`memory_config`: Memory Config of the output tensor

    """
    if len(input_tensor.shape) != 3:
        raise RuntimeError("Input Tensor must have strictly 3 dimensions!")

    if input_tensor.layout != TILE_LAYOUT:
        raise RuntimeError("Input Tensor must be in a TILE_LAYOUT!")

    if not has_storage_type_of(input_tensor, DEVICE_STORAGE_TYPE):
        raise RuntimeError("input_tensor must be on device!")

    batch_size, *_ = input_tensor.shape
    if input_tensor.shape == (batch_size, 384, 1024 * 3):
        batch_size, sequence_size, three_times_hidden_size = input_tensor.shape
        input_tensor = reshape(input_tensor, (batch_size, 1, sequence_size, three_times_hidden_size))

        ttl_input_tensor = input_tensor.value

        query_key_value = ttl.operations.primary.transformers.split_query_key_value_and_split_heads(
            ttl_input_tensor,
            ttl_input_tensor.device().compute_with_storage_grid_size(),
            memory_config,
        )
        query_key_value = (Tensor(ttl_tensor) for ttl_tensor in query_key_value)
        query, key, value = query_key_value
        return query, key, value
    else:
        batch_size, sequence_size, hidden_size = input_tensor.shape

        input_tensor = reshape(input_tensor, (batch_size, 1, sequence_size, hidden_size))
        ttl_input_tensor = input_tensor.value

        if kv_input_tensor is not None:
            kv_input_tensor = reshape(kv_input_tensor, (batch_size, 1, sequence_size, hidden_size * 2))
            ttl_kv_input_tensor = kv_input_tensor.value
        else:
            ttl_kv_input_tensor = None

        query_key_value = ttl.tensor.nlp_create_qkv_heads(
            ttl_input_tensor,
            ttl_kv_input_tensor,
            num_heads=num_heads,
            output_mem_config=memory_config,
        )
        query_key_value = (Tensor(ttl_tensor) for ttl_tensor in query_key_value)
        query, key, value = query_key_value
        return query, key, value


def _torch_attention_softmax(input_tensor: Tensor, *, head_size: int, attention_mask, **_):
    import torch

    if head_size is not None:
        scaler = 1 / (head_size**0.5)
    else:
        scaler = 1.0

    input_tensor *= scaler

    if attention_mask is not None:
        input_tensor += attention_mask[..., :1, :]

    return torch.softmax(input_tensor, -1)


def _fallback_attention_softmax(input_tensor: Tensor, *, head_size: int, attention_mask, **_):
    import ttnn

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    if attention_mask is not None:
        attention_mask = ttnn.from_device(attention_mask)
        attention_mask = ttnn.to_layout(attention_mask, ttnn.ROW_MAJOR_LAYOUT)
        attention_mask = ttnn.to_torch(attention_mask)

    return _torch_attention_softmax(input_tensor, head_size=head_size, attention_mask=attention_mask)


@decorate_operation(torch_function=_fallback_attention_softmax)
def attention_softmax(
    input_tensor: Tensor,
    *,
    head_size: Optional[int],
    attention_mask: Optional[Tensor],
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
) -> Tensor:
    """
    attention_softmax(input_tensor: ttnn.Tensor, *, head_size: int, attention_mask: Optional[Tensor]) -> Tensor

    Divides :attr:`input_tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax

    Args:
        * :attr:`input_tensor`: Input Tensor
        * :attr:`core_grid`: Compute and Storage Core Grid to use for the operation
        * :attr:`memory_config`: Memory Config of the output tensor

    """
    if len(input_tensor.shape) != 4:
        raise RuntimeError("Input Tensor must have strictly 4 dimensions!")

    if input_tensor.layout != TILE_LAYOUT:
        raise RuntimeError("Input Tensor must be in a TILE_LAYOUT!")

    if head_size is not None:
        scaler = 1 / (head_size**0.5)
    else:
        scaler = 1.0

    if attention_mask is not None:
        output_tensor = ttl.tensor.scale_mask_softmax(
            input_tensor.value, scaler, attention_mask.value, output_mem_config=memory_config
        )
        return Tensor(output_tensor)
    else:
        scaled_input_tensor = input_tensor * scaler
        ttl_scaled_input_tensor = scaled_input_tensor.value
        ttl_output_tensor = ttl.tensor.softmax(ttl_scaled_input_tensor, output_mem_config=memory_config)
        return Tensor(ttl_output_tensor)


@decorate_operation(torch_function=_torch_attention_softmax)
def attention_softmax_(
    input_tensor: Tensor,
    *,
    head_size: Optional[int],
    attention_mask: Optional[Tensor],
) -> Tensor:
    """
    attention_softmax_(input_tensor: ttnn.Tensor, *, head_size: int, attention_mask: Optional[Tensor]) -> Tensor

    Divides :attr:`input_tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax. In-Place.

    Args:
        * :attr:`input_tensor`: Input Tensor
        * :attr:`core_grid`: Compute and Storage Core Grid to use for the operation
        * :attr:`memory_config`: Memory Config of the output tensor

    """
    if len(input_tensor.shape) != 4:
        raise RuntimeError("Input Tensor must have strictly 4 dimensions!")

    if input_tensor.layout != TILE_LAYOUT:
        raise RuntimeError("Input Tensor must be in a TILE_LAYOUT!")

    if head_size is not None:
        scaler = 1 / (head_size**0.5)
    else:
        scaler = 1.0

    if attention_mask is not None:
        ttl.operations.primary.transformers.scale_mask_softmax_in_place(
            input_tensor.value, scaler, attention_mask.value
        )
        return input_tensor
    else:
        raise RuntimeError("Cannot apply divide by sqrt(head_size) using in-place version!")


def _torch_concatenate_heads(input_tensor: Tensor, **_):
    import torch

    batch_size, num_heads, sequence_size, head_size = input_tensor.shape

    output_tensor = torch.permute(input_tensor, (0, 2, 1, 3)).contiguous().clone()
    output_tensor = (
        torch.reshape(output_tensor, (batch_size, sequence_size, num_heads * head_size)).contiguous().clone()
    )
    return output_tensor


def _fallback_concatenate_heads(input_tensor: Tensor, **_):
    import ttnn

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return _torch_concatenate_heads(input_tensor)


@decorate_operation(torch_function=_fallback_concatenate_heads)
def concatenate_heads(
    input_tensor: Tensor,
    *,
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
) -> Tensor:
    """
    concatenate_heads(input_tensor: ttnn.Tensor, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tensor

    Takes in a tensor of shape [batch_size, num_heads, sequence_size, head_size], concatenates heads back along the width dimension and return the tensor of [batch_size, sequence_size, num_heads * head_size]

    Args:
        * :attr:`input_tensor`: Input Tensor
        * :attr:`memory_config`: Memory Config of the output tensor

    """
    if len(input_tensor.shape) != 4:
        raise RuntimeError("Input Tensor must have strictly 4 dimensions!")

    if input_tensor.layout != TILE_LAYOUT:
        raise RuntimeError("Input Tensor must be in a TILE_LAYOUT!")

    batch_size, num_heads, sequence_size, head_size = input_tensor.shape

    ttl_input_tensor = input_tensor.value
    ttl_output_tensor = ttl.tensor.nlp_concat_heads(
        ttl_input_tensor,
        memory_config,
    )
    output_tensor = Tensor(ttl_output_tensor)
    output_tensor = reshape(output_tensor, (batch_size, sequence_size, num_heads * head_size))

    return output_tensor
