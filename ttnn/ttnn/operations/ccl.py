# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

from typing import List

import tt_lib as ttl

import ttnn

import typing

THIS_MODULE = sys.modules[__name__]

__all__ = []


@ttnn.register_operation(
    name="ttnn.all_gather",
    validate_input_tensors=None,
    torch_function=None,
)
def all_gather(
    multidevice_tensor: typing.Any,
    dim: int,
    num_links: int,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
):
    per_device_tensor = [x.value for x in multidevice_tensor.get_tensors()]
    all_gather_tensor = ttl.tensor.all_gather(per_device_tensor, dim, num_links, output_mem_config=memory_config)
    output_tensors = [ttnn.Tensor(x) for x in all_gather_tensor]
    return ttnn.DeviceMeshTensor(
        output_tensors, multidevice_tensor.multidevice, ttnn.DeviceMeshTensorMapper(shard_dim=dim), device=True
    )
