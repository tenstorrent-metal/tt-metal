# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import pathlib

import torch

import tt_lib as ttl


tt_dtype_to_torch_dtype = {
    ttl.tensor.DataType.UINT32: torch.int32,
    ttl.tensor.DataType.BFLOAT16: torch.bfloat16,
}


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.BFLOAT16,
    ],
)
def test_tensor_conversion_between_torch_and_tt_tile(tt_dtype, device):
    TILE_HEIGHT = 32
    TILE_WIDTH = 32

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    num_cores_height = 4
    num_cores_width = 1
    num_tiles_per_core_height = 2
    num_tiles_per_core_width = 2

    shard_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(0, num_cores_height * num_cores_width - 1)
            )
        }
    )
    shard_shape = [TILE_HEIGHT * num_tiles_per_core_height, TILE_WIDTH * num_tiles_per_core_width]

    tensor_shape = (
        1,
        num_cores_height,
        TILE_HEIGHT * num_tiles_per_core_height,
        num_cores_width * TILE_WIDTH * num_tiles_per_core_width,
    )
    shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
    shard_halo = False
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)

    if dtype == torch.int32:
        torch_tensor = torch.randint(0, 1024, tensor_shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(tensor_shape, dtype=dtype)
    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype).to(ttl.tensor.Layout.TILE)

    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
    tt_tensor = tt_tensor.to(device, mem_config, shard_spec)
    tt_tensor = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)

    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    assert passing


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.BFLOAT16,
    ],
)
def test_tensor_conversion_between_torch_and_tt_rm(tt_dtype, device):
    TILE_HEIGHT = 32
    TILE_WIDTH = 32

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    num_cores_height = 8
    num_cores_width = 8

    shard_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(num_cores_height - 1, num_cores_width - 1)
            )
        }
    )
    shard_shape = [72, 128]

    tensor_shape = (1, 1, 2304, 256)
    shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
    shard_halo = False
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)

    if dtype == torch.int32:
        torch_tensor = torch.randint(0, 1024, tensor_shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(tensor_shape, dtype=dtype)
    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype)

    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1)
    tt_tensor = tt_tensor.to(device, mem_config, shard_spec)
    tt_tensor = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)

    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    assert passing
