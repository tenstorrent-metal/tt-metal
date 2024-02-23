# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import pathlib

import torch

import tt_lib as ttl

from models.utility_functions import get_debug_tensor

tt_dtype_to_torch_dtype = {
    ttl.tensor.DataType.UINT32: torch.int32,
    ttl.tensor.DataType.UINT16: torch.int16,
    ttl.tensor.DataType.BFLOAT16: torch.bfloat16,
    ttl.tensor.DataType.BFLOAT8_B: torch.float,
}
TILE_WIDTH = 32
TILE_HEIGHT = 32

debug = True


def print_tiles(tiled_tensor, num_tiles_height, num_tiles_width):
    tile_torch_rows = torch.chunk(tiled_tensor, int(num_tiles_height), dim=2)
    row_idx = 0
    for row in tile_torch_rows:
        tiles = torch.chunk(row, int(num_tiles_width), dim=3)
        col_idx = 0
        for tile in tiles:
            tile_idx = row_idx * num_tiles_width + col_idx
            print("Trip Tile " + str(int(tile_idx)) + " with shape " + str(tile.shape))
            print(tile)
            col_idx = col_idx + 1
        row_idx = row_idx + 1


def get_tensor(shape, dtype):
    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)
    return torch_tensor


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.BFLOAT16,
    ],
)
@pytest.mark.parametrize(
    "tensor_shape, shard_scheme, shard_shape, grid_override",
    [
        ([1, 4, 64, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (64, 64), None),
        ([1, 1, 128, 128], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (64, 64), None),
        ([1, 1, 2048, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (512, 64), None),
        ([1, 1, 2048, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64), None),
        ([1, 1, 4096, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64), None),
        ([1, 1, 8192, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64), None),
        ([1, 1, 14336, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64), None),
        ([1, 1, 256, 32], ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, (32, 32), None),
        ([1, 1, 128, 64], ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, (32, 64), None),
        ([1, 1, 2048, 64], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (1024, 64), None),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttl.tensor.ShardOrientation.ROW_MAJOR, ttl.tensor.ShardOrientation.COL_MAJOR],
)
def test_tensor_conversion_between_torch_and_tt_tile(
    tt_dtype, device, tensor_shape, shard_scheme, shard_shape, shard_orientation, grid_override
):
    dtype = tt_dtype_to_torch_dtype[tt_dtype]
    shard_halo = False

    if grid_override == None:
        compute_grid = ttl.tensor.CoreCoord(
            device.compute_with_storage_grid_size().x - 1, device.compute_with_storage_grid_size().y - 1
        )
    else:
        compute_grid = ttl.tensor.CoreCoord(grid_override[0] - 1, grid_override[1] - 1)
    shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), compute_grid)})
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)

    two_d_shape = (tensor_shape[0] * tensor_shape[1] * tensor_shape[2], tensor_shape[3])
    num_tiles_width = (two_d_shape[1]) / TILE_WIDTH
    num_tiles_height = (two_d_shape[0]) / TILE_HEIGHT

    if debug:
        torch_tensor = get_debug_tensor(num_tiles_width, num_tiles_height, dtype)
        torch_tensor
    else:
        torch_tensor = get_tensor(tensor_shape, dtype)
    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype).to(ttl.tensor.Layout.TILE)
    mem_config = ttl.tensor.MemoryConfig(shard_scheme, ttl.tensor.BufferType.L1, shard_spec)
    tt_tensor = tt_tensor.to(device, mem_config)

    tt_tensor = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    assert passing

    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype, device, ttl.tensor.Layout.TILE, mem_config)
    tt_tensor = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    torch_tensor_after_round_trip = tt_tensor.to_torch()

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    assert passing


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
        ttl.tensor.DataType.BFLOAT16,
    ],
)
@pytest.mark.parametrize(
    "tensor_shape, shard_scheme, shard_shape, grid_override",
    [
        ([1, 1, 4, 32], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1, 32), None),
        ([1, 1, 4, 32], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1, 32), (1, 4)),
        ([1, 1, 5, 32], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (2, 32), (1, 3)),
    ],
)
def test_tensor_conversion_between_torch_and_tt_rm(
    tt_dtype, device, tensor_shape, shard_scheme, shard_shape, grid_override
):
    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
    shard_halo = False
    if grid_override == None:
        compute_grid = ttl.tensor.CoreCoord(
            device.compute_with_storage_grid_size().x - 1, device.compute_with_storage_grid_size().y - 1
        )
    else:
        compute_grid = ttl.tensor.CoreCoord(grid_override[0] - 1, grid_override[1] - 1)
    shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), compute_grid)})
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)

    if debug:
        num_pages_width = tensor_shape[2]
        num_pages_height = tensor_shape[3] / shard_shape[1]
        torch_tensor = get_debug_tensor(
            num_pages_width, num_pages_height, dtype, page_width=1, page_height=tensor_shape[-1]
        )
    else:
        torch_tensor = get_tensor(tensor_shape, dtype)

    torch_tensor = torch_tensor.reshape(tensor_shape)

    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype)

    assert list(torch_tensor.size()) == tensor_shape

    mem_config = ttl.tensor.MemoryConfig(shard_scheme, ttl.tensor.BufferType.L1, shard_spec)
    tt_tensor = tt_tensor.to(device, mem_config)
    tt_tensor = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)

    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    assert passing
