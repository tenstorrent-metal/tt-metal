# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from typing import Union, Tuple

import torch
import torch.nn as nn
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


TILE_WIDTH = 32


def get_shard_grid_from_num_cores(ncores: Union[int, Tuple[int, int]]) -> ttnn.experimental.tensor.CoreRangeSet:
    max_grid_size = (9, 12)  ## (y, x)
    if isinstance(ncores, int):
        if ncores % max_grid_size[1] == 0:
            core_grid = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
            grid_coord = ttnn.experimental.tensor.CoreCoord(core_grid.x - 1, core_grid.y - 1)
            return ttnn.experimental.tensor.CoreRangeSet(
                {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord)}
            )
        else:
            if ncores < max_grid_size[1]:
                core_grid = ttnn.CoreGrid(y=1, x=ncores)
                grid_coord = ttnn.experimental.tensor.CoreCoord(core_grid.x - 1, 0)
                return ttnn.experimental.tensor.CoreRangeSet(
                    {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord)}
                )
            else:
                core_grid_1 = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
                core_grid_2 = ttnn.CoreGrid(y=ncores // max_grid_size[1] + 1, x=ncores % max_grid_size[1])
                grid_coord_1 = ttnn.experimental.tensor.CoreCoord(core_grid_1.x - 1, core_grid_1.y - 1)
                grid_coord_2 = ttnn.experimental.tensor.CoreCoord(core_grid_2.x - 1, core_grid_2.y - 1)
                return ttnn.experimental.tensor.CoreRangeSet(
                    {
                        ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord_1),
                        ttnn.experimental.tensor.CoreRange(
                            ttnn.experimental.tensor.CoreCoord(0, grid_coord_2.y), grid_coord_2
                        ),
                    }
                )
    elif isinstance(ncores, tuple):
        ncores_h, ncores_w = ncores
        assert ncores_h <= max_grid_size[0]
        assert ncores_w <= max_grid_size[1]
        return ttnn.experimental.tensor.CoreRangeSet(
            {
                ttnn.experimental.tensor.CoreRange(
                    ttnn.experimental.tensor.CoreCoord(0, 0),
                    ttnn.experimental.tensor.CoreCoord(ncores_w - 1, ncores_h - 1),
                )
            }
        )
    else:
        raise ValueError("Invalid ncores")


@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 8, 8, 640],
        [2, 16, 16, 640],
        [1, 16, 16, 640],
        [2, 8, 8, 1280],
        [2, 16, 16, 1280],
    ],
)
@pytest.mark.parametrize(
    "shard_strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.WIDTH]
)
def test_silu_multi_core(device, input_shape, shard_strategy):
    ## input shape is N C H W
    batch_size, height, width, num_channels = input_shape
    torch.manual_seed(0)
    input = torch.rand(input_shape, dtype=torch.bfloat16)

    torch_result = nn.functional.silu(input)

    tt_input = input
    num_bytes = 2  ## only BFLOAT16 is supported

    ## calculate ncores, corresponding grid_size and in_shard_shape based on the input_shape
    ncores = None
    max_grid_size = (9, 12)  ## (y, x)
    if shard_strategy == ttnn.ShardStrategy.HEIGHT:
        ## nsticks per shard should be divisible by in_w
        max_nshards = min(batch_size * height, max_grid_size[0] * max_grid_size[1])
        nshards = max_nshards
        while nshards > 0:
            if batch_size * height % nshards == 0:
                break
            nshards -= 1
        ncores = nshards
    elif shard_strategy == ttnn.ShardStrategy.WIDTH:
        ## nsticks per shard should be divisible by in_w
        max_nshards_w = min(num_channels, max_grid_size[1])
        nshards_w = max_nshards_w
        while nshards_w > 0:
            ## make sure: 1. nshards_w divides num_channels, and 2. shard_shape[1] is aligned to 32B
            if num_channels % nshards_w == 0 and math.ceil(num_channels * num_bytes / nshards_w) % TILE_WIDTH == 0:
                break
            nshards_w -= 1
        ncores = nshards_w
    elif shard_strategy == ttnn.ShardStrategy.BLOCK:
        max_nshards_h = min(batch_size * height, max_grid_size[0])  ## height along NHW
        max_nshards_w = min(num_channels, max_grid_size[1])  ## width along C
        ## find nshards_h along NHW
        nshards_h = max_nshards_h
        while nshards_h > 0:
            if batch_size * height % nshards_h == 0:
                break
            nshards_h -= 1
        ## find nshards_w along C
        nshards_w = max_nshards_w
        while nshards_w > 0:
            ## make sure: 1. nshards_w divides num_channels, and 2. shard_shape[1] is aligned to 32B
            if num_channels % nshards_w == 0 and math.ceil(num_channels * num_bytes / nshards_w) % TILE_WIDTH == 0:
                break
            nshards_w -= 1
        if nshards_w == 0 or nshards_h == 0:
            raise ValueError("nshards_h or nshards_w is 0")
        ncores = (nshards_h, nshards_w)

    shard_grid = get_shard_grid_from_num_cores(ncores)
    shard_orientation = ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR

    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        tensor_memory_layout = ttnn.types.TensorMemoryLayout.BLOCK_SHARDED
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        tensor_memory_layout = ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED
    elif shard_strategy == ttnn.ShardStrategy.WIDTH:
        tensor_memory_layout = ttnn.types.TensorMemoryLayout.WIDTH_SHARDED

    ## input shard
    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        shard_height = math.ceil(batch_size * height * width / ncores[0])
        shard_width = math.ceil(num_channels / ncores[1])
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        shard_height = math.ceil(batch_size * height * width / ncores)
        shard_width = num_channels
    elif shard_strategy == ttnn.ShardStrategy.WIDTH:
        shard_height = math.ceil(batch_size * height * width)
        shard_width = math.ceil(num_channels / ncores)
    shard_shape = (shard_height, shard_width)

    shard_spec = ttnn.experimental.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, False)
    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.types.BufferType.L1, shard_spec)

    ## output shard
    shard_shape = (shard_height, shard_width)
    shard_spec = ttnn.experimental.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, False)

    input_tensor = ttnn.from_torch(tt_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)

    output_tensor = ttnn.silu(input_tensor, memory_config=in_sharded_mem_config)
    output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    ## compare the results
    assert_with_pcc(torch_result, output_tensor, 0.999)
