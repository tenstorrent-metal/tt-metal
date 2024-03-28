# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@pytest.mark.parametrize(
    "N, C, H, W, num_groups",
    [
        (1, 320, 1, 8192, 32),
        (1, 960, 1, 1024, 32),
        # not fit in L1 for GS
        # (1, 960, 1, 4096, 32),
    ],
)
def test_group_norm_with_block_sharded_v2_8x4_grid(device, N, C, H, W, num_groups):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=4, x=8)

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.ones((C,), dtype=torch.bfloat16)
    torch_bias = torch.zeros((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # input mask
    input_mask_tensor = ttnn.create_groupnorm_input_mask(C, num_groups, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # gamma/beta
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # shard config
    grid_coord = ttnn.experimental.tensor.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord)}
    )
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid, shard_shape, ttnn.experimental.tensor.ShardOrientation.COL_MAJOR, False
    )
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    # groupnorm
    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
    )

    # output tensor
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9997)


@pytest.mark.parametrize(
    "N, C, H, W, num_groups",
    [
        (2, 320, 64, 64, 32),
        (1, 640, 1, 2048, 32),
        (1, 640, 1, 4096, 32),
        (1, 960, 1, 2048, 32),
        (1, 960, 1, 4096, 32),
        (1, 1280, 1, 512, 32),
        (1, 1280, 1, 2048, 32),
        (1, 1920, 1, 512, 32),
        (1, 1920, 1, 2048, 32),
        (1, 2560, 1, 512, 32),
        # not fit in L1 for GS
        # (2, 960, 64, 64, 32),
        # (1, 640, 1, 8192, 32),
    ],
)
def test_group_norm_with_block_sharded_v2_8x8_grid(device, N, C, H, W, num_groups):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=8, x=8)

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.ones((C,), dtype=torch.bfloat16)
    torch_bias = torch.zeros((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # input mask
    input_mask_tensor = ttnn.create_groupnorm_input_mask(C, num_groups, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # gamma/beta
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # shard config
    grid_coord = ttnn.experimental.tensor.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord)}
    )
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid, shard_shape, ttnn.experimental.tensor.ShardOrientation.COL_MAJOR, False
    )
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    # groupnorm
    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
    )

    # output tensor
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9997)
