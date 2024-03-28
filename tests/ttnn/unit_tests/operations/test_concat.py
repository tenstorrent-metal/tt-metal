# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("height", [20, 32])
@pytest.mark.parametrize("width", [4, 32])
@pytest.mark.parametrize("dim", [0, 1])
def test_concat(device, height, width, dim):
    torch_input_tensor_a = torch.rand((height, width), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=dim)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=dim)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize(
    "input_shape_a, shard_shape_a, input_shape_b, shard_shape_b, output_shard_shape, shard_grid",
    (
        (
            (1, 1, 16, 16),
            (8, 16),
            (1, 1, 16, 16),
            (8, 16),
            (8, 32),
            ttnn.experimental.tensor.CoreRangeSet(
                {
                    ttnn.experimental.tensor.CoreRange(
                        ttnn.experimental.tensor.CoreCoord(0, 0), ttnn.experimental.tensor.CoreCoord(0, 1)
                    )
                }
            ),
        ),
        (
            (1, 1, 160, 32),
            (80, 32),
            (1, 1, 160, 32),
            (80, 32),
            (80, 64),
            ttnn.experimental.tensor.CoreRangeSet(
                {
                    ttnn.experimental.tensor.CoreRange(
                        ttnn.experimental.tensor.CoreCoord(0, 0), ttnn.experimental.tensor.CoreCoord(0, 1)
                    )
                }
            ),
        ),
        (
            (1, 1, 160, 32),
            (80, 32),
            (1, 1, 160, 16),
            (80, 16),
            (80, 48),
            ttnn.experimental.tensor.CoreRangeSet(
                {
                    ttnn.experimental.tensor.CoreRange(
                        ttnn.experimental.tensor.CoreCoord(0, 0), ttnn.experimental.tensor.CoreCoord(0, 1)
                    )
                }
            ),
        ),
    ),
)
def test_sharded_concat(
    device, input_shape_a, shard_shape_a, input_shape_b, shard_shape_b, output_shard_shape, shard_grid
):
    input_a_sharded_memory_config = ttnn.create_sharded_memory_config(
        shard_shape_a,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    input_b_sharded_memory_config = ttnn.create_sharded_memory_config(
        shard_shape_b,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        output_shard_shape,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    torch_input_tensor_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(input_shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=3)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_a = ttnn.to_memory_config(input_tensor_a, input_a_sharded_memory_config)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_b = ttnn.to_memory_config(input_tensor_b, input_b_sharded_memory_config)

    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=3, memory_config=output_sharded_memory_config)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
    assert_with_pcc(torch_output_tensor, output)

@pytest.mark.parametrize(
    "batch0, column0, height0, width0, batch1, column1, height1, width1",
    (
        (1, 256, 30, 40,1, 128, 30, 40),
        ( 1, 128, 60, 80, 1, 64, 60, 80),
        (1, 64, 30, 40,1, 128, 30, 40),
        (1, 128, 15, 20,1, 256, 15, 20),
    ),
)

@pytest.mark.parametrize("dim", [1])
def test_concat_yolov8(device, batch0, column0, height0, width0, batch1, column1, height1, width1, dim):
    torch_input_tensor_a = torch.rand((batch0, column0, height0, width0), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((batch1, column1, height1, width1), dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=dim)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=dim)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
