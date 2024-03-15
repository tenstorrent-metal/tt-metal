# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

import tt_lib as ttl

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape, input_shard_config, output_shard_config",
    [
        (
            (1, 1, 8192, 320),
            dict(
                shape=(1, 1, 80, 1024),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 64, 1024),
                core_grid=ttnn.CoreGrid(y=5, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
        (
            (1, 1, 8192, 320),
            dict(
                shape=(1, 1, 64, 1024),
                core_grid=ttnn.CoreGrid(y=5, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 80, 1024),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
        (
            (1, 1, 2048, 320),
            dict(
                shape=(1, 1, 80, 1024),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 64, 1024),
                core_grid=ttnn.CoreGrid(y=5, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
        (
            (1, 1, 2048, 320),
            dict(
                shape=(1, 1, 64, 1024),
                core_grid=ttnn.CoreGrid(y=5, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 80, 1024),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
        (
            (1, 1, 2048, 640),
            dict(
                shape=(1, 1, 80, 256),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 128, 256),
                core_grid=ttnn.CoreGrid(y=5, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
        (
            (1, 1, 2048, 640),
            dict(
                shape=(1, 1, 128, 256),
                core_grid=ttnn.CoreGrid(y=5, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 80, 256),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
        (
            (1, 1, 512, 640),
            dict(
                shape=(1, 1, 80, 64),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 128, 64),
                core_grid=ttnn.CoreGrid(y=5, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
        (
            (1, 1, 512, 640),
            dict(
                shape=(1, 1, 128, 64),
                core_grid=ttnn.CoreGrid(y=5, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 80, 64),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
        (
            (1, 1, 2048, 640),
            dict(
                shape=(1, 1, 80, 256),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 128, 256),
                core_grid=ttnn.CoreGrid(y=5, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
        (
            (1, 1, 2048, 640),
            dict(
                shape=(1, 1, 128, 256),
                core_grid=ttnn.CoreGrid(y=5, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 128, 256),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
        (
            (1, 1, 8192, 640),
            dict(
                shape=(1, 1, 80, 1024),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 128, 1024),
                core_grid=ttnn.CoreGrid(y=5, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
        (
            (1, 1, 64, 64),
            dict(
                shape=(1, 1, 32, 64),
                core_grid=ttnn.CoreGrid(y=2, x=1),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            dict(
                shape=(1, 1, 64, 64),
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
        ),
    ],
)
def test_sd_reshard(device, input_shape, input_shard_config, output_shard_config):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)

    # convert to ttnn, and move to device
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # convert interleaved to sharded
    input_sharded_memory_config = ttnn.create_sharded_memory_config(
        **input_shard_config,
        halo=False,
        use_height_and_width_as_shard_shape=True,
    )
    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        **output_shard_config,
        halo=False,
        use_height_and_width_as_shard_shape=True,
    )
    sharded_input_tensor = ttnn.to_memory_config(input_tensor, input_sharded_memory_config)

    # call shared -> interleaved and interleaved -> sharded
    output_tensor_a = ttnn.to_memory_config(sharded_input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor_a = ttnn.to_memory_config(output_tensor_a, memory_config=output_sharded_memory_config)

    # call reshard op
    output_tensor_b = ttl.tensor.reshard(sharded_input_tensor, output_sharded_memory_config)
    output_tensor_b = ttnn.Tensor(output_tensor_b)

    # convert ttnn tensors to torch tensor
    output_a = ttnn.to_torch(output_tensor_a)
    output_b = ttnn.to_torch(output_tensor_b)
    assert_with_pcc(output_a, output_b, 1.0)
