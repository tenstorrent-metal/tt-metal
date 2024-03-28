# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import tt_lib as ttl
from loguru import logger

from tt_lib.utils import (
    _nearest_y,
)


@pytest.mark.parametrize(
    "batch_size, input_channels, input_height, input_width, num_cores, grid_size, height_sharded",
    (
        (1, 256, 64, 1, 98, (12, 9), True),
        # (8, 512, 512, 28, 28, 2, 2, 80, (10, 8), False),
        # (8, 1024, 1024, 14, 14, 2, 2, 56, (7, 8), False),
        # (16, 256, 256, 56, 56, 2, 2, 98, (12, 9), True),
        # (16, 512, 512, 28, 28, 2, 2, 80, (11, 8), False),
        # (16, 1024, 1024, 14, 14, 2, 2, 56, (9, 8), False),
    ),
)
def test_bcast_sharded(
    device,
    batch_size,
    input_channels,
    input_height,
    input_width,
    num_cores,
    grid_size,
    height_sharded,
):
    # if batch_size > 8 and dtype != ttl.tensor.DataType.BFLOAT8_B:
    #    pytest.skip("Batch > 8 must be run fully bfp8")
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    assert input_channels % 32 == 0
    # assert output_channels % 32 == 0
    # assert stride_h == stride_w
    dtype = ttl.tensor.DataType.BFLOAT16

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    torch.manual_seed(0)
    a_activation_shape = [batch_size, input_channels, input_height, input_width]
    A_pyt = torch.normal(mean=0, std=0.1, size=a_activation_shape).bfloat16()

    b_weights_shape = [input_channels]
    B_pyt = torch.normal(mean=0, std=0.1, size=b_weights_shape).bfloat16()

    # Convert NCHW to NHWC shape
    A_pyt_nhwc = torch.permute(A_pyt, (0, 2, 3, 1))
    A_pyt_nhwc = A_pyt_nhwc.reshape(1, 1, batch_size * input_height * input_width, input_channels)

    a_activation_shape_nhwc = [batch_size, input_height, input_width, input_channels]
    A_cl_host = ttl.tensor.Tensor(A_pyt_nhwc, dtype).reshape(
        1, 1, batch_size * input_height * input_width, input_channels
    )
    num_cores_height_slices = num_cores if height_sharded else grid_size[0]
    input_shape = [1, 1, _nearest_y(batch_size * input_height * input_width, 32), input_channels]
    A_cl_host = A_cl_host.pad(input_shape, (0, 0, 0, 0), 0.0)
    A_interleaved = A_cl_host.to(ttl.tensor.Layout.TILE).to(
        device,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )
    assert A_interleaved.get_legacy_shape()[0] == 1 and A_interleaved.get_legacy_shape()[1] == 1

    # image flattened params
    input_2d_height = A_interleaved.get_legacy_shape()[2]
    input_2d_width = A_interleaved.get_legacy_shape()[3]
    input_2d_height_padded = _nearest_y(input_2d_height, num_cores_height_slices * 32)
    input_shard_height = (int)(input_2d_height_padded / num_cores_height_slices)

    logger.debug(f"input_2d_height={input_2d_height}")
    logger.debug(f"input_2d_width={input_2d_width}")
    sharded_memory_layout = (
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED if height_sharded else ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED
    )
    sharded_memory_orientation = (
        ttl.tensor.ShardOrientation.ROW_MAJOR if height_sharded else ttl.tensor.ShardOrientation.COL_MAJOR
    )
    input_shard_width = input_2d_width if height_sharded else ((int)(input_2d_width / grid_size[1]))
    logger.debug(f"grid_size={grid_size}")
    logger.debug(f"shard_memory_layout={sharded_memory_layout}")
    logger.debug(f"input_shard_height={input_shard_height}, input_shard_width={input_shard_width}")

    A_sharded = ttl.tensor.interleaved_to_sharded(
        A_interleaved,
        grid_size,
        [input_shard_height, input_shard_width],
        sharded_memory_layout,
        sharded_memory_orientation,
    )
    # Prepare weights for simple matmul
    B_tiled_host = ttl.tensor.Tensor(B_pyt, dtype)
    B_tiled = B_tiled_host.to(device)

    # downsample golden output using maxpool
    # out_golden = torch.nn.functional.max_pool2d(A_pyt, 1, stride=stride_h)
    # out_golden_2d_nhwc = torch.permute(out_golden, (0, 2, 3, 1)).reshape(
    #     1, 1, batch_size * output_height * output_width, input_channels
    # )

    # downsample_params = [batch_size, input_height, input_width, stride_h, stride_w]
    sharded_memory_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1
    )
    # Run downsample op
    # A_downampled_sharded = ttl.tensor.downsample(A_sharded, downsample_params, output_dtype=dtype)

    tt_output = ttl.tensor.bcast(
        A_sharded,
        B_tiled,
        ttl.tensor.BcastOpMath.MUL,
        ttl.tensor.BcastOpDim.H,
        output_mem_config=sharded_memory_config,
    )
    print(tt_output)
