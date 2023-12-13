# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger


from models.utility_functions import is_wormhole_b0
import tt_lib as ttl
from models.utility_functions import (
    comp_pcc,
)
import torch

shapes = [
    [1, 1, 25088, 64],
]


@pytest.mark.parametrize("shape", shapes)
def test_move_op(shape, device):
    run_move_op(shape, device)


def run_move_op(shape, device):
    """
    For non_overlap, multi-core is run for num_tiles > 1.
    """
    torch.manual_seed(1234)
    height_sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    dtype = ttl.tensor.DataType.BFLOAT16
    layout = ttl.tensor.Layout.ROW_MAJOR
    shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
    shard_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(11, 7),
            ),
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 8),
                ttl.tensor.CoreCoord(1, 8),
            ),
        }
    )
    assert shape[0] == 1 and shape[1] == 1
    assert shape[2] % 98 == 0 and shape[3] % 32 == 0
    shard_shape = [(int)(shape[2] / 98), shape[3]]
    shard_spec = ttl.tensor.ShardSpec(
        shard_grid,
        shard_shape,
        shard_orientation,
        False,
    )
    # make dummy shape half of shape, so we will test move sharded with overlap
    dummy_shape = [shape[0], shape[1], (int)(shape[2] / 2), shape[3]]
    dummy_shard_shape = [(int)(dummy_shape[2] / 98), dummy_shape[3]]
    dummy_shard_spec = ttl.tensor.ShardSpec(
        shard_grid,
        dummy_shard_shape,
        shard_orientation,
        False,
    )
    dummy_tensor = torch.zeros(dummy_shape)
    # tt_dummy_tensor = ttl.tensor.Tensor(dummy_tensor, dtype)
    # tt_dummy_tensor = tt_dummy_tensor.to(device, height_sharded_mem_config, dummy_shard_spec)
    print("shape=", shape)
    input_volume = shape[2] * shape[3]
    tensor = []
    for val in range(1, input_volume + 1):
        tensor.append(val)
    torch_tensor = torch.tensor(tensor).reshape(shape)
    tt_tensor = ttl.tensor.Tensor(torch_tensor, dtype)
    tt_tensor = tt_tensor.to(device, height_sharded_mem_config, shard_spec)

    # Free up dummy tensor from memory to make available to move
    # tt_dummy_tensor.deallocate()

    output = ttl.tensor.move_sharded(tt_tensor, height_sharded_mem_config)

    tt_host_rm = output.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm = tt_host_rm.to_torch()

    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm, torch_tensor, 0.99)
    logger.info(f"Passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing_pcc
