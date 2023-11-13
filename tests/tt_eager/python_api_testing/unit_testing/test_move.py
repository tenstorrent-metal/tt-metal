# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger


from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0
import tt_lib as ttl
from models.utility_functions import (
    comp_pcc,
)
import torch


def run_move_op(test_id, shape, layout, dtype, in0_mem_config, output_mem_config, device):
    """
    For non_overlap, multi-core is run for num_tiles > 1.
    """
    torch.manual_seed(1234)

    # Dummy tensor to shift input tensor in memory
    if test_id == 0:
        dummy_shape = [1, 1, 32, 32]
    elif test_id == 1:
        dummy_shape = shape  # This will allow output and input buffers to not overlap
    else:
        raise NotImplementedError(f"Unknown test id: {test_id}!")

    dummy_tensor = torch.randn(dummy_shape)
    tt_dummy_tensor = ttl.tensor.Tensor(dummy_tensor, dtype).to(layout).to(device, in0_mem_config)

    torch_tensor = torch.randn(shape)
    tt_tensor = ttl.tensor.Tensor(torch_tensor, dtype).to(layout).to(device, in0_mem_config)

    # Free up dummy tensor from memory to make available to move
    tt_dummy_tensor.deallocate()

    output = ttl.tensor.move(tt_tensor, output_mem_config)

    tt_host_rm = output.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm = tt_host_rm.to_torch()

    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm, torch_tensor, 0.99)
    logger.info(f"Passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing_pcc


shapes = [
    [1, 1, 32, 32],
    [1, 3, 320, 384],
]
if is_wormhole_b0():
    del shapes[1:]


@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "output_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "dtype, layout",
    (
        (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.Layout.TILE),
        (ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR),
        (ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE),
    ),
    ids=["BFLOAT8_B-TILE", "BFLOAT16-RM", "BFLOAT16-TILE"],
)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("test_id", (0, 1), ids=["overlap", "non_overlap"])
def test_move_op(test_id, shape, layout, dtype, in0_mem_config, output_mem_config, device):
    run_move_op(test_id, shape, layout, dtype, in0_mem_config, output_mem_config, device)


def test_move_op_with_program_cache(use_program_cache, device):
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1)
    output_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1)
    dtype = ttl.tensor.DataType.BFLOAT16
    layout = ttl.tensor.Layout.TILE
    shape = [1, 3, 320, 384]

    # Single core because of overlap
    for _ in range(2):
        run_move_op(0, shape, layout, dtype, in0_mem_config, output_mem_config, device)

    # Multi-core
    for _ in range(2):
        run_move_op(1, shape, layout, dtype, in0_mem_config, output_mem_config, device)

    assert ttl.program_cache.num_entries() == 2
