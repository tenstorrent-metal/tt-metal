# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


@pytest.mark.parametrize(
    "input_shape, dim, layout",
    [
        ([4, 1, 33, 256], 0, ttl.tensor.Layout.ROW_MAJOR),
        ([4, 1, 256, 32], 0, ttl.tensor.Layout.TILE),
        ([8, 5, 13, 384], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([8, 5, 32, 384], 3, ttl.tensor.Layout.TILE),
        # MLP AllGather
        ([1, 1, 32, 32768], 3, ttl.tensor.Layout.TILE),
        # Selfout AllGather
        ([1, 2 * 16, 32, 64], 1, ttl.tensor.Layout.TILE),
        # Input AllGather
        ([1, 1, 32, 8192], 3, ttl.tensor.Layout.TILE),
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),
    ],
)
def test_all_gather(devices, input_shape, dim, layout, mem_config, function_level_defaults):
    input_tensor = torch.rand(input_shape).bfloat16()
    input_tensors = torch.chunk(input_tensor, 4, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(
            ttl.tensor.Tensor(t, ttl.tensor.DataType.BFLOAT16).to(layout).to(devices[i], mem_config)
        )

    tt_out_tensors = ttl.tensor.all_gather(tt_input_tensors, dim, mem_config)

    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        eq, output = comp_equal(tt_output_tensor, input_tensor)
        assert eq, f"{i} FAILED: {output}"
