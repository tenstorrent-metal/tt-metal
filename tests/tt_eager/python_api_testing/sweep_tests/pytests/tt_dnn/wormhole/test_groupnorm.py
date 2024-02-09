# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import groupnorm as tt_groupnorm


def run_groupnorm_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config[0] == "SYSTEM_MEMORY":
        in_mem_config[0] = None
    if in_mem_config[1] == "SYSTEM_MEMORY":
        in_mem_config[1] = None
    if in_mem_config[2] == "SYSTEM_MEMORY":
        in_mem_config[2] = None

    print(in_mem_config)

    x = torch.Tensor(size=input_shape[0]).uniform_(-10, 10).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape[1]).uniform_(-10, 10).to(torch.bfloat16)
    z = torch.Tensor(size=input_shape[1]).uniform_(-10, 10).to(torch.bfloat16)

    x = torch.where(x.abs() > 1e-3, x, 1e-3)
    y = torch.where(y.abs() > 1e-3, y, 1e-3)
    z = torch.where(z.abs() > 1e-3, z, 1e-3)

    x_ref = x.detach().clone()
    y_ref = y.detach().clone()
    z_ref = z.detach().clone()

    # compute ref value --------------------------
    ref_value = pytorch_ops.groupnorm(x_ref, y_ref, z_ref)

    # compute tt value ---------------------------
    tt_result = tt_groupnorm(
        x=x,
        y=y,
        z=z,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs -------------
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    assert success


test_sweep_args = [
    (
        [(7, 14, 32, 160), (1, 14, 1, 1), (1, 14, 1, 1)],
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        18784230,
    ),
    (
        [(7, 14, 32, 160), (1, 14, 1, 1), (1, 14, 1, 1)],
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        3378971,
    ),
    (
        [(7, 14, 32, 160), (1, 14, 1, 1), (1, 14, 1, 1)],
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT8_B],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        16005792,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_groupnorm_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_groupnorm_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
