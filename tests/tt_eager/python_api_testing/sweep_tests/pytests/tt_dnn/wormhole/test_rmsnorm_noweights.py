# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import rmsnorm_noweights as tt_rmsnorm


def run_rmsnorm_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config[0] == "SYSTEM_MEMORY":
        in_mem_config[0] = None

    print(in_mem_config)

    x = torch.Tensor(size=input_shape[0]).uniform_(-10, 10)
    x_ref = x.detach().clone()

    # compute ref value --------------------------
    ref_value = pytorch_ops.rmsnorm_noweights(x_ref)

    # compute tt value ---------------------------
    tt_result = tt_rmsnorm(
        x=x,
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
        [(1, 6, 256, 160)],
        [ttl.tensor.DataType.BFLOAT8_B],
        [ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        9767382,
    ),
    (
        [(3, 10, 192, 64)],
        [ttl.tensor.DataType.BFLOAT8_B],
        [ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        17493725,
    ),
    (
        [(5, 11, 96, 128)],
        [ttl.tensor.DataType.BFLOAT8_B],
        [ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        5018076,
    ),
    (
        [(5, 11, 96, 128)],
        [ttl.tensor.DataType.BFLOAT8_B],
        [ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        1296595,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_rmsnorm_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_rmsnorm_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
