# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
from loguru import logger
import pytest
import torch
import tt_lib as ttl


from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests import tt_lib_ops, pytorch_ops
from models.utility_functions import tt2torch_tensor
from tests.tt_eager.python_api_testing.sweep_tests.common import set_slow_dispatch_mode


def run_clamp_bw(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device):
    random.seed(0)
    torch.manual_seed(data_seed)
    # prev_dispatch_mode=#set_slow_dispatch_mode(dispatch_mode)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape).uniform_(-100, 100)

    x = torch.where(x.abs() > 1e-3, x, 1e-3)
    y = torch.where(y.abs() > 1e-3, y, 1e-3)

    ref_value = pytorch_ops.sqrt_bw(x, y)

    tt_result = tt_lib_ops.sqrt_bw(
        x=x,
        y=y,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    # set_slow_dispatch_mode(prev_dispatch_mode)

    assert success


test_sweep_args = [
    (
        (6, 11, 192, 32),
        [
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.DataType.BFLOAT16,
        ],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
            None,
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        6411147,
        "",
    ),
    (
        (6, 11, 192, 32),
        [
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.DataType.BFLOAT16,
        ],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            None,
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        4781318,
        "",
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_clamp_bw_test(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dispatch_mode, device):
    run_clamp_bw(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, dispatch_mode, device)
