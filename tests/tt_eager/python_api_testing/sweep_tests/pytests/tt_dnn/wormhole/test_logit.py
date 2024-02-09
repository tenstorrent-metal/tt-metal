# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.common import set_slow_dispatch_mode
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests import tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_eltwise_logit_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, eps, data_seed, dispatch_mode, device
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = gen_rand(size=input_shape, low=0, high=0.99).to(torch.bfloat16)
    x = torch.where(x.abs() > 1e-3, x, 1e-3)

    # compute ref value
    ref_value = pytorch_ops.logit(x=x, eps=eps)

    tt_result = tt_lib_ops.eltwise_logit(
        x=x,
        eps=eps,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert success


test_sweep_args = [
    (
        (9, 5, 160, 96),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        0.64453125,
        7340822,
        "",
    ),
    (
        (12, 9, 64, 384),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        0.140625,
        12484268,
        "",
    ),
    (
        (3, 11, 181, 76),
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.ROW_MAJOR],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        1000,
        7340822,
        "",
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, eps, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_eltwise_logit(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, eps, data_seed, dispatch_mode, device
):
    random.seed(0)
    run_eltwise_logit_test(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, eps, data_seed, dispatch_mode, device
    )
