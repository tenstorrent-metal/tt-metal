# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import repeat_interleave as tt_repeat_interleave
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand
from tests.tt_eager.python_api_testing.sweep_tests.common import set_slow_dispatch_mode


def run_repeat_interleave_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, repeat, dim, dispatch_mode, device
):
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = gen_rand(size=input_shape, low=-100, high=100).to(torch.bfloat16)
    # compute ref value
    x_ref = x.detach().clone()
    ref_value = pytorch_ops.repeat_interleave(x_ref, repeat=repeat, dim=dim)

    tt_result = tt_repeat_interleave(
        x=x,
        repeat=repeat,
        dim=dim,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_equal(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert success


# 	3	2	6978585	(('TT_METAL_SLOW_DISPATCH_MODE', '1'),)	completed	Max ATOL Delta: 198.0, Max RTOL Delta: inf, PCC: 0.9859791322181275, PCC check failed	fail	NA	NA	NA	Details

test_sweep_args = [
    (
        (1, 10, 230, 38),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        6978585,
        3,
        2,
        "1",
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, repeat, dim, dispatch_mode",
    (test_sweep_args),
)
def test_repeat_interleave_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, repeat, dim, dispatch_mode, device
):
    random.seed(0)
    run_repeat_interleave_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, repeat, dim, dispatch_mode, device
    )
