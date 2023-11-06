# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import complex_recip as pt_complex_recip
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import complex_recip as tt_complex_recip
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_complex
from tests.tt_eager.python_api_testing.sweep_tests.common import set_dispatch_mode


def run_complex_recip_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device
):
    torch.manual_seed(data_seed)
    set_dispatch_mode(dispatch_mode)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand_complex(size=input_shape, low=-100, high=100)
    x_ref = x.detach().clone()

    # get ref result
    ref_value = pt_complex_recip(x_ref)

    tt_result = tt_complex_recip(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (1, 7, 32, 192),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        17155532,
        False,
    ),
    (
        (1, 7, 32, 192),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        16305027,
        False,
    ),
    (
        (1, 5, 128, 192),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        19575052,
        False,
    ),
    (
        (1, 5, 128, 192),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        7329721,
        False,
    ),
    (
        (1, 6, 192, 384),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        5223220,
        False,
    ),
    (
        (1, 6, 192, 384),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        5588697,
        False,
    ),
    (
        (1, 11, 32, 64),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        2841039,
        False,
    ),
    (
        (1, 11, 32, 64),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        6645129,
        False,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_complex_recip_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device
):
    run_complex_recip_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device
    )
