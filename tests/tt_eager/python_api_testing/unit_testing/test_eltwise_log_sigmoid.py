# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
import time
import os
import random
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../../..")

import pytest
import torch

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_log_sigmoid as tt_eltwise_log_sigmoid


def run_eltwise_log_sigmoid_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config= None

    x = torch.Tensor(size=input_shape).uniform_(-10, 10)
    x_ref = x.detach().clone()
    # get ref result
    ref_value = pytorch_ops.log_sigmoid(x_ref)

    tt_result = tt_eltwise_log_sigmoid(
        x=x,
        device=device,
        device_id=0,
        dtype=[dtype],
        layout=[dlayout],
        buffer_type=[in_mem_config],
        output_mem_config=out_mem_config
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args=[
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.DRAM, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 17155532),
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.BufferType.L1, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 16305027),
    ((4, 7, 32, 96), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 13587334),
    ((6, 4, 156, 214), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.BufferType.DRAM, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 19325774),
    ((6, 4, 156, 214), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.BufferType.L1, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 4016313),
    ((6, 4, 156, 214), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 13126809),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (
        test_sweep_args
    ),
)
def test_eltwise_log_sigmoid_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    run_eltwise_log_sigmoid_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
