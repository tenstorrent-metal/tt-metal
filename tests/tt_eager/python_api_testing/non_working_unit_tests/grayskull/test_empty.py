# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose
from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import empty as pt_empty
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import empty as tt_empty


def run_empty_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)
    x_ref = x.detach().clone()

    # get referent value
    ref_value = pt_empty(x_ref)

    tt_result = tt_empty(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    print(f"REF: {ref_value[0, 0, 1:10, 1:10]}")
    print(f"TT: {tt_result[0, 0, 1:10, 1:10]}")

    # compare tt and golden outputs
    success, pcc_value = comp_allclose(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (8, 21, 345, 448),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        6777703,
    ),
    # (
    #     (2, 16, 205, 132),
    #     ttl.tensor.DataType.BFLOAT16,
    #     ttl.tensor.Layout.ROW_MAJOR,
    #     ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    #     ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    #     1412689,
    # ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_empty_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_empty_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
