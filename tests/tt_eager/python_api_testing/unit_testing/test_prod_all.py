# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from functools import partial

import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc

from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)


def get_tensors(input_shape, output_shape, device):
    torch.manual_seed(2023)
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.float32
    npu_layout = ttl.tensor.Layout.TILE

    torch_input = torch.randint(1, 5, input_shape, dtype=cpu_dtype)
    torch_output = torch.randint(1, 5, output_shape, dtype=cpu_dtype)
    torch.set_printoptions(threshold=10000, precision=5, sci_mode=False)
    tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


@pytest.mark.parametrize(
    "shapes",
    (
        ([1, 1, 32, 32]),
        ([1, 4, 32, 32]),
        ([2, 2, 32, 32]),
        ([6, 4, 32, 32]),
        # [[1, 1, 320, 320]], #Fails for all_dimensions = True
        # [[1, 3, 320, 64]], #Fails for all_dimensions = True
    ),
)
def test_prod(shapes, device):
    output_shape = shapes.copy()

    (tt_input, tt_output, torch_input) = get_tensors(shapes, shapes, device)

    torch_output = torch.prod(torch_input)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_cpu = (
        ttl.operations.primary.prod_all(tt_input).cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()
    )
    N, C, H, W = tt_output_cpu.shape
    torch.set_printoptions(threshold=10000, precision=5, sci_mode=False)
    logger.info("Input shape")
    logger.info(torch_input.shape)
    logger.info("TT Output")
    logger.info(tt_output_cpu[0, 0, 0, 0])
    logger.info("Torch Output")
    logger.info(torch_output)

    # test for equivalance
    # TODO(Dongjin) : check while changing rtol after enabling fp32_dest_acc_en
    rtol = atol = 0.12
    # passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)
    passing, output_pcc = comp_allclose_and_pcc(
        torch_output, tt_output_cpu[0, 0, 0, 0], pcc=0.999, rtol=rtol, atol=atol
    )

    logger.info(f"Out passing={passing}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing
