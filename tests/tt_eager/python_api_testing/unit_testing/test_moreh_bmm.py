# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0
from tests.tt_eager.python_api_testing.unit_testing.test_moreh_matmul import get_tensors
from models.utility_functions import comp_allclose_and_pcc


@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "shape",
    (
        [1, 31, 639, 31],
        [5, 95, 415, 65],
        [10, 191, 447, 159],
        [20, 287, 479, 255],
    ),
)
def test_moreh_bmm(shape, device):
    input_shape = [1, shape[0], shape[1], shape[2]]
    mat2_shape = [1, shape[0], shape[2], shape[3]]
    output_shape = [1, shape[0], shape[1], shape[3]]

    # get tensors
    tt_input, tt_mat2, _, _, _, torch_input, torch_mat2, _ = get_tensors(
        input_shape, mat2_shape, output_shape, False, False, False, device)

    # tt bmm
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_out = (
        ttl.operations.primary.moreh_bmm(tt_input, tt_mat2)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # torch bmm
    torch_input = torch_input.reshape(-1, input_shape[2], input_shape[3])
    torch_mat2 = torch_mat2.reshape(-1, mat2_shape[2], mat2_shape[3])
    torch_out = torch.bmm(torch_input, torch_mat2)
    torch_out = torch.unsqueeze(torch_out, dim=0)

    ## test for equivalance
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_out, pcc=0.999)
    logger.info(f"Out passing={passing}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing


@pytest.mark.parametrize(
    "shape",
    (
        [1, 32, 32, 32],
        [3, 31, 31, 31],
        [5, 255, 765, 511],
        [7, 511, 313, 765],
    ),
)
@pytest.mark.parametrize(
    "requires_grad",
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
def test_moreh_bmm_backward(shape, requires_grad, device):
    require_input_grad, require_mat2_grad = requires_grad
    input_shape = [1, shape[0], shape[1], shape[2]]
    mat2_shape = [1, shape[0], shape[2], shape[3]]
    output_shape = [1, shape[0], shape[1], shape[3]]

    # get tensors
    tt_input, tt_mat2, tt_output_grad, tt_input_grad, tt_mat2_grad, torch_input, torch_mat2, torch_output_grad = get_tensors(
        input_shape, mat2_shape, output_shape, require_input_grad,
        require_mat2_grad, False, device)

    # tt bmm fwd, bwd
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    ttl.operations.primary.moreh_bmm_backward(tt_output_grad, tt_input, tt_mat2, tt_input_grad, tt_mat2_grad)

    # torch bmm fwd, bwd
    torch_input = torch_input.reshape(-1, input_shape[2], input_shape[3])
    torch_mat2 = torch_mat2.reshape(-1, mat2_shape[2], mat2_shape[3])
    torch_output_grad = torch_output_grad.reshape(-1, output_shape[2],
                                                  output_shape[3])
    torch_out = torch.bmm(torch_input.requires_grad_(require_input_grad),
                          torch_mat2.requires_grad_(require_mat2_grad))
    torch_out.backward(torch_output_grad)

    # test for equivalance
    rtol = atol = 0.1
    if require_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()

        torch_input_grad = torch.unsqueeze(torch_input.grad, dim=0)
        passing, output_pcc = comp_allclose_and_pcc(torch_input_grad,
                                                    ttcpu_input_grad,
                                                    pcc=0.999,
                                                    rtol=rtol,
                                                    atol=atol)
        logger.info(f"input_grad passing={passing}")
        logger.info(f"input_grad pcc={output_pcc}")
        assert passing

    if require_mat2_grad:
        ttcpu_mat2_grad = tt_mat2_grad.cpu().to(cpu_layout).unpad_from_tile(mat2_shape).to_torch()

        torch_mat2_grad = torch.unsqueeze(torch_mat2.grad, dim=0)
        passing, output_pcc = comp_allclose_and_pcc(torch_mat2_grad,
                                                    ttcpu_mat2_grad,
                                                    pcc=0.999,
                                                    rtol=rtol,
                                                    atol=atol)
        logger.info(f"mat2_grad passing={passing}")
        logger.info(f"mat2_grad pcc={output_pcc}")
        assert passing
