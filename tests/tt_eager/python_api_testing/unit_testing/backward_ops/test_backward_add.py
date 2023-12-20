# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import *
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from loguru import logger


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_add(input_shapes, device):
    in_data, input_tensor = bw_data_gen(input_shapes, device, True)
    other_data, other_tensor = bw_data_gen(input_shapes, device, True)
    grad_data, grad_tensor = bw_data_gen(input_shapes, device)

    tt_output_tensor_on_device = tt_lib.tensor.add_bw(grad_tensor, input_tensor, other_tensor)

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.add(in_data, other_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = list()
    golden_tensor.append(in_data.grad)
    golden_tensor.append(other_data.grad)

    status = compare_results(tt_output_tensor_on_device, golden_tensor, comparison_funcs.comp_pcc)
    assert status
