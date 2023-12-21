# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import *


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
# @pytest.mark.parametrize("threshold",[0.0])
def test_bw_relu(input_shapes, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)

    pyt_y = torch.relu(in_data)

    tt_output_tensor_on_device = tt_lib.tensor.relu_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = list()
    golden_tensor.append(in_data.grad)

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
