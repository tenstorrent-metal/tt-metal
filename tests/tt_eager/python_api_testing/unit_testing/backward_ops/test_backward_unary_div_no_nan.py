# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import compare_pcc, data_gen_with_range


def torch_div_no_nan(input, scalar):
    return torch.where(torch.tensor(scalar) == 0, torch.zeros_like(input), torch.div(input, scalar))


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", [0.05, 0.0, -0.5, 5.12])
def test_bw_unary_div_no_nan(input_shapes, scalar, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 199, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -200, 201, device, required_grad=True)

    tt_output_tensor_on_device = tt_lib.tensor.unary_div_no_nan_bw(grad_tensor, input_tensor, scalar=scalar)

    in_data.retain_grad()

    pyt_y = torch_div_no_nan(in_data, scalar)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]
    golden_tensor[0] = torch.where(torch.isnan(golden_tensor[0]), torch.zeros_like(in_data), golden_tensor[0])

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
