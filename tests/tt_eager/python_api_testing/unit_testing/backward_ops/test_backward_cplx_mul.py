# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import compare_results


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_cplx_sub(input_shapes, device):
    torch.manual_seed(0)
    in_data = torch.randn(input_shapes, dtype=torch.complex64)
    in_data.requires_grad = True

    other_data = torch.randn(input_shapes, dtype=torch.complex64)
    other_data.requires_grad = True

    torch.manual_seed(42)
    grad_data = torch.randn(input_shapes, dtype=torch.complex64)

    in_data_cplx = torch.cat((in_data.real, in_data.imag), dim=3)
    other_data_cplx = torch.cat((other_data.real, other_data.imag), dim=3)
    input_tensor = (
        tt_lib.tensor.Tensor(in_data_cplx, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    other_tensor = (
        tt_lib.tensor.Tensor(other_data_cplx, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    grad_data_cplx = torch.cat((grad_data.real, grad_data.imag), dim=3)
    grad_tensor = (
        tt_lib.tensor.Tensor(grad_data_cplx, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    tt_output_tensor_on_device = tt_lib.tensor.complex_mul_bw(grad_tensor, input_tensor, other_tensor)

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.mul(in_data, other_data)

    pyt_y.backward(gradient=grad_data)

    grad_self_real = torch.real(in_data.grad)
    grad_self_imag = torch.imag(in_data.grad)
    grad_other_real = torch.real(other_data.grad)
    grad_other_imag = torch.imag(other_data.grad)
    golden_tensor = [
        torch.cat((grad_self_real, grad_self_imag), dim=3),
        torch.cat((grad_other_real, grad_other_imag), dim=3),
    ]

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
