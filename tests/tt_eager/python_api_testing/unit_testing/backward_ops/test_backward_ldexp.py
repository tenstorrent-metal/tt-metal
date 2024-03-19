# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_pt_tt, compare_results


def data_gen_pt_tt(input_shapes, device, required_grad=False, val=1):
    pt_tensor = (torch.ones(input_shapes, requires_grad=required_grad) * val).bfloat16()
    tt_tensor = (
        tt_lib.tensor.Tensor(pt_tensor, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    return pt_tensor, tt_tensor


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_bw_ldexp(input_shapes, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True, val=1)
    other_data, other_tensor = data_gen_pt_tt(input_shapes, device, True, val=0)

    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device, False, 1)

    print("input_tensor", input_tensor)
    print("other_tensor", other_tensor)
    print("grad_tensor", grad_tensor)

    tt_output_tensor_on_device = tt_lib.tensor.ldexp_bw(grad_tensor, input_tensor, other_tensor)

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.ldexp(in_data, other_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, other_data.grad]
    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)

    print("tt_output_tensor_on_device", tt_output_tensor_on_device)
    print("golden_tensor", golden_tensor)
    assert comp_pass
