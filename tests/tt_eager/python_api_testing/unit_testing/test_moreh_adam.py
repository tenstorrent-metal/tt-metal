# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim

import tt_lib as ttl
import pytest
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0
from models.utility_functions import comp_pcc
from loguru import logger


@pytest.mark.parametrize(
    "shape",
    (
        (1, 1, 32, 32),   # singl
        (12, 6, 64, 64),   # multi tile
    ),
)
@skip_for_wormhole_b0
def test_moreh_adam(shape, device):
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    x_data = torch.rand((N, C, H, W)).to(torch.bfloat16)
    y_data = torch.rand((N, C, H, W)).to(torch.bfloat16)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.weight = nn.Parameter(torch.randn(N, C, H, W))

        def forward(self, x):
            return torch.mul(x, self.weight)

    model = SimpleModel()
    print("x_data:", x_data)
    cpu_exp_avg = torch.zeros_like(model.weight)
    cpu_exp_avg_sq = torch.zeros_like(model.weight)
    cpu_max_exp_avg_sq = torch.zeros_like(model.weight)

    dev_param = ttl.tensor.Tensor(
        model.weight.reshape(-1).tolist(),
        model.weight.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(device)

    print("weigth:", model.weight)
    print("dev_param:", dev_param)

    criterion = nn.L1Loss()

    optimizer = optim.Adam({model.weight}, lr=0.01, betas=(0.8, 0.888), eps=1e-06, weight_decay=0.1, amsgrad=True)

    optimizer.zero_grad()

    optimizer_state_dict = optimizer.state_dict()

    outputs = model(x_data)

    loss = criterion(outputs, y_data)

    loss.backward()

    print("grad = ", model.weight.grad)
    cpu_grad = model.weight.grad.clone()

    print("before optimizer_state_dict:", optimizer_state_dict)

    print("cpu_grad:", cpu_grad)

    dev_grad = ttl.tensor.Tensor(
        cpu_grad.reshape(-1).tolist(),
        model.weight.grad.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(device)

    dev_exp_avg = ttl.tensor.Tensor(
        cpu_exp_avg.reshape(-1).tolist(),
        cpu_exp_avg.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(device)

    dev_exp_avg_sq = ttl.tensor.Tensor(
        cpu_exp_avg_sq.reshape(-1).tolist(),
        cpu_exp_avg_sq.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(device)

    dev_max_exp_avg_sq = ttl.tensor.Tensor(
        cpu_max_exp_avg_sq.reshape(-1).tolist(),
        cpu_max_exp_avg_sq.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(device)

    optimizer.step()

    optimizer_state_dict = optimizer.state_dict()


    print("")
    print("")
    print("after optimizer_state_dict:", optimizer_state_dict)
    print("Updated weigth:", model.weight)
    print("")
    print("", optimizer_state_dict['state'][0]['max_exp_avg_sq'])

    cpu_max_exp_avg_sq_result = optimizer_state_dict['state'][0]['max_exp_avg_sq']
    dev_max_exp_avg_sq_result = ttl.tensor.Tensor(
        cpu_max_exp_avg_sq_result.reshape(-1).tolist(),
        cpu_max_exp_avg_sq_result.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(device)
    print("dev_max_exp_avg_sq_result:", dev_max_exp_avg_sq_result)

    ret_list_ = ttl.operations.primary.moreh_adam(dev_param, dev_grad, dev_exp_avg, dev_exp_avg_sq
        , 0.01, 0.8, 0.888, 1e-06, 0.1, 1, True, dev_max_exp_avg_sq)

    assert dev_param.shape() == list(model.weight.shape)

    param_result = dev_param.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    max_exp_avg_sq_result = dev_max_exp_avg_sq.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    print("param_result:", param_result.shape)
    print("param_result:", param_result)

    print("max_exp_avg_sq_result:", max_exp_avg_sq_result.shape)
    print("max_exp_avg_sq_result:", max_exp_avg_sq_result)

    passing, out = comp_pcc(model.weight, param_result)
    logger.info(f"Out passing={passing}")
    logger.info(f"Output pcc={out}")
    assert passing
