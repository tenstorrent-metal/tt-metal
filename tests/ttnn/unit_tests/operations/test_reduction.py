# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [-1, -2])
def test_std(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.std(torch_input_tensor, dim=dim, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.std(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [-1, -2])
def test_var(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.var(torch_input_tensor, dim=dim, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.var(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [-1, -2])
def test_max(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.max(torch_input_tensor, dim=dim, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.max(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
def test_max_global(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.max(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.max(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor[0, 0, 0, 0]

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [-1, -2])
def test_min(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.min(torch_input_tensor, dim=dim, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.min(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
def test_min_global(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.min(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.min(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor[0, 0, 0, 0]

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [-1, -2, (2, 1)])
def test_sum(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
def test_sum_global(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor[0, 0, 0, 0]

    assert_with_pcc(torch_output_tensor, output_tensor)
