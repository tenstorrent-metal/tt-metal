# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_reshape(h, w):
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(1, 1, w, h)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    output_tensor = ttnn.reshape(input_tensor, (1, 1, w, h))
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_reshape_negative_1(h, w):
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(-1)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    output_tensor = ttnn.reshape(input_tensor, (-1,))  # TODO: allow passing in -1
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [32, 32])
@pytest.mark.parametrize("c", [2 * 32, 2 * 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("w", [1, 4])
def test_reshape_in_4D(n, c, h, w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(h, w, n, c)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    output_tensor = ttnn.reshape(input_tensor, (h, w, n, c))
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [32, 32])
@pytest.mark.parametrize("c", [2 * 32, 2 * 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("w", [1, 4])
def test_reshape_in_4D_on_device(n, c, h, w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(h, w, n, c)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    output_tensor = ttnn.reshape(input_tensor, (h, w, n, c))
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.skip(reason="Issue #4007")
def test_permute_reshape(device):
    input_shape = (1, 4, 64, 32)
    output_shape = (1, 64, 128)

    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.permute(torch_input_tensor, (0, 2, 1, 3))
    torch_output_tensor = torch.reshape(torch_output_tensor, output_shape)

    tt_input = ttnn.from_torch(torch_input_tensor)
    tt_input = ttnn.to_device(tt_input, device)
    output_tensor = ttnn.permute(tt_input, (0, 2, 1, 3))
    output_tensor = ttnn.reshape(tt_input, output_shape)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)
