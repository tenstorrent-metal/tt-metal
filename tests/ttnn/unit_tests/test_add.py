import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [2 * 32])
def test_add_1D_tensor_and_scalar(device, scalar, size):
    torch_input_tensor = torch.rand((size, ), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + scalar

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.copy_to_device(input_tensor, device)
    output_tensor = input_tensor + scalar
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
    assert output_tensor.shape == (size, )


@pytest.mark.parametrize("s", [3])
@pytest.mark.parametrize("h", [2 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_add_scalar(device, s, h, w):
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + s

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.copy_to_device(input_tensor, device)
    output_tensor = input_tensor + s
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)


@pytest.mark.parametrize("alpha", [0.42])
@pytest.mark.parametrize("scalar_input_tensor_b", [0.5])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("w", [4])
def test_add_scalar_and_alpha(device, alpha, scalar_input_tensor_b, h, w):
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor, scalar_input_tensor_b, alpha=alpha)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.copy_to_device(input_tensor, device)
    output_tensor = ttnn.add(input_tensor, scalar_input_tensor_b, alpha=alpha)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.99999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_add(device, h, w):
    torch_a = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_b = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output = torch.add(torch_a, torch_b)

    a = ttnn.from_torch(torch_a)
    a = ttnn.copy_to_device(a, device)
    b = ttnn.from_torch(torch_b)
    b = ttnn.copy_to_device(b, device)
    tt_output = ttnn.add(a, b)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize("n", [32])
@pytest.mark.parametrize("c", [2 * 32])
@pytest.mark.parametrize("h", [4 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_add_4D(device, n, c, h, w):
    torch_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output = torch.add(torch_a, torch_b)

    a = ttnn.from_torch(torch_a)
    a = ttnn.copy_to_device(a, device)
    b = ttnn.from_torch(torch_b)
    b = ttnn.copy_to_device(b, device)
    tt_output = ttnn.add(a, b)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)
