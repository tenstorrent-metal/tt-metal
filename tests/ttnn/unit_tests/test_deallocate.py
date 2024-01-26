# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_deallocate(device, h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor)

    with pytest.raises(RuntimeError) as exception:
        ttnn.deallocate(input_tensor)
    assert "ttnn.deallocate: Tensor must be on device!" in str(exception.value)

    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = input_tensor + input_tensor

    # Create input_tensor_a reference to the same storage by using reshape which will create input_tensor_a new flyweight
    # (If reshape operation changes, then this test might need to be updated)
    output_tensor_reference = ttnn.reshape(output_tensor, (h, w))

    ttnn.deallocate(output_tensor)
    with pytest.raises(RuntimeError) as exception:
        print(output_tensor_reference)
    assert "Buffer must be allocated on device!" in str(exception.value)
