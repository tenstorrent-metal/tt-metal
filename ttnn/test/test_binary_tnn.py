# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    (
        (torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384]), torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384]), torch.Size([1, 3, 320, 384])),
    ),
)
class TestEltwiseBinary:
    def test_ttnn_squared_difference(self, input_shape_a, input_shape_b):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.squared_difference(a, b)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

        ttnn.close(device)

    @pytest.mark.parametrize("alpha", [1.0, 2.5, -2.5])
    def test_ttnn_subalpha(self, input_shape_a, input_shape_b, alpha):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape_b, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.subalpha(a, b, alpha)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

        ttnn.close(device)
