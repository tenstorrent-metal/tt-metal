# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import pathlib

import torch

import ttnn


@pytest.mark.parametrize("h", [1024])
@pytest.mark.parametrize("w", [1024])
def test_dump_and_load(tmp_path, h, w):
    file_name = tmp_path / pathlib.Path("tensor.bin")

    torch_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_tensor)
    ttnn.dump_tensor(file_name, tensor)

    loaded_tensor = ttnn.load_tensor(file_name)
    loaded_torch_tensor = ttnn.to_torch(loaded_tensor)
    assert torch.allclose(torch_tensor, loaded_torch_tensor)


@pytest.mark.parametrize("h", [1024])
@pytest.mark.parametrize("w", [1024])
def test_dump_and_load_tilized(tmp_path, h, w):
    file_name = tmp_path / pathlib.Path("tensor.bin")

    torch_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_tensor)
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    ttnn.dump_tensor(file_name, tensor)

    loaded_tensor = ttnn.load_tensor(file_name)
    loaded_tensor = ttnn.to_layout(loaded_tensor, ttnn.ROW_MAJOR_LAYOUT)
    loaded_torch_tensor = ttnn.to_torch(loaded_tensor)
    assert torch.allclose(torch_tensor, loaded_torch_tensor)
