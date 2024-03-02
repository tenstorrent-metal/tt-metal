# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib as ttl
import ttnn
from loguru import logger


#######
# Test MultiDevice Initialization, Open/Close
#######
def test_device_mesh_open_close_explicit():
    """Manually open and close multi-device"""
    num_pcie_devices = ttnn.get_num_pcie_devices()
    if num_pcie_devices <= 1:
        pytest.skip("Requires multiple devices to run")

    device_grid, device_ids = ttnn.DeviceGrid(2, 2), ttnn.get_pcie_device_ids()
    multi_device = ttnn.open_device_mesh(device_grid, device_ids)
    ttnn.close_device_mesh(multi_device)


def test_multi_device_open_close_fixture(device_mesh):
    """Using `multi_device` pytest fixture defined in conftest.py"""
    pass


def test_multi_device_open_close_using_context_manager():
    """Using context manager to open and close multi-device"""
    device_grid, device_ids = ttnn.DeviceGrid(2, 2), ttnn.get_device_ids()
    with ttnn.create_device_mesh(device_grid, device_ids) as device_mesh:
        # Do something with multi_device
        pass


#######
# Simple Multi-Device Tensor tests
#######


def test_ttnn_to_and_from_multi_device_shard():
    """MultiDevice APIs: APIs to map tensors onto device-mesh and loopback"""
    from ttnn import ShardedMeshMapper, ConcatMeshComposer

    torch_tensor = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)

    with ttnn.create_device_mesh(ttnn.DeviceGrid(1, 4), ttnn.get_device_ids()) as device_mesh:
        ttnn_tensor = ttnn.from_torch(torch_tensor, mesh_mapper=ShardedMeshMapper(device_mesh, dim=3))
        ttnn_tensor = ttnn.to_device(ttnn_tensor)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
        torch_loop_back_tensor = ttnn.to_torch(ttnn_loop_back_tensor, mesh_composer=ConcatMeshComposer(dim=3))

        assert torch.all(torch_tensor == torch_loop_back_tensor)


def test_ttnn_to_and_from_multi_device_replicate():
    """MultiDevice APIs: APIs to map tensors onto device-mesh and loopback"""
    from ttnn import ShardedMeshMapper, ConcatMeshComposer

    torch_tensor = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)

    with ttnn.create_device_mesh(ttnn.DeviceGrid(1, 4), ttnn.get_device_ids()) as device_mesh:
        ttnn_tensor = ttnn.from_torch(torch_tensor, mesh_mapper=ShardedMeshMapper(device_mesh, dim=3))
        ttnn_tensor = ttnn.to_device(ttnn_tensor)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
        torch_loop_back_tensor = ttnn.to_torch(ttnn_loop_back_tensor, mesh_composer=ConcatMeshComposer(dim=3))

        assert torch.all(torch_tensor == torch_loop_back_tensor)


def test_ttnn_to_and_from_multi_device_all_gather():
    """MultiDevice APIs: APIs to map tensors onto device-mesh and loopback"""
    from ttnn import ShardedMeshMapper, ConcatMeshComposer

    torch_tensor = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)

    with ttnn.create_device_mesh(ttnn.DeviceGrid(1, 4), ttnn.get_device_ids()) as device_mesh:
        ttnn_tensor = ttnn.from_torch(torch_tensor, mesh_mapper=ShardedMeshMapper(device_mesh, dim=3))
        ttnn_tensor = ttnn.to_device(ttnn_tensor)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
        torch_loop_back_tensor = ttnn.to_torch(ttnn_loop_back_tensor, mesh_composer=ConcatMeshComposer(dim=3))

        assert torch.all(torch_tensor == torch_loop_back_tensor)
