# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections import abc

from typing import List, Dict

import ttnn
import torch


DeviceMesh = ttnn._ttnn.multi_device.DeviceMesh


def get_num_pcie_devices() -> int:
    import tt_lib as ttl

    return ttl.device.GetNumPCIeDevices()


def get_pcie_device_ids() -> List[int]:
    import tt_lib as ttl

    num_pcie_devices = ttl.device.GetNumPCIeDevices()
    return list(range(num_pcie_devices))


def get_device_ids() -> List[int]:
    return get_pcie_device_ids()  # change this to full-device range when tt-metal supports it


def open_device_mesh(device_grid: ttnn.DeviceGrid, device_ids: List[int]):
    """
    open_device_mesh(device_grid: ttnn.DeviceGrid, device_ids: int) -> ttnn.DeviceMesh:

    Open a device with the given device_id. If the device is already open, return the existing device.
    """
    assert len(device_ids) > 0
    return ttnn._ttnn.multi_device.open_device_mesh(device_grid=device_grid.as_tuple(), device_ids=device_ids)


def close_device_mesh(device_mesh):
    """
    close_device(multi_device: ttnn.Multi) -> None:

    Close the device and remove it from the device cache.
    """
    return ttnn._ttnn.multi_device.close_device_mesh(device_mesh)


@contextlib.contextmanager
def create_device_mesh(device_grid: ttnn.DeviceGrid, device_ids: List[int]):
    """
    create_device_mesh(device_grid: ttnn.DeviceGrid, device_ids: List[int]) -> ttnn.DeviceMesh

    Context manager for opening and closing a device.
    """
    device_mesh = open_device_mesh(device_grid=device_grid, device_ids=device_ids)
    try:
        yield device_mesh
    finally:
        close_device_mesh(device_mesh)


class MeshMapper(abc.ABC):
    def __init__(self, device_mesh):
        self.device_mesh = device_mesh
        self.device_id_to_tensor = {}

    @abc.abstractmethod
    def map(self, tensor: torch.tensor):
        raise NotImplementedError("Subclasses must implement this method")


class MeshComposer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def compose(self, device_id_to_shard):
        raise NotImplementedError("Subclasses must implement this method")


class ShardedMeshMapper(MeshMapper):
    def __init__(self, device_mesh, dim):
        super().__init__(device_mesh)
        self.shard_dim = dim

    def map(self, tensor: torch.tensor) -> Dict[int, ttnn.Tensor]:
        sliced_tensors = torch.chunk(tensor, self.device_mesh.get_num_devices(), dim=self.shard_dim)
        self.device_id_to_tensor = {
            i: ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for i, input_tensor in enumerate(sliced_tensors)
        }
        return self.device_id_to_tensor


class ReplicateMeshMapper(MeshMapper):
    def __init__(self, device_mesh):
        super().__init__(device_mesh)

    def map(self, tensor: torch.tensor):
        raise NotImplementedError


class ConcatMeshComposer(MeshComposer):
    def __init__(self):
        pass

    def compose(self, device_id_to_tensor) -> torch.Tensor:
        x = [ttnn.to_torch(tt_input_tensor) for tt_input_tensor in device_id_to_tensor.values()]
        return torch.cat(x, dim=self.config.shard_dim)


__all__ = []
