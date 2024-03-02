# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
from enum import Enum

import tt_lib as ttl
import ttnn

DataType = ttl.tensor.DataType
uint16 = DataType.UINT16
uint32 = DataType.UINT32
float32 = DataType.FLOAT32
bfloat16 = DataType.BFLOAT16
bfloat8_b = DataType.BFLOAT8_B

BufferType = ttl.tensor.BufferType
TensorMemoryLayout = ttl.tensor.TensorMemoryLayout
# TODO: MemoryConfig = ttnn._ttnn.types.MemoryConfig
MemoryConfig = ttl.tensor.MemoryConfig
MathFidelity = ttl.tensor.MathFidelity
DRAM_MEMORY_CONFIG = ttnn._ttnn.types.DRAM_MEMORY_CONFIG
L1_MEMORY_CONFIG = ttnn._ttnn.types.L1_MEMORY_CONFIG
L1_BLOCK_SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1)
L1_HEIGHT_SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
L1_WIDTH_SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.WIDTH_SHARDED, BufferType.L1)

Layout = ttl.tensor.Layout
ROW_MAJOR_LAYOUT = Layout.ROW_MAJOR
TILE_LAYOUT = Layout.TILE

StorageType = ttl.tensor.StorageType
DEVICE_STORAGE_TYPE = StorageType.DEVICE

TILE_SIZE = 32

Shape = ttnn._ttnn.types.Shape

Tensor = ttnn._ttnn.types.Tensor


import torch


class MultiDeviceTensor:
    def __init__(self, device_id_to_tensor):
        self.device_id_to_tensor = device_id_to_tensor

    def to_device(self, multi_device):
        for device_id, tensor in self.device_id_to_tensor.items():
            self.device_id_to_tensor[device_id] = ttnn.to_device(tensor, device=multi_device.get_device(device_id))
        return self

    def get_device_tensor(self, device_id):
        return self.device_id_to_tensor[device_id]

    def get_tensors(self):
        return [x for x in self.device_id_to_tensor.values()]


class DeviceMeshTensor:
    def __init__(self, tensor, multidevice, config, device=False):
        self.config = config

        if device == False:
            self.multidevice = multidevice

            sliced_tensors = torch.chunk(tensor, multidevice.get_num_devices(), dim=config.shard_dim)
            self.device_to_tensor = {
                i: ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                for i, input_tensor in enumerate(sliced_tensors)
            }
        else:
            self.multidevice = multidevice
            self.device_to_tensor = {i: tensor[i] for i, t in enumerate(tensor)}

    def to_torch(self):
        x = [ttnn.to_torch(tt_input_tensor) for tt_input_tensor in self.device_to_tensor.values()]
        return torch.cat(x, dim=self.config.shard_dim)

    def to_device(self, multi_device):
        tensors = []
        for i in range(self.multidevice.get_device_ids()):
            device_tensor = ttnn.to_device(self.device_to_tensor[i], device=multi_device.get_device(i))
            self.device_to_tensor[i] = device_tensor
            tensors.append(device_tensor)
        return tensors

    def get_device_tensor(self, device_id):
        return self.device_to_tensor[device_id]

    def get_tensors(self):
        return [x for x in self.device_to_tensor.values()]


@dataclasses.dataclass
class CoreGrid:
    y: int
    x: int

    @property
    def num_cores(self):
        return self.y * self.x


@dataclasses.dataclass
class CoreRange:
    start: CoreGrid
    end: CoreGrid


@dataclasses.dataclass
class DeviceGrid:
    y: int
    x: int

    @property
    def num_devices(self):
        return self.y * self.x

    def as_tuple(self):
        return (self.y, self.x)


@dataclasses.dataclass
class DeviceIds:
    start: DeviceGrid
    end: DeviceGrid


class ShardStrategy(Enum):
    HEIGHT = 1
    WIDTH = 2
    BLOCK = 3


class ShardOrientation(Enum):
    ROW_MAJOR = 1
    COLUMN_MAJOR = 2
