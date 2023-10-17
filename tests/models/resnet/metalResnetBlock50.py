# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Type, Union, Optional, List, Callable
import time
import tt_lib
import torch
import torch.nn as nn
import math
from tests.models.resnet.utils import conv3x3, conv1x1, fold_bn_to_conv, fold_bn_to_conv_weights_bias
from models.utility_functions import pad_by_zero, tt2torch_tensor
from tt_lib.utils import pad_weight

from tt_lib.fused_ops.average_pool import run_avg_pool_on_device_wrapper as TtAvgPool
from tt_lib.fused_ops.max_pool import run_max_pool_on_device_wrapper as TtMaxPool
from tt_lib.fused_ops.max_pool import compute_max_pool_shape
from tt_lib.fused_ops.softmax import softmax as TtSoftmax
from tt_lib.fused_ops.conv import resnet50_first_conv, resnet50_1x1_conv_as_matmul, resnet50_optimized_conv, resnet50_1x1_conv_s2_as_downsample_and_matmul
from models.utility_functions import _nearest_32, profiler
from tt_lib.fallback_ops import fallback_ops


def ResnetLinear(
    in_features: int,
    out_features: int,
    weight: tt_lib.tensor.Tensor,
    bias: Optional[tt_lib.tensor.Tensor] = None,
    transpose: bool = True,
    output_mem_config=tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
    ),
    device=None,
):
    """
    Returns a function for linear operation in resnet with bias.
    """
    if bias is not None:
        assert bias.shape()[-1] == out_features, "bias shape is not as expected"
        if device is not None:
            bias = bias.to(device)

    if transpose:
        assert weight.shape() == [1, 1, out_features, in_features], "weight does not have the expected shape"
        weight_T = tt_lib.tensor.transpose(weight)
    else:
        assert weight.shape() == [1, 1, in_features, out_features], "weight does not have the expected shape"
        weight_T = weight
    if device is not None:
        weight_T = weight_T.to(device)

    def linear_(act):
        ## this uses the systolic 1d matmul with bias fused
        output = tt_lib.tensor.resnet_matmul(act, weight_T, bias, output_mem_config)
        return output

    return linear_


def do_nothing_op(x):
    return x


def _nearest_y(x, y):
    return math.ceil(x / y) * y


def format_tensor(x, target_layout, device, output_mem_config, pad_value=0.0):
    if x.layout() == target_layout:
        return x
    if x.layout() == tt_lib.tensor.Layout.ROW_MAJOR and target_layout == tt_lib.tensor.Layout.TILE:
        x_padded_shape = tt_lib.tensor.pad_to_tile_shape(x.shape(), False, False, True, True)
        if x.shape() != x_padded_shape:
            return tt_lib.tensor.format_input_tensor(
                x, device, x_padded_shape, pad_value, target_layout, output_mem_config
            )
        else:
            return tt_lib.tensor.tilize(x, output_mem_config, use_multicore=True)
    elif x.layout() == tt_lib.tensor.Layout.TILE and target_layout == tt_lib.tensor.Layout.ROW_MAJOR:
        if x.shape() != x.shape_without_padding():
            return tt_lib.tensor.format_output_tensor(
                x, x.shape_without_padding(), device, target_layout, output_mem_config
            )
        else:
            return tt_lib.tensor.untilize(x, output_mem_config, use_multicore=True)
    else:
        assert False


# Local copy of unpad_from_zero to always set output to
def unpad_from_zero(x, desired_shape):
    if x.shape()[-1] == desired_shape[-1] and x.shape()[-2] == desired_shape[-2]:
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if x.layout() != tt_lib.tensor.Layout.ROW_MAJOR:
            x = x.to(tt_lib.tensor.Layout.ROW_MAJOR)
        x = x.unpad(
            (0, 0, 0, 0), (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1)
        )
        x = x.to_torch().to(torch.float)
    return x


def compute_conv_output_shape(conv_params, x_shape):
    H = x_shape[1]
    W = x_shape[2]
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    OH = ((int)((H - R + 2 * P_H) / U)) + 1
    OW = ((int)((W - S + 2 * P_W) / V)) + 1
    return [x_shape[0], OH, OW, K]


# hardcoding matmul config for 1x1 convs
# key: mm act height, mm act width, mm weight width
hardcoded_matmul_config_conv = {
    1: {
        (3136, 64, 64): {
            "compute_with_storage_grid_size": (2, 2),
            "in0_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 49,
            "per_core_N": 1,
        },
        (3136, 64, 256): {
            "compute_with_storage_grid_size": (4, 2),
            "in0_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 49,
            "per_core_N": 2,
        },
        (3136, 256, 64): {
            "compute_with_storage_grid_size": (2, 7),
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 14,
            "per_core_N": 1,
        },
        (3136, 256, 128): {
            "compute_with_storage_grid_size": (4, 7),
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 14,
            "per_core_N": 1,
        },
        (800, 128, 512): {
            "compute_with_storage_grid_size": (4, 2),
            "in0_block_w": 4,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 13,
            "per_core_N": 4,
        },
        (800, 512, 128): {
            "compute_with_storage_grid_size": (4, 4),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 7,
            "per_core_N": 1,
        },
        (800, 512, 256): {
            "compute_with_storage_grid_size": (8, 4),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 7,
            "per_core_N": 1,
        },
        (224, 256, 1024): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 4,
        },
        (224, 1024, 256): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 32,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 1,
        },
        (224, 1024, 512): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 32,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 2,
        },
        (64, 512, 2048): {
            "compute_with_storage_grid_size": (8, 2),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 8,
        },
        (64, 2048, 512): {
            "compute_with_storage_grid_size": (8, 2),
            "in0_block_w": 64,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 2,
        },
    },
    2: {
        (6272, 64, 64): {
            "compute_with_storage_grid_size": (2, 4),
            "in0_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 49,
            "per_core_N": 1,
        },
        (6272, 64, 256): {
            "compute_with_storage_grid_size": (4, 4),
            "in0_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 49,
            "per_core_N": 2,
        },
        (6272, 256, 64): {
            "compute_with_storage_grid_size": (2, 9),  # (x,y)
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 22,  # across y
            "per_core_N": 1,  # across x
        },
        (6272, 256, 128): {
            "compute_with_storage_grid_size": (4, 9),
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 22,
            "per_core_N": 1,
        },
        (1568, 128, 512): {
            "compute_with_storage_grid_size": (4, 4),
            "in0_block_w": 4,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 13,
            "per_core_N": 4,
        },
        (1568, 512, 128): {
            "compute_with_storage_grid_size": (4, 9),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 6,
            "per_core_N": 1,
        },
        (1568, 512, 256): {
            "compute_with_storage_grid_size": (8, 9),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 6,
            "per_core_N": 1,
        },
        (416, 256, 1024): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 2,
            "per_core_N": 4,
        },
        (416, 1024, 256): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 32,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 2,
            "per_core_N": 1,
        },
        (416, 1024, 512): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 32,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 2,
            "per_core_N": 2,
        },
        (128, 512, 2048): {
            "compute_with_storage_grid_size": (8, 4),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 8,
        },
        (128, 2048, 512): {
            "compute_with_storage_grid_size": (8, 4),
            "in0_block_w": 64,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 2,
        },
    },
    8: {
        (25088, 64, 64): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=2,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=8,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (25088, 64, 256): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=8,
            per_core_N=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (25088, 256, 64): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=8,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (25088, 256, 128): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=8,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (6272, 128, 512): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=2,
            per_core_N=16,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (6272, 256, 512): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=2,
            per_core_N=16,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (6272, 512, 128): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=16,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=2,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (6272, 512, 256): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=2,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=20,
            per_core_N=1,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 256, 1024): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=5,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 256): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=4,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=5,
            per_core_N=1,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 512): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=5,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 512, 1024): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=5,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 512): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=7,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (416, 512, 2048): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=2,
            per_core_N=8,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (416, 2048, 512): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=8,
            out_subblock_h=2,
            out_subblock_w=2,
            per_core_M=2,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
    },
}

hardcoded_conv_blocking_and_parallelization_config = {
    1: {
        (3136, 64): [64 * 3, 64, 64, 64, 64, 64, (7, 7), 64, 64],
        (800, 128): [128, 32, 128, 32, 64, 32, (5, 5), 32, 128],
        (224, 256): [256, 32, 128, 32, 128, 32, (1, 7), 32, 256],
        (64, 512): [512, 32, 64, 32, 64, 32, (1, 2), 32, 512],
        # bypass convs
        (3136, 256): [128, 64, 64, 64, 64, 64, (7, 7), 64, 256],
        (800, 512): [256, 32, 64, 32, 64, 32, (5, 5), 32, 512],
        (224, 1024): [512, 32, 128, 32, 64, 32, (1, 7), 32, 1024],
        (64, 2048): [1024, 32, 128, 32, 64, 32, (1, 2), 32, 2048],
    },
    2: {
        (6272, 64): [64 * 3, 128, 64, 128, 64, 128, (7, 7), 128, 64],
        (1568, 128): [128, 32, 128, 32, 64, 32, (7, 7), 32, 128],
        (416, 256): [256, 64, 128, 64, 128, 64, (7, 1), 64, 256],
        (128, 512): [512, 32, 64, 32, 64, 32, (1, 4), 32, 512],
        # bypass convs
        (6272, 256): [128, 128, 64, 128, 64, 128, (7, 7), 128, 256],
        (1568, 512): [256, 32, 64, 32, 64, 32, (7, 7), 32, 512],
        (416, 1024): [512, 64, 128, 64, 64, 64, (7, 1), 64, 1024],
        (128, 2048): [1024, 64, 128, 64, 64, 64, (1, 2), 64, 2048],
    },
    8: {
        (25088, 64): [64 * 3, 256, 64, 128, 64, 256, (12, 9), 256, 64],
        (6272, 128): [128, 64, 128, 64, 128, 64, (12, 9), 64, 128],
        (1568, 256): [256, 160, 32, 32, 32, 160, (10, 8), 160, 32],
        (416, 512): [512, 64, 64, 32, 32, 64, (7, 8), 64, 64],
        # bypass convs
        (6272, 512): [256, 64, 512, 32, 256, 64, (12, 9), 64, 512],
        (1568, 1024): [512, 160, 128, 32, 64, 160, (10, 8), 160, 128],
        (416, 2048): [512, 64, 256, 32, 32, 64, (7, 8), 64, 256],
    },
}


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device=None,
        state_dict=None,
        base_address=None,
        fold_batchnorm=False,
        downsample_conv_on_tt=None,
        norm_layer_after_downsample_conv_on_tt=None,
        downsample_params=[],
        storage_in_dram=True,
        input_shape=[],
        batch_size=1,
        sharded=None,
        out_sharded=False,
        act_block_w_equals_input_channels_x_filter_width=False,
        use_downsample_op_and_mm_for_conv1x1_s2=False
    ) -> None:
        super().__init__()
        self.device = device
        self.state_dict = state_dict
        self.base_address = base_address
        self.fold_batchnorm = fold_batchnorm
        self.downsample_conv_on_tt = downsample_conv_on_tt
        self.norm_layer_after_downsample_conv_on_tt = norm_layer_after_downsample_conv_on_tt
        self.downsample_params = downsample_params
        self.storage_in_dram = storage_in_dram
        if self.storage_in_dram:
            self.memory_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            )
        else:
            self.memory_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
            )
        if sharded is not None:
            self.sharded_memory_config = tt_lib.tensor.MemoryConfig(sharded, tt_lib.tensor.BufferType.L1)
        else:
            self.sharded_memory_config = self.memory_config
        self.out_memory_config = self.sharded_memory_config if out_sharded else self.memory_config
        if use_downsample_op_and_mm_for_conv1x1_s2:
            assert sharded

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        conv1_weight = state_dict[f"{base_address}.conv1.weight"]
        conv1_bias = None

        self.bn1 = norm_layer(width)
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address}.bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address}.bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address}.bn1.num_batches_tracked"], requires_grad=False
        )
        self.bn1.eval()

        conv2_weight = state_dict[f"{base_address}.conv2.weight"]
        conv2_bias = None

        self.bn2 = norm_layer(width)
        self.bn2.weight = nn.Parameter(state_dict[f"{self.base_address}.bn2.weight"])
        self.bn2.bias = nn.Parameter(state_dict[f"{self.base_address}.bn2.bias"])
        self.bn2.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_mean"])
        self.bn2.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_var"])
        self.bn2.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address}.bn2.num_batches_tracked"], requires_grad=False
        )
        self.bn2.eval()

        conv3_weight = state_dict[f"{base_address}.conv3.weight"]
        conv3_bias = None

        self.bn3 = norm_layer(planes * self.expansion)
        self.bn3.weight = nn.Parameter(state_dict[f"{self.base_address}.bn3.weight"])
        self.bn3.bias = nn.Parameter(state_dict[f"{self.base_address}.bn3.bias"])
        self.bn3.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn3.running_mean"])
        self.bn3.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn3.running_var"])
        self.bn3.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address}.bn3.num_batches_tracked"], requires_grad=False
        )
        self.bn3.eval()

        self.relu = tt_lib.tensor.relu_without_autoformat
        self.downsample = downsample
        self.stride = stride

        if self.fold_batchnorm:
            conv1_weight, conv1_bias = fold_bn_to_conv_weights_bias(conv1_weight, self.bn1)
            conv2_weight, conv2_bias = fold_bn_to_conv_weights_bias(conv2_weight, self.bn2)
            conv3_weight, conv3_bias = fold_bn_to_conv_weights_bias(conv3_weight, self.bn3)
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        self.module_input_shape = input_shape
        self.conv1_params = [width, inplanes, 1, 1, 1, 1, 0, 0, dilation, groups]
        self.conv1_output_shape = compute_conv_output_shape(self.conv1_params, input_shape)
        conv1_as_mm_padded_act_height = _nearest_32(
            self.conv1_output_shape[0] * self.conv1_output_shape[1] * self.conv1_output_shape[2]
        )
        assert (conv1_as_mm_padded_act_height, inplanes, width) in hardcoded_matmul_config_conv[batch_size]
        matmul_config = hardcoded_matmul_config_conv[batch_size][(conv1_as_mm_padded_act_height, inplanes, width)]
        # 1x1 conv with stride 1 padding 0 is run using regular matmul
        self.conv1 = resnet50_1x1_conv_as_matmul(
            conv1_weight.reshape(-1).tolist(),
            self.conv1_params,
            self.device,
            conv1_bias.tolist(),
            matmul_config,
            fuse_relu=True,
            output_mem_config=self.sharded_memory_config,
        )

        self.conv2_params = [width, width, 3, 3, stride, stride, 1, 1, dilation, groups]
        self.conv2_output_shape = compute_conv_output_shape(self.conv2_params, self.conv1_output_shape)
        conv2_output_padded_face_size = _nearest_32(
            self.conv2_output_shape[0] * self.conv2_output_shape[1] * self.conv2_output_shape[2]
        )
        assert (conv2_output_padded_face_size, width) in hardcoded_conv_blocking_and_parallelization_config[batch_size]
        [
            act_block_w_datums,
            act_block_h_datums,
            weight_block_w_datums,
            out_subblock_h_datums,
            out_subblock_w_datums,
            out_block_h_datums,
            grid_size,
            per_core_act_h,
            per_core_weight_w,
        ] = hardcoded_conv_blocking_and_parallelization_config[batch_size][(conv2_output_padded_face_size, width)]
        assert per_core_act_h % 32 == 0
        per_core_act_h_ntiles = (int)(per_core_act_h / 32)
        per_core_weight_w_ntiles = (int)(per_core_weight_w / 32)
        if act_block_w_equals_input_channels_x_filter_width:
            assert act_block_w_datums == width * 3
        else:
            assert act_block_w_datums == width
        self.conv2 = resnet50_optimized_conv(
            conv2_weight.reshape(-1).tolist(),
            self.conv2_params,
            self.device,
            [act_block_h_datums, act_block_w_datums],
            [act_block_w_datums, weight_block_w_datums],
            [out_subblock_h_datums, out_subblock_w_datums],
            out_block_h_datums,
            grid_size,
            per_core_act_h_ntiles,
            per_core_weight_w_ntiles,
            conv2_bias.tolist(),
            True,
            output_mem_config=self.sharded_memory_config,
        )

        self.conv3_params = [planes * self.expansion, width, 1, 1, 1, 1, 0, 0, dilation, groups]
        self.conv3_output_shape = compute_conv_output_shape(self.conv3_params, self.conv2_output_shape)
        conv3_as_mm_padded_act_height = _nearest_32(
            self.conv3_output_shape[0] * self.conv3_output_shape[1] * self.conv3_output_shape[2]
        )
        matmul_config = None
        assert (conv3_as_mm_padded_act_height, width, planes * self.expansion) in hardcoded_matmul_config_conv[
            batch_size
        ]
        # print("Setting matmul config for 1x1 conv (third conv in module)")
        matmul_config = hardcoded_matmul_config_conv[batch_size][
            (conv3_as_mm_padded_act_height, width, planes * self.expansion)
        ]
        # 1x1 conv with stride 1 padding 0 is run using regular matmul
        self.conv3 = resnet50_1x1_conv_as_matmul(
            conv3_weight.reshape(-1).tolist(),
            self.conv3_params,
            self.device,
            conv3_bias.tolist(),
            matmul_config,
            output_mem_config=self.sharded_memory_config,
        )
        self.conv3_output_shape = compute_conv_output_shape(self.conv3_params, self.conv2_output_shape)

        self.downsample_or_noop = self.downsample_conv_on_tt
        if self.downsample_or_noop is None:
            self.downsample_or_noop = do_nothing_op
        else:
            if (not use_downsample_op_and_mm_for_conv1x1_s2) and (self.downsample_params[2] != 1 or self.downsample_params[4] != 1 or self.downsample_params[6] != 0):
                # this downsample conv requires row major input
                def downsample_conv_op_wrapper(op):
                    def downsample_conv_op_with_formatting(x):
                        x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
                        x = x.reshape(
                            self.module_input_shape[0],
                            self.module_input_shape[1],
                            self.module_input_shape[2],
                            self.module_input_shape[3],
                        )
                        return op(x)

                    return downsample_conv_op_with_formatting

                self.downsample_or_noop = downsample_conv_op_wrapper(self.downsample_conv_on_tt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("This module input shape - ", self.module_input_shape)
        # conv1 is 1x1 conv
        # print("Running conv1")
        out = self.conv1(x)
        # print("conv1 output shape - ", self.conv1_output_shape)
        # print("Running ds or nop")
        ds_out = self.downsample_or_noop(x)
        # Relu after conv1 is fused with the 1x1 conv (matmul)
        # out = self.relu(out, self.memory_config)
        # print("Running untilize op")
        out = format_tensor(out, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
        out = out.reshape(
            self.conv1_output_shape[0],
            self.conv1_output_shape[1],
            self.conv1_output_shape[2],
            self.conv1_output_shape[3],
        )

        # print("Running conv2")
        out = self.conv2(out)
        # out = self.relu(out, self.memory_config)  ## fused with conv2
        # conv3 is 1x1 conv
        # print("Running conv3")
        out = self.conv3(out)

        fused_activations = [tt_lib.tensor.FusibleActivation.RELU]
        # print("Running eltwise add")
        out = tt_lib.tensor.add_without_autoformat(
            out, ds_out, fused_activations, output_mem_config=self.out_memory_config
        )
        # out = self.relu(out, self.memory_config)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Bottleneck,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device=None,
        state_dict=None,
        base_address=None,
        fold_batchnorm=False,
        storage_in_dram=True,
        conv_input_face_shape_hw=[224, 224],
        batch_size=1,
        sharded=False,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_address_with_dot = base_address  # this is root layer, no dot is needed
        self.state_dict = state_dict
        self.fold_batchnorm = fold_batchnorm
        self.storage_in_dram = storage_in_dram
        self.conv_input_face_shape_hw = conv_input_face_shape_hw
        self.batch_size = batch_size
        self.sharded = sharded
        if self.storage_in_dram:
            self.memory_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            )
        else:
            self.memory_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
            )
        if sharded:
            self.sharded_memory_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1
            )
        else:
            self.sharded_memory_config = self.memory_config
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        conv1_weight = state_dict[f"{self.base_address_with_dot}conv1.weight"]
        conv1_bias = None

        self.bn1 = norm_layer(self.inplanes)  # batch norm
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address_with_dot}bn1.num_batches_tracked"], requires_grad=False
        )
        self.bn1.eval()

        if self.fold_batchnorm:
            conv1_weight, conv1_bias = fold_bn_to_conv_weights_bias(conv1_weight, self.bn1)
            self.bn1 = nn.Identity()

        self.conv1_params = [self.inplanes, 3, 7, 7, 2, 2, 3, 3, 1, groups]
        if batch_size == 1:
            act_block_h_datums = 256
            grid_size = (7, 7)
            per_core_act_h_ntiles = 8
        elif batch_size == 2:
            act_block_h_datums = 256
            grid_size = (7, 7)
            per_core_act_h_ntiles = 16
        elif batch_size == 8:
            # 7,7 multi core config triggers non-deterministic output
            # grid_size = (7,7)
            # per_core_act_h_ntiles = 64
            # act_block_h_datums = 256
            # grid_size = (7,8)
            # per_core_act_h_ntiles = 56
            act_block_h_datums = 1024
            grid_size = (12, 9)
            per_core_act_h_ntiles = 32

        self.conv1 = resnet50_first_conv(
            conv1_weight.reshape(-1).tolist(),
            self.conv1_params,
            self.device,
            [act_block_h_datums, 32],
            [32, 64],
            [32, 64],
            act_block_h_datums,
            grid_size,
            per_core_act_h_ntiles,
            conv1_bias.tolist(),
            8,
            fuse_relu=True,
            out_mem_config=self.sharded_memory_config if sharded else None,
        )
        self.conv1_output_shape = compute_conv_output_shape(
            self.conv1_params,
            [batch_size, self.conv_input_face_shape_hw[0], self.conv_input_face_shape_hw[1], self.inplanes],
        )
        self.relu = tt_lib.tensor.relu_without_autoformat
        # self.maxpool = fallback_ops.MaxPool2d(kernel_size=3, stride=2, padding=1, channels_last=True, reshape_2d=True)
        # self.maxpool = TtMaxPool(self.device, kernel_size=3, stride=2, padding=1, output_mem_config=self.memory_config, nblocks=8, channels_last=True, reshape_2d=True)
        self.maxpool = TtMaxPool(
            self.device,
            kernel_size=3,
            stride=2,
            padding=1,
            output_mem_config=self.sharded_memory_config,
            nblocks=1,
            channels_last=True,
            reshape_2d=True,
        )
        self.maxpool_output_shape = compute_max_pool_shape(3, 2, 1, self.conv1_output_shape)
        self.layer1, self.layer1_output_shape = self._make_layer(
            block,
            64,
            layers[0],
            name="layer1",
            state_dict=state_dict,
            layer_input_shape=self.maxpool_output_shape,
            batch_size=batch_size,
            sharded=tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED if sharded else None,
            out_sharded=True,
            act_block_w_equals_input_channels_x_filter_width=True,
        )
        self.layer2, self.layer2_output_shape = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            name="layer2",
            state_dict=state_dict,
            layer_input_shape=self.layer1_output_shape,
            batch_size=batch_size,
            sharded=tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED if sharded else None,
            out_sharded=False,
            use_downsample_op_and_mm_for_conv1x1_s2=True if sharded else False,
        )
        self.layer3, self.layer3_output_shape = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            name="layer3",
            state_dict=state_dict,
            layer_input_shape=self.layer2_output_shape,
            batch_size=batch_size,
            sharded=tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED if sharded else None,
            out_sharded=False,
            use_downsample_op_and_mm_for_conv1x1_s2=True if sharded else False,
        )
        self.layer4, self.layer4_output_shape = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            name="layer4",
            state_dict=state_dict,
            layer_input_shape=self.layer3_output_shape,
            batch_size=batch_size,
            sharded=tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED if sharded else None,
            out_sharded=True,
        )

        # All modules in RN50 are unrolled here. One variable for each module. Only specific number of modules supported - layers MUST equal to [3, 4, 6, 3]
        assert layers == [3, 4, 6, 3]
        self.layer1_module1 = self.layer1[0]
        self.layer1_module2 = self.layer1[1]
        self.layer1_module3 = self.layer1[2]

        self.layer2_module1 = self.layer2[0]
        self.layer2_module2 = self.layer2[1]
        self.layer2_module3 = self.layer2[2]
        self.layer2_module4 = self.layer2[3]

        self.layer3_module1 = self.layer3[0]
        self.layer3_module2 = self.layer3[1]
        self.layer3_module3 = self.layer3[2]
        self.layer3_module4 = self.layer3[3]
        self.layer3_module5 = self.layer3[4]
        self.layer3_module6 = self.layer3[5]

        self.layer4_module1 = self.layer4[0]
        self.layer4_module2 = self.layer4[1]
        self.layer4_module3 = self.layer4[2]

        self.avgpool = TtAvgPool(self.device)

        fc_weight = pad_weight(state_dict[f"{self.base_address_with_dot}fc.weight"])
        fc_weight = torch.transpose(fc_weight, 3, 2)
        fc_weight = tt_lib.tensor.Tensor(
            fc_weight.reshape(-1).tolist(),
            fc_weight.shape,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        ).to(tt_lib.tensor.Layout.TILE)
        fc_bias = pad_weight(state_dict[f"{self.base_address_with_dot}fc.bias"])
        fc_bias = tt_lib.tensor.Tensor(
            fc_bias.reshape(-1).tolist(), fc_bias.shape, tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.ROW_MAJOR
        ).to(tt_lib.tensor.Layout.TILE)
        self.fc = ResnetLinear(
            512 * block.expansion,
            1024,
            fc_weight,
            fc_bias,
            transpose=False,
            output_mem_config=self.memory_config,
            device=self.device,
        )  # num_classes = 1000
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Bottleneck,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        name: str = None,
        state_dict=None,
        layer_input_shape=[],
        batch_size=1,
        sharded=None,
        out_sharded=False,
        act_block_w_equals_input_channels_x_filter_width=False,
        use_downsample_op_and_mm_for_conv1x1_s2=False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        self.downsample_conv_on_tt = None
        self.norm_layer_after_downsample_conv_on_tt = None
        if sharded is not None:
            self.ds_conv_output_memory_config = tt_lib.tensor.MemoryConfig(sharded, tt_lib.tensor.BufferType.L1)
        else:
            self.ds_conv_output_memory_config = self.memory_config
        if use_downsample_op_and_mm_for_conv1x1_s2:
            assert sharded
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            nl = norm_layer(planes * block.expansion)
            nl.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.weight"])
            nl.bias = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.bias"])
            nl.running_mean = nn.Parameter(
                state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_mean"]
            )
            nl.running_var = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_var"])
            nl.num_batches_tracked = nn.Parameter(
                state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.num_batches_tracked"],
                requires_grad=False,
            )
            nl.eval()
            downsample_conv_weight = state_dict[f"{self.base_address_with_dot}{name}.0.downsample.0.weight"]
            downsample_conv_bias = None

            if self.fold_batchnorm:
                downsample_conv_weight, downsample_conv_bias = fold_bn_to_conv_weights_bias(downsample_conv_weight, nl)
                nl = nn.Identity()

            # With single buffered input CB, these shapes work -
            # hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_downsample_conv = {
            #     (3136, 256) : [128, 128, 128, 64] ,
            #     (800, 512) : [128, 128, 128, 64] ,
            #     (224, 1024) : [64, 128, 64, 64],
            #     (64, 2048) : [64, 128, 64, 64] ,
            # }

            downsample_output_channels = planes * block.expansion
            self.downsample_params = [
                downsample_output_channels,
                self.inplanes,
                1,
                1,
                stride,
                stride,
                0,
                0,
                self.dilation,
                1,
            ]
            self.downsample_conv_output_shape = compute_conv_output_shape(self.downsample_params, layer_input_shape)
            is_downsample_1x1_conv = stride == 1
            is_1x1_downsample_conv_sanity_check = (
                self.downsample_params[2] == 1
                and self.downsample_params[3] == 1
                and self.downsample_params[4] == 1
                and self.downsample_params[5] == 1
                and self.downsample_params[6] == 0
                and self.downsample_params[7] == 0
            )
            assert is_1x1_downsample_conv_sanity_check == is_downsample_1x1_conv
            downsample_output_padded_face_size = _nearest_32(
                self.downsample_conv_output_shape[0]
                * self.downsample_conv_output_shape[1]
                * self.downsample_conv_output_shape[2]
            )
            matmul_config = None
            if is_downsample_1x1_conv:
                assert (
                    downsample_output_padded_face_size,
                    self.inplanes,
                    downsample_output_channels,
                ) in hardcoded_matmul_config_conv[batch_size]
                # print("Setting matmul config for 1x1 conv (downsample stride 1 conv in module)")
                matmul_config = hardcoded_matmul_config_conv[batch_size][
                    (downsample_output_padded_face_size, self.inplanes, downsample_output_channels)
                ]
                self.downsample_conv_on_tt = resnet50_1x1_conv_as_matmul(
                    downsample_conv_weight.reshape(-1).tolist(),
                    self.downsample_params,
                    self.device,
                    downsample_conv_bias.tolist(),
                    matmul_config,
                    output_mem_config=self.ds_conv_output_memory_config,
                )
            elif use_downsample_op_and_mm_for_conv1x1_s2:
                assert (
                    downsample_output_padded_face_size,
                    self.inplanes,
                    downsample_output_channels,
                ) in hardcoded_matmul_config_conv[batch_size]
                matmul_config = hardcoded_matmul_config_conv[batch_size][
                    (downsample_output_padded_face_size, self.inplanes, downsample_output_channels)
                ]
                assert stride == 2
                downsample_op_params = [batch_size, layer_input_shape[1], layer_input_shape[2], stride, stride]
                #print("Calling ds op and matmul op, input shape - ", layer_input_shape)
                self.downsample_conv_on_tt = resnet50_1x1_conv_s2_as_downsample_and_matmul(
                    downsample_conv_weight.reshape(-1).tolist(),
                    self.downsample_params,
                    downsample_op_params, # used by downsample op
                    self.device,
                    downsample_conv_bias.tolist(),
                    matmul_config,
                    self.ds_conv_output_memory_config
                )
            else:
                assert (
                    downsample_output_padded_face_size,
                    downsample_output_channels,
                ) in hardcoded_conv_blocking_and_parallelization_config[batch_size]
                [
                    act_block_w_datums,
                    act_block_h_datums,
                    weight_block_w_datums,
                    out_subblock_h_datums,
                    out_subblock_w_datums,
                    out_block_h_datums,
                    grid_size,
                    per_core_act_h,
                    per_core_weight_w,
                ] = hardcoded_conv_blocking_and_parallelization_config[batch_size][
                    (downsample_output_padded_face_size, downsample_output_channels)
                ]
                assert per_core_act_h % 32 == 0
                per_core_act_h_ntiles = (int)(per_core_act_h / 32)
                per_core_weight_w_ntiles = (int)(per_core_weight_w / 32)
                assert self.inplanes % act_block_w_datums == 0
                self.downsample_conv_on_tt = resnet50_optimized_conv(
                    downsample_conv_weight.reshape(-1).tolist(),
                    self.downsample_params,
                    self.device,
                    [act_block_h_datums, act_block_w_datums],
                    [act_block_w_datums, weight_block_w_datums],
                    [out_subblock_h_datums, out_subblock_w_datums],
                    out_block_h_datums,
                    grid_size,
                    per_core_act_h_ntiles,
                    per_core_weight_w_ntiles,
                    downsample_conv_bias.tolist(),
                    output_mem_config=self.ds_conv_output_memory_config,
                )
            self.norm_layer_after_downsample_conv_on_tt = nl

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                device=self.device,
                state_dict=self.state_dict,
                base_address=f"{self.base_address_with_dot}{name}.0",
                fold_batchnorm=self.fold_batchnorm,
                downsample_conv_on_tt=self.downsample_conv_on_tt,
                norm_layer_after_downsample_conv_on_tt=self.norm_layer_after_downsample_conv_on_tt,
                downsample_params=self.downsample_params,
                storage_in_dram=self.storage_in_dram,
                input_shape=layer_input_shape,
                batch_size=batch_size,
                sharded=sharded,
                out_sharded=sharded,
                act_block_w_equals_input_channels_x_filter_width=act_block_w_equals_input_channels_x_filter_width,
                use_downsample_op_and_mm_for_conv1x1_s2=use_downsample_op_and_mm_for_conv1x1_s2
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            previous_layer = layers[-1]
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    device=self.device,
                    state_dict=self.state_dict,
                    base_address=f"{self.base_address_with_dot}{name}.{_}",
                    fold_batchnorm=self.fold_batchnorm,
                    storage_in_dram=self.storage_in_dram,
                    input_shape=previous_layer.conv3_output_shape,
                    batch_size=batch_size,
                    sharded=sharded,
                    out_sharded=True if _ != blocks - 1 else out_sharded,
                    act_block_w_equals_input_channels_x_filter_width=act_block_w_equals_input_channels_x_filter_width,
                )
            )
        last_layer_shape = layers[-1].conv3_output_shape
        return layers, last_layer_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra_padding_for_32B_alignment = 25
        x = torch.nn.functional.pad(x, (3, 4 + extra_padding_for_32B_alignment, 3, 3, 0, 1))
        # permute_key="permute_key"
        # input_tensor_construct_key="input_tensor_construct_key"
        # misc_key = "misc_key"
        # assert x.shape[3] == 224 and x.shape[2] == 224
        # profiler.start(permute_key)

        # print("padding x shape - ", x.shape)
        x = torch.permute(x, (0, 2, 3, 1))
        # profiler.end(permute_key)
        # profiler.start(input_tensor_construct_key)
        x = tt_lib.tensor.Tensor(x, tt_lib.tensor.DataType.BFLOAT16)
        # profiler.end(input_tensor_construct_key)

        original_A_cl_host_shape = x.shape()
        x = x.reshape(x.shape()[0], x.shape()[1], 1, x.shape()[2] * x.shape()[3])
        # print("A_cl_host shape after re-shape (only for transfer)", x.shape())

        x = x.to(self.device, self.memory_config)  # to l1
        # re-shape back to original shape (N, H, W, C)
        x = x.reshape(
            original_A_cl_host_shape[0],
            original_A_cl_host_shape[1],
            original_A_cl_host_shape[2],
            original_A_cl_host_shape[3],
        )
        # print("A_cl_device shape into OP", x.shape())
        # print("Running conv1")
        # time.sleep(5)
        x = self.conv1(x)
        # time.sleep(5)
        # x = x.reshape(1, 1, x.shape()[0]*x.shape()[1]*x.shape()[2], x.shape()[3]);
        # print("Printing relu after conv1")
        # Relu is fused with conv1
        # x = self.relu(x, self.memory_config)
        # tt_lib.device.DumpDeviceMemoryState(self.device)
        # x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM))
        x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
        # time.sleep(2)
        x = x.reshape(
            self.conv1_output_shape[0],
            self.conv1_output_shape[1],
            self.conv1_output_shape[2],
            self.conv1_output_shape[3],
        )
        # print("Running maxpool")
        x = self.maxpool(x)
        # print("Done maxpool")
        x = x.reshape(
            1,
            1,
            self.maxpool_output_shape[0] * self.maxpool_output_shape[1] * self.maxpool_output_shape[2],
            self.maxpool_output_shape[3],
        )
        x = format_tensor(x, tt_lib.tensor.Layout.TILE, self.device, self.sharded_memory_config)

        x = self.layer1_module1(x)
        x = self.layer1_module2(x)
        x = self.layer1_module3(x)

        x = self.layer2_module1(x)
        x = self.layer2_module2(x)
        x = self.layer2_module3(x)
        x = self.layer2_module4(x)
        if self.sharded:
            grid_size = (10, 8)
            x = tt_lib.tensor.interleaved_to_sharded(
                x,
                grid_size,
                [math.ceil((x.shape()[-2] // 32) / grid_size[0]) * 32, x.shape()[-1] // grid_size[1]],
                tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                tt_lib.tensor.ShardOrientation.COL_MAJOR,
            )
        x = self.layer3_module1(x)
        x = self.layer3_module2(x)
        x = self.layer3_module3(x)
        x = self.layer3_module4(x)
        x = self.layer3_module5(x)
        x = self.layer3_module6(x)
        if self.sharded:
            grid_size = (7, 8)
            x = tt_lib.tensor.interleaved_to_sharded(
                x,
                grid_size,
                [math.ceil((x.shape()[-2] // 32) / grid_size[0]) * 32, x.shape()[-1] // grid_size[1]],
                tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                tt_lib.tensor.ShardOrientation.COL_MAJOR,
            )

        x = self.layer4_module1(x)
        x = self.layer4_module2(x)
        x = self.layer4_module3(x)

        unpadded_shape = x.shape_without_padding()
        x = tt_lib.tensor.untilize(x, self.memory_config, use_multicore=True)
        x = tt_lib.tensor.unpad(
            x,
            (0, 0, 0, 0),
            (unpadded_shape[0] - 1, unpadded_shape[1] - 1, unpadded_shape[2] - 1, unpadded_shape[3] - 1),
            output_mem_config=self.memory_config,
        )

        x = x.reshape(self.batch_size, x.shape()[1], (int)(x.shape()[2] / self.batch_size), x.shape()[3])

        unpadded_shape = x.shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]
        x = tt_lib.tensor.pad(
            x, padded_shape, [0, 0, 0, 0], 0, output_mem_config=self.memory_config, use_multicore=True
        )
        x = tt_lib.tensor.tilize(x, output_mem_config=self.memory_config, use_multicore=True)

        x = self.avgpool(x, self.memory_config)

        unpadded_shape_end = [x.shape()[0] - 1, x.shape()[1] - 1, 1 - 1, x.shape()[3] - 1]
        x = tt_lib.tensor.untilize(x, self.memory_config, use_multicore=True)
        x = tt_lib.tensor.unpad(x, (0, 0, 0, 0), unpadded_shape_end, output_mem_config=self.memory_config)

        x = x.reshape(1, x.shape()[1], self.batch_size * x.shape()[2], x.shape()[3])

        unpadded_shape = x.shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]
        x = tt_lib.tensor.pad(
            x, padded_shape, [0, 0, 0, 0], 0, output_mem_config=self.memory_config, use_multicore=True
        )
        x = tt_lib.tensor.tilize(x, output_mem_config=self.memory_config, use_multicore=True)

        x = self.fc(x)
        x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
        x = x.reshape(self.batch_size, x.shape()[1], (int)(x.shape()[2] / self.batch_size), x.shape()[3])
        x = x.cpu()
        # assert x.layout() != tt_lib.tensor.Layout.ROW_MAJOR
        # x = x.to(tt_lib.tensor.Layout.ROW_MAJOR)
        desired_shape = [x.shape()[0], x.shape()[1], 1, 1000]
        x = x.unpad(
            (0, 0, 0, 0), (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1)
        )
        # tt_to_torch_key = "tt_to_torch_key"
        # profiler.start(tt_to_torch_key)
        x = x.to_torch().to(torch.float)
        # profiler.end(tt_to_torch_key)
        # print("Printing profiler")
        # profiler.print()
        # print("Done printing profiler")
        return x

    def run_first_part(self, x):
        extra_padding_for_32B_alignment = 25
        x = torch.nn.functional.pad(x, (3, 4 + extra_padding_for_32B_alignment, 3, 3, 0, 1))
        # permute_key="permute_key"
        # input_tensor_construct_key="input_tensor_construct_key"
        # misc_key = "misc_key"
        # assert x.shape[3] == 224 and x.shape[2] == 224
        # profiler.start(permute_key)

        # print("padding x shape - ", x.shape)
        x = torch.permute(x, (0, 2, 3, 1))
        # profiler.end(permute_key)
        # profiler.start(input_tensor_construct_key)
        x = tt_lib.tensor.Tensor(x, tt_lib.tensor.DataType.BFLOAT16)
        # profiler.end(input_tensor_construct_key)

        original_A_cl_host_shape = x.shape()
        x = x.reshape(x.shape()[0], x.shape()[1], 1, x.shape()[2] * x.shape()[3])
        # print("A_cl_host shape after re-shape (only for transfer)", x.shape())

        x = x.to(self.device, self.memory_config)  # to l1
        # re-shape back to original shape (N, H, W, C)
        x = x.reshape(
            original_A_cl_host_shape[0],
            original_A_cl_host_shape[1],
            original_A_cl_host_shape[2],
            original_A_cl_host_shape[3],
        )
        # print("A_cl_device shape into OP", x.shape())
        # print("Running conv1")
        # time.sleep(5)
        x = self.conv1(x)
        # time.sleep(5)
        # x = x.reshape(1, 1, x.shape()[0]*x.shape()[1]*x.shape()[2], x.shape()[3]);
        # print("Printing relu after conv1")
        # Relu is fused with conv1
        # x = self.relu(x, self.memory_config)
        # tt_lib.device.DumpDeviceMemoryState(self.device)

        # x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM))
        x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
        # time.sleep(2)
        x = x.reshape(
            self.conv1_output_shape[0],
            self.conv1_output_shape[1],
            self.conv1_output_shape[2],
            self.conv1_output_shape[3],
        )

        conv1_output_on_host = x.cpu().to_torch().to(torch.float)

        # Permute from nhwc to nchw
        conv1_output_on_host = torch.transpose(conv1_output_on_host, 2, 3)
        conv1_output_on_host = torch.transpose(conv1_output_on_host, 1, 2)
        # print("Running maxpool")
        # time.sleep(2)
        x = self.maxpool(x)
        max_pool_output_on_host = x.cpu().to_torch().to(torch.float)
        max_pool_output_on_host = torch.reshape(max_pool_output_on_host, self.maxpool_output_shape)
        # Permute from nhwc to nchw
        max_pool_output_on_host = torch.transpose(max_pool_output_on_host, 2, 3)
        max_pool_output_on_host = torch.transpose(max_pool_output_on_host, 1, 2)
        return conv1_output_on_host, max_pool_output_on_host

        # print("Done maxpool")
        x = x.reshape(
            1,
            1,
            self.maxpool_output_shape[0] * self.maxpool_output_shape[1] * self.maxpool_output_shape[2],
            self.maxpool_output_shape[3],
        )
        x = format_tensor(x, tt_lib.tensor.Layout.TILE, self.device, self.memory_config)

        x = self.layer1_module1(x)
        x = self.layer1_module2(x)
        x = self.layer1_module3(x)

        x = self.layer2_module1(x)
        x = self.layer2_module2(x)
        x = self.layer2_module3(x)
        x = self.layer2_module4(x)

        x = self.layer3_module1(x)
        x = self.layer3_module2(x)
        x = self.layer3_module3(x)
        x = self.layer3_module4(x)
        x = self.layer3_module5(x)
        x = self.layer3_module6(x)

        x = self.layer4_module1(x)
        x = self.layer4_module2(x)
        x = self.layer4_module3(x)

        x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
        x = x.reshape(self.batch_size, x.shape()[1], (int)(x.shape()[2] / self.batch_size), x.shape()[3])

        unpadded_shape = x.shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]
        x = tt_lib.tensor.pad(x, padded_shape, [0, 0, 0, 0], 0, output_mem_config=self.memory_config)
        x = tt_lib.tensor.tilize(x, output_mem_config=self.memory_config)

        x = self.avgpool(x, self.memory_config)

        unpadded_shape_end = [x.shape()[0] - 1, x.shape()[1] - 1, 1 - 1, x.shape()[3] - 1]
        x = tt_lib.tensor.untilize_with_unpadding(
            x,
            output_tensor_end=unpadded_shape_end,
            output_tensor_start=[0, 0, 0, 0],
            output_mem_config=self.memory_config,
        )
        x = x.reshape(1, x.shape()[1], self.batch_size * x.shape()[2], x.shape()[3])

        unpadded_shape = x.shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]
        x = tt_lib.tensor.pad(x, padded_shape, [0, 0, 0, 0], 0, output_mem_config=self.memory_config)
        x = tt_lib.tensor.tilize(x, output_mem_config=self.memory_config)

        x = self.fc(x)
        x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
        x = x.reshape(self.batch_size, x.shape()[1], (int)(x.shape()[2] / self.batch_size), x.shape()[3])
        x = x.cpu()
        # assert x.layout() != tt_lib.tensor.Layout.ROW_MAJOR
        # x = x.to(tt_lib.tensor.Layout.ROW_MAJOR)
        desired_shape = [x.shape()[0], x.shape()[1], 1, 1000]
        x = x.unpad(
            (0, 0, 0, 0), (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1)
        )
        # tt_to_torch_key = "tt_to_torch_key"
        # profiler.start(tt_to_torch_key)
        x = x.to_torch().to(torch.float)
        # profiler.end(tt_to_torch_key)
        # print("Printing profiler")
        # profiler.print()
        # print("Done printing profiler")
        return x, conv1_output_on_host, max_pool_output_on_host
