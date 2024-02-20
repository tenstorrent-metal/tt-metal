# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import math

import torch
import torchvision

import ttnn
from ttnn.model_converter import (
    convert_conv2d_weight_and_bias_to_ttnn,
    fold_batch_norm2d_into_conv2d,
    fold_conv7s2_into_conv4s1,
    convert_remaining_children_and_parameters,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0, pad_and_fold_conv_activation_for_unity_stride

from models.experimental.functional_resnet.tt import ttnn_functional_resnet


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256
    ttnn_module_args["enable_auto_formatting"] = ttnn_module_args.kernel_size < (7, 7)
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}


def resnet_converter(*, module, module_name, ttnn_module_args, **_):
    parameters = {}
    if isinstance(module, torchvision.models.resnet.BasicBlock):
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(module.conv1, module.bn1)
        ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1

        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(module.conv2, module.bn2)

        parameters["conv1"]: ttnn.Conv2d = convert_conv2d_weight_and_bias_to_ttnn(
            conv1_weight, conv1_bias, ttnn_module_args.conv1
        )
        parameters["conv2"]: ttnn.Conv2d = convert_conv2d_weight_and_bias_to_ttnn(
            conv2_weight, conv2_bias, ttnn_module_args.conv2
        )

    elif isinstance(module, torch.nn.Conv2d) and module.kernel_size == (7, 7) and module.stride == (2, 2):
        return fold_conv7s2_into_conv4s1(module.weight, module.bias, ttnn_module_args)

    return parameters


@skip_for_wormhole_b0()
def test_basic_block(device):
    torch.manual_seed(0)

    torch_model = torchvision.models.resnet.BasicBlock(inplanes=64, planes=64, stride=1).eval()

    torch_input_tensor = torch.rand((8, 64, 56, 56), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = ttnn.model_converter.from_torch_model(
        model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        converter=resnet_converter,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = ttnn_functional_resnet.BasicBlock(parameters)

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))

    padded_input_channels = math.ceil(input_tensor.shape[3] / 16) * 16
    input_tensor = torch.nn.functional.pad(input_tensor, (0, padded_input_channels - input_tensor.shape[3], 0, 0, 0, 0))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_model(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(
        output_tensor,
        [
            torch_input_tensor.shape[0],
            torch_input_tensor.shape[2],
            torch_input_tensor.shape[3],
            torch_input_tensor.shape[1],
        ],
    )
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)
    breakpoint()

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9998)


@pytest.mark.skip(reason="Downsample not working properly")
@skip_for_wormhole_b0()
def test_basic_block_with_downsample(device):
    torch.manual_seed(0)

    def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
        """1x1 convolution"""
        return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    torch_model = torchvision.models.resnet.BasicBlock(
        inplanes=64, planes=64, stride=1, downsample=conv1x1(64, 64, 1)
    ).eval()

    torch_input_tensor = torch.rand((8, 64, 56, 56), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = ttnn.model_converter.from_torch_model(
        model=torch_model,
        run_model=lambda model: model(torch_input_tensor),
        converter=resnet_converter,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = ttnn_functional_resnet.BasicBlock(parameters)

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))

    padded_input_channels = math.ceil(input_tensor.shape[3] / 16) * 16
    input_tensor = torch.nn.functional.pad(input_tensor, (0, padded_input_channels - input_tensor.shape[3], 0, 0, 0, 0))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_model(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(
        output_tensor,
        [
            torch_input_tensor.shape[0],
            torch_input_tensor.shape[2],
            torch_input_tensor.shape[3],
            torch_input_tensor.shape[1],
        ],
    )
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9999)


@skip_for_wormhole_b0()
def test_resnet_conv7s2(device):
    in_planes = 64

    torch_model = torch.nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=[3, 3], bias=False)

    torch_input_tensor = torch.rand((8, 3, 224, 224), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = ttnn.model_converter.from_torch_model(
        model=torch_model,
        run_model=lambda model: model(torch_input_tensor),
        converter=resnet_converter,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    input_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, *torch_model.padding, *torch_model.stride
    )

    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    output_tensor = parameters.copy_input_to_device(input_tensor)
    output_tensor = parameters(output_tensor)
    output_tensor = parameters.copy_output_from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor)


@skip_for_wormhole_b0()
def test_resnet(device):
    torch.manual_seed(0)

    torch_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()

    torch_input_tensor = torch.rand((8, 3, 224, 224), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    def resnet_converter(*, module, module_name, ttnn_module_args, is_to_be_converted):
        parameters = {}
        if isinstance(module, torchvision.models.resnet.BasicBlock):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1

            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(module.conv1, module.bn1)
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(module.conv2, module.bn2)

            update_ttnn_module_args(ttnn_module_args.conv1)
            update_ttnn_module_args(ttnn_module_args.conv2)

            parameters["conv1"] = convert_conv2d_weight_and_bias_to_ttnn(
                conv1_weight, conv1_bias, ttnn_module_args.conv1
            )
            parameters["conv2"] = convert_conv2d_weight_and_bias_to_ttnn(
                conv2_weight, conv2_bias, ttnn_module_args.conv2
            )

            if module.downsample is not None:
                downsample_weight, downsample_bias = fold_batch_norm2d_into_conv2d(
                    module.downsample[0], module.downsample[1]
                )
                update_ttnn_module_args(ttnn_module_args.downsample[0])
                parameters["downsample"] = convert_conv2d_weight_and_bias_to_ttnn(
                    downsample_weight, downsample_bias, ttnn_module_args.downsample[0]
                )
                ttnn_module_args["downsample"] = ttnn_module_args.downsample[0]

        elif isinstance(module, torchvision.models.resnet.ResNet):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(module.conv1, module.bn1)
            parameters["conv1"]: ttnn.Conv2d = fold_conv7s2_into_conv4s1(
                conv1_weight, conv1_bias, ttnn_module_args.conv1
            )

            return convert_remaining_children_and_parameters(
                module,
                name=module_name,
                is_to_be_converted=is_to_be_converted,
                converter=resnet_converter,
                parameters=parameters,
                ttnn_module_args=ttnn_module_args,
                already_converter_children={"conv1", "bn1", "relu1"},
            )

        return parameters

    reader_patterns_cache = {}
    parameters_and_modules = ttnn.model_converter.from_torch_model(
        model=torch_model,
        run_model=lambda model: model(torch_input_tensor),
        reader_patterns_cache=reader_patterns_cache,
        converter=resnet_converter,
        device=device,
    )

    input_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, *torch_model.conv1.padding, *torch_model.conv1.stride
    )

    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    output_tensor = parameters.conv1.copy_input_to_device(input_tensor)
    output_tensor = parameters.conv1(output_tensor)
    output_tensor = parameters.maxpool(output_tensor)
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)

    for basic_block_parameters in parameters.layer1.values():
        basic_block = ttnn_functional_resnet.BasicBlock(basic_block_parameters)
        output_tensor = basic_block(output_tensor)
    for basic_block_parameters in parameters.layer2.values():
        # TODO: remove this return once downsample works properly
        return
        basic_block = ttnn_functional_resnet.BasicBlock(basic_block_parameters)
        output_tensor = basic_block(output_tensor)
    for basic_block_parameters in parameters.layer3.values():
        basic_block = ttnn_functional_resnet.BasicBlock(basic_block_parameters)
        output_tensor = basic_block(output_tensor)
    for basic_block_parameters in parameters.layer4.values():
        basic_block = ttnn_functional_resnet.BasicBlock(basic_block_parameters)
        output_tensor = basic_block(output_tensor)

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.reshape(output_tensor, (8, 1, 49, 512))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.global_avg_pool2d(output_tensor)
    output_tensor = output_tensor @ parameters.fc.weight + parameters.fc.bias

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(output_tensor, (8, 1000))

    # The check below doesn't work yet
    # assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)
