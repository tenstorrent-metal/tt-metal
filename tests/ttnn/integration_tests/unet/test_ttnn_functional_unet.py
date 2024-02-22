# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn

from torchview import draw_graph

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0

from models.experimental.functional_unet.tt import ttnn_functional_unet

import ttnn


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256
    # ttnn_module_args["enable_auto_formatting"] = ttnn_module_args.kernel_size < (7, 7)


def custom_preprocessor(model, name, ttnn_module_args):
    parameters = {}
    print("ttnn_module_args: ", ttnn_module_args)
    if isinstance(model, UNet):
        # ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c1["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c1_2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c1["deallocate_activation"] = True  #
        ttnn_module_args.c1_2["deallocate_activation"] = True  #
        ttnn_module_args.c1["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 64}
        ttnn_module_args.c1_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 64}
        print("\n\n\n\nttnn_module_args.p1: ")
        print(ttnn_module_args.p1)

        print("the type of ttnn_module_args.p1 is: ", type(ttnn_module_args.p1))
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.c1_2, model.b1_2)

        update_ttnn_module_args(ttnn_module_args.c1)
        update_ttnn_module_args(ttnn_module_args.c1_2)

        parameters["c1"] = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.c1)
        parameters["c1_2"] = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.c1_2)
        parameters["p1"] = {}
        ttnn_module_args.p1["parallel_config_override"] = {
            "grid_size": parameters["c1"]["parallel_config"],
            "ncores_nhw": parameters["c1"]["num_cores_nhw"],
        }

        print(f'parameters["c1"]: {parameters["c1"]}')

        # print("parameters['p1']: ", parameters["p1"])

    #        if model.downsample is not None:
    #            downsample_weight, downsample_bias = fold_batch_norm2d_into_conv2d(
    #                model.downsample[0], model.downsample[1]
    #            )
    #            update_ttnn_module_args(ttnn_module_args.downsample[0])
    #            parameters["downsample"] = preprocess_conv2d(
    #                downsample_weight, downsample_bias, ttnn_module_args.downsample[0]
    #            )
    #            ttnn_module_args["downsample"] = ttnn_module_args.downsample[0]

    return parameters


## Define a convolutional block
# def conv_block(in_channels, out_channels):
#    return nn.Sequential(
#        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#        nn.BatchNorm2d(out_channels),
#        nn.ReLU(inplace=True),
#        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#        nn.BatchNorm2d(out_channels),
#        nn.ReLU(inplace=True)
#    )
## Define an upsample block
# def upsample_block(in_channels, out_channels):
#    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
## Define the U-Net model using Sequential


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Contracting Path
        # self.c1 = conv_block(3, 16)
        self.c1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.r1 = nn.ReLU(inplace=True)
        self.c1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.b1_2 = nn.BatchNorm2d(16)
        self.r1_2 = nn.ReLU(inplace=True)

        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

    #        self.c2 = conv_block(16, 16)
    #        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
    #        self.c3 = conv_block(16, 32)
    #        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
    #        self.c4 = conv_block(32, 32)
    #        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)
    #        # Bottleneck
    #        self.bottleneck = conv_block(32, 64)
    #        # Expansive Path
    #        self.u4 = upsample_block(64, 64)
    #        self.c5 = conv_block(96, 32)
    #        self.u3 = upsample_block(32, 32)
    #        self.c6 = conv_block(64, 32)
    #        self.u2 = upsample_block(32, 32)
    #        self.c7 = conv_block(48, 16)
    #        self.u1 = upsample_block(16, 16)
    #        self.c8 = conv_block(32, 16)
    #        # Output layer
    #        self.output_layer = nn.Conv2d(16, 1, kernel_size=1)
    def forward(self, x):
        # Contracting Path
        c1 = self.c1(x)
        b1 = self.b1(c1)
        r1 = self.r1(b1)
        c1_2 = self.c1_2(r1)
        b1_2 = self.b1_2(c1_2)
        r1_2 = self.r1_2(b1_2)
        p1 = self.p1(r1_2)
        #        c2 = self.c2(p1)
        #        p2 = self.p2(c2)
        #        c3 = self.c3(p2)
        #        p3 = self.p3(c3)
        #        c4 = self.c4(p3)
        #        p4 = self.p4(c4)
        #        # Bottleneck
        #        bottleneck = self.bottleneck(p4)
        #        # Expansive Path
        #        u4 = self.u4(bottleneck)
        #        c5 = self.c5(torch.cat([u4, c4], dim=1))
        #        u3 = self.u3(c5)
        #        c6 = self.c6(torch.cat([u3, c3], dim=1))
        #        u2 = self.u2(c6)
        #        c7 = self.c7(torch.cat([u2, c2], dim=1))
        #        u1 = self.u1(c7)
        #        c8 = self.c8(torch.cat([u1, c1], dim=1))
        #        # Output layer
        #        output = self.output_layer(c8)

        # return output
        # return r1_2
        return p1


# Example usage
model = UNet()
# input_tensor = torch.randn(1, 3, 1056, 160)  # Batch size of 1, 3 channels (RGB), 256x256 input
input_tensor = torch.randn(2, 3, 1056, 160)  # Batch size of 1, 3 channels (RGB), 256x256 input
output_tensor = model(input_tensor)
print("\n\n\n")
print("output_tensor size is: ", output_tensor.size())
print("\n\n\n")
# model_graph = draw_graph(
#     model,
#     # input_size=(1, 3, 1056, 160),
#     input_size=(2, 3, 1056, 160),
#     dtypes=[torch.float32],
#     expand_nested=True,
#     graph_name="unetSeqEdit2",
#     depth=2,
#     directory=".",
# )
# model_graph.visual_graph.render(format="pdf")

device_id = 0
device = ttnn.open(device_id)

torch.manual_seed(0)

# torch_model = BasicBlock(inplanes=64, planes=64, stride=1).eval()
torch_model = UNet()
print("\n\n\n\nthe torch model type is: ")
print(type(torch_model))
print("\n\n\n\n")
for layer in torch_model.children():
    print(layer)
new_state_dict = {}
# new_state_dict = torch_model.state_dict()
for name, parameter in torch_model.state_dict().items():
    if isinstance(parameter, torch.FloatTensor):
        # new_state_dict[name] = torch.rand_like(parameter)
        new_state_dict[name] = parameter + 100.0
print("new_state_dict keys: ", new_state_dict.keys())
print("\n\n\n\n")
# print("new_state_dict[c1.0.weight]: ", new_state_dict["c1.0.weight"])
torch_model.load_state_dict(new_state_dict)

# torch_input_tensor = torch.rand((8, 64, 56, 56), dtype=torch.float32)
# torch_output_tensor = torch_model(torch_input_tensor)
# torch_input_tensor = torch.randn(1, 3, 1056, 160)  # Batch size of 1, 3 channels (RGB),  1056x160 input
torch_input_tensor = torch.randn(2, 3, 1056, 160)  # Batch size of 2, 3 channels (RGB), 1056x160 input
torch_output_tensor = torch_model(torch_input_tensor)

reader_patterns_cache = {}
parameters = preprocess_model(
    initialize_model=lambda: torch_model,
    run_model=lambda model: model(torch_input_tensor),
    custom_preprocessor=custom_preprocessor,
    reader_patterns_cache=reader_patterns_cache,
    device=device,
)

print("\n\n\n")
print("the parameters now are: ")
print(parameters)
ttnn_model = ttnn_functional_unet.UNet(parameters)
#
output_tensor = ttnn_model.torch_call(torch_input_tensor)
#
print("torch_output_tensor.size: ", torch_output_tensor.size())
print("ttnn_output_tensor.size: ", output_tensor.size())
assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)
