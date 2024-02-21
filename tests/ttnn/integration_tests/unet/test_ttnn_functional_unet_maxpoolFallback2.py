# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn

from torchview import draw_graph

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0

from models.experimental.functional_unet.tt import ttnn_functional_unet_maxpoolFallback2

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

        ttnn_module_args.c2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c2_2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c2["deallocate_activation"] = True  #
        ttnn_module_args.c2_2["deallocate_activation"] = True  #
        # ttnn_module_args.c2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 64}
        # ttnn_module_args.c2_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 64}
        ttnn_module_args.c2["conv_blocking_and_parallelization_config_override"] = None
        ttnn_module_args.c2_2["conv_blocking_and_parallelization_config_override"] = None
        ttnn_module_args.c3["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c3_2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c3["deallocate_activation"] = True  #
        ttnn_module_args.c3_2["deallocate_activation"] = True  #
        ttnn_module_args.c3["conv_blocking_and_parallelization_config_override"] = None
        ttnn_module_args.c3_2["conv_blocking_and_parallelization_config_override"] = None
        ttnn_module_args.c4["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c4_2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c4["deallocate_activation"] = True  #
        ttnn_module_args.c4_2["deallocate_activation"] = True  #
        ttnn_module_args.c4["conv_blocking_and_parallelization_config_override"] = None
        ttnn_module_args.c4_2["conv_blocking_and_parallelization_config_override"] = None

        ttnn_module_args.bnc["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.bnc_2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.bnc["deallocate_activation"] = True  #
        ttnn_module_args.bnc_2["deallocate_activation"] = True  #
        ttnn_module_args.bnc["conv_blocking_and_parallelization_config_override"] = None
        ttnn_module_args.bnc_2["conv_blocking_and_parallelization_config_override"] = None

        ttnn_module_args.c5["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c5_2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c5["deallocate_activation"] = True  #
        ttnn_module_args.c5_2["deallocate_activation"] = True  #
        ttnn_module_args.c5["conv_blocking_and_parallelization_config_override"] = None
        ttnn_module_args.c5_2["conv_blocking_and_parallelization_config_override"] = None

        ttnn_module_args.c6["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c6_2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c6["deallocate_activation"] = True  #
        ttnn_module_args.c6_2["deallocate_activation"] = True  #
        ttnn_module_args.c6["conv_blocking_and_parallelization_config_override"] = None
        ttnn_module_args.c6_2["conv_blocking_and_parallelization_config_override"] = None

        ttnn_module_args.c7["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c7_2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c7["deallocate_activation"] = True  #
        ttnn_module_args.c7_2["deallocate_activation"] = True  #
        ttnn_module_args.c7["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
        # ttnn_module_args.c7_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
        ttnn_module_args.c7_2["conv_blocking_and_parallelization_config_override"] = None

        ttnn_module_args.c8["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c8_2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c8["deallocate_activation"] = True  #
        ttnn_module_args.c8_2["deallocate_activation"] = True  #
        ttnn_module_args.c8["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
        ttnn_module_args.c8_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}

        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
        conv1_2_weight, conv1_2_bias = fold_batch_norm2d_into_conv2d(model.c1_2, model.b1_2)
        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.c2, model.b2)
        conv2_2_weight, conv2_2_bias = fold_batch_norm2d_into_conv2d(model.c2_2, model.b2_2)
        conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.c3, model.b3)
        conv3_2_weight, conv3_2_bias = fold_batch_norm2d_into_conv2d(model.c3_2, model.b3_2)
        conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
        conv4_2_weight, conv4_2_bias = fold_batch_norm2d_into_conv2d(model.c4_2, model.b4_2)
        convbn_weight, convbn_bias = fold_batch_norm2d_into_conv2d(model.bnc, model.bnb)
        convbn_2_weight, convbn_2_bias = fold_batch_norm2d_into_conv2d(model.bnc_2, model.bnb_2)
        conv5_weight, conv5_bias = fold_batch_norm2d_into_conv2d(model.c5, model.b5)
        conv5_2_weight, conv5_2_bias = fold_batch_norm2d_into_conv2d(model.c5_2, model.b5_2)
        conv6_weight, conv6_bias = fold_batch_norm2d_into_conv2d(model.c6, model.b6)
        conv6_2_weight, conv6_2_bias = fold_batch_norm2d_into_conv2d(model.c6_2, model.b6_2)
        conv7_weight, conv7_bias = fold_batch_norm2d_into_conv2d(model.c7, model.b7)
        conv7_2_weight, conv7_2_bias = fold_batch_norm2d_into_conv2d(model.c7_2, model.b7_2)
        conv8_weight, conv8_bias = fold_batch_norm2d_into_conv2d(model.c8, model.b8)
        conv8_2_weight, conv8_2_bias = fold_batch_norm2d_into_conv2d(model.c8_2, model.b8_2)

        update_ttnn_module_args(ttnn_module_args.c1)
        update_ttnn_module_args(ttnn_module_args.c1_2)
        update_ttnn_module_args(ttnn_module_args.c2)
        update_ttnn_module_args(ttnn_module_args.c2_2)
        update_ttnn_module_args(ttnn_module_args.c3)
        update_ttnn_module_args(ttnn_module_args.c3_2)
        update_ttnn_module_args(ttnn_module_args.c4)
        update_ttnn_module_args(ttnn_module_args.c4_2)
        update_ttnn_module_args(ttnn_module_args.bnc)
        update_ttnn_module_args(ttnn_module_args.bnc_2)
        update_ttnn_module_args(ttnn_module_args.c5)
        update_ttnn_module_args(ttnn_module_args.c5_2)
        update_ttnn_module_args(ttnn_module_args.c6)
        update_ttnn_module_args(ttnn_module_args.c6_2)
        update_ttnn_module_args(ttnn_module_args.c7)
        update_ttnn_module_args(ttnn_module_args.c7_2)
        update_ttnn_module_args(ttnn_module_args.c8)
        update_ttnn_module_args(ttnn_module_args.c8_2)

        parameters["c1"] = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.c1)
        parameters["c1_2"] = preprocess_conv2d(conv1_2_weight, conv1_2_bias, ttnn_module_args.c1_2)
        parameters["c2"] = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.c2)
        parameters["c2_2"] = preprocess_conv2d(conv2_2_weight, conv2_2_bias, ttnn_module_args.c2_2)
        parameters["c3"] = preprocess_conv2d(conv3_weight, conv3_bias, ttnn_module_args.c3)
        parameters["c3_2"] = preprocess_conv2d(conv3_2_weight, conv3_2_bias, ttnn_module_args.c3_2)
        parameters["c4"] = preprocess_conv2d(conv4_weight, conv4_bias, ttnn_module_args.c4)
        parameters["c4_2"] = preprocess_conv2d(conv4_2_weight, conv4_2_bias, ttnn_module_args.c4_2)
        parameters["bnc"] = preprocess_conv2d(convbn_weight, convbn_bias, ttnn_module_args.bnc)
        parameters["bnc_2"] = preprocess_conv2d(convbn_2_weight, convbn_2_bias, ttnn_module_args.bnc_2)
        parameters["c5"] = preprocess_conv2d(conv5_weight, conv5_bias, ttnn_module_args.c5)
        parameters["c5_2"] = preprocess_conv2d(conv5_2_weight, conv5_2_bias, ttnn_module_args.c5_2)
        parameters["c6"] = preprocess_conv2d(conv6_weight, conv6_bias, ttnn_module_args.c6)
        parameters["c6_2"] = preprocess_conv2d(conv6_2_weight, conv6_2_bias, ttnn_module_args.c6_2)
        parameters["c7"] = preprocess_conv2d(conv7_weight, conv7_bias, ttnn_module_args.c7)
        parameters["c7_2"] = preprocess_conv2d(conv7_2_weight, conv7_2_bias, ttnn_module_args.c7_2)
        parameters["c8"] = preprocess_conv2d(conv8_weight, conv8_bias, ttnn_module_args.c8)
        parameters["c8_2"] = preprocess_conv2d(conv8_2_weight, conv8_2_bias, ttnn_module_args.c8_2)
        # parameters["p1"] = {}

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

        self.c2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(16)
        self.r2 = nn.ReLU(inplace=True)
        self.c2_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.b2_2 = nn.BatchNorm2d(16)
        self.r2_2 = nn.ReLU(inplace=True)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(32)
        self.r3 = nn.ReLU(inplace=True)
        self.c3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b3_2 = nn.BatchNorm2d(32)
        self.r3_2 = nn.ReLU(inplace=True)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(32)
        self.r4 = nn.ReLU(inplace=True)
        self.c4_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b4_2 = nn.BatchNorm2d(32)
        self.r4_2 = nn.ReLU(inplace=True)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bnc = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bnb = nn.BatchNorm2d(64)
        self.bnr = nn.ReLU(inplace=True)
        self.bnc_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bnb_2 = nn.BatchNorm2d(64)
        self.bnr_2 = nn.ReLU(inplace=True)

        self.u4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.c5 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.b5 = nn.BatchNorm2d(32)
        self.r5 = nn.ReLU(inplace=True)
        self.c5_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b5_2 = nn.BatchNorm2d(32)
        self.r5_2 = nn.ReLU(inplace=True)
        self.u3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.c6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.b6 = nn.BatchNorm2d(32)
        self.r6 = nn.ReLU(inplace=True)
        self.c6_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b6_2 = nn.BatchNorm2d(32)
        self.r6_2 = nn.ReLU(inplace=True)
        self.u2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.c7 = nn.Conv2d(48, 16, kernel_size=3, padding=1)
        self.b7 = nn.BatchNorm2d(16)
        self.r7 = nn.ReLU(inplace=True)
        self.c7_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.b7_2 = nn.BatchNorm2d(16)
        self.r7_2 = nn.ReLU(inplace=True)
        self.u1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.c8 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.b8 = nn.BatchNorm2d(16)
        self.r8 = nn.ReLU(inplace=True)
        self.c8_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.b8_2 = nn.BatchNorm2d(16)
        self.r8_2 = nn.ReLU(inplace=True)

    #    self.u4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    #    self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

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

        print("the output in torch before maxpool: ", r1_2[1, :2, :2, :2])
        p1 = self.p1(r1_2)

        c2 = self.c2(p1)
        b2 = self.b2(c2)
        r2 = self.r2(b2)
        c2_2 = self.c2_2(r2)
        b2_2 = self.b2_2(c2_2)
        r2_2 = self.r2_2(b2_2)
        p2 = self.p2(r2_2)

        c3 = self.c3(p2)
        b3 = self.b3(c3)
        r3 = self.r3(b3)
        c3_2 = self.c3_2(r3)
        b3_2 = self.b3_2(c3_2)
        r3_2 = self.r3_2(b3_2)
        p3 = self.p3(r3_2)

        c4 = self.c4(p3)
        b4 = self.b4(c4)
        r4 = self.r4(b4)
        c4_2 = self.c4_2(r4)
        b4_2 = self.b4_2(c4_2)
        r4_2 = self.r4_2(b4_2)
        p4 = self.p4(r4_2)

        bnc = self.bnc(p4)
        bnb = self.bnb(bnc)
        bnr = self.bnr(bnb)
        bnc_2 = self.bnc_2(bnr)
        bnb_2 = self.bnb_2(bnc_2)
        bnr_2 = self.bnr_2(bnb_2)
        u4 = self.u4(bnr_2)
        conc1 = torch.cat([u4, r4_2], dim=1)

        c5 = self.c5(conc1)
        b5 = self.b5(c5)
        r5 = self.r5(b5)
        c5_2 = self.c5_2(r5)
        b5_2 = self.b5_2(c5_2)
        r5_2 = self.r5_2(b5_2)

        u3 = self.u3(r5_2)
        conc2 = torch.cat([u3, r3_2], dim=1)

        c6 = self.c6(conc2)
        b6 = self.b6(c6)
        r6 = self.r6(b6)
        c6_2 = self.c6_2(r6)
        b6_2 = self.b6_2(c6_2)
        r6_2 = self.r6_2(b6_2)

        u2 = self.u2(r6_2)
        conc3 = torch.cat([u2, r2_2], dim=1)

        c7 = self.c7(conc3)
        b7 = self.b7(c7)
        r7 = self.r7(b7)
        c7_2 = self.c7_2(r7)
        b7_2 = self.b7_2(c7_2)
        r7_2 = self.r7_2(b7_2)

        u1 = self.u1(r7_2)
        conc4 = torch.cat([u1, r1_2], dim=1)

        c8 = self.c8(conc4)
        b8 = self.b8(c8)
        r8 = self.r8(b8)
        c8_2 = self.c8_2(r8)
        b8_2 = self.b8_2(c8_2)
        r8_2 = self.r8_2(b8_2)

        # u4 = self.u4(bnr_2)
        # p1 = self.p1(bnb_2)
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
        return r8_2
        # return p1


## Example usage
# model = UNet()
## input_tensor = torch.randn(1, 3, 1056, 160)  # Batch size of 1, 3 channels (RGB), 256x256 input
# input_tensor = torch.randn(2, 3, 1056, 160)  # Batch size of 1, 3 channels (RGB), 256x256 input
# output_tensor = model(input_tensor)
# print("\n\n\n")
# print("output_tensor size is: ", output_tensor.size())
# print("\n\n\n")
# model_graph = draw_graph(
#    model,
#    # input_size=(1, 3, 1056, 160),
#    input_size=(2, 3, 1056, 160),
#    dtypes=[torch.float32],
#    expand_nested=True,
#    graph_name="unetSeqEditmore",
#    depth=2,
#    directory=".",
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
        # new_state_dict[name] = torch.rand_like(parameter) + 100.0
        new_state_dict[name] = parameter + 100.0
print("new_state_dict keys: ", new_state_dict.keys())
print("\n\n\n\n")
# print("new_state_dict[c1.0.weight]: ", new_state_dict["c1.0.weight"])
torch_model.load_state_dict(new_state_dict)

# torch_input_tensor = torch.rand((8, 64, 56, 56), dtype=torch.float32)
# torch_output_tensor = torch_model(torch_input_tensor)
# torch_input_tensor = torch.randn(1, 3, 1056, 160)  # Batch size of 1, 3 channels (RGB),  1056x160 input
torch_input_tensor = torch.randn(2, 3, 1056, 160)  # Batch size of 2, 3 channels (RGB), 1056x160 input
# torch_input_tensor = torch.randn(2, 3, 528, 80)  # Batch size of 2, 3 channels (RGB), 1056x160 input
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
ttnn_model = ttnn_functional_unet_maxpoolFallback2.UNet(parameters)
#
output_tensor = ttnn_model.torch_call(torch_input_tensor)
#
print("torch_output_tensor.size: ", torch_output_tensor.size())
print("ttnn_output_tensor.size: ", output_tensor.size())
print("torch_output_tensor some values: ", torch_output_tensor[1, :2, :2, :2])
print("ttnn_output_tensor some values: ", output_tensor[1, :2, :2, :2])
assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9999)
