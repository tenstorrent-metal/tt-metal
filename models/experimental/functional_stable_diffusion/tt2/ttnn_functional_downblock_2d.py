# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_resnetblock2d import resnetBlock2D
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_downsample_2d import downsample_2d


class downblock2d:
    def __init__(
        self,
        device,
        parameters,
        reader_patterns_cache,
        batch_size,
        input_height,
        input_width,
        compute_kernel_config=None,
    ):
        self.device = device
        self.parameters = parameters
        self.resnets = [
            resnetBlock2D(
                device,
                resnet,
                reader_patterns_cache,
                batch_size,
                input_height,
                input_width,
                compute_kernel_config=compute_kernel_config,
            )
            for resnet in parameters.resnets
        ]
        # self.downsample_2d = downsample_2d(device, parameters.downsamplers[0], reader_patterns_cache, batch_size, input_height, input_width)

        self.output_height = self.resnets[-1].output_height
        self.output_width = self.resnets[-1].output_width

    def __call__(
        self,
        temb,
        hidden_states,
        in_channels,
        out_channels,
        temb_channels,
        dropout=0.0,
        num_layers=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        resnet_pre_norm=True,
        output_scale_factor=1.0,
        add_downsample=False,
        downsample_padding=1,
        dtype: Optional[ttnn.DataType] = None,
        compute_kernel_config=None,
    ):
        output_states = ()
        for i, resnet in enumerate(self.resnets):
            in_channels = in_channels if i == 0 else out_channels
            hidden_states = resnet(
                input_tensor=hidden_states,
                temb=temb,
                in_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                out_channels=out_channels,
                pre_norm=resnet_pre_norm,
                eps=resnet_eps,
                up=False,
                down=False,
                dtype=dtype,
            )

            output_states += (hidden_states,)

        if add_downsample:
            assert False, "Bug in model implementation"
            downsamplers = [
                self.downsample_2d(
                    in_channels=in_channels,
                    hidden_states=hidden_states,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                    name="op",
                    dtype=dtype,
                    compute_kernel_config=compute_kernel_config,
                )
            ]

            for downsampler in downsamplers:
                hidden_states = downsampler
                output_states += (hidden_states,)

        return hidden_states, output_states
