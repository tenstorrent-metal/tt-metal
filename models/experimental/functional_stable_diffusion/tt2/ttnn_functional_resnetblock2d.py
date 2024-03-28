# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
import os
import torch
from typing import Optional, Dict
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    pre_process_input,
    post_process_output,
    permute_conv_parameters,
)
import time


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return input


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


config_override = {
    (320, 320, 64, 64): {"act_block_h": 64},
    (640, 640, 32, 32): {"act_block_h": 64},
    (640, 1920, 32, 32): {"act_block_h": 32},
    (640, 1280, 32, 32): {"act_block_h": 32},
    (1280, 1920, 16, 16): {"act_block_h": 32},
    (1280, 1280, 32, 32): {"act_block_h": 32},
    (1280, 1280, 16, 16): {"act_block_h": 32},
    (320, 960, 64, 64): {"act_block_h": 32},
    (640, 960, 32, 32): {"act_block_h": 32},
    (320, 640, 64, 64): {"act_block_h": 32},
    (640, 320, 64, 64): {"act_block_h": 64},
    (640, 640, 64, 64): {"act_block_h": 32},
}

split_chunks = {
    (320, 960, 64, 64): 2,
    (640, 1920, 32, 32): 3,
    (640, 1280, 32, 32): 2,
    (640, 960, 32, 32): 2,
    (1280, 1920, 16, 16): 3,
    (1280, 2560, 8, 8): 2,
    (1280, 2560, 16, 16): 2,
}


class resnetBlock2D:
    def __init__(
        self,
        device,
        parameters,
        reader_patterns_cache,
        batch_size,
        input_height,
        input_width,
        compute_kernel_config=None,
        group_norm_on_device=True,
    ):
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.device = device
        self.parameters = parameters
        self.conv1s = []
        self.fallback_on_groupnorm = os.environ.get("FALLBACK_ON_GROUPNORM", "0") == "1"
        parameters.conv1.weight, parameters.conv1.bias = permute_conv_parameters(
            parameters.conv1.weight, parameters.conv1.bias
        )
        out_channels = parameters.conv1.bias.shape[-1]
        in_channels = parameters.conv1.weight.shape[1]

        parameters.conv1.bias = torch.reshape(parameters.conv1.bias, (1, 1, 1, out_channels))
        conv1_split_chunks = 1
        if (out_channels, in_channels, input_height, input_width) in split_chunks:
            conv1_split_chunks = split_chunks[(out_channels, in_channels, input_height, input_width)]
        split_input_channels = in_channels // conv1_split_chunks
        if conv1_split_chunks > 1:
            print(f"Splitting: {(out_channels, in_channels, input_height, input_width)} into: {conv1_split_chunks}")
        if conv1_split_chunks == 1:
            split_weight_tensors = [parameters.conv1.weight]
        else:
            split_weight_tensors = torch.split(parameters.conv1.weight, split_input_channels, 1)

        for i in range(conv1_split_chunks):
            tt_weight_tensor = ttnn.from_torch(split_weight_tensors[i], ttnn.float32)
            if i == 0:
                tt_bias_tensor = ttnn.from_torch(parameters.conv1.bias, ttnn.float32)
            else:
                # TODO: fix no bias in conv error
                torch_bias_zeros_tensor = torch.zeros(parameters.conv1.bias.shape, dtype=torch.bfloat16).float()
                tt_bias_tensor = ttnn.from_torch(torch_bias_zeros_tensor, ttnn.float32)
            conv1_config_override = {}
            if (out_channels, in_channels, input_height, input_width) in config_override:
                conv1_config_override = config_override[(out_channels, in_channels, input_height, input_width)]
            self.conv1s.append(
                ttnn.Conv2d(
                    split_input_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    dtype=ttnn.bfloat8_b,
                    device=device,
                    use_1d_systolic_array=False,
                    batch_size=batch_size,
                    input_height=input_height,
                    input_width=input_width,
                    reader_patterns_cache=reader_patterns_cache,
                    weight=tt_weight_tensor,
                    bias=tt_bias_tensor,
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    weights_dtype=ttnn.bfloat8_b,
                    conv_blocking_and_parallelization_config_override=conv1_config_override,
                    use_shallow_conv_variant=False,
                    compute_kernel_config=compute_kernel_config,
                    # enable_auto_formatting=(conv1_split_chunks > 1) or not group_norm_on_device,
                    # reallocate_halo_output=True,
                )
            )

        use_in_shortcut = True if "conv_shortcut" in parameters else False
        if use_in_shortcut:
            parameters.conv_shortcut.weight, parameters.conv_shortcut.bias = permute_conv_parameters(
                parameters.conv_shortcut.weight, parameters.conv_shortcut.bias
            )

            convs_input_height = input_height
            convs_input_width = input_width
            parameters.conv_shortcut.bias = torch.reshape(parameters.conv_shortcut.bias, (1, 1, 1, out_channels))
            tt_weight_tensor = ttnn.from_torch(parameters.conv_shortcut.weight, ttnn.float32)
            tt_bias_tensor = ttnn.from_torch(parameters.conv_shortcut.bias, ttnn.float32)
            self.conv_shortcut = ttnn.Conv2d(
                parameters.conv_shortcut.weight.shape[1],
                parameters.conv_shortcut.weight.shape[0],
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dtype=ttnn.bfloat8_b,
                device=device,
                use_1d_systolic_array=False,
                batch_size=batch_size,
                input_height=convs_input_height,
                input_width=convs_input_width,
                reader_patterns_cache=reader_patterns_cache,
                weight=tt_weight_tensor,
                bias=tt_bias_tensor,
                math_fidelity=ttnn.MathFidelity.LoFi,
                weights_dtype=ttnn.bfloat8_b,
                use_shallow_conv_variant=False,
                compute_kernel_config=compute_kernel_config,
            )
            self.output_height = self.conv_shortcut.output_height
            self.output_width = self.conv_shortcut.output_width

        conv2_input_height = self.conv1s[0].output_height
        conv2_input_width = self.conv1s[0].output_width
        parameters.conv2.weight, parameters.conv2.bias = permute_conv_parameters(
            parameters.conv2.weight, parameters.conv2.bias
        )
        parameters.conv2.bias = torch.reshape(parameters.conv2.bias, (1, 1, 1, out_channels))
        tt_weight_tensor = ttnn.from_torch(parameters.conv2.weight, ttnn.float32)
        tt_bias_tensor = ttnn.from_torch(parameters.conv2.bias, ttnn.float32)
        conv2_config_override = {}
        if (out_channels, out_channels, input_height, input_width) in config_override:
            conv2_config_override = config_override[(out_channels, out_channels, input_height, input_width)]
        if use_in_shortcut:
            conv2_config_override["grid_size"] = self.conv_shortcut.conv.grid_size
            conv2_config_override["per_core_out_matrix_height"] = self.conv_shortcut.conv.per_core_out_matrix_height
            conv2_config_override["per_core_weight_matrix_width"] = self.conv_shortcut.conv.per_core_out_matrix_width
        self.conv2 = ttnn.Conv2d(
            parameters.conv2.weight.shape[1],
            parameters.conv2.weight.shape[0],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dtype=ttnn.bfloat8_b,
            device=device,
            use_1d_systolic_array=False,
            batch_size=batch_size,
            input_height=conv2_input_height,
            input_width=conv2_input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=tt_weight_tensor,
            bias=tt_bias_tensor,
            math_fidelity=ttnn.MathFidelity.LoFi,
            weights_dtype=ttnn.bfloat8_b,
            conv_blocking_and_parallelization_config_override=conv2_config_override,
            use_shallow_conv_variant=False,
            deallocate_activation=True,
            compute_kernel_config=compute_kernel_config,
        )

        self.groups = 32
        if use_in_shortcut:
            assert self.conv2.conv.output_sharded_memory_config == self.conv_shortcut.conv.output_sharded_memory_config
        (
            self.first_gn_expected_input_sharded_memory_config,
            self.first_group_norm_core_grid,
        ) = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
            device=self.device,
            num_channels=in_channels,
            num_groups=self.groups,
            input_nhw=batch_size * input_height * input_width,
            is_height_sharded=False,
        )
        (
            self.second_gn_expected_input_sharded_memory_config,
            self.second_group_norm_core_grid,
        ) = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
            device=self.device,
            num_channels=out_channels,
            num_groups=self.groups,
            input_nhw=batch_size * input_height * input_width,
            is_height_sharded=False,
        )

        self.output_height = self.conv2.output_height
        self.output_width = self.conv2.output_width
        assert self.input_height == self.output_height
        assert self.input_width == self.output_width
        out_channels = parameters.conv1.bias.shape[-1]
        in_channels = parameters.conv1.weight.shape[1]

        if not self.fallback_on_groupnorm:
            print("!!!!!!! GN op")
            # self.parameters.norm1.weight = ttnn.create_group_norm_weight_bias_rm(
            #     ttnn.to_torch(self.parameters.norm1.weight), in_channels, self.groups
            # )
            # self.parameters.norm1.bias = ttnn.create_group_norm_weight_bias_rm(
            #     ttnn.to_torch(self.parameters.norm1.bias), in_channels, self.groups
            # )
            if (
                self.first_gn_expected_input_sharded_memory_config.memory_layout
                == ttnn.types.TensorMemoryLayout.BLOCK_SHARDED
            ):
                num_cores_across_channel = self.first_group_norm_core_grid.y
            elif (
                self.first_gn_expected_input_sharded_memory_config.memory_layout
                == ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED
            ):
                num_cores_across_channel = 1
            else:
                num_cores_across_channel = int(self.first_group_norm_core_grid.x * self.first_group_norm_core_grid.y)

            self.parameters.norm1.weight = ttnn.create_group_norm_weight_bias_rm(
                ttnn.to_torch(self.parameters.norm1.weight), in_channels, num_cores_across_channel
            )
            self.parameters.norm1.bias = ttnn.create_group_norm_weight_bias_rm(
                ttnn.to_torch(self.parameters.norm1.bias), in_channels, num_cores_across_channel
            )

            self.norm1_input_mask_torch_tensor = ttnn.create_groupnorm_input_mask(
                in_channels, self.groups, num_cores_across_channel
            )

            self.norm1_input_mask = ttnn.from_torch(
                self.norm1_input_mask_torch_tensor,
                dtype=ttnn.DataType.BFLOAT8_B,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            self.parameters.norm1.weight = ttnn.from_torch(
                self.parameters.norm1.weight,
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.parameters.norm1.bias = ttnn.from_torch(
                self.parameters.norm1.bias,
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # self.parameters.norm2.weight = ttnn.create_group_norm_weight_bias_rm(
            #     ttnn.to_torch(self.parameters.norm2.weight), out_channels, self.groups
            # )
            # self.parameters.norm2.bias = ttnn.create_group_norm_weight_bias_rm(
            #     ttnn.to_torch(self.parameters.norm2.bias), out_channels, self.groups
            # )
            if (
                self.second_gn_expected_input_sharded_memory_config.memory_layout
                == ttnn.types.TensorMemoryLayout.BLOCK_SHARDED
            ):
                num_cores_across_channel = self.second_group_norm_core_grid.y
            elif (
                self.second_gn_expected_input_sharded_memory_config.memory_layout
                == ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED
            ):
                num_cores_across_channel = 1
            else:
                num_cores_across_channel = int(self.second_group_norm_core_grid.x * self.second_group_norm_core_grid.y)

            self.parameters.norm2.weight = ttnn.create_group_norm_weight_bias_rm(
                ttnn.to_torch(self.parameters.norm2.weight), out_channels, num_cores_across_channel
            )
            self.parameters.norm2.bias = ttnn.create_group_norm_weight_bias_rm(
                ttnn.to_torch(self.parameters.norm2.bias), out_channels, num_cores_across_channel
            )
            self.parameters.norm2.weight = ttnn.from_torch(
                self.parameters.norm2.weight,
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.parameters.norm2.bias = ttnn.from_torch(
                self.parameters.norm2.bias,
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            self.norm2_input_mask_torch_tensor = ttnn.create_groupnorm_input_mask(
                out_channels, self.groups, num_cores_across_channel
            )
            self.norm2_input_mask = ttnn.from_torch(
                self.norm2_input_mask_torch_tensor,
                dtype=ttnn.DataType.BFLOAT8_B,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def __call__(
        self,
        input_tensor,
        *,
        temb,
        in_channels,
        temb_channels=1280,
        groups: int = 32,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        out_channels: Optional[int] = None,
        non_linearity="silu",
        pre_norm=True,
        eps=1e-5,
        up=False,
        down=False,
        use_in_shortcut: Optional[bool] = None,
        dtype: Optional[ttnn.DataType] = None,
    ):
        assert groups == self.groups
        if non_linearity == "mish":
            assert False, "Mish is not implemented!"
        else:
            nonlinearity = ttnn.silu

        print(input_tensor.get_layout())
        print(input_tensor.get_dtype())

        out_channels = in_channels if out_channels is None else out_channels
        # input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        # convert input tensor to tile layout for eltwise add in the end

        input_tensor_in_dram = ttnn.get_memory_config(input_tensor) == ttnn.DRAM_MEMORY_CONFIG
        if not input_tensor_in_dram:
            input_tensor_dram = ttnn.to_memory_config(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(input_tensor)
        else:
            input_tensor_dram = input_tensor

        # if not input_tensor_in_dram:
        #     input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # ttnn.dump_device_memory_state(self.device, prefix="GN_resnet_3_")

        if ttnn.get_memory_config(hidden_states) != self.first_gn_expected_input_sharded_memory_config:
            # if ttnn.is_sharded(hidden_states):
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
            hidden_states = ttnn.reshape(
                hidden_states, (self.conv2.batch_size, 1, self.conv2.input_height * self.conv2.input_width, in_channels)
            )
            hidden_states = ttnn.to_memory_config(hidden_states, self.first_gn_expected_input_sharded_memory_config)

            # ttnn.dump_device_memory_state(self.device, prefix="GN_resnet_4_")

        if self.fallback_on_groupnorm:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(
                hidden_states, (self.conv2.batch_size, self.conv2.input_height, self.conv2.input_width, in_channels)
            )
            hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))
            hidden_states = ttnn.operations.normalization._fallback_group_norm(
                hidden_states,
                num_groups=groups,
                weight=self.parameters.norm1.weight,
                bias=self.parameters.norm1.bias,
                epsilon=eps,
            )
            hidden_states = pre_process_input(self.device, hidden_states)
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, use_multicore=True)
        else:
            hidden_states = ttnn.group_norm(
                hidden_states,
                num_groups=groups,
                input_mask=self.norm1_input_mask,
                weight=self.parameters.norm1.weight,
                bias=self.parameters.norm1.bias,
                epsilon=eps,
                memory_config=ttnn.get_memory_config(hidden_states),
                core_grid=self.first_group_norm_core_grid,
            )
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
            hidden_states = ttnn.reshape(
                hidden_states,
                (1, 1, self.conv2.batch_size * self.conv2.input_height * self.conv2.input_width, in_channels),
            )
        hidden_states = nonlinearity(hidden_states, memory_config=ttnn.get_memory_config(hidden_states))

        if up:
            assert False, "Up block within residual block is not implemented!"
        elif down:
            assert False, "Down block within residual block is not implemented"

        conv1_split_chunks = len(self.conv1s)
        if conv1_split_chunks > 1:
            split_hidden_states = []
            output_tensor_start_width_dim = 0
            in_channels = self.parameters.conv1.weight.shape[1]
            split_input_channels = in_channels // conv1_split_chunks

            output_tensor_end_width_dim = split_input_channels
            for i in range(conv1_split_chunks):
                split_hidden_states.append(
                    hidden_states[:, :, :, output_tensor_start_width_dim:output_tensor_end_width_dim]
                )
                output_tensor_start_width_dim += split_input_channels
                output_tensor_end_width_dim += split_input_channels

        if conv1_split_chunks == 1:
            hidden_states = ttnn.to_memory_config(hidden_states, self.conv1s[0].conv.input_sharded_memory_config)
            hidden_states = self.conv1s[0](hidden_states)
        else:
            for i in range(conv1_split_chunks):
                split_hidden_states[i] = ttnn.to_memory_config(
                    split_hidden_states[i], self.conv1s[i].conv.input_sharded_memory_config
                )
                split_hidden_states[i] = self.conv1s[i](split_hidden_states[i])
                if i != 0:
                    split_hidden_states[i] = ttnn.add(
                        split_hidden_states[i],
                        split_hidden_states[i - 1],
                        memory_config=self.conv1s[i].conv.output_sharded_memory_config,
                    )
                    ttnn.deallocate(split_hidden_states[i - 1])
            hidden_states = split_hidden_states[-1]
            split_hidden_states = []

        if temb is not None:
            temb = nonlinearity(temb)
            if temb_channels is not None:
                if time_embedding_norm == "default":
                    time_emb_proj_out_channels = out_channels
                elif time_embedding_norm == "scale_shift":
                    time_emb_proj_out_channels = out_channels * 2
                else:
                    raise ValueError(f"unknown time_embedding_norm : {time_embedding_norm} ")
                temb = ttnn.linear(
                    temb,
                    self.parameters.time_emb_proj.weight,
                    bias=self.parameters.time_emb_proj.bias,
                    core_grid=temb.device().core_grid,
                )

            temb = ttnn.permute(temb, (2, 0, 1, 3))

        if temb is not None and time_embedding_norm == "default":
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
            hidden_states = ttnn.clone(
                hidden_states, memory_config=ttnn.get_memory_config(hidden_states), dtype=ttnn.bfloat16
            )
            hidden_states = ttnn.reshape(
                hidden_states,
                (self.conv2.batch_size, 1, self.conv2.input_height * self.conv2.input_width, out_channels),
            )
            hidden_states = hidden_states + temb

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        # print("GN2 shape - ", (self.conv2.batch_size, out_channels, self.conv2.input_height, self.conv2.input_width))
        hidden_states = ttnn.reshape(
            hidden_states,
            (self.conv2.batch_size, 1, self.conv2.input_height * self.conv2.input_width, out_channels),
        )
        if self.fallback_on_groupnorm:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
            hidden_states = ttnn.reshape(
                hidden_states,
                (self.conv1s[0].batch_size, self.conv1s[0].input_height, self.conv1s[0].input_width, out_channels),
            )
            hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))
            hidden_states = ttnn.operations.normalization._fallback_group_norm(
                hidden_states,
                num_groups=groups,
                weight=self.parameters.norm2.weight,
                bias=self.parameters.norm2.bias,
                epsilon=eps,
            )

            hidden_states = pre_process_input(self.device, hidden_states)
        else:
            hidden_states = ttnn.to_memory_config(hidden_states, self.second_gn_expected_input_sharded_memory_config)
            # print(f"Resnetblock GN2: memory_config={ttnn.get_memory_config(hidden_states)}")
            hidden_states = ttnn.group_norm(
                hidden_states,
                num_groups=groups,
                input_mask=self.norm2_input_mask,
                weight=self.parameters.norm2.weight,
                bias=self.parameters.norm2.bias,
                epsilon=eps,
                memory_config=self.second_gn_expected_input_sharded_memory_config,
                core_grid=self.second_group_norm_core_grid,
            )
        hidden_states = nonlinearity(hidden_states, memory_config=ttnn.get_memory_config(hidden_states))

        hidden_states = ttnn.to_memory_config(hidden_states, self.conv2.conv.input_sharded_memory_config)
        hidden_states = self.conv2(hidden_states)
        use_in_shortcut = in_channels != out_channels if use_in_shortcut is None else use_in_shortcut

        if not input_tensor_in_dram:
            input_tensor = ttnn.to_memory_config(input_tensor_dram, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            input_tensor = input_tensor_dram

        # if not input_tensor_in_dram:
        #     input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)

        if use_in_shortcut:
            if ttnn.get_memory_config(input_tensor) != self.conv_shortcut.conv.input_sharded_memory_config:
                if ttnn.is_sharded(input_tensor):
                    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
                input_tensor = ttnn.to_memory_config(input_tensor, self.conv_shortcut.conv.input_sharded_memory_config)
            input_tensor = self.conv_shortcut(input_tensor)

        if ttnn.get_memory_config(input_tensor) != ttnn.get_memory_config(hidden_states):
            if ttnn.is_sharded(input_tensor):
                input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.get_memory_config(hidden_states))
        output_tensor = ttnn.add(input_tensor, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        if output_scale_factor != 1.0:
            assert False  # Do we need this?
            output_sc_recip = 1 / output_scale_factor
            output_sc_recip = ttnn.from_torch(
                torch.full([1, 1, 1, 1], output_sc_recip), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
            )
            output_sc_recip = ttnn.to_device(output_sc_recip, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
            output_tensor = ttnn.mul(output_tensor, output_sc_recip, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(input_tensor)
        ttnn.deallocate(hidden_states)
        return output_tensor
