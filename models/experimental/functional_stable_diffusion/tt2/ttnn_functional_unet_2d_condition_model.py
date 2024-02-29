# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
import torch.nn as nn
import math
import ttnn
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
from loguru import logger
from tt_lib.fallback_ops import fallback_ops

from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_embeddings import TtTimestepEmbedding
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_unet_mid_block_2d_cross_attn import (
    unet_mid_block_2d_cross_attn,
)
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_cross_attention_down_block_2d import (
    cross_attention_down_block_2d,
)
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_cross_attn_upblock import (
    cross_attention_upblock2d,
)
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_downblock_2d import downblock2d
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_upblock_2d import upblock_2d
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    run_ttnn_conv_with_pre_and_post_tensor_formatting,
    pre_process_input_new,
    post_process_output,
)


def permute_conv_weights(weight, bias):
    weight = ttnn.to_layout(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_layout(bias, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = ttnn.to_torch(bias)
    return weight, bias


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


class UNet2DConditionModel:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        input_height,
        input_width,
        reader_patterns_cache,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: str = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
    ):
        self.device = device
        self.parameters = parameters

        parameters.conv_in.weight, parameters.conv_in.bias = permute_conv_weights(
            parameters.conv_in.weight, parameters.conv_in.bias
        )
        parameters.conv_in.bias = torch.reshape(parameters.conv_in.bias, (1, 1, 1, parameters.conv_in.bias.shape[-1]))
        tt_weight_tensor = ttnn.from_torch(parameters.conv_in.weight, ttnn.float32)
        tt_bias_tensor = ttnn.from_torch(parameters.conv_in.bias, ttnn.float32)
        # breakpoint()
        out_channels = parameters.conv_in.weight.shape[0]
        in_channels = parameters.conv_in.weight.shape[1]

        print(f"CIN: height: {input_height}, width: {input_width}, dim: {2 * input_height * input_width}")
        self.conv_in = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dtype=ttnn.bfloat8_b,
            device=device,
            use_1d_systolic_array=True if in_channels < 320 else False,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=tt_weight_tensor,
            bias=tt_bias_tensor,
            math_fidelity=ttnn.MathFidelity.LoFi,
            weights_dtype=ttnn.bfloat8_b,
            conv_blocking_and_parallelization_config_override={},
            use_shallow_conv_variant=False,
        )
        # breakpoint()
        self.down_blocks = []
        input_height = self.conv_in.output_height
        input_width = self.conv_in.output_height
        print(f"D-1: height: {input_height}, width: {input_width}, dim: {2 * input_height * input_width}")
        self.down_block_types = down_block_types
        for i, down_block_type in enumerate(down_block_types):
            if down_block_type == "CrossAttnDownBlock2D":
                down_block = cross_attention_down_block_2d(
                    device,
                    parameters.down_blocks[i],
                    reader_patterns_cache,
                    batch_size,
                    input_height,
                    input_width,
                    group_norm_on_device=True if i == 0 else False,
                )
            elif down_block_type == "DownBlock2D":
                down_block = downblock2d(
                    device, parameters.down_blocks[i], reader_patterns_cache, batch_size, input_height, input_width
                )
            else:
                assert False

            self.down_blocks.append(down_block)
            input_height = down_block.output_height
            input_width = down_block.output_width
            print(f"D{i}:  height: {input_height}, width: {input_width}, dim: {2 * input_height * input_width}")

        assert mid_block_type == "UNetMidBlock2DCrossAttn"
        self.mid_block = unet_mid_block_2d_cross_attn(
            device, parameters.mid_block, reader_patterns_cache, batch_size, input_height, input_width
        )
        input_height = self.mid_block.output_height
        input_width = self.mid_block.output_width
        print(f"MID: height: {input_height}, width: {input_width}, dim: {2 * input_height * input_width}")

        self.up_blocks = []
        self.up_block_types = up_block_types
        for i, up_block_type in enumerate(up_block_types):
            if up_block_type == "CrossAttnUpBlock2D":
                up_block = cross_attention_upblock2d(
                    device, parameters.up_blocks[i], reader_patterns_cache, batch_size, input_height, input_width
                )
            elif up_block_type == "UpBlock2D":
                up_block = upblock_2d(
                    device, parameters.up_blocks[i], reader_patterns_cache, batch_size, input_height, input_width
                )
            else:
                assert False

            self.up_blocks.append(up_block)
            input_height = up_block.output_height
            input_width = up_block.output_width
            print(f"UP{i}: height: {input_height}, width: {input_width}, dim: {2 * input_height * input_width}")

        parameters.conv_out.weight, parameters.conv_out.bias = permute_conv_weights(
            parameters.conv_out.weight, parameters.conv_out.bias
        )
        parameters.conv_out.bias = torch.reshape(
            parameters.conv_out.bias, (1, 1, 1, parameters.conv_out.bias.shape[-1])
        )
        tt_weight_tensor = ttnn.from_torch(parameters.conv_out.weight, ttnn.float32)
        tt_bias_tensor = ttnn.from_torch(parameters.conv_out.bias, ttnn.float32)
        out_channels = parameters.conv_out.weight.shape[0]
        in_channels = parameters.conv_out.weight.shape[1]

        print(f"COU: height: {input_height}, width: {input_width}, dim: {2 * input_height * input_width}")
        self.conv_out = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dtype=ttnn.bfloat8_b,
            device=device,
            use_1d_systolic_array=True,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=tt_weight_tensor,
            bias=tt_bias_tensor,
            math_fidelity=ttnn.MathFidelity.LoFi,
            weights_dtype=ttnn.bfloat8_b,
            conv_blocking_and_parallelization_config_override={"act_block_h": 64},
            use_shallow_conv_variant=False,
            enable_auto_formatting=True,
            deallocate_activation=True,
        )

        self.emb = TtTimestepEmbedding(parameters.time_embedding)

    def __call__(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        config,
        class_labels=None,
        attention_mask=None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        return_dict: bool = True,
        reader_patterns_cache: Optional[Dict] = None,
        dtype: Optional[ttnn.DataType] = None,
    ):
        num_upsamplers = len(block_out_channels) - 1
        default_overall_up_factor = 2**num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        time_embed_dim = block_out_channels[0] * 4
        sample_shape_list = list(sample.shape)
        if any(s % default_overall_up_factor != 0 for s in sample_shape_list[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            assert False, "attention mask is always None"
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if config.center_input_sample:
            assert False, "We are not centering"
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep

        # # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = timesteps.expand(sample.shape[0]) # Nonte: IS ON TORCH

        # Note: keep this code for future references; this is constant propped currently!
        # t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        # t_emb = t_emb.to(dtype=self.dtype)

        t_emb = timestep
        timestep_input_dim = block_out_channels[0]
        emb = self.emb(t_emb)

        if class_embed_type is None and num_class_embeds is not None:
            assert False, "We do not support embedding"
            class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            assert False, "We do not support TimestepEmbedding"
            class_embedding = TtTimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            assert False, "We do not support Identity"
            class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            class_embedding = None

        if class_embedding is not None:
            assert False, "This should not be triggerred!"
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if config.class_embed_type == "timestep":
                class_labels = time_proj(class_labels)

            class_emb = class_embedding(class_labels)
            emb = emb + class_emb

        sample = pre_process_input_new(self.device, sample)
        # sample in l1 interelaved and tiled and nhwc
        sample = ttnn.to_memory_config(sample, self.conv_in.conv.input_sharded_memory_config)
        sample = self.conv_in(sample)
        sample = ttnn.reallocate(sample)

        # con_in completes

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(self.down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(self.down_block_types)

        # 3.down
        sample_copied_to_dram = ttnn.to_memory_config(sample, ttnn.DRAM_MEMORY_CONFIG)
        sample_copied_to_dram = post_process_output(
            self.device,
            sample_copied_to_dram,
            self.conv_in.batch_size,
            self.conv_in.output_height,
            self.conv_in.output_width,
            self.conv_in.out_channels,
        )
        down_block_res_samples = (sample_copied_to_dram,)
        output_channel = block_out_channels[0]
        for i, (down_block_type, down_block) in enumerate(zip(self.down_block_types, self.down_blocks)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if down_block_type == "CrossAttnDownBlock2D":
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    config=config,
                    resnet_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim[i],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    group_norm_sharded_config={
                        "shard_strategy": self.conv_in.conv.input_shard_scheme,
                        "shard_orientation": self.conv_in.conv.input_shard_orientation,
                        "grid_size": self.conv_in.conv.grid_size,
                    }
                    if i == 0
                    else None,
                )
            elif down_block_type == "DownBlock2D":
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=emb,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    dtype=dtype,
                )
            else:
                assert (
                    False
                ), f"CrossAttnDownBlock2D, and DownBlock2D are the only down blocks implemented! you requested {down_block_type}"

            down_block_res_samples += res_samples

        # 4.mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            config=config,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

        # 5.up
        num_upsamplers = 0

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, (up_block_type, up_block) in enumerate(zip(self.up_block_types, self.up_blocks)):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer

            if not is_final_block:
                add_upsample = True
            else:
                add_upsample = False

            if up_block_type == "UpBlock2D" or up_block_type == "CrossAttnUpBlock2D":
                resnets = layers_per_block + 1
            res_samples = down_block_res_samples[-resnets:]
            down_block_res_samples = down_block_res_samples[:-resnets]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if up_block_type == "CrossAttnUpBlock2D":
                sample = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    config=config,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=reversed_attention_head_dim[i],
                    dual_cross_attention=dual_cross_attention,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
            elif up_block_type == "UpBlock2D":
                sample = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
            else:
                assert (
                    False
                ), f"CrossAttnUpBlock2D, and UpBlock2D are the only up blocks implemented! you requested {up_block_type}"

        # 6.post-process
        sample = ttnn.group_norm(
            sample,
            num_groups=norm_num_groups,
            epsilon=norm_eps,
            weight=self.parameters.conv_norm_out.weight,
            bias=self.parameters.conv_norm_out.bias,
        )
        sample = ttnn.silu(sample)

        sample = run_ttnn_conv_with_pre_and_post_tensor_formatting(
            self.device,
            self.conv_out,
            sample,
            self.conv_out.batch_size,
            self.conv_out.input_height,
            self.conv_out.input_width,
            self.conv_out.out_channels,
        )

        # con_in completes

        return sample
