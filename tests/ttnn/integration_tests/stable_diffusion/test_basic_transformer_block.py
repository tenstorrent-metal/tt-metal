# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from diffusers import StableDiffusionPipeline
import ttnn
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_basic_transformer_block import (
    basic_transformer_block as ttnn_basic_transformer_block,
)
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_basic_transformer_block import (
    basic_transformer_block as tt2_ttnn_basic_transformer_block,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    pre_process_input_new,
    post_process_output,
)


@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index, attention_head_dim",
    [
        (
            1,
            2,
            1024,
            320,
            0,
            40,
        ),
        (
            1,
            2,
            256,
            640,
            1,
            80,
        ),
        (
            1,
            2,
            64,
            1280,
            2,
            160,
        ),
        (
            1,
            2,
            16,
            1280,
            2,
            160,
        ),
    ],
)
def test_basic_transformer_block_256x256(device, model_name, N, C, H, W, index, attention_head_dim):
    torch.manual_seed(0)

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    model = pipe.unet
    model.eval()
    config = model.config
    basic_transformer = pipe.unet.down_blocks[index].attentions[1].transformer_blocks[0]

    hidden_states_shape = torch.Size([N, C, H, W])
    hidden_states = torch.rand(hidden_states_shape) * 0.01
    encoder_hidden_states_shape = [1, 2, 77, 768]
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    timestep = None
    attention_mask = None
    cross_attention_kwargs = None
    class_labels = None
    only_cross_attention = False

    torch_output = basic_transformer(hidden_states.squeeze(0), encoder_hidden_states.squeeze(0))

    parameters = preprocess_model_parameters(initialize_model=lambda: basic_transformer, device=device)

    hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16)
    hidden_states = ttnn.to_device(hidden_states, device)
    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, dtype=ttnn.bfloat16)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device)

    ttnn_output = ttnn_basic_transformer_block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        class_labels=class_labels,
        parameters=parameters,
        config=config,
        device=device,
        only_cross_attention=only_cross_attention,
        attention_head_dim=attention_head_dim,
    )

    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output.unsqueeze(0), ttnn_output, pcc=0.98)


@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index, attention_head_dim",
    [
        (
            1,
            2,
            4096,
            320,
            3,
            40,
        ),
        (
            1,
            2,
            1024,
            640,
            2,
            80,
        ),
        (
            1,
            2,
            256,
            1280,
            1,
            160,
        ),
        (
            1,
            2,
            64,
            1280,
            1,
            160,
        ),
    ],
)
def test_basic_transformer_block_512x512(device, model_name, N, C, H, W, index, attention_head_dim):
    torch.manual_seed(0)

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    model = pipe.unet
    model.eval()
    config = model.config
    basic_transformer = pipe.unet.up_blocks[index].attentions[1].transformer_blocks[0]

    hidden_states_shape = torch.Size([N, C, H, W])
    hidden_states = torch.rand(hidden_states_shape) * 0.01
    encoder_hidden_states_shape = [1, 2, 77, 768]
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    timestep = None
    attention_mask = None
    cross_attention_kwargs = None
    class_labels = None

    torch_output = basic_transformer(hidden_states.squeeze(0), encoder_hidden_states.squeeze(0))

    parameters = preprocess_model_parameters(initialize_model=lambda: basic_transformer, device=device)
    model = tt2_ttnn_basic_transformer_block(device, parameters)

    hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    hidden_states = ttnn.to_device(hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_output = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        class_labels=class_labels,
        config=config,
        attention_head_dim=attention_head_dim,
    )

    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output.unsqueeze(0), ttnn_output, pcc=0.98)
