# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline
import pytest
from tqdm.auto import tqdm

from tests.ttnn.utils_for_testing import assert_with_pcc
from diffusers import LMSDiscreteScheduler
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_stable_diffusion.custom_preprocessing import custom_preprocessor
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_unet_2d_condition_model import (
    UNet2DConditionModel,
)
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_unet_2d_condition_model import (
    UNet2DConditionModel as UNet2D,
)

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)

scheduler.set_timesteps(1)


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def constant_prop_time_embeddings(timesteps, batch_size, time_proj):
    timesteps = timesteps[None]
    timesteps = timesteps.expand(batch_size)
    t_emb = time_proj(timesteps)
    return t_emb


@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width",
    [
        (2, 4, 32, 32),
    ],
)
def test_unet_2d_condition_model_256x256(device, batch_size, in_channels, input_height, input_width):
    # setup pytorch model
    torch.manual_seed(0)
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    model = pipe.unet
    model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=custom_preprocessor, device=device
    )

    timestep_shape = [1, 1, 2, 320]
    encoder_hidden_states_shape = [1, 2, 77, 768]
    class_labels = None
    attention_mask = None
    cross_attention_kwargs = None
    return_dict = True
    config = model.config

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]

    input = torch.randn(hidden_states_shape)
    timestep = [i for i in tqdm(scheduler.timesteps)][0]
    ttnn_timestep = constant_prop_time_embeddings(timestep, batch_size, model.time_proj)
    ttnn_timestep = ttnn_timestep.unsqueeze(0).unsqueeze(0)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    torch_output = model(input, timestep=timestep, encoder_hidden_states=encoder_hidden_states.squeeze(0)).sample

    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, ttnn.TILE_LAYOUT)
    input = ttnn.to_device(input, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_timestep = ttnn.from_torch(ttnn_timestep, ttnn.bfloat16)
    ttnn_timestep = ttnn.to_layout(ttnn_timestep, ttnn.TILE_LAYOUT)
    ttnn_timestep = ttnn.to_device(ttnn_timestep, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, ttnn.bfloat16)
    encoder_hidden_states = ttnn.to_layout(encoder_hidden_states, ttnn.TILE_LAYOUT)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_output = UNet2DConditionModel(
        input,
        timestep=ttnn_timestep,
        encoder_hidden_states=encoder_hidden_states,
        class_labels=class_labels,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        parameters=parameters,
        device=device,
        config=config,
    )
    ttnn_output = ttnn_to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)


@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width",
    [
        (2, 4, 64, 64),
    ],
)
def test_unet_2d_condition_model_512x512(device, batch_size, in_channels, input_height, input_width):
    # setup pytorch model
    torch.manual_seed(0)
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    model = pipe.unet
    model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=custom_preprocessor, device=device
    )

    timestep_shape = [1, 1, 2, 320]
    encoder_hidden_states_shape = [1, 2, 77, 768]
    class_labels = None
    attention_mask = None
    cross_attention_kwargs = None
    return_dict = True
    config = model.config

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]

    input = torch.randn(hidden_states_shape)
    timestep = [i for i in tqdm(scheduler.timesteps)][0]
    ttnn_timestep = constant_prop_time_embeddings(timestep, batch_size, model.time_proj)
    ttnn_timestep = ttnn_timestep.unsqueeze(0).unsqueeze(0)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    torch_output = model(input, timestep=timestep, encoder_hidden_states=encoder_hidden_states.squeeze(0)).sample
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_device(input, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    input = ttnn.to_layout(input, ttnn.TILE_LAYOUT, ttnn.bfloat8_b)

    ttnn_timestep = ttnn.from_torch(ttnn_timestep, ttnn.bfloat16)
    ttnn_timestep = ttnn.to_layout(ttnn_timestep, ttnn.TILE_LAYOUT)
    ttnn_timestep = ttnn.to_device(ttnn_timestep, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, ttnn.bfloat16)
    encoder_hidden_states = ttnn.to_layout(encoder_hidden_states, ttnn.TILE_LAYOUT)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    reader_patterns_cache = {}
    model = UNet2D(device, parameters, batch_size, input_height, input_width, reader_patterns_cache)
    ttnn_output = model(
        input,
        timestep=ttnn_timestep,
        encoder_hidden_states=encoder_hidden_states,
        class_labels=class_labels,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        config=config,
    )
    ttnn_output = ttnn_to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
