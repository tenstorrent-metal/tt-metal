# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline
import pytest
import ttnn

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_upsample_2d import upsample2d as ttnn_upsample2d
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_upsample_2d import (
    upsample2d as tt2_ttnn_upsample2d,
)
from models.experimental.functional_stable_diffusion.custom_preprocessing import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import torch_random
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    pre_process_input_new,
    post_process_output,
)


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return input


@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index",
    [
        (2, 1280, 4, 4, 0),
        (2, 1280, 8, 8, 1),
        (2, 640, 16, 16, 2),
    ],
)
@pytest.mark.parametrize("scale_factor", [2])
def test_upsample2d_256x256(device, scale_factor, batch_size, in_channels, input_height, input_width, index):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[index]
    resnet_upsampler = unet_upblock.upsamplers[0]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.up_blocks[index].upsamplers[0]

    input_shape = batch_size, in_channels, input_height, input_width
    out_channels = in_channels

    input = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output = resnet_upsampler(input)

    tt_input_tensor = ttnn.from_torch(input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_up = ttnn_upsample2d(
        device,
        tt_input_tensor,
        parameters,
        in_channels,
        out_channels,
        scale_factor,
    )
    torch_up = ttnn.to_torch(tt_up)

    assert_with_pcc(torch_output, torch_up, 0.99)


@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index",
    [
        (2, 1280, 8, 8, 0),
        (2, 1280, 16, 16, 1),
        (2, 640, 32, 32, 2),
    ],
)
@pytest.mark.parametrize("scale_factor", [2])
def test_upsample2d_512x512(device, scale_factor, batch_size, in_channels, input_height, input_width, index):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[index]
    resnet_upsampler = unet_upblock.upsamplers[0]
    reader_patterns_cache = {}

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.up_blocks[index].upsamplers[0]
    model = tt2_ttnn_upsample2d(device, parameters, reader_patterns_cache, batch_size, input_height, input_width)

    input_shape = batch_size, in_channels, input_height, input_width
    out_channels = in_channels

    input = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output = resnet_upsampler(input)

    tt_input_tensor = ttnn.from_torch(input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_input_tensor = pre_process_input_new(device, tt_input_tensor)
    tt_up = model(
        tt_input_tensor,
        in_channels,
        out_channels,
    )
    tt_up = post_process_output(device, tt_up, batch_size, input_height * 2, input_width * 2, in_channels)
    torch_up = ttnn.to_torch(tt_up)

    assert_with_pcc(torch_output, torch_up, 0.99)
