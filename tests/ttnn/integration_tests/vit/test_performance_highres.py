# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from loguru import logger
import torch
import math
import transformers
from datasets import load_dataset
from transformers import AutoImageProcessor

import ttnn

from models.experimental.functional_vit.tt import ttnn_functional_vit_highres
from models.experimental.functional_vit.tt import ttnn_optimized_functional_vit_highres

from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report


def get_expected_times(functional_vit):
    return {
        ttnn_functional_vit_highres: (12, 17),
        ttnn_optimized_functional_vit_highres: (12, 0.08),
    }[functional_vit]


def interpolate_pos_encoding(
    position_embeddings: torch.Tensor, patch_size, num_patches, height: int, width: int
) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
    resolution images.

    Source:
    https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
    """

    # num_patches = embeddings.shape[1] - 1
    num_positions = position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
        return position_embeddings
    class_pos_embed = position_embeddings[:, 0]
    patch_pos_embed = position_embeddings[:, 1:]
    dim = position_embeddings.shape[-1]
    h0 = height // patch_size
    w0 = width // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    h0, w0 = h0 + 0.1, w0 + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed,
        scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
        mode="bicubic",
        align_corners=False,
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [1280])
# @pytest.mark.parametrize("functional_vit", [ttnn_functional_vit_highres, ttnn_optimized_functional_vit_highres])
@pytest.mark.parametrize("functional_vit", [ttnn_optimized_functional_vit_highres])
# @pytest.mark.parametrize("functional_vit", [ ttnn_functional_vit_highres])
def test_performance(device, use_program_cache, model_name, batch_size, image_size, functional_vit):
    disable_persistent_kernel_cache()

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTModel(config).eval()

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0].resize((image_size, image_size))
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_pixel_values = image_processor(
        image, return_tensors="pt", do_resize=False, do_center_crop=False
    ).pixel_values.to(torch.bfloat16)

    # torch_pixel_values = torch.rand((1, 3, 1280, 1280))
    torch_attention_mask = (
        None  # torch.zeros(1, sequence_size) if functional_vit == ttnn_optimized_functional_vit else None
    )

    if functional_vit == ttnn_functional_vit_highres:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_vit == ttnn_optimized_functional_vit_highres:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_vit: {functional_vit}")

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        custom_preprocessor=functional_vit.custom_preprocessor,
        device=device,
    )

    # TODO: integrate it in preprocess_model_parameters
    model_state_dict = model.state_dict()
    torch_cls_token = torch.nn.Parameter(model_state_dict["embeddings.cls_token"]).to(torch.bfloat16)
    init_position_embeddings = torch.nn.Parameter(model_state_dict["embeddings.position_embeddings"]).to(torch.bfloat16)
    patch_size = 16
    tot_patch_count = (image_size // patch_size) * (image_size // patch_size)
    torch_position_embeddings = torch.nn.Parameter(
        interpolate_pos_encoding(init_position_embeddings, patch_size, tot_patch_count, image_size, image_size)
    )

    cls_token = ttnn.from_torch(torch_cls_token, layout=ttnn.TILE_LAYOUT, device=device)
    position_embeddings = ttnn.from_torch(torch_position_embeddings, layout=ttnn.TILE_LAYOUT, device=device)

    torch_pixel_values = torch_pixel_values.to(torch.bfloat16)
    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    durations = []
    for _ in range(2):
        start = time.time()
        tt_output = functional_vit.vit(
            config,
            pixel_values,
            position_embeddings,
            cls_token,
            attention_mask=None,
            parameters=parameters,
        )
        tt_output = ttnn.from_device(tt_output)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times(functional_vit)
    prep_perf_report(
        model_name=tt_model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")
