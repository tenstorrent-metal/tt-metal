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

from models.experimental.functional_vit.tt import ttnn_functional_vit
from models.experimental.functional_vit.tt import ttnn_optimized_functional_vit

from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report


def get_expected_times(functional_vit):
    return {
        ttnn_functional_vit: (12, 17),
        ttnn_optimized_functional_vit: (12, 0.08),
    }[functional_vit]


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [224])
# @pytest.mark.parametrize("functional_vit", [ttnn_functional_vit, ttnn_optimized_functional_vit])
@pytest.mark.parametrize("functional_vit", [ttnn_optimized_functional_vit])
# @pytest.mark.parametrize("functional_vit", [ttnn_functional_vit])
def test_performance(device, use_program_cache, model_name, batch_size, image_size, functional_vit):
    disable_persistent_kernel_cache()

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTModel(config).eval()

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(torch.bfloat16)

    if functional_vit == ttnn_functional_vit:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_vit == ttnn_optimized_functional_vit:
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
    torch_position_embeddings = torch.nn.Parameter(model_state_dict["embeddings.position_embeddings"]).to(
        torch.bfloat16
    )

    cls_token = ttnn.from_torch(torch_cls_token, layout=ttnn.TILE_LAYOUT, device=device)
    position_embeddings = ttnn.from_torch(torch_position_embeddings, layout=ttnn.TILE_LAYOUT, device=device)

    torch_pixel_values = torch_pixel_values.to(torch.bfloat16)
    pixel_values = ttnn.from_torch(torch_pixel_values, layout=ttnn.TILE_LAYOUT, device=device)

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
