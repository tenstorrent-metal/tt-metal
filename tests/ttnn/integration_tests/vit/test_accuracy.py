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
from models.experimental.vit.vit_helper_funcs import get_data_loader, get_batch

from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report
import ast
from pathlib import Path


def get_expected_times(functional_vit):
    return {
        ttnn_functional_vit: (12, 17),
        ttnn_optimized_functional_vit: (12, 0.08),
    }[functional_vit]


def get_imagenet_label_dict():
    path = "models/sample_data/imagenet_class_labels.txt"
    with open(path, "r") as file:
        class_labels = ast.literal_eval(file.read())
    return class_labels


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [224])
# @pytest.mark.parametrize("functional_vit", [ttnn_functional_vit, ttnn_optimized_functional_vit])
# @pytest.mark.parametrize("functional_vit", [ttnn_optimized_functional_vit])
@pytest.mark.parametrize("functional_vit", [ttnn_functional_vit])
def test_performance(
    device, use_program_cache, model_name, batch_size, image_size, functional_vit, model_location_generator
):
    disable_persistent_kernel_cache()

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model = model.to(torch.bfloat16)

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
        initialize_model=lambda: model,
        custom_preprocessor=functional_vit.custom_preprocessor,
        device=device,
    )

    torch_pixel_values = torch_pixel_values.to(torch.bfloat16)
    pixel_values = ttnn.from_torch(torch_pixel_values, layout=ttnn.TILE_LAYOUT, device=device)

    iterations = 50
    imagenet_label_dict = get_imagenet_label_dict()

    data_loader = get_data_loader("ImageNet_data", batch_size, iterations)
    correct = 0
    for iter in range(iterations):
        predictions = []
        inputs, labels = get_batch(data_loader, image_processor)

        torch_pixel_values = inputs.to(torch.bfloat16)
        tt_inputs = ttnn.from_torch(torch_pixel_values, layout=ttnn.TILE_LAYOUT, device=device)

        tt_output = functional_vit.vit(
            config,
            tt_inputs,
            attention_mask=None,
            parameters=parameters,
        )
        tt_output = ttnn.from_device(tt_output)
        print(tt_output.shape)
        prediction = ttnn.to_torch(tt_output[0][0]).argmax(dim=-1)
        for i in range(batch_size):
            predictions.append(imagenet_label_dict[prediction[i].item()])
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1
        del tt_output, tt_inputs, inputs, labels, predictions

        enable_persistent_kernel_cache()

    accuracy = correct / (batch_size * iterations)
    logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")
