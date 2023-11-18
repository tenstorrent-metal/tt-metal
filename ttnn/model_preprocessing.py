# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import pathlib
from typing import Optional

from loguru import logger
import numpy as np
import torch

import ttnn

TILE_HEIGHT = 32
TILE_WIDTH = 32


@dataclass
class ParametersConfig:
    linear_weight_dtype: ttnn.DataType = ttnn.bfloat16
    linear_bias_dtype: ttnn.DataType = ttnn.bfloat16
    layernorm_parameter_dtype: ttnn.DataType = ttnn.bfloat16
    embedding_weight_dtype: ttnn.DataType = ttnn.bfloat16


def pad_tensor(tensor, height_multiple=TILE_HEIGHT, width_multiple=TILE_WIDTH):
    if len(tensor.shape) > 1:
        *_, height, width = tensor.shape
        if height % height_multiple == 0 and width % width_multiple == 0:
            return tensor

        padded_height = int(np.ceil(height / height_multiple)) * height_multiple
        padded_width = int(np.ceil(width / width_multiple)) * width_multiple
        tensor = ttnn.core._reshape_to_4D(tensor)
        *batch_sizes, _, _ = tensor.shape
        tensor = ttnn.Tensor(tensor._tensor.pad(batch_sizes + [padded_height, padded_width], [0, 0, 0, 0], 0.0))
        # tensor = ttnn.reshape(tensor, tuple(original_batch_sizes + [padded_height, padded_width])) # TODO: re-enable this line once the correct `shape_without_padding` is being returned
    else:
        (width,) = tensor.shape
        if width % width_multiple == 0:
            return tensor

        padded_width = int(np.ceil(width / width_multiple)) * width_multiple
        tensor = ttnn.core._reshape_to_4D(tensor)
        *batch_sizes, height, _ = tensor.shape
        tensor = ttnn.Tensor(tensor._tensor.pad(batch_sizes + [height, padded_width], [0, 0, 0, 0], 0.0))
        # tensor = ttnn.reshape(tensor, (padded_width,)) # TODO: re-enable this line once the correct `shape_without_padding` is being returned
    return tensor


def preprocess_linear_weight(parameters_config, weight):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=parameters_config.linear_weight_dtype)
    weight = pad_tensor(weight)
    weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT)
    return weight


def preprocess_linear_bias(parameters_config, bias):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=parameters_config.linear_bias_dtype)
    bias = pad_tensor(bias)
    bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
    return bias


def preprocess_layernorm_parameter(parameters_config, parameter):
    parameter = parameter.reshape((1, -1))
    parameter = ttnn.from_torch(parameter, dtype=parameters_config.layernorm_parameter_dtype)
    parameter = pad_tensor(parameter)
    parameter = ttnn.to_layout(parameter, ttnn.TILE_LAYOUT)
    return parameter


def preprocess_embedding_weight(parameters_config, weight):
    weight = ttnn.from_torch(weight, dtype=parameters_config.embedding_weight_dtype)
    return weight


def default_preprocessor(parameters_config, torch_model, full_name):
    parameters = {}
    if isinstance(torch_model, torch.nn.Linear):
        parameters[f"{full_name}weight"] = preprocess_linear_weight(parameters_config, torch_model.weight)
        if torch_model.bias is not None:
            parameters[f"{full_name}bias"] = preprocess_linear_bias(parameters_config, torch_model.bias)
    elif isinstance(torch_model, torch.nn.LayerNorm):
        parameters[f"{full_name}weight"] = preprocess_layernorm_parameter(parameters_config, torch_model.weight)
        parameters[f"{full_name}bias"] = preprocess_layernorm_parameter(parameters_config, torch_model.bias)
    elif isinstance(torch_model, torch.nn.Embedding):
        parameters[f"{full_name}weight"] = preprocess_embedding_weight(parameters_config, torch_model.weight)
    return parameters


def _preprocess_model_parameters(
    model_name,
    parameters_config,
    torch_model,
    *,
    prefix="",
    is_to_be_converted,
    custom_preprocessor=None,
):
    parameters = {}

    named_children = list(torch_model.named_children())

    if not named_children:
        for name, parameter in torch_model.named_parameters():
            full_name = f"{prefix}{name}"
            parameters[full_name] = parameter

    for name, child in named_children:
        full_name = f"{prefix}{name}."

        use_default_preprocessor = True
        if custom_preprocessor is not None:
            custom_preprocessor_parameters = custom_preprocessor(
                parameters_config=parameters_config,
                torch_model=child,
                full_name=full_name,
            )
            if custom_preprocessor_parameters:
                parameters.update(custom_preprocessor_parameters)
                # Custom preprocessor didn't handle this case, so, try using default preprocessor
                use_default_preprocessor = False

        if use_default_preprocessor:
            if not is_to_be_converted(child, full_name):
                child_parameters = _preprocess_model_parameters(
                    model_name,
                    parameters_config,
                    child,
                    prefix=full_name,
                    is_to_be_converted=is_to_be_converted,
                    custom_preprocessor=custom_preprocessor,
                )
                parameters.update(child_parameters)
            else:
                default_preprocessor_parameters = default_preprocessor(parameters_config, child, full_name)
                if default_preprocessor_parameters:
                    parameters.update(default_preprocessor_parameters)
                else:
                    child_parameters = _preprocess_model_parameters(
                        model_name,
                        parameters_config,
                        child,
                        prefix=full_name,
                        is_to_be_converted=is_to_be_converted,
                        custom_preprocessor=custom_preprocessor,
                    )
                    parameters.update(child_parameters)

    return parameters


def _load_parameters(model_cache_path: pathlib.Path) -> dict:
    parameters = {}
    logger.info(f"Loading model weights from cache: {model_cache_path}")
    for file_name in model_cache_path.glob("*"):
        extension = file_name.suffix
        name = file_name.stem
        if extension == ".bin":
            parameters[name] = ttnn.load_tensor(file_name)
        elif extension == ".pt":
            parameters[name] = torch.load(file_name)
        else:
            raise RuntimeError("Unrecognized file extension!")
    return parameters


def _dump_parameters(model_cache_path: pathlib.Path, parameters: dict) -> None:
    model_cache_path.mkdir(parents=True)
    for name, tensor in parameters.items():
        file_path = str(model_cache_path / name)
        if isinstance(tensor, ttnn.Tensor):
            file_name = file_path + ".bin"
            ttnn.dump_tensor(file_name, tensor)
        elif isinstance(tensor, torch.nn.Parameter):
            file_name = file_path + ".pt"
            torch.save(tensor, file_name)
        else:
            raise RuntimeError(f"Unsupported type: {type(tensor)}")


def preprocess_model_parameters(
    model_name,
    parameters_config,
    torch_model,
    *,
    prefix="",
    is_to_be_converted,
    custom_preprocessor=None,
    device: Optional[ttnn.Device] = None,
    use_model_cache: bool = False,
):
    model_cache_path = ttnn.MODEL_CACHE_PATH / model_name

    if use_model_cache and model_cache_path.exists():
        parameters = _load_parameters(model_cache_path)
    else:
        parameters = _preprocess_model_parameters(
            model_name,
            parameters_config,
            torch_model,
            prefix=prefix,
            is_to_be_converted=is_to_be_converted,
            custom_preprocessor=custom_preprocessor,
        )
        if use_model_cache:
            _dump_parameters(model_cache_path, parameters)

    if device is not None:
        for name, parameter in list(parameters.items()):
            if isinstance(parameter, ttnn.Tensor):
                parameters[name] = ttnn.to_device(parameter, device)
            else:
                parameters[name] = parameter

    return parameters
