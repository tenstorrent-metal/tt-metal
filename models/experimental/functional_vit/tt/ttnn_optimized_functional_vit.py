# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import transformers
import torch

import ttnn
from ttnn.model_preprocessing import (
    preprocess_linear_weight,
    preprocess_linear_bias,
)

from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask

# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/vit/modeling_vit.py


def vit_patch_embeddings(
    config,
    pixel_values,
    *,
    parameters,
):
    batch_size, img_c, img_h, img_w = pixel_values.shape
    patch_size = 16
    patch_count = img_h // patch_size  # 14
    patch_size_sq = int(patch_size * patch_size)  # 256
    patch_size_sq_trpl = int(patch_size_sq * img_c)  # 768
    patch_count_sq = int(patch_count * patch_count)  # 196

    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.ROW_MAJOR_LAYOUT)

    pixel_values = ttnn.reshape(pixel_values, (1, img_c, img_h, patch_count, patch_size))
    pixel_values = ttnn.reshape(pixel_values, (1, img_c, patch_count, patch_size, patch_count, patch_size))
    pixel_values = ttnn.reshape(pixel_values, (1, img_c, patch_count, patch_count, patch_size, patch_size))
    # pixel_values = ttnn.permute(pixel_values, (0, 1, 2, 4, 3, 5))
    pixel_values = ttnn.reshape(pixel_values, (1, img_c, patch_count_sq, patch_size_sq))
    pixel_values = ttnn.permute(pixel_values, (0, 2, 1, 3))
    pixel_values = ttnn.reshape(pixel_values, (1, patch_count_sq, patch_size_sq_trpl))

    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)
    # pixel_values = pixel_values @ parameters.projection.weight
    # pixel_values = pixel_values + parameters.projection.bias

    patch_embedding_output = ttnn.linear(
        pixel_values,
        parameters.projection.weight,
        bias=parameters.projection.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=2, x=2),
    )
    ttnn.deallocate(pixel_values)

    return patch_embedding_output


def vit_embeddings(
    config,
    pixel_values,
    position_embeddings,
    cls_tokens,
    *,
    parameters,
):
    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters=parameters.patch_embeddings)
    embedding_output = ttnn.concat((cls_tokens, patch_embeddings), dim=1)
    embedding_output = embedding_output + position_embeddings
    embedding_output = ttnn.pad(embedding_output, ((0, 0), (0, 27), (0, 0)), 0)

    return embedding_output


def vit_layernorm_before(
    config,
    hidden_states,
    *,
    parameters,
):
    attention_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return attention_output


def vit_layernorm_after(
    config,
    hidden_states,
    *,
    parameters,
):
    attention_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return attention_output


def vit_attention(
    config,
    hidden_states,
    attention_mask,
    query_key_value_weight,
    query_key_value_bias,
    self_output_weight,
    self_output_bias,
    *,
    num_cores_x=8,
):
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    batch_size = 8

    query_key_value_output = ttnn.linear(
        hidden_states,
        query_key_value_weight,
        bias=query_key_value_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=2, x=2),
    )

    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value_output,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value_output)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=2, x=2),
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attention_probs = ttnn.transformer.attention_softmax_(
        attention_scores, attention_mask=attention_mask, head_size=head_size
    )

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=2, x=2),
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        self_output_weight,
        bias=self_output_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=2, x=2),
    )
    ttnn.deallocate(context_layer)
    self_output = ttnn.reallocate(self_output)

    return self_output


def vit_intermediate(
    hidden_states,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=2, x=2),
        activation="gelu",
    )
    ttnn.deallocate(hidden_states)

    return output


def vit_output(
    config,
    hidden_states,
    residual,
    *,
    parameters,
):
    # output = hidden_states @ parameters.dense.weight
    # output = output + parameters.dense.bias

    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=2, x=2),
    )
    ttnn.deallocate(hidden_states)

    output = output + residual
    ttnn.deallocate(residual)

    return output


def vit_feedforward(
    config,
    hidden_states,
    attention_output,
    *,
    parameters,
):
    intermediate = vit_intermediate(hidden_states, parameters=parameters.intermediate)
    hidden_states = vit_output(config, intermediate, attention_output, parameters=parameters.output)
    return hidden_states


def vit_layer(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    layernorm_before_output = vit_layernorm_before(
        config,
        hidden_states,
        parameters=parameters,
    )
    attention_output = vit_attention(
        config,
        layernorm_before_output,
        attention_mask,
        parameters.attention.attention.query_key_value.weight,
        parameters.attention.attention.query_key_value.bias,
        parameters.attention.output.dense.weight,
        parameters.attention.output.dense.bias,
    )
    attention_output = attention_output + hidden_states
    layernorm_after_output = vit_layernorm_after(
        config,
        attention_output,
        parameters=parameters,
    )
    feedforward_output = vit_feedforward(
        config,
        layernorm_after_output,
        attention_output,
        parameters=parameters,
    )

    return feedforward_output


def vit_encoder(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    encoder_input = hidden_states
    encoder_output = None
    for encoder_parameters in parameters.layer:
        encoder_output = vit_layer(
            config,
            encoder_input,
            attention_mask,
            parameters=encoder_parameters,
        )
        encoder_input = encoder_output
    return encoder_output


def vit(
    config,
    pixel_values,
    position_embeddings,
    cls_tokens,
    attention_mask,
    *,
    parameters,
):
    embeddings_output = vit_embeddings(
        config, pixel_values, position_embeddings, cls_tokens, parameters=parameters.embeddings
    )

    hidden_states = vit_encoder(
        config,
        embeddings_output,
        attention_mask=None,
        parameters=parameters.encoder,
    )

    # Final LayerNorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm.weight,
        bias=parameters.layernorm.bias,
    )

    # Pooler
    pooler_output = output @ parameters.pooler.dense.weight
    pooler_output = pooler_output + parameters.pooler.dense.bias
    pooler_output = ttnn.tanh(pooler_output)

    return pooler_output


def preprocess_inputs(
    input_ids,
    token_type_ids,
    attention_mask,
    device,
):
    batch_size, _ = input_ids.shape

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, 0, 0, 0, 0, 0, 0, batch_size - 1))
        attention_mask = ttnn.from_torch(
            attention_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    return input_ids, token_type_ids, attention_mask


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.vit.modeling_vit.ViTPatchEmbeddings):
        weight = torch_model.projection.weight
        bias = torch_model.projection.bias

        three_times_hidden_size, _, _, _ = weight.shape
        hidden_size = three_times_hidden_size // 3

        preprocessed_weight = torch.reshape(weight, (three_times_hidden_size, 3, hidden_size))
        preprocessed_weight = torch.permute(preprocessed_weight, (1, 2, 0))
        preprocessed_weight = torch.reshape(preprocessed_weight, (three_times_hidden_size, three_times_hidden_size))

        parameters = {"projection": {}}
        parameters["projection"]["weight"] = ttnn.from_torch(
            preprocessed_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        parameters["projection"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    if hasattr(torch_model, "query") and hasattr(torch_model, "key") and hasattr(torch_model, "value"):
        qkv_weight = torch.cat(
            [
                torch_model.query.weight,
                torch_model.key.weight,
                torch_model.value.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [torch_model.query.bias, torch_model.key.bias, torch_model.value.bias],
            dim=0,
        )

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat16)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat16)

    return parameters
