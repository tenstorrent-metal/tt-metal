# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import functional as F
import transformers
import pytest
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import comp_pcc, divup
import tt_lib as ttl
import ttnn

from models.demos.ttnn_falcon7b.tt.falcon_rotary_embedding import TtFalconRotaryEmbedding
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config, get_tt_cache_path
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor

torch.manual_seed(0)


@pytest.mark.parametrize(
    "model_name, input_shape, expected_pcc",
    (
        (
            "tiiuae/falcon-7b-instruct",
            (1, 1, 128, 64),
            0.9999,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_ttnn_falcon_rotary_embeddings(
    device,
    model_name,
    input_shape,
    expected_pcc,
    model_config_str,
):
    torch.manual_seed(0)

    config = transformers.FalconConfig.from_pretrained(model_name)
    torch_model = transformers.models.falcon.modeling_falcon.FalconRotaryEmbedding(config.head_dim).eval()

    batch, num_kv_heads, query_length, head_dim = input_shape
    torch_value_layer = torch.rand(batch, num_kv_heads, query_length, head_dim, dtype=torch.float32)
    torch_query_layer = torch.rand(batch, config.num_attention_heads, query_length, head_dim, dtype=torch.float32)
    torch_key_layer = torch.rand(batch, num_kv_heads, query_length, head_dim, dtype=torch.float32)
    torch_cos, torch_sin = torch_model.forward(torch_value_layer, seq_len=query_length)
    torch_query_embed, torch_key_embed = transformers.models.falcon.modeling_falcon.apply_rotary_pos_emb(
        torch_query_layer, torch_key_layer, torch_cos, torch_sin, None
    )

    model_config = get_model_config(model_config_str)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=device,
        custom_preprocessor=create_custom_preprocessor(
            model_config,
            tt_cache_path=get_tt_cache_path(f"{model_name}"),
            device=device,
        ),
    )

    ttnn_model = TtFalconRotaryEmbedding(parameters, model_config=model_config)

    ttnn_query_layer = ttnn.from_torch(
        torch_query_layer, device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
    )
    ttnn_key_layer = ttnn.from_torch(
        torch_key_layer, device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
    )
    ttnn_query_embed = ttnn_model(ttnn_query_layer)
    ttnn_key_embed = ttnn_model(ttnn_key_layer)

    query_embed_pcc = assert_with_pcc(torch_query_embed, ttnn.to_torch(ttnn_query_embed), expected_pcc)
    key_embed_pcc = assert_with_pcc(torch_key_embed, ttnn.to_torch(ttnn_key_embed), expected_pcc)
    logger.success(f"Query Embeddings Passed: pcc: {query_embed_pcc}, expected: {expected_pcc}")
    logger.success(f"Key Embeddings Passed: pcc: {key_embed_pcc}, expected: {expected_pcc}")
