# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.demos.ttnn_falcon7b.tt.falcon_mlp import TtFalconMLP
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config, get_tt_cache_path
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
import transformers

from loguru import logger


@pytest.mark.parametrize(
    "model_name, batch, seq_len, expected_pcc",
    (
        (
            "tiiuae/falcon-7b-instruct",
            1,
            128,
            0.995,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_ttnn_falcon_mlp(
    device,
    model_name,
    batch,
    seq_len,
    expected_pcc,
    model_config_str,
):
    torch.manual_seed(0)

    configuration = transformers.FalconConfig.from_pretrained(model_name)
    torch_model = transformers.models.falcon.modeling_falcon.FalconMLP(configuration).eval()
    torch_input = (torch.rand(batch, 1, seq_len, configuration.hidden_size) * 2) - 1
    torch_output = torch_model(torch_input)

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

    ttnn_model = TtFalconMLP(device, model_config, parameters)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=model_config["DEFAULT_DTYPE"],
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_output = ttnn_model(ttnn_input)

    passed, pcc = assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output).to(torch_output.dtype), expected_pcc)
    logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")
