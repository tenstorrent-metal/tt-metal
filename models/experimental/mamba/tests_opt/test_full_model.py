# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.experimental.mamba.tt_opt.full_model import MambaTT
from models.experimental.mamba.tt_opt import model_config
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


class MambaPytorch(torch.nn.Module):
    def __init__(self, hf_reference_model, num_layers=None):
        super().__init__()
        self.embedding = hf_reference_model.embedding

        if num_layers is None:
            self.layers = hf_reference_model.layers
        else:
            self.layers = hf_reference_model.layers[:num_layers]

        self.norm_f = hf_reference_model.norm_f
        self.lm_head = hf_reference_model.lm_head

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        x = self.lm_head(x)
        return x


@pytest.mark.parametrize(
    "model_version, batch, pcc",
    (
        (
            "state-spaces/mamba-370m",
            32,
            0.99,
        ),
    ),
)
def test_mamba_model_inference(
    model_version: MambaPretrainedModelName,
    batch: int,
    pcc: float,
):
    torch.manual_seed(10)

    reference_model = MambaDecode.from_pretrained(model_version, batch_size=batch)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    input_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
    input_ids = input_ids.repeat(batch, 1)

    reference_output = MambaPytorch(reference_model)(input_ids)

    device = ttnn.open_device(device_id=0)

    config = model_config.create_model_config(batch, reference_model.args.d_model)
    mamba_model = MambaTT(reference_model, device, config, tt_cache_path=f"/tmp/{model_version}")

    tt_output = mamba_model(input_ids)

    ttnn.close_device(device)

    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba SSM output failed")
        assert does_pass, f"PCC value is lower than {pcc}"
