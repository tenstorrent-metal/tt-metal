# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from models.experimental.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.experimental.mamba.tt_opt.mamba_one_step_ssm import TtMambaSSM
from models.experimental.mamba.tt_opt import model_config
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


class PytorchMambaSSM(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.block = hf_reference_model.layers[layer_num].mixer
        self.block.eval()

    def forward(self, x):
        result = self.block.ssm(x)
        return result


@pytest.mark.parametrize(
    "model_version, batch, pcc",
    (
        (
            "state-spaces/mamba-370m",
            1,
            0.99,
        ),
    ),
)
def test_mamba_ssm_inference(
    model_version: MambaPretrainedModelName,
    batch,
    pcc: float,
):
    torch.manual_seed(0)

    LAYER_NUM = 0

    reference_model = MambaDecode.from_pretrained(model_version)
    reference_model.args.batch_size = batch

    d_in = reference_model.args.d_model * reference_model.args.expand
    input = torch.rand(batch, 1, d_in)

    reference_output = PytorchMambaSSM(reference_model, LAYER_NUM)(input)

    residual_block = reference_model.layers[LAYER_NUM]
    assert not isinstance(residual_block, torch.Tensor), "Expected torch.Module"

    device = ttnn.open_device(device_id=0)

    config = model_config.create_model_config(batch, d_in)

    model = TtMambaSSM(reference_model.args, device, residual_block.state_dict(), batch, d_in, config)
    tt_input = input.view(1, 1, batch, d_in)
    tt_input = ttnn.to_device(
        ttnn.from_torch(tt_input, layout=ttnn.TILE_LAYOUT), device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_output = model(tt_input)
    tt_output = ttnn.to_torch(tt_output)
    ttnn.close_device(device)
    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba SSM output failed")
        assert does_pass, f"PCC value is lower than {pcc}"