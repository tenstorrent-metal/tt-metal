# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
import ttnn
from models.demos.falcon7b.tt.falcon_causallm import TtFalconCausalLM
from models.demos.falcon7b.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
)

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc

from .common import create_custom_preprocessor

torch.manual_seed(0)


def run_test_FalconCausalLM_inference(
    device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    pcc,
    model_config,
    model_location_generator,
):
    configuration = transformers.FalconConfig.from_pretrained(model_version)
    model = transformers.models.falcon.modeling_falcon.FalconCausalLM(configuration).eval()

    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    model_input = torch.arange(seq_len * batch).reshape(batch, seq_len)

    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        past_key_values = None
        tt_layer_past = ()
        k_cache = torch.zeros(batch, configuration.max_position_embeddings, head_dim)
        v_cache = torch.zeros(batch, configuration.max_position_embeddings, head_dim)
        for i in range(num_layers):
            tt_k_cache = ttnn.from_torch(
                k_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
            )
            tt_v_cache = ttnn.from_torch(
                v_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
            )
            tt_layer_past += ((tt_k_cache, tt_v_cache),)
        position_ids = None

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        past_key_values = ()
        tt_layer_past = ()
        for i in range(num_layers):
            k_cache = torch.rand(batch, 1, kv_cache_len, head_dim)
            v_cache = torch.rand(batch, 1, kv_cache_len, head_dim)
            past_key_values += ((k_cache, v_cache),)

            tt_k_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
            tt_v_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
            tt_k_cache[:, :, :kv_cache_len, :] = k_cache
            tt_v_cache[:, :, :kv_cache_len, :] = v_cache
            tt_k_cache = ttnn.from_torch(
                tt_k_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
            )
            tt_v_cache = ttnn.from_torch(
                tt_v_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
            )
            tt_layer_past += ((tt_k_cache, tt_v_cache),)

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    # Prepare output -----------------------------------------------------------------------
    pytorch_FalconCausalLM = PytorchFalconCausalLM(hugging_face_reference_model, num_layers)
    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=model_input, past_key_values=past_key_values, use_cache=use_cache
    )

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    parameters = preprocess_model_parameters(
        initialize_model=lambda: hugging_face_reference_model,
        device=device,
    )
    tt_FalconCausalLM = TtFalconCausalLM(
        device,
        num_layers,
        configuration,
        max_position_embeddings,
        model_config,
        parameters=parameters,
    )

    # TODO: Generate embeddings and attention_mask on device
    if llm_mode == "prefill":
        tt_outs = []
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_embeddings=tt_embeddings[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=True,
            )
            tt_outs.append(ttnn.to_torch(tt_out).squeeze(1))
        tt_out = torch.vstack(tt_outs)

    elif llm_mode == "decode":
        tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=True,
        )
        tt_out = ttnn.to_torch(tt_out).squeeze(1)
        tt_out = tt_out.transpose(0, 1)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output: {output_pcc}")

    for i in range(num_layers):
        tt_layer_pres = (
            ttnn.to_torch(tt_layer_present[i][0]),
            ttnn.to_torch(tt_layer_present[i][1]),
        )
        if llm_mode == "prefill":
            pytorch_layer_pres = pytorch_layer_present[i]
            tt_layer_pres = (
                tt_layer_pres[0][:, :, :kv_len, :],
                tt_layer_pres[1][:, :, :kv_len, :],
            )
        elif llm_mode == "decode":
            pytorch_layer_pres = (
                pytorch_layer_present[i][0][:, :, kv_cache_len, :],
                pytorch_layer_present[i][1][:, :, kv_cache_len, :],
            )
            tt_layer_pres = (
                tt_layer_pres[0][:, :, kv_cache_len, :],
                tt_layer_pres[1][:, :, kv_cache_len, :],
            )

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[0], tt_layer_pres[0], pcc)
        logger.info(f"K Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[1], tt_layer_pres[1], pcc)
        logger.info(f"V Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

    if does_pass:
        logger.info("Falcon CausalLM Passed!")
    else:
        logger.warning("Falcon CausalLM Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 2, 128, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq128", "decode_batch32"],
)
@pytest.mark.parametrize(
    "num_layers, pcc",
    ((2, 0.98), (32, 0.86)),
    ids=["layers_2", "layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_FalconCausalLM_inference(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    pcc,
    request,
    model_config_str,
    model_location_generator,
    device,
):
    model_config = get_model_config(model_config_str)

    tt_lib.profiler.set_profiler_location(f"falcon-7b_{request.node.callspec.id}")

    run_test_FalconCausalLM_inference(
        device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        pcc,
        model_config,
        model_location_generator,
    )
