# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.experimental.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.experimental.falcon40b.tt.falcon_causallm import TtFalconCausalLM

from models.experimental.falcon40b.tt.model_config import (
    get_model_config,
)

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    nearest_32,
    skip_for_grayskull,
    get_devices_for_t3000,
)


class PytorchFalconCausalLM(torch.nn.Module):
    def __init__(self, hf_reference_model, num_layers):
        super().__init__()
        self.model = hf_reference_model
        self.model.transformer.h = self.model.transformer.h[:num_layers]

        # Disable dropout
        self.model.eval()

    def forward(self, input_ids, past_key_values, use_cache):
        result = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=False,
        )

        return result


def run_test_FalconCausalLM_inference(
    devices,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    out_pcc,
    cache_pcc,
    token_pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
):
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, num_hidden_layers=num_layers
    )

    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    num_attention_heads = configuration.num_attention_heads
    num_kv_heads = configuration.num_kv_heads
    use_cache = True
    use_global_cos_sin_cache = True

    if 1:
        model_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        model_input = torch.stack([torch.arange(seq_len)] * batch).reshape(batch, seq_len)

    # Generate dummy kv_cache --------------------------------------------------------------
    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        past_key_values = None
        tt_layer_past = ()
        tt_k_cache_host = torch.zeros(batch, num_kv_heads, max_position_embeddings, head_dim)
        tt_v_cache_host = torch.zeros(batch, num_kv_heads, max_position_embeddings, head_dim)
        tt_k_cache_host = torch.chunk(tt_k_cache_host, len(devices), 1)
        tt_v_cache_host = torch.chunk(tt_v_cache_host, len(devices), 1)

        for i in range(num_layers):
            tt_k_cache = []
            tt_v_cache = []
            for i in range(len(devices)):
                tt_k_cache.append(
                    torch2tt_tensor(
                        tt_k_cache_host[i],
                        devices[i],
                        tt_lib.tensor.Layout.TILE,
                        model_config["KV_CACHE_MEMCFG"],
                        model_config["KV_CACHE_DTYPE"],
                    )
                )
                tt_v_cache.append(
                    torch2tt_tensor(
                        tt_v_cache_host[i],
                        devices[i],
                        tt_lib.tensor.Layout.TILE,
                        model_config["KV_CACHE_MEMCFG"],
                        model_config["KV_CACHE_DTYPE"],
                    )
                )
            tt_layer_past += ((tt_k_cache, tt_v_cache),)

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        past_key_values = ()
        tt_layer_past = ()
        for i in range(num_layers):
            k_cache = torch.rand(batch, num_kv_heads, kv_cache_len, head_dim)
            v_cache = torch.rand(batch, num_kv_heads, kv_cache_len, head_dim)
            past_key_values += (
                (
                    torch.repeat_interleave(k_cache, num_attention_heads // num_kv_heads, 1),
                    (torch.repeat_interleave(v_cache, num_attention_heads // num_kv_heads, 1)),
                ),
            )

            tt_k_cache_host = torch.zeros(batch, num_kv_heads, max_position_embeddings, head_dim)
            tt_v_cache_host = torch.zeros(batch, num_kv_heads, max_position_embeddings, head_dim)
            tt_k_cache_host[:, :, :kv_cache_len, :] = k_cache
            tt_v_cache_host[:, :, :kv_cache_len, :] = v_cache
            tt_k_cache_host = torch.chunk(tt_k_cache_host, len(devices), 1)
            tt_v_cache_host = torch.chunk(tt_v_cache_host, len(devices), 1)

            tt_k_cache = []
            tt_v_cache = []
            for j in range(len(devices)):
                tt_k_cache.append(
                    torch2tt_tensor(
                        tt_k_cache_host[j],
                        devices[j],
                        tt_lib.tensor.Layout.TILE,
                        model_config["KV_CACHE_MEMCFG"],
                        model_config["KV_CACHE_DTYPE"],
                    )
                )
                tt_v_cache.append(
                    torch2tt_tensor(
                        tt_v_cache_host[j],
                        devices[j],
                        tt_lib.tensor.Layout.TILE,
                        model_config["KV_CACHE_MEMCFG"],
                        model_config["KV_CACHE_DTYPE"],
                    )
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
    tt_FalconCausalLM = TtFalconCausalLM(
        devices,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        use_global_cos_sin_cache=use_global_cos_sin_cache,
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
                use_cache=use_cache,
            )
            tt_outs.append(torch.cat([tt2torch_tensor(tt_o).squeeze(1) for tt_o in tt_out], -1))
        tt_out = torch.vstack(tt_outs)

    elif llm_mode == "decode":
        attention_mask_memconfig = model_config["ATTN_MASK_MEMCFG"]
        num_max_tokens = nearest_32(kv_cache_len + 1)
        if attention_mask_memconfig.is_sharded():
            attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
            attn_mask_shard_shape[-1] = num_max_tokens
            attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

        tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
        tt_embeddings = [
            tt_embeddings[i].to(devices[i], model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"]) for i in range(len(devices))
        ]
        tt_attention_mask = [tt_attention_mask[i].to(devices[i], attention_mask_memconfig) for i in range(len(devices))]
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        tt_out = torch.cat([tt2torch_tensor(tt_o).squeeze(1) for tt_o in tt_out], -1)
        tt_out = tt_out.transpose(0, 1)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, out_pcc)
    logger.info(f"Output: {output_pcc}")

    for i in range(num_layers):
        pytorch_layer_pres = pytorch_layer_present[i]
        tt_layer_pres = (
            torch.cat([tt2torch_tensor(tt_layer_p) for tt_layer_p in tt_layer_present[i][0]], 1),
            torch.cat([tt2torch_tensor(tt_layer_p) for tt_layer_p in tt_layer_present[i][1]], 1),
        )
        tt_layer_pres = (
            torch.repeat_interleave(
                tt_layer_pres[0][:, :, :kv_len, :],
                configuration.num_attention_heads // configuration.num_kv_heads,
                1,
            ),
            torch.repeat_interleave(
                tt_layer_pres[1][:, :, :kv_len, :],
                configuration.num_attention_heads // configuration.num_kv_heads,
                1,
            ),
        )

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[0], tt_layer_pres[0], cache_pcc)
        logger.info(f"K Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[1], tt_layer_pres[1], cache_pcc)
        logger.info(f"V Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

        if llm_mode == "decode":
            does_pass2, output_pcc = comp_pcc(
                pytorch_layer_pres[0][:, :, kv_len - 1 : kv_len, :],
                tt_layer_pres[0][:, :, kv_len - 1 : kv_len, :],
                token_pcc,
            )
            logger.info(f"K Cache Layer {i} new token: {output_pcc}")

            does_pass = does_pass and does_pass2

            does_pass2, output_pcc = comp_pcc(
                pytorch_layer_pres[1][:, :, kv_len - 1 : kv_len, :],
                tt_layer_pres[1][:, :, kv_len - 1 : kv_len, :],
                token_pcc,
            )
            logger.info(f"V Cache Layer {i} new token: {output_pcc}")

            does_pass = does_pass and does_pass2

    if does_pass:
        logger.info("Falcon CausalLM Passed!")
    else:
        logger.warning("Falcon CausalLM Failed!")
        assert does_pass


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", (4, 8))
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 2, 32, 0),
        ("prefill", 2, 64, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq32", "prefill_seq64", "decode_batch32"],
)
@pytest.mark.parametrize(
    "num_layers",
    (1,),
    ids=["layers_1"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-40b-instruct",),
    ids=["falcon_40b"],
)
@pytest.mark.parametrize(
    "model_config_str, out_pcc, cache_pcc, token_pcc",
    [("BFLOAT8_B-SHARDED", 0.99, 0.99, 0.99), ("BFLOAT16-SHARDED", 0.99, 0.99, 0.99)],
)
def test_FalconCausalLM_inference(
    num_devices,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    out_pcc,
    cache_pcc,
    token_pcc,
    request,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    all_devices,
    # use_program_cache, # TODO: enable program cache as soon as multi chip correctness is verified
):
    if llm_mode == "prefill":
        if model_config_str == "BFLOAT16-SHARDED":
            pytest.skip("Prefill is only tested for BFLOAT8_B!")
        elif model_config_str == "BFLOAT8_B-SHARDED":
            # TODO: Investigate why PCC is lower for prefill?
            out_pcc = 0.96

    input_shape = [batch, seq_len]
    model_config = get_model_config(model_config_str, llm_mode, input_shape, num_devices)
    devices = get_devices_for_t3000(all_devices, num_devices)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    tt_lib.profiler.set_profiler_location(f"falcon-40b_{request.node.callspec.id}")

    run_test_FalconCausalLM_inference(
        devices,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        out_pcc,
        cache_pcc,
        token_pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
