# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn
from transformers.models import bloom

from models.experimental.functional_bloom.tt import ttnn_functional_bloom
from models.utility_functions import torch_random, skip_for_wormhole_b0


from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.int64)
    torch_alibi = bloom.modeling_bloom.build_alibi_tensor(torch_attention_mask, config.n_head, dtype=torch.float32)

    torch_output, *_ = model(
        torch_hidden_states,
        torch_residual,
        torch_alibi,
        torch_attention_mask,
    )

    parameters = ttnn.model_converter.from_torch_model(
        model=lambda: model,
        device=device,
        converter=ttnn_functional_bloom.converter,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_attention_mask.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    alibi = ttnn_functional_bloom.build_alibi_tensor(torch_attention_mask, config.n_head, dtype=torch.bfloat16)
    alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_functional_bloom.bloom_attention(
        config,
        hidden_states,
        residual,
        alibi,
        attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_mlp(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomMLP(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)

    torch_output = model(torch_hidden_states, torch_residual)

    parameters = ttnn.model_converter.from_torch_model(model=lambda: model, device=device)

    hidden_states = ttnn.from_torch(torch_hidden_states.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_functional_bloom.bloom_mlp(
        hidden_states,
        residual,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9998)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_block(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomBlock(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.int64)
    torch_alibi = bloom.modeling_bloom.build_alibi_tensor(torch_attention_mask, config.n_head, dtype=torch.float32)

    torch_output, *_ = model(
        torch_hidden_states,
        torch_alibi,
        torch_attention_mask,
    )

    parameters = ttnn.model_converter.from_torch_model(
        model=lambda: model,
        device=device,
        converter=ttnn_functional_bloom.converter,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_attention_mask.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    alibi = ttnn_functional_bloom.build_alibi_tensor(torch_attention_mask, config.n_head, dtype=torch.bfloat16)
    alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_functional_bloom.bloom_block(
        config,
        hidden_states,
        alibi,
        attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.997)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomModel.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.int64)
    torch_output = model(input_ids=torch_input_ids, attention_mask=torch_attention_mask).last_hidden_state

    parameters = ttnn.model_converter.from_torch_model(
        model=lambda: model,
        device=device,
        converter=ttnn_functional_bloom.converter,
    )

    padded_input_ids, alibi, causal_mask = ttnn_functional_bloom.preprocess_inputs(
        input_ids=torch_input_ids,
        attention_mask=torch_attention_mask,
        num_heads=config.n_head,
        device=device,
    )

    output = ttnn_functional_bloom.bloom(
        config,
        padded_input_ids,
        alibi,
        causal_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.93)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_for_question_answering(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    config.position_embedding_type = "none"
    model = bloom.modeling_bloom.BloomForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.int64)
    torch_output = model(input_ids=torch_input_ids, attention_mask=torch_attention_mask)

    parameters = ttnn.model_converter.from_torch_model(
        model=lambda: model,
        device=device,
        converter=ttnn_functional_bloom.converter,
    )

    padded_input_ids, alibi, causal_mask = ttnn_functional_bloom.preprocess_inputs(
        input_ids=torch_input_ids,
        attention_mask=torch_attention_mask,
        num_heads=config.n_head,
        device=device,
    )

    output = ttnn_functional_bloom.bloom_for_question_answering(
        config,
        padded_input_ids,
        alibi,
        causal_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)
    start_logits = output[..., 0]
    end_logits = output[..., 1]

    assert_with_pcc(torch_output.start_logits, start_logits, 0.876)
    assert_with_pcc(torch_output.end_logits, end_logits, 0.895)
