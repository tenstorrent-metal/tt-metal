# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def ttnn_optimized_multi_head_attention(
    hidden_states,
    attention_mask,
    query_key_value_weight,
    query_key_value_bias,
    self_output_weight,
    self_output_bias,
    *,
    num_heads,
    num_cores_x=12,
):
    batch_size, _, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query_key_value_output = ttnn.linear(
        hidden_states,
        query_key_value_weight,
        bias=query_key_value_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=(batch_size, num_cores_x),
    )

    (
        query,
        key,
        value,
    ) = ttnn.nlp.split_query_key_value_and_split_heads(
        query_key_value_output,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=(batch_size, num_cores_x),
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value_output)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=(batch_size, num_cores_x),
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attention_probs = ttnn.nlp.attention_softmax_(attention_scores, attention_mask=attention_mask, head_size=head_size)

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=(batch_size, num_cores_x),
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.nlp.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        self_output_weight,
        bias=self_output_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=(batch_size, num_cores_x),
    )
    ttnn.deallocate(context_layer)

    return self_output


def ttnn_optimized_feedforward(hidden_states, ff1_weight, ff1_bias, ff2_weight, ff2_bias, num_cores_x=12):
    batch_size, *_ = hidden_states.shape

    num_cores_x = 12
    ff1_output = ttnn.linear(
        hidden_states,
        ff1_weight,
        bias=ff1_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=(batch_size, num_cores_x),
        activation="gelu",
    )

    ff2_output = ttnn.linear(
        ff1_output,
        ff2_weight,
        bias=ff2_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=(batch_size, num_cores_x),
    )
    ttnn.deallocate(ff1_output)

    return ff2_output


def ttnn_optimized_bert_encoder(
    hidden_states,
    attention_mask,
    parameters,
    *,
    num_heads,
):
    multi_head_attention_output = ttnn_optimized_multi_head_attention(
        hidden_states,
        attention_mask,
        parameters.attention.self.query_key_value.weight,
        parameters.attention.self.query_key_value.bias,
        parameters.attention.output.dense.weight,
        parameters.attention.output.dense.bias,
        num_heads=num_heads,
    )

    multi_head_attention_add_and_layer_norm_output = ttnn.layer_norm(
        hidden_states,
        residual_input_tensor=multi_head_attention_output,
        weight=parameters.attention.output.LayerNorm.weight,
        bias=parameters.attention.output.LayerNorm.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(hidden_states)
    ttnn.deallocate(multi_head_attention_output)

    feedforward_output = ttnn_optimized_feedforward(
        multi_head_attention_add_and_layer_norm_output,
        parameters.intermediate.dense.weight,
        parameters.intermediate.dense.bias,
        parameters.output.dense.weight,
        parameters.output.dense.bias,
    )

    feedforward_add_and_layer_norm_output = ttnn.layer_norm(
        multi_head_attention_add_and_layer_norm_output,
        residual_input_tensor=feedforward_output,
        weight=parameters.output.LayerNorm.weight,
        bias=parameters.output.LayerNorm.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(multi_head_attention_add_and_layer_norm_output)
    ttnn.deallocate(feedforward_output)

    return feedforward_add_and_layer_norm_output


def ttnn_optimized_bert(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    num_heads,
):
    word_embeddings = ttnn.embedding(
        input_ids,
        parameters.bert.embeddings.word_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.deallocate(input_ids)

    token_type_embeddings = ttnn.embedding(
        token_type_ids,
        parameters.bert.embeddings.token_type_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.deallocate(token_type_ids)

    embeddings = word_embeddings + token_type_embeddings
    ttnn.deallocate(word_embeddings)
    ttnn.deallocate(token_type_embeddings)

    encoder_input = ttnn.layer_norm(
        embeddings,
        weight=parameters.bert.embeddings.LayerNorm.weight,
        bias=parameters.bert.embeddings.LayerNorm.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(embeddings)

    encoder_output = None
    for encoder_parameters in parameters.bert.encoder.layer:
        encoder_output = ttnn_optimized_bert_encoder(
            encoder_input,
            attention_mask,
            encoder_parameters,
            num_heads=num_heads,
        )
        encoder_output = ttnn.reallocate(encoder_output)
        encoder_input = encoder_output

    return encoder_output


def ttnn_optimized_bert_for_question_answering(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    num_heads,
):
    bert_output = ttnn_optimized_bert(
        input_ids,
        token_type_ids,
        attention_mask,
        parameters,
        num_heads=num_heads,
    )

    qa_outputs = ttnn.linear(
        bert_output,
        parameters.qa_outputs.weight,
        bias=parameters.qa_outputs.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return qa_outputs
