import time

import torch
import transformers
import ttnn


def bert_embeddings(input_tensor, token_type_ids, position_ids, *, parameters):
    output_tensor = ttnn.embedding(input_tensor, parameters.word_embeddings.weight, layout=ttnn.TILE_LAYOUT)
    output_tensor += ttnn.embedding(token_type_ids, parameters.token_type_embeddings.weight, layout=ttnn.TILE_LAYOUT)
    output_tensor += ttnn.embedding(
        position_ids,
        parameters.position_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output_tensor = ttnn.layer_norm(
        output_tensor,
        weight=parameters.LayerNorm.weight,
        bias=parameters.LayerNorm.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return output_tensor


def bert_layer(hidden_states, *, parameters):
    query_key_value = ttnn.linear(
        hidden_states,
        parameters.attention.self.query_key_value.weight,
        bias=parameters.attention.self.query_key_value.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=8, x=12),
    )

    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value, num_heads=12, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    attention_scores = ttnn.matmul(query, key, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=8, x=12))
    attention_weights = ttnn.matmul(
        attention_scores, value, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=8, x=12)
    )
    context_layer = ttnn.transformer.concatenate_heads(attention_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
    return context_layer


def bert_encoder(hidden_states, *, parameters):
    for layer in parameters.layer:
        hidden_states = bert_layer(hidden_states, parameters=layer)
    return hidden_states


def bert_model(input_tensor, position_ids, token_type_ids, *, parameters):
    output_tensor = bert_embeddings(input_tensor, token_type_ids, position_ids, parameters=parameters.embeddings)
    output_tensor = bert_encoder(output_tensor, parameters=parameters.encoder)

    return output_tensor


def bert_converter(*, module, **_):
    import torch
    from ttnn.model_converter import (
        convert_torch_linear_bias_to_ttnn,
        convert_torch_linear_weight_to_ttnn,
    )

    parameters = {}
    if hasattr(module, "query") and hasattr(module, "key") and hasattr(module, "value"):
        qkv_weight = torch.cat(
            [
                module.query.weight,
                module.key.weight,
                module.value.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [module.query.bias, module.key.bias, module.value.bias],
            dim=0,
        )

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = convert_torch_linear_weight_to_ttnn(qkv_weight, dtype=ttnn.bfloat16)
        parameters["query_key_value"]["bias"] = convert_torch_linear_bias_to_ttnn(qkv_bias, dtype=ttnn.bfloat16)
    return parameters


def main():
    model_name = "bert-base-uncased"
    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 1
    model = transformers.BertModel.from_pretrained(model_name, config=config)

    with ttnn.tracer.trace():
        torch_input_tensor = torch.randint(0, 1000, (8, 128))
        torch_output_tensor = model(torch_input_tensor)
    ttnn.tracer.visualize(torch_output_tensor, file_name="torch_bert_trace.svg")

    with ttnn.manage_device(device_id=0) as device, ttnn.tracer.trace():
        parameters_conv2d_maxpool2d_dict = ttnn.model_converter.from_torch_model(
            model=model,
            device=device,
            converter=bert_converter,
        )

        position_ids = torch.randint(0, 2, (8, 128))
        position_ids = ttnn.from_torch(position_ids, device=device)

        token_type_ids = torch.randint(0, 2, (8, 128))
        token_type_ids = ttnn.from_torch(token_type_ids, device=device)

        # non-optimized = 128.94 ms
        for _ in range(21):
            start_time = time.time()
            input_tensor = ttnn.from_torch(torch_input_tensor, device=device)
            output_tensor = bert_model(input_tensor, position_ids, token_type_ids, parameters=parameters)
            output_tensor = ttnn.to_torch(output_tensor)
            end_time = time.time()
            print("Time taken to preprocess model: ", (end_time - start_time) / 8 * 1000, "ms")

    ttnn.tracer.visualize(torch_output_tensor, file_name="bert_trace.svg")


if __name__ == "__main__":
    main()
