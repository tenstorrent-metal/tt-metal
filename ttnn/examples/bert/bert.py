# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import transformers

import ttnn
from ttnn.model_preprocessing import preprocess_model

import networkx as nx


def embedding(input_ids, *, parameters):
    output = ttnn.embedding(input_ids, weight=parameters.weight, layout=ttnn.TILE_LAYOUT)
    return output


def bert_embeddings(input_ids, token_type_ids, *, parameters):
    position_ids = torch.randint(0, 2, (1, 128))
    position_ids = ttnn.from_torch(position_ids, device=input_ids.device())

    word_embeddings = embedding(input_ids, parameters=parameters.word_embeddings)
    token_type_embeddings = embedding(token_type_ids, parameters=parameters.token_type_embeddings)
    position_embeddings = embedding(position_ids, parameters=parameters.position_embeddings)
    output = word_embeddings + token_type_embeddings
    output += position_embeddings
    output = ttnn.layer_norm(output, weight=parameters.LayerNorm.weight, bias=parameters.LayerNorm.bias)
    return output


def bert_attention(hidden_states, *, parameters):
    output = hidden_states @ parameters.self.query.weight
    output = output + parameters.self.query.bias
    return output


def bert_layer(hidden_states, *, parameters):
    output = bert_attention(hidden_states, parameters=parameters.attention)
    return output


def bert_encoder(hidden_states, *, parameters):
    output = bert_layer(hidden_states, parameters=parameters.layer[0])
    return output


def bert_model(input_ids, *, parameters):
    token_type_ids = torch.randint(0, 2, (1, 128))
    token_type_ids = ttnn.from_torch(token_type_ids, device=input_ids.device())
    embeddings = bert_embeddings(input_ids, token_type_ids, parameters=parameters.embeddings)
    output = bert_encoder(embeddings, parameters=parameters.encoder)
    return output


def main():
    model_name = "bert-base-uncased"
    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 1
    model = transformers.BertModel.from_pretrained(model_name, config=config).eval()
    input_tensor = torch.randint(0, 1000, (1, 128))

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    parameters = preprocess_model(
        model_name=model_name,
        initialize_model=lambda: model,
        run_model=lambda model: model(input_tensor),
        reader_patterns_cache={},
        device=device,
    )

    with ttnn.tracer.trace():
        input_tensor = torch.randint(0, 1000, (1, 128))
        output_tensor = model(input_tensor)

    graph = ttnn.tracer.get_graph(output_tensor, flatten=True)
    operations = set()
    for node in nx.topological_sort(graph):
        operation = graph.nodes[node]["operation"]
        if isinstance(operation, ttnn.tracer.TorchFunction):
            operations.add(str(operation))

    node_to_tensor_map = {input_tensor.node: torch.zeros_like(input_tensor)}

    for node in nx.topological_sort(graph):
        if node in node_to_tensor_map:
            continue

        input_nodes = [None for _ in graph.in_edges(node)]
        for input_node, _, edge_data in graph.in_edges(node, data=True):
            input_nodes[edge_data["sink_input_index"]] = input_node

        input_tensors = [node_to_tensor_map[input_node] for input_node in input_nodes]

        operation = graph.nodes[node]["operation"]
        if isinstance(operation, ttnn.tracer.TorchFunction):
            arg_name_value_pairs = operation.arg_name_value_pairs
            function_args = [
                arg
                for arg_name, arg in arg_name_value_pairs
                if isinstance(arg_name, ttnn.tracer.PositionalArgument)
                and not isinstance(arg, ttnn.tracer.InputTensorIndex)
            ]
            function_kwargs = {
                arg_name: arg
                for arg_name, arg in arg_name_value_pairs
                if not isinstance(arg_name, ttnn.tracer.PositionalArgument)
                and not isinstance(arg, ttnn.tracer.InputTensorIndex)
            }
            if operation.function == torch.nn.functional.layer_norm:
                function_kwargs["weight"] = input_tensors[1]
                function_kwargs["bias"] = input_tensors[2]
                input_tensors = input_tensors[:1]
            node_to_tensor_map[node] = operation.function(*input_tensors, *function_args, **function_kwargs)
        elif isinstance(operation, ttnn.tracer.TorchTensor):
            node_to_tensor_map[node] = operation.tensor
        elif isinstance(operation, ttnn.tracer.TorchParameter):
            node_to_tensor_map[node] = operation.parameter

    for node in nx.topological_sort(graph):
        if len(list(graph.successors(node))) == 0:
            output_tensor = node_to_tensor_map[node]
            print(output_tensor)

    for operation in operations:
        print(operation)
        # if operation == "embedding":
        #     ttnn.register_function("embedding", embedding)
        # elif operation == "bert_embeddings":
        #     ttnn.register_function("bert_embeddings", bert_embeddings)
        # elif operation == "bert_attention":
        #     ttnn.register_function("bert_attention", bert_attention)
        # elif operation == "bert_layer":
        #     ttnn.register_function("bert_layer", bert_layer)
        # elif operation == "bert_encoder":
        #     ttnn.register_function("bert_encoder", bert_encoder)
        # elif operation == "bert_model":
        #     ttnn.register_function("bert_model", bert_model)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
