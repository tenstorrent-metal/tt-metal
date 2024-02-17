# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import io
import re


import torch
import transformers

import ttnn

import networkx as nx


def to_snake_case(name):
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return name


visited = set()


def get_module_input_nodes(module_operation):
    return [module_input.node for module_input in module_operation.inputs]


def get_module_input_tensor_names(module_operation):
    module_input_nodes = get_module_input_nodes(module_operation)
    input_tensor_names = []
    for node in module_input_nodes:
        operation = module_operation.graph.nodes[node]["operation"]
        input_tensor_names.append(f"{operation.name}")
    return input_tensor_names


def module_to_source_code(module_operation, prefix=""):
    graph = module_operation.graph
    string_io = io.StringIO()

    input_tensor_names = get_module_input_tensor_names(module_operation)
    input_tensor_names_as_string = ", ".join(input_tensor_names)

    module_name = to_snake_case(f"{type(module_operation.module).__name__}")
    string_io.write(f"def {module_name}(config, {input_tensor_names_as_string}, *, parameters):\n")

    module_input_nodes = get_module_input_nodes(module_operation)
    node_to_variable = {}
    for module_input, name in zip(module_input_nodes, input_tensor_names):
        node_to_variable[module_input] = name

    index = 0
    for node in nx.topological_sort(graph):
        if node in module_input_nodes:
            continue

        operation = graph.nodes[node]["operation"]

        input_nodes = [input_node for input_node, _ in graph.in_edges(node)]
        input_variables = [node_to_variable[input_node] for input_node in input_nodes]
        input_variables_as_string = ", ".join(input_variables)

        created_variable = True
        if len(input_nodes) == 1:
            input_node = input_nodes[0]
            if len(list(graph.successors(input_node))) == 1:
                created_variable = False
                variable = input_variables[0]
                node_to_variable[node] = variable

        if created_variable:
            variable = f"tensor_{index}"
            index += 1
            node_to_variable[node] = variable

        if isinstance(operation, ttnn.tracer.TorchParameter):
            torchtrail_name = getattr(operation.parameter, "torchtrail_name", None)
            torchtrail_name = torchtrail_name.replace(f"{prefix}", "")
            string_io.write(f"    {variable} = parameters{torchtrail_name}\n")
        elif isinstance(operation, ttnn.tracer.TorchTensor):
            string_io.write(f"    {variable} = {operation.tensor}\n")
        elif isinstance(operation, ttnn.tracer.TorchFunction):
            function_args = []
            function_kwargs = []
            for arg_name, arg in operation.arg_name_value_pairs:
                if isinstance(arg_name, ttnn.tracer.PositionalArgumentName):
                    if not isinstance(arg, ttnn.tracer.InputTensorIndex):
                        function_args.append(f"{arg}")
                else:
                    if not isinstance(arg, ttnn.tracer.InputTensorIndex):
                        function_kwargs.append(f"{arg_name}={arg}")
            function_args_as_string = ", ".join(function_args)
            function_kwargs_as_string = ", ".join(function_kwargs)

            all_arguments = [input_variables_as_string]
            if function_args_as_string != "":
                all_arguments.append(function_args_as_string)
            if function_kwargs_as_string != "":
                all_arguments.append(function_kwargs_as_string)
            arguments_string = ", ".join(all_arguments)
            string_io.write(f"    {variable} = {operation}({arguments_string})\n")

        elif isinstance(operation, ttnn.tracer.TorchModule):
            module_name = to_snake_case(f"{type(operation.module).__name__}")
            parameters_string = operation.module.torchtrail_name.replace(f"{prefix}", "")
            string_io.write(
                f"    {variable} = {module_name}(config, {input_variables_as_string}, parameters=parameters{parameters_string})\n"
            )
        else:
            raise ValueError(f"Unknown operation type: {operation}")

    output = string_io.getvalue()
    print(output)


def autogenerate_code(graph):
    for node in nx.topological_sort(graph):
        operation = graph.nodes[node]["operation"]
        if isinstance(operation, ttnn.tracer.TorchModule):
            module = operation.module
            full_name = f"{type(module).__module__}.{type(module).__name__}"

            if full_name in visited:
                continue
            visited.add(full_name)

            module_to_source_code(operation, module.torchtrail_name)
            autogenerate_code(operation.graph)


def test_autogenerate_torch_bert():
    model_name = "bert-base-uncased"
    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 1
    model = transformers.BertModel.from_pretrained(model_name, config=config).eval()
    input_tensor = torch.randint(0, 1000, (1, 128))

    with ttnn.tracer.trace():
        input_tensor = torch.randint(0, 1000, (1, 128))
        outputs = model(input_tensor)

    flattened_graph = ttnn.tracer.get_graph(outputs, flatten=True)
    node_to_tensor_map = {input_tensor.node: input_tensor}
    ttnn.tracer.visualize(flattened_graph, file_name="bert_graph.svg", verbose=True)

    for node in nx.topological_sort(flattened_graph):
        if node in node_to_tensor_map:
            continue

        input_nodes = [None for _ in flattened_graph.in_edges(node)]
        for input_node, _, edge_data in flattened_graph.in_edges(node, data=True):
            input_nodes[edge_data["sink_input_index"]] = input_node

        input_tensors = [node_to_tensor_map[input_node] for input_node in input_nodes]

        operation = flattened_graph.nodes[node]["operation"]
        if isinstance(operation, ttnn.tracer.TorchFunction):
            arg_name_value_pairs = operation.arg_name_value_pairs
            function_args = [
                arg
                for arg_name, arg in arg_name_value_pairs
                if isinstance(arg_name, ttnn.tracer.PositionalArgumentName)
                and not isinstance(arg, ttnn.tracer.InputTensorIndex)
            ]
            function_kwargs = {
                arg_name: arg
                for arg_name, arg in arg_name_value_pairs
                if not isinstance(arg_name, ttnn.tracer.PositionalArgumentName)
                and not isinstance(arg, ttnn.tracer.InputTensorIndex)
            }
            if operation.function == torch.nn.functional.layer_norm:
                function_kwargs["weight"] = input_tensors[1]
                function_kwargs["bias"] = input_tensors[2]
                input_tensors = input_tensors[:1]
            node_to_tensor_map[node] = operation.function(*input_tensors, *function_args, **function_kwargs)
        elif isinstance(operation, ttnn.tracer.TorchParameter):
            node_to_tensor_map[node] = operation.parameter
        elif isinstance(operation, ttnn.tracer.TorchTensor):
            node_to_tensor_map[node] = operation.tensor
        else:
            raise ValueError(f"Unknown operation type: {operation}")

    for node in nx.topological_sort(flattened_graph):
        if len(list(flattened_graph.successors(node))) == 0:
            print(f"Node: {node}")
            eval_output_tensor = node_to_tensor_map[node]
            assert torch.allclose(outputs.pooler_output, eval_output_tensor)

    graph = ttnn.tracer.get_graph(outputs)
    autogenerate_code(graph)
