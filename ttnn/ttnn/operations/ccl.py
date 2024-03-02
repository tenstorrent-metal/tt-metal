# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

from typing import List

import tt_lib as ttl

import ttnn

import torch

THIS_MODULE = sys.modules[__name__]

__all__ = []


#            R"doc(Performs all gather on a list of tensors that form one tensor that is distributed across devices.
# The output is a list of a tensor which has been duplciated across the input devices.)doc"
@ttnn.register_operation(
    name="ttnn.all_gather",
    validate_input_tensors=None,  # TODO(cfjchu): fix me
    torch_function=None,  # TODO(cfjchu): fix me
)
def all_gather(
    multidevice_tensor: ttnn.DeviceMeshTensor,
    dim: int,
    num_links: int,
    *,
    # layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
):
    r"""
            py::arg("input_tensors"), py::arg("dim"), py::arg("num_links") = 1, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            R"doc(Performs all gather on a list of tensors that form one tensor that is distributed across devices. The output is a list of a tensor which has been duplciated across the input devices.)doc"
    embedding(inxput_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Retrieves word embeddings using input_tensor. The input_tensor is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.

    Args:
        * :attr:`input_tensor`: the indices ttnn.Tensor
        * :attr:`weight`: the embeddings ttnn.Tensor that correspond to the indices ttnn.Tensor

    Example::
        >>> device_id = 0
        >>> device = ttnn.open(device_id)
        >>> input_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]), dtype=ttnn.uint32), device)
        >>> # an embedding matrix containing 10 tensors of size 4
        >>> weight = ttnn.to_device(ttnn.from_torch(torch.rand(10, 4), dtype=ttnn.bfloat16), device)
        >>> ttnn.embedding(input_tensor, weight)
        ttnn.Tensor([ [[1, 0.106445, 0.988281, 0.59375],
            [0.212891, 0.964844, 0.199219, 0.996094],
            [3.78362e-38, 0, 7.89785e-39, 0],
            [8.04479e-38, 0, 1.25815e-38, 0]],
           [[2.71833e-38, 0, 3.59995e-38, 0],
            [7.60398e-38, 0, 1.83671e-38, 0],
            [2.22242e-38, 0, 1.88263e-38, 0],
            [1.35917e-38, 0, 4.49994e-39, 0]]], dtype=bfloat16 )

    """
    """

        attn_output = tt_lib.tensor.all_gather(
            attn_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            output_mem_config=self.model_config["DEFAULT_MEMCFG"],
        )
    """
    """

    if len(input_tensor.shape) != 2:
        raise RuntimeError("Input Tensor must have rank of 2!")
    if len(weight.shape) not in {2, 4}:
        raise RuntimeError("Weight Tensor must either have rank of 2 or 4!")

    *_, hidden_embedding_dim = tuple(weight.shape)
    weight = ttnn.unsqueeze_to_4D(weight)

    batch_size, sentence_size = input_tensor.shape
    input_tensor = ttnn.reshape(input_tensor, shape=(batch_size, 1, 1, sentence_size))

    tilized = layout == ttnn.TILE_LAYOUT
    embeddings = ttnn.Tensor(
        ttl.tensor.embeddings(input_tensor.value, weight.value, tilized, output_mem_config=memory_config)
    )
    embeddings = ttnn.reshape(embeddings, shape=(batch_size, sentence_size, hidden_embedding_dim))

    return embeddings

        // Multi-Device ops
        m_tensor.def("all_gather", &all_gather,
            py::arg("input_tensors"), py::arg("dim"), py::arg("num_links") = 1, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            R"doc(Performs all gather on a list of tensors that form one tensor that is distributed across devices. The output is a list of a tensor which has been duplciated across the input devices.)doc"
        );
    """
    per_device_tensor = [x.value for x in multidevice_tensor.get_tensors()]
    all_gather_tensor = ttl.tensor.all_gather(per_device_tensor, dim, num_links, output_mem_config=memory_config)
    output_tensors = [ttnn.Tensor(x) for x in all_gather_tensor]
    return ttnn.DeviceMeshTensor(
        output_tensors, multidevice_tensor.multidevice, ttnn.DeviceMeshTensorMapper(shard_dim=dim), device=True
    )
