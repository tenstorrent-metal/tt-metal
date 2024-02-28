# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import tt_lib as ttl
from ttnn.operations.core import squeeze, unsqueeze_to_4D
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import is_tile_dim_alligned


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def concatenate_qkv(q, k, v):
    dim = -1
    device = k.device
    memory_config = ttnn.get_memory_config(k)

    if q is not None:
        q = ttnn_to_torch(q)
        assert is_tile_dim_alligned(q.shape[dim])

    k = ttnn_to_torch(k)
    v = ttnn_to_torch(v)

    assert is_tile_dim_alligned(k.shape[dim])
    assert is_tile_dim_alligned(v.shape[dim])

    if q is not None:
        qkv = torch.cat([q, k, v], dim=-1)
    else:
        qkv = torch.cat([k, v], dim=-1)
    qkv = ttnn.from_torch(qkv, ttnn.bfloat16)
    qkv = ttnn.to_layout(qkv, layout=ttnn.TILE_LAYOUT)
    qkv = ttnn.to_device(qkv, device, memory_config=memory_config)
    return qkv


class cross_attention:
    def __init__(self, device, parameters):
        self.fused_qkv = parameters.to_q.weight.shape[0] == parameters.to_k.weight.shape[0]
        if self.fused_qkv:
            parameters["qkv"] = ttnn.model_preprocessing.ParameterDict()
            parameters.qkv["weight"] = concatenate_qkv(
                parameters.to_q.weight, parameters.to_k.weight, parameters.to_v.weight
            )

            for key in ["to_q", "to_k", "to_v"]:
                assert "bias" not in parameters[key]
                del parameters[key]
        else:
            parameters["kv"] = ttnn.model_preprocessing.ParameterDict()
            parameters.kv["weight"] = concatenate_qkv(None, parameters.to_k.weight, parameters.to_v.weight)

        self.device = device
        self.parameters = parameters

    def prepare_attention_mask(self, attention_mask, target_length, heads=8):
        head_size = heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            assert False, "Attention Mask has always been None, This is not implemented!"

        return attention_mask

    def batch_to_head_dim(self, tensor, heads=8):
        head_size = heads
        _, batch_size, seq_len, dim = tensor.shape
        tensor = ttnn.to_layout(
            tensor, layout=ttnn.ROW_MAJOR_LAYOUT
        )  # TILE_LAYOUT is not compatible with tensor shape, hence we used ROW_MAJOR_LAYOUT.
        tensor = ttnn.reshape(tensor, (batch_size // head_size, head_size, seq_len, dim))
        tensor = ttnn.permute(tensor, (0, 2, 1, 3))
        tensor = ttnn.reshape(tensor, (1, batch_size // head_size, seq_len, dim * head_size))
        tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
        return tensor

    def head_to_batch_dim(self, tensor, heads=8):
        head_size = heads
        _, batch_size, seq_len, dim = tensor.shape
        tensor = ttnn.to_layout(
            tensor, layout=ttnn.ROW_MAJOR_LAYOUT
        )  # TILE_LAYOUT is not compatible with tensor shape, hence we used ROW_MAJOR_LAYOUT.
        tensor = ttnn.reshape(tensor, (batch_size, seq_len, head_size, dim // head_size))
        tensor = ttnn.permute(tensor, (0, 2, 1, 3))
        tensor = ttnn.reshape(tensor, (1, batch_size * head_size, seq_len, dim // head_size))
        tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
        return tensor

    def get_attention_scores(self, query, t_key, attention_mask=None, scale=None, device=None):
        # t_key = ttnn.permute(key, (0, 1, 3, 2))
        attention_scores = ttnn.matmul(query, t_key)

        attention_scores = ttnn.mul(attention_scores, scale)

        if attention_mask is not None:
            attention_scores = ttnn.add(attention_scores, attention_mask)

        attention_scores = ttnn.softmax(attention_scores, dim=-1)

        return attention_scores

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        query_dim: int = None,
        cross_attention_dim=None,
        heads: int = 8,
        dim_head: int = 64,
        attention_mask=None,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_kwargs={},
    ):
        # sequence_length = hidden_states.shape[-2]
        # attention_mask = self.prepare_attention_mask(attention_mask, sequence_length)

        if len(hidden_states.shape) == 4:
            hidden_states = squeeze(hidden_states, 0)
        if encoder_hidden_states and len(encoder_hidden_states.shape) == 4:
            encoder_hidden_states = squeeze(encoder_hidden_states, 0)

        if self.fused_qkv:
            qkv_out = ttnn.matmul(
                hidden_states,
                self.parameters.qkv.weight,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )
            query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv_out,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                num_heads=heads,
            )
            del qkv_out
        else:
            q_proj = ttnn.matmul(hidden_states, self.parameters.to_q.weight, memory_config=ttnn.L1_MEMORY_CONFIG)
            kv_proj = ttnn.matmul(encoder_hidden_states, self.parameters.kv.weight, memory_config=ttnn.L1_MEMORY_CONFIG)
            query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(q_proj, kv_proj, num_heads=heads)
            breakpoint()
            del kv_proj
            del q_proj
            attention_mask = torch.ones((1, 1, 1, key.shape[-1])) * 1e-9
            attention_mask[:, :, :77, :] = 0
            attention_mask[:, :, 256 : 256 + 77, :] = 0
            attention_mask = ttnn.from_torch(
                attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )

        scale = torch.ones((1, 1)) * dim_head**-0.5
        scale = ttnn.from_torch(scale, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        attention_probs = self.get_attention_scores(query, key, attention_mask, scale=scale, device=self.device)

        padding_needed = attention_probs.shape[-1] - value.shape[-2]
        if padding_needed > 0:
            value = ttnn.pad(value, ((0, padding_needed), (0, 0)), 0)

        hidden_states = ttnn.matmul(attention_probs, value)

        hidden_states = ttnn.transformer.concatenate_heads(
            hidden_states, memory_config=ttnn.get_memory_config(hidden_states)
        )

        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters.to_out[0].weight,
            bias=self.parameters.to_out[0].bias,
            memory_config=ttnn.get_memory_config(hidden_states),
        )

        if len(hidden_states.shape) == 3:
            hidden_states = unsqueeze_to_4D(hidden_states)

        return hidden_states
