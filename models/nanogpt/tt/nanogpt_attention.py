# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import tt_lib
import math
from tt_lib.fallback_ops import fallback_ops
from models.helper_funcs import Linear
import math

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    torch_to_tt_tensor,
)


class TtCausalSelfAttention(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        self.out_mem_config_l1 = tt_lib.tensor.MemoryConfig(
            True, tt_lib.tensor.BufferType.L1
        )

        assert config.n_embd % config.n_head == 0

        self.config = config
        self.block_size = 1024

        self.device = device
        # Get the weights
        self.tt_weight_c_attn = state_dict[f"{base_address}.c_attn.weight"]
        self.tt_weight_c_proj = state_dict[f"{base_address}.c_proj.weight"]

        # Push weights to Ttp device
        self.tt_weight_c_attn = torch_to_tt_tensor(self.tt_weight_c_attn, self.device)

        self.tt_weight_c_proj = torch_to_tt_tensor(self.tt_weight_c_proj, self.device)

        self.tt_weight_c_attn = tt_lib.tensor.transpose(self.tt_weight_c_attn)
        self.tt_weight_c_proj = tt_lib.tensor.transpose(self.tt_weight_c_proj)

        # Load biases
        self.tt_bias_c_attn = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.c_attn.bias"], self.device
        )

        self.tt_bias_c_proj = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.c_proj.bias"], self.device
        )
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size
            ),
        )

        self.c_attn = Linear(
            self.config.n_embd,
            3 * config.n_embd,
            self.tt_weight_c_attn,
            self.tt_bias_c_attn,
            output_mem_config=self.out_mem_config_l1,
        )
        self.c_proj = Linear(
            self.config.n_embd,
            self.config.n_embd,
            self.tt_weight_c_proj,
            self.tt_bias_c_proj,
            output_mem_config=self.out_mem_config_l1,
        )

    def const_tensor(self, shape, value):
        return tt_lib.tensor.full(
            shape, value, output_mem_config=self.out_mem_config_l1
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        (
            _,
            B,
            T,
            C,
        ) = x.shape()  # batch size, sequence length, embedding dimensionality (n_embd)

        x1 = self.c_attn(x)

        pt_x1 = tt_to_torch_tensor(x1)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = pt_x1.split(self.n_embd, dim=3)

        k = torch_to_tt_tensor_rm(k, self.device)
        q = torch_to_tt_tensor_rm(q, self.device)
        v = torch_to_tt_tensor_rm(v, self.device)

        k = fallback_ops.reshape(k, B, T, self.n_head, C // self.n_head)
        k = tt_lib.tensor.transpose(k, 1, 2, output_mem_config=self.out_mem_config_l1)

        q = fallback_ops.reshape(q, B, T, self.n_head, C // self.n_head)
        q = tt_lib.tensor.transpose(q, 1, 2, output_mem_config=self.out_mem_config_l1)

        v = fallback_ops.reshape(v, B, T, self.n_head, C // self.n_head)
        v = tt_lib.tensor.transpose(v, 1, 2, output_mem_config=self.out_mem_config_l1)

        # manual implementation of attention
        key_layer_transposed = tt_lib.tensor.transpose(
            k, output_mem_config=self.out_mem_config_l1
        )
        att = tt_lib.tensor.bmm(
            q, key_layer_transposed, output_mem_config=self.out_mem_config_l1
        )

        const_att = self.const_tensor(att.shape(), 1.0 / math.sqrt(k.shape()[-1]))

        att = tt_lib.tensor.mul(
            att, const_att, output_mem_config=self.out_mem_config_l1
        )

        att = tt_to_torch_tensor(att)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        tt_att = torch_to_tt_tensor_rm(att, self.device, put_on_device=False)

        tt_att = fallback_ops.softmax(tt_att, dim=-1)

        tt_y = tt_lib.tensor.bmm(tt_att, v, output_mem_config=self.out_mem_config_l1)

        tt_y = tt_lib.tensor.transpose_hc(
            tt_y, output_mem_config=self.out_mem_config_l1
        )
        tt_y = fallback_ops.reshape(tt_y, 1, B, T, C)

        # output projection
        x2 = self.c_proj(tt_y)
        return x2
