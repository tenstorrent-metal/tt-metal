# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from loguru import logger
import math

from transformers import T5Model
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)


def gelu_new(x, device):
    x = tt2torch_tensor(x)
    x = (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )
    x = torch2tt_tensor(x, device)

    return x


class TtT5DenseGatedActDense(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.device = device
        self.mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.L1)

        enc_dec = "decoder" if config["is_decoder"] else "encoder"

        self.wi_0_weights = torch2tt_tensor(
            state_dict[f"{base_address}.wi_0.weight"], device
        )
        self.wi_1_weights = torch2tt_tensor(
            state_dict[f"{base_address}.wi_1.weight"], device
        )
        self.wo_weights = torch2tt_tensor(
            state_dict[f"{base_address}.wo.weight"], device
        )

        self.wi_0_weights = tt_lib.tensor.transpose(self.wi_0_weights)
        self.wi_1_weights = tt_lib.tensor.transpose(self.wi_1_weights)
        self.wo_weights = tt_lib.tensor.transpose(self.wo_weights)

        # self.dropout = nn.Dropout(config["dropout_rate"])
        self.act = gelu_new

    def forward(self, hidden_states):
        hidden_gelu = self.act(
            tt_lib.tensor.matmul(hidden_states, self.wi_0_weights), self.device
        )
        hidden_linear = tt_lib.tensor.matmul(hidden_states, self.wi_1_weights, output_mem_config = self.mem_config)
        hidden_states = tt_lib.tensor.mul(hidden_gelu, hidden_linear, output_mem_config = self.mem_config)
        # hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        # if (
        #    isinstance(self.wo.weight, torch.Tensor)
        #    and hidden_states.dtype != self.wo.weight.dtype
        #    and self.wo.weight.dtype != torch.int8
        # ):
        #    hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = tt_lib.tensor.matmul(hidden_states, self.wo_weights, output_mem_config = self.mem_config)
        return hidden_states
