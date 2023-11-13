# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import numpy as np
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from tests.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)

import tt_lib
from tt_lib.fallback_ops import fallback_ops

from transformers import RobertaForMaskedLM
from transformers import AutoTokenizer

mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.L1)


class TtRobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(
        self,
        config,
        state_dict,
        base_address,
        device,
        fall_back_to_torch_gelu=True,
        fall_back_to_torch_linear=True,
    ):
        super().__init__()
        self.device = device
        self.fall_back_to_torch_gelu = fall_back_to_torch_gelu
        self.fallback_to_torch_linear = fall_back_to_torch_linear

        self.dense_weight = torch2tt_tensor(
            state_dict[f"{base_address}.dense.weight"], self.device
        )
        self.dense_bias = torch2tt_tensor(
            state_dict[f"{base_address}.dense.bias"], self.device
        )

        self.gamma = torch2tt_tensor(
            state_dict[f"{base_address}.layer_norm.weight"], self.device
        )
        self.beta = torch2tt_tensor(
            state_dict[f"{base_address}.layer_norm.bias"], self.device
        )
        self.layer_norm = fallback_ops.LayerNorm(
            weights=self.gamma,
            biases=self.beta,
            eps=config.layer_norm_eps,
            normalized_shape=config.hidden_size,
        )

        self.decoder_weight = torch2tt_tensor(
            state_dict[f"{base_address}.decoder.weight"], self.device
        )
        self.bias = torch2tt_tensor(state_dict[f"{base_address}.bias"], self.device)

        self.decoder_bias = torch2tt_tensor(
            state_dict[f"{base_address}.decoder.bias"], self.device
        )
        if self.fallback_to_torch_linear:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
            self.decoder.weight.data = state_dict[f"{base_address}.decoder.weight"]
            self.decoder.bias.data = state_dict[f"{base_address}.decoder.bias"]

    def linear(self, x, weight, bias):
        weight = tt_lib.tensor.transpose(weight)
        x = tt_lib.tensor.matmul(x, weight, output_mem_config=mem_config)
        x = tt_lib.tensor.bcast(
            x,
            bias,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
            output_mem_config=mem_config,
        )
        return x

    def forward(self, features, **kwargs):
        x = self.linear(features, self.dense_weight, self.dense_bias)

        if self.fall_back_to_torch_gelu:
            torch_input = tt2torch_tensor(x)
            torch_x = nn.functional.gelu(torch_input)
            x = torch2tt_tensor(torch_x, self.device)
        else:
            x = tt_lib.tensor.gelu(tt_input)

        x = self.layer_norm(x)

        if self.fallback_to_torch_linear:
            # project back to size of vocabulary with bias
            torch_input = tt2torch_tensor(x).to(torch.float)
            torch_x = self.decoder(torch_input)
            x = torch2tt_tensor(torch_x, self.device)
        else:
            x = self.linear(x, self.decoder_weight, self.decoder_bias)

        return x
