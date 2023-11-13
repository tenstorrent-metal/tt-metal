# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from functools import partial
import tt_lib

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
from models.helper_funcs import Linear as TTLinear
from models.utility_functions import pad_by_zero


class TtRobertaSelfOutput(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        self.mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.L1
        )
        self.device = device

        self.dense_weight = pad_by_zero(
            state_dict[f"{base_address}.dense.weight"], self.device
        )[0]
        self.dense_bias = pad_by_zero(
            state_dict[f"{base_address}.dense.bias"], self.device
        )[0]

        gamma = pad_by_zero(
            state_dict[f"{base_address}.LayerNorm.weight"], self.device
        )[0]
        beta = pad_by_zero(state_dict[f"{base_address}.LayerNorm.bias"], self.device)[0]
        self.LayerNorm = partial(
            tt_lib.tensor.layernorm, eps=config.layer_norm_eps, gamma=gamma, beta=beta
        )

        # TODO: Add dropout when supported
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense_linear = TTLinear(
            self.dense_weight.shape()[-1],
            self.dense_weight.shape()[-2],
            self.dense_weight,
            self.dense_bias,
        )

    def linear(self, x, weight, bias):
        weight = tt_lib.tensor.transpose(weight)
        x = tt_lib.tensor.matmul(x, weight, output_mem_config=self.mem_config)
        x = tt_lib.tensor.bcast(
            x,
            bias,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
            self.mem_config,
        )
        return x

    def forward(
        self, hidden_states: tt_lib.tensor.Tensor, input_tensor: tt_lib.tensor.Tensor
    ) -> tt_lib.tensor.Tensor:
        hidden_states = self.dense_linear(hidden_states)
        # TODO: Add dropout when supported
        # hidden_states = self.dropout(hidden_states)
        hidden_states = tt_lib.tensor.add(
            hidden_states, input_tensor, output_mem_config=self.mem_config
        )
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
