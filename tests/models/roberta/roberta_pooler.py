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

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from tests.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from models.helper_funcs import Linear as TTLinear
from models.utility_functions import pad_by_zero
import tt_lib
from tt_lib.fallback_ops import fallback_ops


# Copied from transformers.models.bert.modeling_bert.BertPooler
class TtRobertaPooler(nn.Module):
    def __init__(
        self,
        config,
        state_dict,
        base_address,
        device,
    ):
        super().__init__()
        self.mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.L1)
        self.device = device

        self.dense_weight = pad_by_zero(
            state_dict[f"{base_address}.dense.weight"], self.device
        )[0]
        self.dense_bias = pad_by_zero(
            state_dict[f"{base_address}.dense.bias"], self.device
        )[0]
        self.dense_linear = TTLinear(
            self.dense_weight.shape()[-1],
            self.dense_weight.shape()[-2],
            self.dense_weight,
            self.dense_bias,
        )

    def linear(self, x, weight, bias):
        weight = tt_lib.tensor.transpose(weight)
        x = tt_lib.tensor.matmul(x, weight, self.mem_config)
        x = tt_lib.tensor.bcast(
            x,
            bias,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
            self.mem_config,
        )
        return x

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        # Convert to torch to slice the tensor
        hidden_states_to_torch = tt2torch_tensor(hidden_states)
        hidden_states_to_torch = hidden_states_to_torch.squeeze(0)
        first_token_tensor = hidden_states_to_torch[:, 0]
        tt_first_token_tensor = torch2tt_tensor(first_token_tensor, self.device)

        pooled_output = self.dense_linear(tt_first_token_tensor)
        pooled_output = tt_lib.tensor.tanh(pooled_output, self.mem_config)
        return pooled_output
