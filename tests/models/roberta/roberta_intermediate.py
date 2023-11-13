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

from transformers import RobertaModel


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class TtRobertaIntermediate(nn.Module):
    def __init__(
        self, config, state_dict, base_address, device, fall_back_to_torch_gelu=True
    ):
        super().__init__()
        self.mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.L1)
        self.device = device

        self.fall_back_to_torch_gelu = fall_back_to_torch_gelu

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
        x = tt_lib.tensor.matmul(x, weight, output_mem_config=self.mem_config)
        x = tt_lib.tensor.bcast(
            x,
            bias,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
            output_mem_config=self.mem_config,
        )
        return x

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_linear(hidden_states)
        if self.fall_back_to_torch_gelu:
            torch_hidden_states = tt2torch_tensor(hidden_states)
            torch_hidden_states = torch.nn.functional.gelu(torch_hidden_states)
            hidden_states = torch2tt_tensor(torch_hidden_states, self.device)
        else:
            hidden_states = tt_lib.tensor.gelu(
                hidden_states, output_mem_config=self.mem_config
            )
        return hidden_states
