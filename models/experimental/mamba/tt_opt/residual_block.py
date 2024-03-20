# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from typing import Callable

from models.experimental.mamba.reference.args import ModelArgs
from models.experimental.mamba.tt_opt.mamba_block import TtMambaBlock

class TtResidualBlock(torch.nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        device,
        load_fn: Callable,
        state_dict,
        num_users,
        hidden_size,
        configs,
        tt_cache_path
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.args = args

        #rms_norm_weight_name = "norm.weight"
        #self.rms_norm_weights = load_fn(rms_norm_weight_name)

        self.tt_mamba_block = TtMambaBlock(self.args,self.device,load_fn,self.state_dict, num_users, hidden_size, configs, tt_cache_path)

    def forward(self, x):
        mamba_input = x
        #mamba_input = ttnn.rms_norm(x, self.rms_norm_weights, epsilon=self.args.eps)
        mamba_input = self.tt_mamba_block(mamba_input)
        x = ttnn.add(x, mamba_input)
        return x
