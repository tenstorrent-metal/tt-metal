# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import os
from pathlib import Path
from typing import Callable

from models.experimental.mamba.tt_opt.residual_block import TtResidualBlock

class TtTensorLoader:
    def __init__(self, state_dict, device, tt_cache_path: str = ""):
        self.state_dict = state_dict
        self.tt_cache_path = tt_cache_path
        self.device = device

        if len(tt_cache_path) > 0 and not os.path.exists(self.tt_cache_path):
            os.makedirs(self.tt_cache_path)

    def get_tensor_loader(self, layer_num):
        def load_tt_tensor(
            name: str,
            tm_fn: Callable = lambda x: x,
            postfix: str = "",
            device: ttnn.device = self.device,
            tt_layout=ttnn.TILE_LAYOUT,
            tt_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tt_dtype=ttnn.bfloat16,
            torch_tensor=None,
        ):
            tensor_name = f"layers.{layer_num}.{name}"

            tensor_cache_filepath = Path(self.tt_cache_path) / (tensor_name + postfix + ".bin")

            if tensor_cache_filepath.exists() and (len(self.tt_cache_path) > 0):
                tt_tensor = ttnn.load_tensor(str(tensor_cache_filepath)).to(device, tt_memory_config)
            else:
                if torch_tensor is None:
                    torch_tensor = self.state_dict[tensor_name]
                torch_tensor = tm_fn(torch_tensor)
                tt_tensor = ttnn.as_tensor(
                    torch_tensor,
                    device=device,
                    layout=tt_layout,
                    memory_config=tt_memory_config,
                    dtype=tt_dtype,
                    cache_file_name=str(tensor_cache_filepath),
                )
                if len(self.tt_cache_path) > 0:
                    ttnn.dump_tensor(
                        str(tensor_cache_filepath),
                        tt_tensor.cpu(),
                    )
            return tt_tensor

        return load_tt_tensor


class MambaTT(torch.nn.Module):
    def __init__(
        self,
        reference_model,
        tt_cache_path,
        num_layers,
        device,
        num_users,
        hidden_size,
        configs
    ):
        super().__init__()
        print(f"Initalizing MambaTT with {num_layers} layers")
        self.args = reference_model.args
        self.device = device
        #self.embedding = reference_model.embedding
        loader = TtTensorLoader(reference_model.state_dict(), self.device, tt_cache_path=tt_cache_path)

        self.layers = [TtResidualBlock(self.args, device, loader.get_tensor_loader(i), reference_model.layers[i].state_dict(), num_users, hidden_size, configs, tt_cache_path) for i in range(num_layers)]

        self.layers = [TtResidualBlock(self.args, device, loader.get_tensor_loader(i), reference_model.layers[i].state_dict(), num_users, hidden_size, configs, tt_cache_path) for i in range(num_layers)]

        self.norm_f = reference_model.norm_f

        self.lm_head = reference_model.lm_head



    def forward(self, x):
        #x = self.embedding(x)
        #x = x.unsqueeze(1)  # (BS, 1, 1, E)
        x = ttnn.from_torch(
            x,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        for layer in self.layers:
            x = layer(x)

        #x = ttnn.to_torch(x).squeeze(1).to(torch.float32)
        #x = self.norm_f(x)
        #x = self.lm_head(x)

        return x
