# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from typing import Callable

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.helper_funcs import Linear
from models.experimental.mamba.reference.args import ModelArgs
from models.experimental.mamba.tt_opt.mamba_one_step_ssm import TtMambaSSM

class TtMambaBlock(torch.nn.Module):
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
        self.num_users = num_users
        self.hidden_size = hidden_size
        self.configs = configs

        # ssm wt
        if self.args.d_model == self.hidden_size:
            print('**********using ssm proj wts')
            in_proj_weight_name = "mixer.in_proj.weight"
            ssm_proj = torch.transpose(self.state_dict[in_proj_weight_name][: self.args.d_inner, :], -1, -2)
            self.ssm_proj = ttnn.as_tensor(ssm_proj, layout=ttnn.TILE_LAYOUT, device=self.device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, cache_file_name=tt_cache_path + "ssm_proj.bin")
        else:
            self.ssm_proj = ttnn.from_torch(torch.rand(1,1,self.hidden_size,2*self.hidden_size), layout=ttnn.TILE_LAYOUT, device=self.device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # mlp wt
        if self.args.d_model == self.hidden_size:
            print('**********using mlp proj wts')
            mlp_proj_weight_name = "mixer.in_proj.weight"
            mlp_proj = torch.transpose(self.state_dict[mlp_proj_weight_name][self.args.d_inner :, :], -1, -2)
            self.mlp_proj = ttnn.as_tensor(mlp_proj, layout=ttnn.TILE_LAYOUT, device=self.device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, cache_file_name=tt_cache_path + "mlp_proj.bin")
        else:
            self.mlp_proj = ttnn.from_torch(torch.rand(1,1,self.hidden_size,2*self.hidden_size), layout=ttnn.TILE_LAYOUT, device=self.device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # down proj wt
        if self.args.d_model == self.hidden_size:
            print('**********using down proj wts')
            down_proj_weight_name = "mixer.out_proj.weight"
            down_proj = torch.transpose(self.state_dict[down_proj_weight_name], -1, -2)
            self.down_proj = ttnn.as_tensor(down_proj, layout=ttnn.TILE_LAYOUT, device=self.device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, cache_file_name=tt_cache_path + "down_proj.bin")
        else:
            self.down_proj = ttnn.from_torch(torch.rand(1,1,self.hidden_size*2,self.hidden_size), layout=ttnn.TILE_LAYOUT, device=self.device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # conv states
        self.conv_states = []
        for i in range(4):
            conv_state = torch.zeros((1,1,self.num_users,self.hidden_size*2))
            self.conv_states.append(ttnn.as_tensor(conv_state, layout=ttnn.TILE_LAYOUT, device=self.device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, cache_file_name=tt_cache_path + f"conv_state_{i}.bin"))
        self.conv_wts = []
        conv1d_weight_name = "mixer.conv1d.weight"
        for i in range(4):
            if self.args.d_model == self.hidden_size:
                print('**********using conv wts')
                conv_wts = torch.transpose(self.state_dict[conv1d_weight_name][:, :, i], -1, -2).unsqueeze(0).unsqueeze(0)
                print('**********conv wts', conv_wts.shape)
                self.conv_wts.append(ttnn.as_tensor(conv_wts, layout=ttnn.TILE_LAYOUT, device=self.device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, cache_file_name=tt_cache_path + f"conv_wts_{i}.bin"))
            else:
                self.conv_wts.append(ttnn.from_torch(torch.rand(1,1,1,self.hidden_size*2,), layout=ttnn.TILE_LAYOUT, device=self.device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16))
        # conv bias
        if self.args.d_model == self.hidden_size:
            print('**********using conv bias')
            conv1d_bias_name = "mixer.conv1d.bias"
            conv_bias = self.state_dict[conv1d_bias_name].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            print('**********conv bias', conv_bias.shape)
            self.conv_bias = ttnn.as_tensor(conv_bias, layout=ttnn.TILE_LAYOUT, device=self.device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, cache_file_name=tt_cache_path + "conv_bias.bin")
        else:
            self.conv_bias = ttnn.from_torch(torch.rand(1,1,1,self.hidden_size*2), layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


        self.tt_ssm = TtMambaSSM(self.args,self.device,load_fn,self.state_dict, num_users, hidden_size, configs, tt_cache_path)

    def forward(self, x):
        x_input = x # b, e=d_model
        x = ttnn.linear(x, self.ssm_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        # left shift conv states
        ttnn.deallocate(self.conv_states[0])
        for i in range(3):
            self.conv_states[i] = self.conv_states[i + 1]
        self.conv_states[3] = x

        # do the convolution
        conv_wts = ttnn.repeat_interleave(self.conv_wts[0], self.num_users, dim=2)
        x = ttnn.mul(conv_wts, self.conv_states[0], memory_config=ttnn.L1_MEMORY_CONFIG)
        for i in range(1,4):
            print('**********', self.conv_wts[i].shape, self.conv_states[i].shape)
            conv_wts = ttnn.repeat_interleave(self.conv_wts[i], self.num_users, dim=2)
            prod = ttnn.mul(conv_wts, self.conv_states[i], memory_config=ttnn.L1_MEMORY_CONFIG)
            x = ttnn.add(x, prod, memory_config=ttnn.L1_MEMORY_CONFIG)
        conv_bias = ttnn.repeat_interleave(self.conv_bias, self.num_users, dim=2)
        x = ttnn.add(x, conv_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.silu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        print('**********', x.shape)
        x = self.tt_ssm(x)
        res = ttnn.linear(x_input, self.mlp_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.mul(x, res, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(res)
        x = ttnn.linear(x, self.down_proj, memory_config=ttnn.L1_MEMORY_CONFIG)

        return x
