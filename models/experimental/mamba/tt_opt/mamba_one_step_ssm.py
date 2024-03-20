# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

import ttnn
from typing import Callable

from models.utility_functions import torch2tt_tensor
from models.helper_funcs import Linear
from models.experimental.mamba.reference.args import ModelArgs


class TtMambaSSM(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, state_dict, num_users, hidden_size, configs, tt_cache_path):
        super().__init__()


        self.state_dict = state_dict
        self.device = device
        self.args = args

        # hidden state
        self.num_users = num_users
        self.hidden_size = hidden_size * 2
        self.configs = configs
        self.n = 32
        self.rank = self.args.dt_rank


        self.tt_hidden_state = ttnn.zeros(
                (1, 1, self.num_users, self.hidden_size*self.n),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        """
        We need to split up the x_proj weights because in the reference
        implementation they perform the linear operation for dt, B, and C in a
        single step. Here we can't do that because it would involve fallback op
        slicing, so we break up the weights ahead of time and do the linear ops
        separately.
        """

        # delta rank weight
        if self.hidden_size == self.args.d_inner and self.rank == self.args.dt_rank:
            print('***********using delta rank weight')
            x_proj_weight_name = "mixer.x_proj.weight"
            delta_t_proj = torch.transpose(self.state_dict[x_proj_weight_name][: self.args.dt_rank, :], -1, -2)
            self.delta_t_proj = ttnn.as_tensor(
                delta_t_proj,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                cache_file_name=tt_cache_path + "/delta_t_proj.bin",
            )

        else:
            self.delta_t_proj = ttnn.from_torch(
                torch.rand(1, 1, self.hidden_size, self.rank),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )

        # delta full weight
        if self.hidden_size == self.args.d_inner and self.rank == self.args.dt_rank:
            print('***********using delta full weight')
            dt_proj_weight_name = "mixer.dt_proj.weight"
            dt_proj_bias_name = "mixer.dt_proj.bias"
            dt_proj_weights = torch.transpose(self.state_dict[dt_proj_weight_name], -1, -2)
            self.dt_proj_weights = ttnn.as_tensor(
                dt_proj_weights,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                cache_file_name=tt_cache_path + "/dt_proj_weights.bin",
            )
            self.dt_proj_bias = ttnn.as_tensor(
                self.state_dict[dt_proj_bias_name],
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                cache_file_name=tt_cache_path + "/dt_proj_bias.bin",
            )
        else:
            self.dt_proj_weights = ttnn.from_torch(
                torch.rand(1, 1, self.rank, self.hidden_size),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            self.dt_proj_bias = ttnn.from_torch(
                torch.rand(1, 1, 1, self.hidden_size),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
        # B
        if self.hidden_size == self.args.d_inner:
            print('***********using B weight')
            B_proj_weights = torch.transpose(self.state_dict[x_proj_weight_name][self.args.dt_rank : (self.args.dt_rank + self.args.d_state), :], -1, -2)
            # pad
            B_proj_weights = F.pad(B_proj_weights, (0, 16), "constant", 0)
            self.B_proj_weights = ttnn.as_tensor(
                B_proj_weights,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                cache_file_name=tt_cache_path + "/B_proj_weights.bin",
            )
        else:
            self.B_proj_weights = ttnn.from_torch(
                torch.rand(
                    1,
                    1,
                    self.hidden_size,
                    self.n,
                ),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
        assert self.n == 32
        B_intermediate_tranform_weights = torch.eye(self.n).repeat(1, self.hidden_size).unsqueeze(0).unsqueeze(0)
        self.B_intermediate = ttnn.as_tensor(
            B_intermediate_tranform_weights,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            cache_file_name=tt_cache_path + "/B_intermediate_weights.bin",
        )

        # A
        if self.hidden_size == self.args.d_inner:
            print('***********using A weight')
            A_weight_name = "mixer.A_log"
            def preprocess_A(x):
                x = -torch.exp(x.float())
                # padding wit inf
                x = F.pad(x, (0, 16), "constant", float("-inf"))
                x = x.reshape(1, self.hidden_size*32)  # (1, 2en)
                return x.repeat(self.num_users, 1) # b, 2en
            
            A = preprocess_A(self.state_dict[A_weight_name])
            
            self.A = ttnn.as_tensor(
                A,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                cache_file_name=tt_cache_path + "/A.bin",
            )

        else:
            self.A = ttnn.from_torch(torch.rand(1, 1, self.num_users, self.hidden_size*self.n), layout=ttnn.TILE_LAYOUT, device=self.device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # C
        if self.hidden_size == self.args.d_inner:
            print('***********using C weight')
            x_proj_weight_name = "mixer.x_proj.weight"
            C_proj_weights = torch.transpose(self.state_dict[x_proj_weight_name][(self.args.dt_rank + self.args.d_state) :, :], -1, -2)
            # pad
            C_proj_weights = F.pad(C_proj_weights, (0, 16), "constant", 0)
            self.C_proj = ttnn.as_tensor(
                C_proj_weights,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                cache_file_name=tt_cache_path + "/C_proj_weights.bin",
            )
        else:
            self.C_proj = ttnn.from_torch(
                    torch.rand(
                        1,
                        1,
                        self.hidden_size,
                        self.n,
                    ),
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                )

        # C pad
        # C_pad = torch.zeros(1,1,self.num_users,self.n)
        # self.C_pad = ttnn.from_torch(C_pad, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # D
        if self.hidden_size == self.args.d_inner:
            print('***********using D weight')
            D_weight_name = "mixer.D"
            D = self.state_dict[D_weight_name]
            D = D.repeat(self.num_users, 1)
            self.D = ttnn.as_tensor(
                D,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=tt_cache_path + "/D.bin",
            )
        else:
            self.D = ttnn.from_torch(
                torch.rand(1, 1, self.num_users, self.hidden_size),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def forward(self, x):
        # delta
        delta_t_proj = ttnn.to_memory_config(self.delta_t_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        print('**********delta_t_proj shape', delta_t_proj.shape, x.shape)
        delta_t0 = ttnn.linear(x, delta_t_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(delta_t_proj)

        dt_proj_weights = ttnn.to_memory_config(self.dt_proj_weights, memory_config=self.configs["sharded_rank"])
        delta_t1 = ttnn.linear(
            delta_t0, self.dt_proj_weights, bias=self.dt_proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(delta_t0)
        ttnn.deallocate(dt_proj_weights)
        delta_t2 = ttnn.softplus(delta_t1, parameter1=1.0, parameter2=20.0, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(delta_t1)

        # calculate abar
        delta_t3 = ttnn.repeat_interleave(delta_t2, self.n, dim=3)
        delta_t4 = ttnn.to_memory_config(delta_t3, memory_config=self.configs["sharded_large"])
        #ttnn.deallocate(delta_t3)
        abar0 = ttnn.to_memory_config(self.A, memory_config=self.configs["sharded_large"])
        abar1 = ttnn.mul(delta_t4, abar0, memory_config=self.configs["sharded_large"])
        ttnn.deallocate(abar0)
        ttnn.deallocate(delta_t4)
        abar2 = ttnn.to_memory_config(abar1, memory_config=ttnn.L1_MEMORY_CONFIG)
        #ttnn.deallocate(abar1)
        abar3 = ttnn.exp(abar2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(abar2)
        abar4 = ttnn.to_memory_config(abar3, memory_config=self.configs["sharded_large"])
        #ttnn.deallocate(abar3)

        # multiply abar and hidden_state
        hidden_state0 = ttnn.to_memory_config(self.tt_hidden_state, memory_config=self.configs["sharded_large"])
        amulh0 = ttnn.mul(abar4, hidden_state0, memory_config=self.configs["sharded_large"])

        # deallocate abar and hidden_state
        ttnn.deallocate(abar4)
        ttnn.deallocate(hidden_state0)

        # B
        B_proj_weights = ttnn.to_memory_config(self.B_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        B0 = ttnn.linear(x, B_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(B_proj_weights)
        B1 = ttnn.repeat(B0, ttnn.Shape([1, 1, 1, self.hidden_size], [1, 1, 32, self.hidden_size]))
        ttnn.deallocate(B0)
        B2 = ttnn.to_memory_config(B1, memory_config=self.configs["sharded_large"])
        #ttnn.deallocate(B1)

        # bbar
        #delta_t3 = ttnn.repeat_interleave(delta_t2, self.n, dim=3)
        ttnn.deallocate(delta_t2)
        delta_t4 = ttnn.to_memory_config(delta_t3, memory_config=self.configs["sharded_large"])
        #ttnn.deallocate(delta_t3)
        bbar0 = ttnn.mul(delta_t4, B2, memory_config=self.configs["sharded_large"])
        ttnn.deallocate(delta_t4)
        ttnn.deallocate(B2)

        # multiply bbar and x
        x0 = ttnn.repeat_interleave(x, self.n, dim=3)
        x1 = ttnn.to_memory_config(x0, memory_config=self.configs["sharded_large"])
        #ttnn.deallocate(x0)
        bmulx0 = ttnn.mul(bbar0, x1, memory_config=self.configs["sharded_large"])

        # deallocate bbar
        ttnn.deallocate(bbar0)
        ttnn.deallocate(x1)

        # add amulh and bmulx
        hidden_state1 = ttnn.add(amulh0, bmulx0, memory_config=self.configs["sharded_large"])
        ttnn.deallocate(self.tt_hidden_state)
        self.tt_hidden_state = ttnn.to_memory_config(hidden_state1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # deallocate amulh and bmulx
        ttnn.deallocate(amulh0)
        ttnn.deallocate(bmulx0)

        # compute C
        C_proj = ttnn.to_memory_config(self.C_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        C0 = ttnn.linear(x, C_proj, memory_config=ttnn.L1_MEMORY_CONFIG)  # b,n
        ttnn.deallocate(C_proj)
        C1 = ttnn.permute(C0, (0, 2, 3, 1))  # b,n,1
        ttnn.deallocate(C0)
        '''
        print("**********C0 shape", C0.shape)
        C1 = ttnn.concat([C0, self.C_pad], dim=3)  # b,32
        C2 = ttnn.concat([self.C_pad, C0], dim=3)  # b,32
        ttnn.deallocate(C0)
        C3 = ttnn.permute(C1, (0, 2, 3, 1))  # b,32,1
        ttnn.deallocate(C1)
        C4 = ttnn.permute(C2, (0, 2, 3, 1))  # b,32,1
        ttnn.deallocate(C2)
        C5 = ttnn.concat([C3, C4], dim=3)  # b,32,2
        print("**********C5 shape", C5.shape)
        ttnn.deallocate(C3)
        ttnn.deallocate(C4)
        '''

        # hidden state @ C
        hidden_state1 = ttnn.to_memory_config(hidden_state1, memory_config=ttnn.L1_MEMORY_CONFIG)
        #ttnn.deallocate(hidden_state1)
        hidden_state3 = ttnn.to_torch(hidden_state1)
        ttnn.deallocate(hidden_state1)
        #hidden_state3 = ttnn.reshape(hidden_state2, (1, self.num_users, self.hidden_size, self.n))  # b, d, 32
        hidden_state3 = hidden_state3.reshape(1, self.num_users, self.hidden_size, self.n)  # b, d, 32
        hidden_state3 = ttnn.from_torch(hidden_state3, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        C2 = ttnn.matmul(hidden_state3, C1, memory_config=ttnn.L1_MEMORY_CONFIG)  # b, d, 1
        ttnn.deallocate(C1)
        C3 = ttnn.permute(C2, (0, 3, 1, 2)) # b, d
        ttnn.deallocate(C2)
        '''
        hidden_state2 = ttnn.to_memory_config(hidden_state1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(hidden_state1)
        hidden_state3 = ttnn.reshape(hidden_state2, (1, self.num_users, self.hidden_size // 2, 32))  # b, d/2, 32
        C6 = ttnn.matmul(hidden_state3, C5)  # b, d/2, 2
        print("**********C6 shape", C6.shape, hidden_state3.shape, C5.shape)
        ttnn.deallocate(hidden_state3)
        ttnn.deallocate(C5)
        C7 = ttnn.permute(C6, (0, 3, 2, 1))  # 2, d/2, b
        ttnn.deallocate(C6)
        C8 = ttnn.reshape(C7, (1, 1, self.hidden_size, self.num_users))  # 1, 1, d, b
        C9 = ttnn.permute(C8, (0, 1, 3, 2))  # 1, 1, b, d
        ttnn.deallocate(C7)
        ttnn.deallocate(C9)
        '''

        # x * D
        xD = ttnn.mul(x, self.D, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x)

        # add xD and x
        print("**********xD shape", xD.shape, C3.shape)
        output = ttnn.add(xD, C3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(xD)
        ttnn.deallocate(C3)

        return output
