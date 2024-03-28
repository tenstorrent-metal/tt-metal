# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import ttnn
from models.demos.falcon7b.tt.model_utils import get_weights_cached
from torch import nn


class TtFalconMLP(nn.Module):
    def __init__(
        self, devices, state_dict, base_url, layer_num, hidden_size: int, model_config, tt_cache_path, seq_len
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.hidden_size = hidden_size
        self.model_config = model_config
        self.padding_size = 4608  # Experimentally determined padding size for 2048 sequence length
        self.seq_len = seq_len
        layer_name = f"{base_url}.{layer_num}"
        dense_h_to_4h_str = f"{layer_name}.mlp.dense_h_to_4h.weight"
        dense_4h_to_h_str = f"{layer_name}.mlp.dense_4h_to_h.weight"

        self.dense_h_to_4h_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            dense_h_to_4h_str,
            weight_config_str="DENSE_H_TO_4H_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[dense_h_to_4h_str], -2, -1) if state_dict else None),
            custom_output_shape=(1, 1, self.padding_size, 4 * self.padding_size) if self.seq_len == 2048 else None,
        )
        self.dense_4h_to_h_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            dense_4h_to_h_str,
            weight_config_str="DENSE_4H_TO_H_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[dense_4h_to_h_str], -2, -1) if state_dict else None),
            custom_output_shape=(1, 1, 4 * self.padding_size, self.padding_size) if self.seq_len == 2048 else None,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        hidden_states = []

        if self.seq_len == 2048:
            # pad x last dim with zeros to match self.padding_size
            # concat with zeros tensor since it's faster
            for device_id in range(len(x)):
                # create tensor, put it on device and concat with existing tensor
                tt_padding = (
                    torch.zeros((1, 1, x[device_id].shape[-2], self.padding_size - x[device_id].shape[-1]))
                    .bfloat16()
                    .float()
                )
                # move tensor to device
                tt_padding = ttnn.from_torch(
                    tt_padding,
                    ttnn.bfloat16,
                    device=self.devices[device_id],
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                # concat tensors
                x[device_id] = ttnn.concat([x[device_id], tt_padding], dim=3)

            grid_size = (8, 8)
            out_shape = [(1, 1, x[i].shape[-2], self.dense_4h_to_h_weights[i].shape[-1]) for i in range(len(x))]
            out_tensors = [torch.zeros(out_shape[i]).bfloat16().float() for i in range(len(x))]
            out_tt = [
                ttnn.from_torch(
                    out_tensors[i],
                    ttnn.bfloat16,
                    device=self.devices[i],
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for i in range(len(x))
            ]

            dram_interleaved_mem_cfg = tt_lib.tensor.MemoryConfig(
                memory_layout=tt_lib.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=tt_lib.tensor.BufferType.DRAM,
            )
            padded_hidden_size = 4608
            num_slices = 2
            for slice_idx in range(num_slices):
                slices = [
                    tt_lib.tensor.interleaved_to_sharded_partial(
                        x[device_id],
                        grid_size,
                        [self.seq_len // num_slices // grid_size[1], padded_hidden_size // grid_size[0]],
                        num_slices,
                        slice_idx,
                        tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                        tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                    )
                    for device_id in range(len(x))
                ]

                prog_cfg = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    in0_block_w=padded_hidden_size // grid_size[1] // 32 // 6,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    per_core_M=x[0].shape[-2] // (num_slices * grid_size[1] * 32),  # 32
                    per_core_N=4 * padded_hidden_size // (grid_size[0] * 32),  # 12
                    transpose_mcast=False,
                    fused_activation=[tt_lib.tensor.FusibleActivation.GELU, True],
                )
                kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
                    math_fidelity=tt_lib.tensor.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                )

                hidden_states = [
                    ttnn.matmul(
                        slices[device_id],
                        self.dense_h_to_4h_weights[device_id],
                        program_config=prog_cfg,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                        compute_kernel_config=kernel_config,
                    )
                    for device_id in range(len(x))
                ]
                for i in range(len(x)):
                    slices[i].deallocate()

                prog_cfg = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    in0_block_w=4 * padded_hidden_size // grid_size[1] // 32 // 9,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    per_core_M=x[0].shape[-2] // (num_slices * grid_size[1] * 32),  # 4
                    per_core_N=padded_hidden_size // (grid_size[0] * 32),  # 18
                    transpose_mcast=False,
                    fused_activation=None,
                )

                out_data = [
                    ttnn.matmul(
                        hidden_states[device_id],
                        self.dense_4h_to_h_weights[device_id],
                        program_config=prog_cfg,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                        compute_kernel_config=kernel_config,
                    )
                    for device_id in range(len(x))
                ]
                for i in range(len(x)):
                    hidden_states[i].deallocate()

                for device_id in range(len(x)):
                    tt_lib.tensor.sharded_to_interleaved_partial(
                        out_data[device_id], out_tt[device_id], num_slices, slice_idx, dram_interleaved_mem_cfg
                    )
                    out_data[device_id].deallocate()

            # remove padding from output
            hidden_states = [out_tensor[..., : self.hidden_size] for out_tensor in out_tt]

        else:
            for device_id in range(len(x)):
                hidden_states.append(
                    tt_lib.tensor.falcon_dense_h_to_4h_matmul(
                        x[device_id],
                        self.dense_h_to_4h_weights[device_id],
                        fused_activation=[tt_lib.tensor.FusibleActivation.GELU, True],
                        output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
                        output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
                    )
                )
                x[device_id].deallocate()
            for device_id in range(len(x)):
                hidden_states[device_id] = tt_lib.tensor.falcon_dense_4h_to_h_matmul(
                    hidden_states[device_id],
                    self.dense_4h_to_h_weights[device_id],
                    output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                    packer_l1_acc=True,
                )

        # return TT Tensor
        return hidden_states
