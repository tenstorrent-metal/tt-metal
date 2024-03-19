# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
from loguru import logger

from models.utility_functions import torch2tt_tensor, is_wormhole_b0


class TtFalconMLP(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = hidden_size
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        dense_h_to_4h_str = f"{layer_name}.mlp.dense_h_to_4h.weight"
        dense_4h_to_h_str = f"{layer_name}.mlp.dense_4h_to_h.weight"

        if (
            tt_cache_path / f"{dense_h_to_4h_str}_{self.model_config['DENSE_H_TO_4H_MM_WEIGHTS_DTYPE'].name}.bin"
        ).exists():
            self.dense_h_to_4h_weights = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{dense_h_to_4h_str}_{self.model_config['DENSE_H_TO_4H_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"])
        else:
            self.dense_h_to_4h_weights = torch2tt_tensor(
                torch.transpose(
                    self.state_dict[dense_h_to_4h_str],
                    -2,
                    -1,
                ),
                self.device,
                tt_memory_config=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_DTYPE"],
            )
            tt_lib.tensor.dump_tensor(
                str(
                    tt_cache_path
                    / f"{dense_h_to_4h_str}_{self.model_config['DENSE_H_TO_4H_MM_WEIGHTS_DTYPE'].name}.bin"
                ),
                self.dense_h_to_4h_weights.cpu(),
            )

        if (
            tt_cache_path / f"{dense_4h_to_h_str}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
        ).exists():
            self.dense_4h_to_h_weights = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{dense_4h_to_h_str}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"])
        else:
            self.dense_4h_to_h_weights = torch2tt_tensor(
                torch.transpose(
                    self.state_dict[dense_4h_to_h_str],
                    -2,
                    -1,
                ),
                self.device,
                tt_memory_config=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_DTYPE"],
            )
            tt_lib.tensor.dump_tensor(
                str(
                    tt_cache_path
                    / f"{dense_4h_to_h_str}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
                ),
                self.dense_4h_to_h_weights.cpu(),
            )

    def forward(self, x: tt_lib.tensor.Tensor, llm_mode: str) -> tt_lib.tensor.Tensor:
        if is_wormhole_b0() and x.shape()[-2] == 1024 and llm_mode == "prefill":
            compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
                math_fidelity=tt_lib.tensor.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )

            program_config = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                in0_block_w=2,
                per_core_M=32,
                per_core_N=9,
                out_subblock_h=2,
                out_subblock_w=3,
                fuse_batch=True,
                fused_activation=[tt_lib.tensor.FusibleActivation.GELU, True],
                mcast_in0=True,
            )

            hidden_states = tt_lib.operations.primary.matmul(
                x,
                self.dense_h_to_4h_weights,
                program_config=program_config,
                output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
                compute_kernel_config=compute_kernel_config,
            )
        # elif is_wormhole_b0() and x.shape()[-2] == 2048 and llm_mode == "prefill":
        #     compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        #         math_fidelity=tt_lib.tensor.MathFidelity.LoFi,
        #         math_approx_mode=True,
        #         fp32_dest_acc_en=False,
        #         packer_l1_acc=True,
        #     )

        #     # Dimensions:
        #     # A = [2048, 4544]  = [64, 142]
        #     # B = [4544, 18176] = [142, 568]
        #     program_config = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        #         compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
        #         in0_block_w=1,
        #         per_core_M=8,
        #         per_core_N=71,
        #         out_subblock_h=8,
        #         out_subblock_w=1,
        #         transpose_mcast=False,
        #         fused_activation=[tt_lib.tensor.FusibleActivation.GELU, True],
        #     )

        #     hidden_states = tt_lib.operations.primary.matmul(
        #         x,
        #         self.dense_h_to_4h_weights,
        #         program_config=program_config,
        #         output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
        #         output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
        #         compute_kernel_config=compute_kernel_config,
        #     )

        else:
            hidden_states = tt_lib.tensor.falcon_dense_h_to_4h_matmul(
                x,
                self.dense_h_to_4h_weights,
                fused_activation=[tt_lib.tensor.FusibleActivation.GELU, True],
                output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
            )
            print("WIIIIIIIIIIIIN")

        x.deallocate()

        # 4h_to_h matmul
        if is_wormhole_b0() and hidden_states.shape()[-2] == 1024 and llm_mode == "prefill":
            compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
                math_fidelity=tt_lib.tensor.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            program_config = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                in0_block_w=8,
                per_core_M=4,
                per_core_N=18,
                out_subblock_h=1,
                out_subblock_w=6,
                transpose_mcast=False,
                fused_activation=None,
            )

            hidden_states = tt_lib.operations.primary.matmul(
                hidden_states,
                self.dense_4h_to_h_weights,
                program_config=program_config,
                output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                compute_kernel_config=compute_kernel_config,
            )
        elif is_wormhole_b0() and hidden_states.shape()[-2] == 2048 and llm_mode == "prefill":
            compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
                math_fidelity=tt_lib.tensor.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            program_config = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                in0_block_w=1,
                per_core_M=8,
                per_core_N=18,
                out_subblock_h=8,
                out_subblock_w=1,
                transpose_mcast=False,
                fused_activation=None,
            )
            hidden_states = tt_lib.operations.primary.matmul(
                hidden_states,
                self.dense_4h_to_h_weights,
                program_config=program_config,
                output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                compute_kernel_config=compute_kernel_config,
            )
        else:
            hidden_states = tt_lib.tensor.falcon_dense_4h_to_h_matmul(
                hidden_states,
                self.dense_4h_to_h_weights,
                output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                packer_l1_acc=True,
            )

        return hidden_states
