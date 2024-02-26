# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import ttnn


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
            loaded_tensor = ttnn.load_tensor(
                str(
                    tt_cache_path
                    / f"{dense_h_to_4h_str}_{self.model_config['DENSE_H_TO_4H_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            )
            self.dense_h_to_4h_weights = ttnn.to_device(
                loaded_tensor, device, memory_config=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"]
            )
        else:
            self.dense_h_to_4h_weights = ttnn.from_torch(
                torch.transpose(
                    self.state_dict[dense_h_to_4h_str],
                    -2,
                    -1,
                ),
                dtype=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_DTYPE"],
                device=self.device,
                memory_config=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"],
                layout=ttnn.TILE_LAYOUT,
            )
            self.dense_h_to_4h_weights = ttnn.unsqueeze_to_4D(self.dense_h_to_4h_weights)
            ttnn.dump_tensor(
                str(
                    tt_cache_path
                    / f"{dense_h_to_4h_str}_{self.model_config['DENSE_H_TO_4H_MM_WEIGHTS_DTYPE'].name}.bin"
                ),
                ttnn.from_device(self.dense_h_to_4h_weights),
            )

        if (
            tt_cache_path / f"{dense_4h_to_h_str}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
        ).exists():
            loaded_tensor = ttnn.load_tensor(
                str(
                    tt_cache_path
                    / f"{dense_4h_to_h_str}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            )
            self.dense_4h_to_h_weights = ttnn.to_device(
                loaded_tensor, device, memory_config=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"]
            )
        else:
            self.dense_4h_to_h_weights = ttnn.from_torch(
                torch.transpose(
                    self.state_dict[dense_4h_to_h_str],
                    -2,
                    -1,
                ),
                dtype=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_DTYPE"],
                device=self.device,
                memory_config=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"],
                layout=ttnn.TILE_LAYOUT,
            )
            self.dense_4h_to_h_weights = ttnn.unsqueeze_to_4D(self.dense_4h_to_h_weights)
            ttnn.dump_tensor(
                str(
                    tt_cache_path
                    / f"{dense_4h_to_h_str}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
                ),
                ttnn.from_device(self.dense_4h_to_h_weights),
            )

    def forward(self, x: ttnn.experimental.tensor.Tensor) -> ttnn.experimental.tensor.Tensor:
        """
        Tensor falcon_dense_h_to_4h_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, std::optional<UnaryWithParam> fused_activation, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
        auto seq_len = input_tensor_a.shape()[2];
        if (seq_len > 1024) {
            TT_FATAL(not fused_activation.has_value());
            // TODO: Check support for seq_len == 128, 256, 512, ..., 2048
            TT_FATAL(seq_len % TILE_HEIGHT == 0, "Falcon mm's seq_len must be a multiple of 32!");
            TT_FATAL(seq_len >=  128, "Falcon mm's seq_len must be greater than 128!");
            TT_FATAL((input_tensor_a.shape() == Shape({1, 1, seq_len, 4544})), "Unsupported input shape");
            TT_FATAL((input_tensor_b.shape() == Shape({1, 1, 4544, 18176})), "Unsupported input shape");
            TT_FATAL(!fused_activation.has_value());
            return operation::run_with_autoformat(Matmul{.bcast_batch=true, .output_mem_config=mem_config, .output_dtype=output_dtype.value_or(input_tensor_a.dtype())}, {input_tensor_a, input_tensor_b}).at(0);
        } else {
            auto program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, true, fused_activation, true, mem_config.is_sharded());
            return operations::primary::matmul_1d(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
        }
        }

        """

        hidden_states = ttnn.experimental.tensor.falcon_dense_h_to_4h_matmul(
            x,
            self.dense_h_to_4h_weights,
            fused_activation=[ttnn.experimental.tensor.FusibleActivation.GELU, True],
            output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
        )
        ttnn.deallocate(x)

        hidden_states = ttnn.experimental.tensor.falcon_dense_4h_to_h_matmul(
            hidden_states,
            self.dense_4h_to_h_weights,
            output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
        )

        # return TT Tensor
        return hidden_states
