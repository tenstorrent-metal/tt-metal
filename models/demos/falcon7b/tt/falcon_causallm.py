# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from torch import nn
from typing import Optional, Tuple

import tt_lib
import ttnn

from models.demos.falcon7b.tt.falcon_model import TtFalconModelShared
from models.utility_functions import torch2tt_tensor


class TtFalconCausalLM(TtFalconModelShared):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        parameters,
    ):
        assert base_url == "", "base_url should be empty at the root of the model!"

        super().__init__(
            device=device,
            state_dict=state_dict,
            base_url=f"transformer",
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            parameters=parameters,
        )
        self.model_config = model_config
        self.lm_head_weights = parameters.lm_head.weight

    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        llm_mode: str,
        attention_mask: tt_lib.tensor.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> tt_lib.tensor.Tensor:
        hidden_states, presents = super().forward(
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )

        """
            auto seq_len = input_tensor_a.shape()[2];

    if (seq_len > 512) {
        // TODO: Check support for seq_len == 128, 256, 512, ..., 2048
        TT_FATAL(seq_len % TILE_HEIGHT == 0, "Falcon mm's seq_len must be a multiple of 32!");
        TT_FATAL(seq_len >=  128, "Falcon mm's seq_len must be greater than 128!");
        TT_FATAL((input_tensor_a.shape() == Shape({1, 1, seq_len, 4544})), "Unsupported input shape");
        TT_FATAL((input_tensor_b.shape() == Shape({1, 1, 4544, 65024})), "Unsupported input shape");
        return operation::run_with_autoformat(Matmul{.bcast_batch=true, .output_mem_config=mem_config, .output_dtype=output_dtype.value_or(input_tensor_a.dtype())}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
    } else {
        auto program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, true, std::nullopt, true, mem_config.is_sharded());
        return operations::primary::matmul_1d(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
    }
        lm_logits = ttnn.matmul(
            hidden_states,
            self.lm_head_weights,
            memory_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
            use_1d_systolic_array=True
        )
        """
        ##"""
        hidden_states = ttnn.unsqueeze_to_4D(hidden_states)
        lm_logits = ttnn.experimental.tensor.falcon_lm_head_matmul(
            hidden_states,
            self.lm_head_weights,
            bias=None,
            output_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
        )
        # """

        return lm_logits, presents
