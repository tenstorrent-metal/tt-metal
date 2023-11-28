# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from functools import partial

import tt_lib
from tt_lib.utils import pad_weight


def feed_forward(
    ffn_dim,
    hidden_dim,
    ff1_weighta,
    ff1_biasa,
    ff2_weighta,
    ff2_biasa,
    device,
    model_config,
):
    # Weights pre-transposed on host​. No on-the fly transpose of W.
    # activation = [1, 9, 384, 1024]
    # ff1_weighta = [1, 1, 1024, 4096]
    # output = [1, 9, 384, 4096]
    if "OP13_FF1_MM_CONFIG" in model_config:
        ff1_matmul = partial(
            tt_lib.operations.primary.matmul,
            program_config=model_config["OP13_FF1_MM_CONFIG"],
            output_mem_config=model_config["OP13_FF1_MM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP13_FF1_MM_OUTPUT_DTYPE"],
        )
    else:
        ff1_matmul = partial(
            tt_lib.tensor.bert_large_ff1_matmul,
            fused_activation=(tt_lib.tensor.FusibleActivation.GELU, True),
            output_mem_config=model_config["OP13_FF1_MM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP13_FF1_MM_OUTPUT_DTYPE"],
        )
    if "OP14_FF2_MM_CONFIG" in model_config:
        ff2_matmul = partial(
            tt_lib.operations.primary.matmul,
            program_config=model_config["OP14_FF2_MM_CONFIG"],
            output_mem_config=model_config["OP14_FF2_MM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP14_FF2_MM_OUTPUT_DTYPE"],
        )
    else:
        ff2_matmul = partial(
            tt_lib.tensor.bert_large_ff2_matmul,
            output_mem_config=model_config["OP14_FF2_MM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP14_FF2_MM_OUTPUT_DTYPE"],
        )

    def op13_MM_bias_gelu(activation, ff1_weighta, ff1_biasa):
        output_plus_bias_act = ff1_matmul(
            activation,
            ff1_weighta,
            bias=ff1_biasa,
        )
        return output_plus_bias_act

    # activation = [1, 9, 384, 4096]
    # ff2_weighta = [1, 1, 4096, 1024]
    # output = [1, 9, 384, 1024]
    def op14_MM_bias(activation, ff2_weighta, ff2_biasa):
        output_plus_bias = ff2_matmul(
            activation,
            ff2_weighta,
            bias=ff2_biasa,
        )
        return output_plus_bias

    def feed_forward_(activation: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        ff1_output_plus_bias_act = op13_MM_bias_gelu(activation, ff1_weighta, ff1_biasa)

        # Don't deallocate activations here since it is used by more ops in encoder

        ff2_output_plus_bias = op14_MM_bias(ff1_output_plus_bias_act, ff2_weighta, ff2_biasa)
        ff1_output_plus_bias_act.deallocate()

        return ff2_output_plus_bias

    return feed_forward_


class TtFeedForwardModel:
    def __init__(self, encoder_idx, state_dict, device, model_config, tt_cache_path):
        # FF1 params
        layer_name = f"bert.encoder.layer.{encoder_idx}"
        encoder_ff1_str = f"{layer_name}.intermediate.dense"
        encoder_ff2_str = f"{layer_name}.output.dense"
        if tt_cache_path is not None:
            encoder0_ff1_weight = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"{encoder_ff1_str}.weight_{model_config['OP13_FF1_MM_WEIGHTS_DTYPE'].name}.bin")
            ).to(device, model_config["OP13_FF1_MM_WEIGHTS_MEMCFG"])
            encoder0_ff1_bias = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"{encoder_ff1_str}.bias_{model_config['OP13_FF1_MM_BIAS_DTYPE'].name}.bin")
            ).to(device, model_config["OP13_FF1_MM_BIAS_MEMCFG"])
            encoder0_ff1_weight_shape = encoder0_ff1_weight.shape()

            encoder0_ff2_weight = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"{encoder_ff2_str}.weight_{model_config['OP14_FF2_MM_WEIGHTS_DTYPE'].name}.bin")
            ).to(device, model_config["OP14_FF2_MM_WEIGHTS_MEMCFG"])
            encoder0_ff2_bias = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"{encoder_ff2_str}.bias_{model_config['OP14_FF2_MM_BIAS_DTYPE'].name}.bin")
            ).to(device, model_config["OP14_FF2_MM_BIAS_MEMCFG"])
        else:
            encoder0_ff1_weight = pad_weight(
                torch.transpose(
                    state_dict[f"{encoder_ff1_str}.weight"],
                    -2,
                    -1,
                )
            )
            encoder0_ff1_bias = pad_weight(state_dict[f"{encoder_ff1_str}.bias"])

            encoder0_ff1_weight_shape = encoder0_ff1_weight.shape
            encoder0_ff1_bias_shape = encoder0_ff1_bias.shape

            encoder0_ff1_weight = (
                tt_lib.tensor.Tensor(
                    encoder0_ff1_weight.reshape(-1).tolist(),
                    encoder0_ff1_weight.shape,
                    model_config["OP13_FF1_MM_WEIGHTS_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(device, model_config["OP13_FF1_MM_WEIGHTS_MEMCFG"])
            )
            encoder0_ff1_bias = (
                tt_lib.tensor.Tensor(
                    encoder0_ff1_bias.reshape(-1).tolist(),
                    encoder0_ff1_bias.shape,
                    model_config["OP13_FF1_MM_BIAS_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(device, model_config["OP13_FF1_MM_BIAS_MEMCFG"])
            )

            # FF2 params
            encoder0_ff2_weight = pad_weight(
                torch.transpose(
                    state_dict[f"{encoder_ff2_str}.weight"],
                    -2,
                    -1,
                )
            )
            encoder0_ff2_bias = pad_weight(state_dict[f"{encoder_ff2_str}.bias"])

            encoder0_ff2_weight_shape = encoder0_ff2_weight.shape
            encoder0_ff2_bias_shape = encoder0_ff2_bias.shape

            encoder0_ff2_weight = (
                tt_lib.tensor.Tensor(
                    encoder0_ff2_weight.reshape(-1).tolist(),
                    encoder0_ff2_weight.shape,
                    model_config["OP14_FF2_MM_WEIGHTS_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(device, model_config["OP14_FF2_MM_WEIGHTS_MEMCFG"])
            )
            encoder0_ff2_bias = (
                tt_lib.tensor.Tensor(
                    encoder0_ff2_bias.reshape(-1).tolist(),
                    encoder0_ff2_bias.shape,
                    model_config["OP14_FF2_MM_BIAS_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(device, model_config["OP14_FF2_MM_BIAS_MEMCFG"])
            )

        self.ffn = feed_forward(
            *encoder0_ff1_weight_shape[-2:],
            encoder0_ff1_weight,
            encoder0_ff1_bias,
            encoder0_ff2_weight,
            encoder0_ff2_bias,
            device,
            model_config,
        )

    def __call__(self, activation: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        return self.ffn(activation)
