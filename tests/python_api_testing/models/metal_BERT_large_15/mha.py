import pytest
from loguru import logger
import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from transformers import BertForQuestionAnswering
import numpy as np

from tests.python_api_testing.models.conftest import model_location_generator_
from libs import tt_lib as ttl
from libs.tt_lib.utils import pad_activation, pad_weight, print_diff_argmax
from libs.tt_lib.fused_ops.softmax import softmax
from utility_functions import enable_compile_cache, comp_pcc, comp_allclose, profiler
from tests.python_api_testing.models.metal_BERT_large_15.utils import (
    run_matmul_with_dataformat,
)


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = (
        ttl.tensor.Tensor(
            py_tensor.reshape(-1).tolist(),
            size,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(tt_device)
    )

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    host = ttl.device.GetHost()
    tt_output = tt_tensor.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    py_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())
    return py_output


def mha(qw, qb, kw, kb, vw, vb, hidden_dim, num_heads, device):
    assert isinstance(num_heads, int) and num_heads > 0

    # Weights pre-transposed on host​. No on-the fly transpose of W​
    qw = torch.transpose(qw, -1, -2)
    kw = torch.transpose(kw, -1, -2)
    vw = torch.transpose(vw, -1, -2)

    qkv_weight = torch.cat((qw, kw, vw), -1)
    qkv_bias = torch.cat((qb, kb, vb), -1)

    qkv_weight = torch2tt_tensor(qkv_weight, device)
    qkv_bias = torch2tt_tensor(qkv_bias, device)

    # Used to scale down the input to the softmax
    freciprocal_of_sqrt_hidden_dim = 1 / math.sqrt(hidden_dim // num_heads)
    reciprocal_of_sqrt_hidden_dim_tensor = ttl.tensor.Tensor(
        [1 / math.sqrt(hidden_dim // num_heads)] + [0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )

    def make_attention_heads(x):
        if num_heads == 1:
            return x
        else:
            # ref code from modeling_bert.py:
            #    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
            #        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            #        x = x.view(new_x_shape)
            #        return x.permute(0, 2, 1, 3)

            untilized_x = ttl.tensor.untilize(x)
            reshaped_unt = ttl.tensor.reshape(
                untilized_x,
                x.shape()[0],
                x.shape()[2],
                num_heads,
                x.shape()[3] // num_heads,
            )

            # N, 128, 2, 64
            transposed = ttl.tensor.transpose_hc(reshaped_unt)
            # N, 2, 128, 64
            retilized = ttl.tensor.tilize(transposed)
            return retilized

    def multiply_by_sqrt_hidden_dim(x):
        return ttl.tensor.bcast(
            x,
            reciprocal_of_sqrt_hidden_dim_tensor,
            ttl.tensor.BcastOpMath.MUL,
            ttl.tensor.BcastOpDim.HW,
        )

    def op1_qkv_fused(activation, qkv_weight, qkv_bias):
        # profiler.start("___op1_qkv_fused")
        qkv = run_matmul_with_dataformat(
            ttl.tensor.bert_large_fused_qkv_matmul,
            ttl.tensor.DataType.BFLOAT16,
            device,
            activation,
            qkv_weight,
        )
        qkv = ttl.tensor.bcast(
            qkv, qkv_bias, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H
        )
        # profiler.end("___op1_qkv_fused")

        return qkv

    def op2_split(qkv):
        # profiler.start("___op2_split")
        Q, K, V = ttl.tensor.bert_large_split_fused_qkv(qkv)
        """
        # Old TM on host with split
        qkv = tt2torch_tensor(qkv)
        hidden_dim = qkv.shape[-1] // 3

        (Q, K, V) = torch.split(qkv, hidden_dim, -1)

        Q = torch2tt_tensor(Q, device)
        K = torch2tt_tensor(K, device)
        V = torch2tt_tensor(V, device)
        """
        # profiler.end("___op2_split")

        return Q, K, V

    def op3_create_heads(Q):
        # profiler.start("___op3_make_attention_heads")
        q_heads = ttl.tensor.bert_large_create_q_head(Q)
        """
        # Old TM with reshape and transpose
        q_heads = make_attention_heads(Q)
        """
        # profiler.end("___op3_make_attention_heads")

        return q_heads

    def op4_create_heads(K):
        # profiler.start("___op4_make_attention_heads")
        # NOTE: This merges in transpose_hw (op6)
        k_heads = ttl.tensor.bert_large_create_k_head(K)
        """
        # Old TM with reshape and transpose
        k_heads = make_attention_heads(K)
        """
        # profiler.end("___op4_make_attention_heads")

        return k_heads

    def op5_create_heads(V):
        # profiler.start("___op5_make_attention_heads")
        v_heads = ttl.tensor.bert_large_create_v_head(V)
        """
        # Old TM with reshape and transpose
        v_heads = make_attention_heads(V)
        """
        # profiler.end("___op5_make_attention_heads")

        return v_heads

    def op6_transpose_hw(K):
        # profiler.start("___op6_transpose_hw")
        kt = ttl.tensor.transpose(K)
        # profiler.end("___op6_transpose_hw")

        return kt

    def op7_bmm(Q_heads, K_T_heads):
        # profiler.start("___op7_bmm")
        qkt = run_matmul_with_dataformat(
            ttl.tensor.bert_large_pre_softmax_bmm,
            ttl.tensor.DataType.BFLOAT16,
            device,
            Q_heads,
            K_T_heads,
        )
        # profiler.end("___op7_bmm")

        return qkt

    def op8_scale_mask_softmax(qkt, attention_mask):
        # Attention scores computation
        # profiler.start("___op8_scale_mask_softmax")

        N, C, H, W = qkt.shape()

        # ref = op8_scale_mask_softmax_ref(qkt, attention_mask)

        new_shape = [N, 1, C * H, W]
        ttl.tensor.reshape(qkt, *new_shape)

        if attention_mask is not None:
            attention_scores = ttl.tensor.scale_mask_softmax_in_place(
                freciprocal_of_sqrt_hidden_dim, attention_mask, qkt
            )
        else:
            attention_score_input = multiply_by_sqrt_hidden_dim(qkt)
            attention_scores = ttl.tensor.softmax_in_place(attention_score_input)
        ttl.tensor.reshape(
            attention_scores, N, C, H, W
        )  # Reshape back to original shape
        # profiler.end("___op8_scale_mask_softmax")

        return attention_scores

    def op8_scale_mask_softmax_ref(qkt, attention_mask):
        # Attention scores computation
        # profiler.start("___op8_scale_mask_softmax")

        N, C, H, W = qkt.shape()
        new_shape = [N, 1, C * H, W]
        ttl.tensor.reshape(qkt, *new_shape)
        attention_score_input = multiply_by_sqrt_hidden_dim(qkt)

        if attention_mask is not None:
            attention_score_input = ttl.tensor.bcast(
                attention_score_input,
                attention_mask,
                ttl.tensor.BcastOpMath.ADD,
                ttl.tensor.BcastOpDim.H,
            )

        attention_scores = softmax(attention_score_input)
        ttl.tensor.reshape(
            attention_scores, N, C, H, W
        )  # Reshape back to original shape
        # profiler.end("___op8_scale_mask_softmax")

        return attention_scores

    def op9_bmm(attention_scores, V_heads):
        # profiler.start("___op9_bmm")
        weighted_activation = run_matmul_with_dataformat(
            ttl.tensor.bert_large_post_softmax_bmm,
            ttl.tensor.DataType.BFLOAT16,
            device,
            attention_scores,
            V_heads,
        )
        # profiler.end("___op9_bmm")

        return weighted_activation

    def op10_unmake_attention_heads(x):
        if num_heads == 1:
            # profiler.start("___op10_unmake_attention_heads")
            # profiler.end("___op10_unmake_attention_heads")
            return x
        else:
            """
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            debug_state["context_reshaped"] = context_layer.clone()

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            """

            # profiler.start("___op10_unmake_attention_heads")
            ctx = ttl.tensor.transpose_hc(x)
            ushape = ctx.shape()
            reshaped = ttl.tensor.reshape(
                ctx, ushape[0], 1, ushape[1], ushape[2] * ushape[3]
            )
            retval = ttl.tensor.tilize(reshaped)
            # profiler.end("___op10_unmake_attention_heads")

            return retval

    def mha_(activation, attention_mask):
        # profiler.start("__mha")
        qkv = op1_qkv_fused(activation, qkv_weight, qkv_bias)
        Q, K, V = op2_split(qkv)

        Q_heads = op3_create_heads(Q)
        K_T_heads = op4_create_heads(K)
        V_heads = op5_create_heads(V)

        """
        # No longer needed as op4 already returns K_head transposed
        K_T_heads = op6_transpose_hw(K_heads)
        """
        qkt = op7_bmm(Q_heads, K_T_heads)

        attention_scores = op8_scale_mask_softmax(qkt, attention_mask)
        weighted_activation = op9_bmm(attention_scores, V_heads)

        res = op10_unmake_attention_heads(
            weighted_activation
        )  # [N, num heads, seq len, hid size / num heads] -> [N, seq len, hid size]
        # profiler.end("__mha")

        return res

    return mha_


class TtMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, config, encoder_idx, state_dict, device):
        super().__init__()
        qw = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.weight"]
        )
        qb = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.bias"]
        )
        kw = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.weight"]
        )
        kb = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.bias"]
        )
        vw = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.weight"]
        )
        vb = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.bias"]
        )

        # Hidden dim
        hidden_dim = qw.shape[-1]

        self.mha = mha(
            qw, qb, kw, kb, vw, vb, hidden_dim, config.num_attention_heads, device
        )

    def forward(self, activation, attention_mask=None):
        result = self.mha(activation, attention_mask)
        return result


class PytorchMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.mha = hugging_face_reference_model.bert.encoder.layer[0].attention.self

        # Disable dropout
        self.mha.eval()

    def forward(self, x):
        result = self.mha(x)[0]
        return result


def run_mha_inference(
    model_version, batch, seq_len, on_weka, pcc, model_location_generator
):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    # Initialize the device
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    if on_weka:
        model_name = str(
            model_location_generator(
                "tt_dnn-models/Bert/BertForQuestionAnswering/models/"
            )
            / model_version
        )
    else:
        model_name = model_version

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(
        model_name, torchscript=False
    )
    tt_mha_model = TtMultiHeadAttentionModel(
        hugging_face_reference_model.config,
        0,
        hugging_face_reference_model.state_dict(),
        device,
    )
    pytorch_mha_model = PytorchMultiHeadAttentionModel(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    mha_input = (
        torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size)
        * 2
    ) - 1
    pytorch_out = pytorch_mha_model(mha_input.squeeze(1)).unsqueeze(1)

    pad_mha_input = pad_activation(mha_input)
    tt_mha_input = ttl.tensor.Tensor(
        pad_mha_input.reshape(-1).tolist(),
        pad_mha_input.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE)
    tt_mha_input = tt_mha_input.to(device)

    tt_out = tt_mha_model(tt_mha_input).to(host)
    tt_out1 = torch.Tensor(tt_out.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        tt_out.shape()
    )

    ttl.device.CloseDevice(device)

    passing, output = comp_pcc(pytorch_out, tt_out1, pcc)
    logger.info(f"Output {output}")
    _, output = comp_allclose(
        pytorch_out, tt_out1, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Output {output}")
    if not passing:
        logger.error(f"Output PCC < {pcc}")

    # print_diff_argmax(pytorch_out, tt_out1)
    # assert np.allclose(pytorch_out.detach().numpy(), tt_out1, 1e-5, 0.17)


@pytest.mark.parametrize(
    "model_version, batch, seq_len, on_weka, pcc",
    (("phiyodr/bert-large-finetuned-squad2", 9, 384, True, 0.99),),
)
def test_mha_inference(
    model_version, batch, seq_len, on_weka, pcc, model_location_generator
):
    # enable_compile_cache()

    run_mha_inference(
        model_version, batch, seq_len, on_weka, pcc, model_location_generator
    )


if __name__ == "__main__":
    test_mha_inference(
        "phiyodr/bert-large-finetuned-squad2",
        9,
        384,
        True,
        0.99,
        model_location_generator_,
    )
