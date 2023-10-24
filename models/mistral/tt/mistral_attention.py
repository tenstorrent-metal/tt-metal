# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, Tuple
import tt_lib
from tt_lib import fallback_ops
from models.mistral.tt.mistral_configuration import TtModelArgs
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.helper_funcs import Linear as TtLinear


class TtAttention(nn.Module):
    def __init__(
        self,
        args: TtModelArgs,
        base_address=None,
        device=None,
        state_dict=None,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.base_address = base_address
        self.state_dict = state_dict

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = self.args.sliding_window

        self.scale = self.args.head_dim**-0.5

        self.wq_weights = torch_to_tt_tensor_rm(self.state_dict[f"{base_address}wq.weight"], self.device)
        self.wq = TtLinear(
            args.dim,
            args.n_heads * args.head_dim,
            self.wq_weights,
        )

        self.wk_weights = torch_to_tt_tensor_rm(self.state_dict[f"{base_address}wk.weight"], self.device)
        self.wk = TtLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            self.wk_weights,
        )

        self.wv_weights = torch_to_tt_tensor_rm(self.state_dict[f"{base_address}wv.weight"], self.device)
        self.wv = TtLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            self.wv_weights,
        )

        self.wo_weights = torch_to_tt_tensor_rm(self.state_dict[f"{base_address}wo.weight"], self.device)
        self.wo = TtLinear(
            args.n_heads * args.head_dim,
            args.dim,
            self.wo_weights,
        )

        self.cache_k = tt_lib.tensor.empty(
            (args.max_batch_size, args.sliding_window, self.n_kv_heads, self.args.head_dim)
        )
        self.cache_v = tt_lib.tensor.empty(
            (args.max_batch_size, args.sliding_window, self.n_kv_heads, self.args.head_dim)
        )

    def forward(
        self,
        x: tt_lib.tensor.Tensor,
        freqs_cis: tt_lib.tensor.Tensor,
        positions: tt_lib.tensor.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tt_lib.tensor.Tensor:
        _, bsz, seqlen, _ = x.shape()
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = fallback_ops.reshape(xq, bsz, seqlen, self.n_heads, self.args.head_dim)

        xk = fallback_ops.reshape(xk, bsz, seqlen, self.n_kv_heads, self.args.head_dim)

        xv = fallback_ops.reshape(xv, bsz, seqlen, self.n_kv_heads, self.args.head_dim)

        xq = tt_to_torch_tensor(xq).to(torch.float32)
        xk = tt_to_torch_tensor(xk).to(torch.float32)
        xv = tt_to_torch_tensor(xv).to(torch.float32)

        freqs_cis = tt_to_torch_tensor(freqs_cis).squeeze(0).squeeze(0)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # The cache is a rotating buffer
        positions = tt_to_torch_tensor(positions).squeeze(0).squeeze(0).squeeze(0)
        scatter_pos = (positions[-self.sliding_window :] % self.sliding_window)[None, :, None, None]
        scatter_pos = scatter_pos.to(torch.int64)
        scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.args.head_dim)
        self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window :])
        self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window :])

        if positions.shape[0] > 1:
            # prefill
            key, value = repeat_kv(xk, xv, self.repeats)
        else:
            cur_pos = positions[-1].item() + 1
            key, value = repeat_kv(self.cache_k[:bsz, :cur_pos, ...], self.cache_v[:bsz, :cur_pos, ...], self.repeats)

        xq = torch_to_tt_tensor_rm(xq, self.device)
        query = tt_lib.tensor.transpose_hc(xq)
        key = tt_lib.tensor.transpose_hc(key)
        value = tt_lib.tensor.transpose_hc(value)

        key = tt_lib.tensor.transpose(key)
        scores = tt_lib.tensor.bmm(query, key)
        scores = tt_lib.tensor.mul_unary(scores, self.scale)

        scores = tt_to_torch_tensor(scores)
        # mask = tt_to_torch_tensor(mask).squeeze(0).squeeze(0).squeeze(0)
        if mask is not None:
            scores += mask[None, None, ...]

        query = tt_to_torch_tensor(query)
        scores = torch_to_tt_tensor_rm(scores, self.device, put_on_device=False)

        scores = tt_lib.operations.primary.softmax_in_place(scores) #last-dim
        output = tt_lib.tensor.bmm(scores, value)  # (bs, n_local_heads, slen, head_dim)

        output = tt_lib.tensor.transpose_hc(output)

        output = fallback_ops.reshape(output, 1, bsz, seqlen, -1)
        return self.wo(output)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int) -> tt_lib.tensor.Tensor:
    dim = 2
    keys = tt_lib.tensor.repeat_interleave(keys, repeats, dim)
    values = tt_lib.tensors.repeat_interleave(values, repeats, dim)
    return keys, values
