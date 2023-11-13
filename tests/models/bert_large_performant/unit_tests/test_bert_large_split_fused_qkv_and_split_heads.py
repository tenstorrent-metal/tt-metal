# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

import numpy as np

import tt_lib as ttl
from models.utility_functions import (
    comp_pcc,
)
import torch


def run_split_fused_qkv_and_split_heads_test(
    device, batch, dtype, in0_mem_config, out_mem_config
):
    torch.manual_seed(1234)

    a_shape = [batch, 1, 384, 3072]

    A = torch.randn(a_shape)

    a_t = (
        ttl.tensor.Tensor(
            A.flatten().tolist(),
            a_shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in0_mem_config)
    )

    q, k, v = ttl.operations.primary.transformers.split_fused_qkv_and_split_heads(
        a_t, ttl.tensor.CoreCoord(12, 9), out_mem_config
    )

    # Check memory of inputs and outputs
    assert a_t.memory_config().buffer_storage == in0_mem_config.buffer_storage
    assert q.memory_config().buffer_storage == out_mem_config.buffer_storage
    assert k.memory_config().buffer_storage == out_mem_config.buffer_storage
    assert v.memory_config().buffer_storage == out_mem_config.buffer_storage
    logger.debug(f"in0: {a_t.memory_config().buffer_storage} and {a_t.dtype()}")
    logger.debug(f"q: {q.memory_config().buffer_storage} and {q.dtype()}")
    logger.debug(f"k: {k.memory_config().buffer_storage} and {k.dtype()}")
    logger.debug(f"v: {v.memory_config().buffer_storage} and {v.dtype()}")

    assert q.shape() == [batch, 16, 384, 64]
    assert k.shape() == [batch, 16, 64, 384]
    assert v.shape() == [batch, 16, 384, 64]

    tt_host_rm_q = q.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_q = tt_host_rm_q.to_torch()
    tt_host_rm_k = k.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_k = tt_host_rm_k.to_torch()
    tt_host_rm_v = v.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_v = tt_host_rm_v.to_torch()

    (ref_q, ref_k, ref_v) = torch.split(A, 1024, dim=-1)

    ref_q = ref_q.reshape([batch, 384, 16, 64]).transpose(-3, -2)
    ref_k = ref_k.reshape([batch, 384, 16, 64]).transpose(-3, -2).transpose(-2, -1)
    ref_v = ref_v.reshape([batch, 384, 16, 64]).transpose(-3, -2)

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, 0.99)
    logger.info(f"Q passing={passing_pcc_q}")
    logger.info(f"Q output pcc={output_pcc_q}")
    assert passing_pcc_q
    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, 0.99)
    logger.info(f"K passing={passing_pcc_k}")
    logger.info(f"K output pcc={output_pcc_k}")
    assert passing_pcc_k
    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, 0.99)
    logger.info(f"V passing={passing_pcc_v}")
    logger.info(f"V output pcc={output_pcc_v}")
    assert passing_pcc_v


import pytest


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch",
    (9, 8, 7),
    ids=[
        "batch_9",
        "batch_8",
        "batch_7",
    ],
)
def test_split_fused_qkv_and_split_heads(
    device, batch, dtype, in0_mem_config, out_mem_config, request
):
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/BERT_large_create_qvk_heads_tm_{request.node.callspec.id}"
    )
    run_split_fused_qkv_and_split_heads_test(
        device, batch, dtype, in0_mem_config, out_mem_config
    )


def test_split_fused_qkv_and_split_heads_with_program_cache(device, use_program_cache):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    dram_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)
    for _ in range(2):
        run_split_fused_qkv_and_split_heads_test(
            device, 9, dtype, dram_mem_config, dram_mem_config
        )

    dram_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1)
    for _ in range(2):
        run_split_fused_qkv_and_split_heads_test(
            device, 9, dtype, dram_mem_config, dram_mem_config
        )

    assert ttl.program_cache.num_entries() == 2
