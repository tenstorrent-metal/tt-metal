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


def run_bert_large_concatenate_heads_test(
    device, batch, dtype, in0_mem_config, out_mem_config
):
    torch.manual_seed(1234)

    a_shape = [batch, 16, 384, 64]

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

    out = ttl.operations.primary.transformers.concatenate_heads(
        a_t, ttl.tensor.CoreCoord(12, 9), out_mem_config
    )

    # Check memory of inputs and outputs
    assert a_t.memory_config().buffer_storage == in0_mem_config.buffer_storage
    assert out.memory_config().buffer_storage == out_mem_config.buffer_storage

    logger.debug(f"in0: {a_t.memory_config().buffer_storage} and {a_t.dtype()}")
    logger.debug(f"out: {out.memory_config().buffer_storage} and {out.dtype()}")

    assert out.shape() == [batch, 1, 384, 1024]
    tt_host_rm_out = out.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_out = tt_host_rm_out.to_torch()

    ref_out = torch.transpose(A, -3, -2).reshape([batch, 1, 384, 1024])
    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm_out, ref_out, 0.99)
    logger.info(f"passing={passing_pcc}")
    logger.info(f"output pcc={output_pcc}")
    assert passing_pcc


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
def test_bert_large_concatenate_heads_test(
    device, batch, dtype, in0_mem_config, out_mem_config, request
):
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/BERT_large_concat_heads_tm_{request.node.callspec.id}"
    )
    run_bert_large_concatenate_heads_test(
        device, batch, dtype, in0_mem_config, out_mem_config
    )


def test_bert_large_concatenate_heads_with_program_cache(device, use_program_cache):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    dram_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)
    for _ in range(2):
        run_bert_large_concatenate_heads_test(
            device, 9, dtype, dram_mem_config, dram_mem_config
        )

    dram_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1)
    for _ in range(2):
        run_bert_large_concatenate_heads_test(
            device, 9, dtype, dram_mem_config, dram_mem_config
        )

    assert ttl.program_cache.num_entries() == 2
