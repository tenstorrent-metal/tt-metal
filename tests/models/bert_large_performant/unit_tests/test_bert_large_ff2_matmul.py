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
import pytest


def run_bert_large_ff2_matmul_test(
    device, dtype, in0_mem_config, in1_mem_config, bias_mem_config, out_mem_config
):
    torch.manual_seed(1234)

    a_shape = [9, 1, 384, 4096]
    b_shape = [1, 1, 4096, 1024]
    bias_shape = [1, 1, 1, 1024]
    bias_pad_shape = [1, 1, 32, 1024]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95
    BIAS = torch.randint(-20, 20, bias_shape, dtype=torch.float)

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
    b_t = (
        ttl.tensor.Tensor(
            B.flatten().tolist(),
            b_shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in1_mem_config)
    )

    if bias_mem_config is not None:
        bias_t = (
            ttl.tensor.Tensor(
                BIAS.flatten().tolist(),
                bias_shape,
                dtype,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .pad(bias_pad_shape, [0, 0, 0, 0], 0)
            .to(ttl.tensor.Layout.TILE)
            .to(device, bias_mem_config)
        )
    else:
        bias_t = None

    t2 = ttl.tensor.bert_large_ff2_matmul(a_t, b_t, bias_t, out_mem_config)

    # Check memory of inputs and outputs
    assert a_t.memory_config().buffer_storage == in0_mem_config.buffer_storage
    assert b_t.memory_config().buffer_storage == in1_mem_config.buffer_storage
    if bias_mem_config is not None:
        assert bias_t.memory_config().buffer_storage == bias_mem_config.buffer_storage
    assert t2.memory_config().buffer_storage == out_mem_config.buffer_storage
    logger.debug(f"in0 is on: {a_t.memory_config().buffer_storage}")
    logger.debug(f"in1 is on: {b_t.memory_config().buffer_storage}")
    if bias_mem_config is not None:
        logger.debug(f"bias is on: {bias_t.memory_config().buffer_storage}")
    logger.debug(f"out is on: {t2.memory_config().buffer_storage}")

    assert t2.shape() == [9, 1, 384, 1024]
    tt_host_rm = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm = tt_host_rm.to_torch()

    ref_bmm = torch.matmul(A, B)
    if bias_mem_config is not None:
        ref_bmm = ref_bmm + BIAS
    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    logger.info(f"Passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "bias_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1),
        None,
    ),
    ids=["bias_DRAM", "bias_L1", "bias_None"],
)
@pytest.mark.parametrize(
    "in1_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1),
    ),
    ids=["in1_DRAM", "in1_L1"],
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
def test_bert_large_ff2_matmul_test(
    device,
    dtype,
    in0_mem_config,
    in1_mem_config,
    bias_mem_config,
    out_mem_config,
    request,
):
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/BERT_large_ff2_matmul_{request.node.callspec.id}"
    )
    run_bert_large_ff2_matmul_test(
        device, dtype, in0_mem_config, in1_mem_config, bias_mem_config, out_mem_config
    )


def test_bert_large_ff2_matmul_with_program_cache(device, use_program_cache):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    dram_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.DRAM)
    for _ in range(2):
        run_bert_large_ff2_matmul_test(
            device,
            dtype,
            dram_mem_config,
            dram_mem_config,
            dram_mem_config,
            dram_mem_config,
        )

    dram_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1)
    for _ in range(2):
        run_bert_large_ff2_matmul_test(
            device,
            dtype,
            dram_mem_config,
            dram_mem_config,
            dram_mem_config,
            dram_mem_config,
        )

    assert ttl.program_cache.num_entries() == 2
