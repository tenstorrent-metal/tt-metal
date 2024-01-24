# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import tt2torch_tensor, comp_pcc
import torch


"""
Falcon-7B shapes + functionality
"""


def run_nlp_create_qkv_heads_falcon7b_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    in0_shape = [batch, 1, seq_len, 4672]

    A = torch.randn(in0_shape)

    in0_t = ttl.tensor.Tensor(A, dtype).to(ttl.tensor.Layout.TILE).to(device, in0_mem_config)

    q, k, v = ttl.tensor.nlp_create_qkv_heads_falcon7b(in0_t, out_mem_config)

    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    assert k.memory_config().buffer_type == out_mem_config.buffer_type
    assert v.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {in0_t.memory_config().buffer_type} and {in0_t.dtype()}")
    logger.debug(f"q: {q.memory_config().buffer_type} and {q.dtype()}")
    logger.debug(f"k: {k.memory_config().buffer_type} and {k.dtype()}")
    logger.debug(f"v: {v.memory_config().buffer_type} and {v.dtype()}")

    assert list(q.shape()) == [batch, 71, seq_len, 64]
    assert list(k.shape()) == [batch, 1, seq_len, 64]
    assert list(v.shape()) == [batch, 1, seq_len, 64]

    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    (ref_q, ref_k, ref_v) = torch.split(A, [4544, 64, 64], dim=-1)
    # Additional shuffling for Q head
    ref_q = torch.reshape(ref_q, [batch, seq_len, 71, 64]).transpose(-3, -2)

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.info(f"Q passing={passing_pcc_q}")
    logger.info(f"Q output pcc={output_pcc_q}")
    assert passing_pcc_q
    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    logger.info(f"K passing={passing_pcc_k}")
    logger.info(f"K output pcc={output_pcc_k}")
    assert passing_pcc_k
    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)
    logger.info(f"V passing={passing_pcc_v}")
    logger.info(f"V output pcc={output_pcc_v}")
    assert passing_pcc_v


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch, seq_len",
    ((1, 32), (1, 64), (1, 128)),
    ids=[
        "batch1_seq32",
        "batch1_seq64",
        "batch1_seq128",
    ],
)
def test_nlp_create_qkv_heads_falcon7b_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, request, device):
    ttl.profiler.set_profiler_location(f"nlp_create_qkv_heads_falcon7b_tm_{request.node.callspec.id}")
    run_nlp_create_qkv_heads_falcon7b_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, device)


def test_nlp_create_qkv_heads_falcon7b_with_program_cache(use_program_cache, device):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    for _ in range(2):
        run_nlp_create_qkv_heads_falcon7b_test(1, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    for _ in range(2):
        run_nlp_create_qkv_heads_falcon7b_test(1, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    assert ttl.program_cache.num_entries() == 2


"""
Generic shapes + functionality
"""


def run_nlp_create_qkv_heads_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    transpose_k_heads,
    read_from_input_tensor_kv,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    torch.manual_seed(1234)

    if read_from_input_tensor_kv:
        in0_shape = [batch, 1, seq_len, num_q_heads * head_dim]
        in1_shape = [batch, 1, seq_len, 2 * num_kv_heads * head_dim]
        A = torch.randn(in0_shape)
        B = torch.randn(in1_shape)
        in0_t = ttl.tensor.Tensor(A, dtype).to(ttl.tensor.Layout.TILE).to(device, in_mem_config)
        in1_t = ttl.tensor.Tensor(B, dtype).to(ttl.tensor.Layout.TILE).to(device, in_mem_config)
    else:
        in0_shape = [batch, 1, seq_len, (num_q_heads + 2 * num_kv_heads) * head_dim]
        A = torch.randn(in0_shape)
        in0_t = ttl.tensor.Tensor(A, dtype).to(ttl.tensor.Layout.TILE).to(device, in_mem_config)

    q, k, v = ttl.tensor.nlp_create_qkv_heads(
        in0_t,
        in1_t if read_from_input_tensor_kv else None,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k_heads,
        output_mem_config=out_mem_config,
    )

    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_type == in_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    assert k.memory_config().buffer_type == out_mem_config.buffer_type
    assert v.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {in0_t.memory_config().buffer_type} and {in0_t.dtype()}")
    logger.debug(f"q: {q.memory_config().buffer_type} and {q.dtype()}")
    logger.debug(f"k: {k.memory_config().buffer_type} and {k.dtype()}")
    logger.debug(f"v: {v.memory_config().buffer_type} and {v.dtype()}")

    assert list(q.shape()) == [batch, num_q_heads, seq_len, head_dim]
    if transpose_k_heads:
        assert list(k.shape()) == [batch, num_kv_heads, head_dim, seq_len]
    else:
        assert list(k.shape()) == [batch, num_kv_heads, seq_len, head_dim]
    assert list(v.shape()) == [batch, num_kv_heads, seq_len, head_dim]

    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    if read_from_input_tensor_kv:
        ref_q = A
        (ref_k, ref_v) = torch.split(B, [num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)
    else:
        (ref_q, ref_k, ref_v) = torch.split(
            A, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
        )

    # Additional shuffling for Q, K, V heads
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    if transpose_k_heads:
        ref_k = ref_k.transpose(-2, -1)

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.info(f"Q passing={passing_pcc_q}")
    logger.info(f"Q output pcc={output_pcc_q}")

    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    logger.info(f"K passing={passing_pcc_k}")
    logger.info(f"K output pcc={output_pcc_k}")

    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)
    logger.info(f"V passing={passing_pcc_v}")
    logger.info(f"V output pcc={output_pcc_v}")
    assert passing_pcc_q
    assert passing_pcc_k
    assert passing_pcc_v


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["in_DRAM", "in_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_q_heads, num_kv_heads, transpose_k_heads, read_from_input_tensor_kv",
    (
        (1, 128, 64, 71, 1, False, False),
        (111, 64, 96, 5, 3, True, False),
        (5, 1024, 64, 8, 8, True, True),
    ),
)
def test_nlp_create_qkv_heads_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    transpose_k_heads,
    read_from_input_tensor_kv,
    dtype,
    in_mem_config,
    out_mem_config,
    request,
    device,
):
    run_nlp_create_qkv_heads_test(
        batch,
        seq_len,
        head_dim,
        num_q_heads,
        num_kv_heads,
        transpose_k_heads,
        read_from_input_tensor_kv,
        dtype,
        in_mem_config,
        out_mem_config,
        device,
    )


def test_nlp_create_qkv_heads_with_program_cache(use_program_cache, device):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    for _ in range(2):
        run_nlp_create_qkv_heads_test(5, 1024, 64, 4, 2, True, False, dtype, mem_config, mem_config, device)
        # Same in0_shape to make sure cache misses if we have additional optional tensor works
        run_nlp_create_qkv_heads_test(5, 1024, 64, 8, 8, True, True, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    assert ttl.program_cache.num_entries() == 2


def run_sharded_nlp_create_qkv_heads_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    read_from_input_tensor_kv,
    dtype,
    device,
):
    torch.manual_seed(1234)
    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = seq_len * batch // 32
    if num_cores == 1:
        pytest.skip("Issue #4706: Can't write 1 core sharded tensors directly to device")
    shard_grid = ttl.tensor.CoreRangeSet(ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True))

    if read_from_input_tensor_kv:
        in0_shape = [seq_len, 1, batch, num_q_heads * head_dim]
        in1_shape = [seq_len, 1, batch, 2 * num_kv_heads * head_dim]
        A = torch.randn(in0_shape)
        B = torch.randn(in1_shape)
        in0_shard_spec = ttl.tensor.ShardSpec(
            shard_grid,
            [
                32,
                in0_shape[-1],
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        )
        in1_shard_spec = ttl.tensor.ShardSpec(
            shard_grid,
            [
                32,
                in1_shape[-1],
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        )
        in0_t = ttl.tensor.Tensor(A, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config, in0_shard_spec)
        in1_t = ttl.tensor.Tensor(B, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config, in1_shard_spec)
    else:
        in0_shape = [seq_len, 1, batch, (num_q_heads + 2 * num_kv_heads) * head_dim]
        A = torch.randn(in0_shape)
        in0_shard_spec = ttl.tensor.ShardSpec(
            shard_grid,
            [
                32,
                in0_shape[-1],
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        )
        in0_t = ttl.tensor.Tensor(A, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config, in0_shard_spec)

    q, k, v = ttl.tensor.nlp_create_qkv_heads(
        in0_t,
        in1_t if read_from_input_tensor_kv else None,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        output_mem_config=mem_config,
    )

    assert list(q.shape()) == [seq_len, num_q_heads, batch, head_dim]
    assert list(k.shape()) == [seq_len, num_kv_heads, batch, head_dim]
    assert list(v.shape()) == [seq_len, num_kv_heads, batch, head_dim]

    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    if read_from_input_tensor_kv:
        ref_q = A
        (ref_k, ref_v) = torch.split(B, [num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)
    else:
        (ref_q, ref_k, ref_v) = torch.split(
            A, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
        )

    # Additional shuffling for Q, K, V heads
    ref_q = torch.reshape(ref_q, [seq_len, batch, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [seq_len, batch, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [seq_len, batch, num_kv_heads, head_dim]).transpose(-3, -2)

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.info(f"Q passing={passing_pcc_q}")
    logger.info(f"Q output pcc={output_pcc_q}")

    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    logger.info(f"K passing={passing_pcc_k}")
    logger.info(f"K output pcc={output_pcc_k}")

    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)
    logger.info(f"V passing={passing_pcc_v}")
    logger.info(f"V output pcc={output_pcc_v}")
    assert passing_pcc_q
    assert passing_pcc_k
    assert passing_pcc_v


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_q_heads, num_kv_heads, read_from_input_tensor_kv",
    (
        (32, 1, 64, 16, 1, False),
        (32, 1, 64, 16, 1, True),
        (32, 1, 64, 32, 1, False),
        (32, 1, 64, 32, 1, True),
        (32, 1, 64, 32, 32, False),
        (32, 1, 64, 32, 32, True),
    ),
)
def test_sharded_nlp_create_qkv_heads_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    read_from_input_tensor_kv,
    dtype,
    device,
):
    run_sharded_nlp_create_qkv_heads_test(
        batch,
        seq_len,
        head_dim,
        num_q_heads,
        num_kv_heads,
        read_from_input_tensor_kv,
        dtype,
        device,
    )


def test_sharded_nlp_create_qkv_heads_with_program_cache(use_program_cache, device):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    for _ in range(2):
        run_sharded_nlp_create_qkv_heads_test(32, 1, 64, 16, 8, False, dtype, device)
        # Same in0_shape to make sure cache misses if we have additional optional tensor works
        run_sharded_nlp_create_qkv_heads_test(32, 1, 64, 32, 1, True, dtype, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    assert ttl.program_cache.num_entries() == 2
