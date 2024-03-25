# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import pytest
import math

import tt_lib as ttl
from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from models.utility_functions import print_diff_argmax, comp_pcc
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero
from models.utility_functions import skip_for_wormhole_b0

# only use certain tests for CI to reduce run time
# grid_sizes = [(i, j) for i in range(1, 13) for j in range(1, 9)] # (1,1) to (12,8)
grid_sizes = [[1, 1], [1, 8], [12, 1], [12, 8]]
# seq_lens = [int(i*32) for i in range(1, 25)] # 32 - 768
seq_lens = [32, 64, 256, 384, 512]


@pytest.mark.parametrize(
    "scale_mask",
    [True, False],
    ids=["yes_mask", "no-mask"],
)
@pytest.mark.parametrize(
    "seq_len",
    seq_lens,
    ids=[f"seq_len_{x}" for x in seq_lens],
)
@pytest.mark.parametrize(
    "grid_size",
    grid_sizes,
    ids=[f"grid_size_{x}_{y}" for x, y in grid_sizes],
)
@pytest.mark.parametrize(
    "casual_mask",
    [True, False],
    ids=["yes_causal", "no-causal"],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.DataType.BFLOAT16,
    ),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
def test_softmax(device, in_dtype, casual_mask, grid_size, seq_len, scale_mask):
    torch.manual_seed(0)

    fuse_head = 768 // seq_len if 768 // seq_len > 0 else 1

    batch = grid_size[0]
    num_cores_r = grid_size[1]
    input_shape = (batch, num_cores_r * fuse_head, seq_len, seq_len)
    M = input_shape[2] * input_shape[1]
    K = input_shape[3] * batch

    hidden_dim = 1024
    num_heads = num_cores_r * fuse_head
    # scale = 1.0
    scale = 1 / math.sqrt(hidden_dim // num_heads)

    if casual_mask == False:
        # attention_mask = torch.zeros(batch, 1, 1, seq_len)
        attention_mask = torch.rand(batch, 1, 1, seq_len)
        attention_mask = (attention_mask > 0.5).float()
        attention_mask = attention_mask.reshape(batch, 1, -1, 32)
        attention_mask_t = ttl.tensor.Tensor(
            attention_mask,
            ttl.tensor.DataType.BFLOAT16,
        ).to(device, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1))
    else:
        # attention_mask = torch.zeros(batch, 1, seq_len, seq_len)
        attention_mask = torch.rand(batch, 1, seq_len, seq_len)
        attention_mask = (attention_mask > 0.5).float()
        attention_mask32 = tilize_to_list(pad_weight(attention_mask))
        attention_mask_t = ttl.tensor.Tensor(
            attention_mask32,
            [batch, 1, seq_len, seq_len],
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.TILE,
            device,
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        )

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    in1_t = torch2tt_tensor(input_tensor, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    in1_t_shard = ttl.tensor.interleaved_to_sharded(
        in1_t,
        grid_size,
        [M // grid_size[1], K // grid_size[0]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    block_w = seq_len // 32
    block_h = seq_len // 32 * fuse_head
    subblock_w = 1
    # calculate the max subblock size < 8
    upper_limit = min(block_w // 2, 8)
    for i in range(upper_limit, 0, -1):
        if block_w % i == 0:
            subblock_w = i
            break

    program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        im_data_format=ttl.tensor.DataType.BFLOAT16,
    )

    if scale_mask:
        tt_output_sharded = ttl.operations.primary.transformers.scale_mask_softmax_in_place(
            in1_t_shard, scale, attention_mask_t, program_config=program_config, is_causal_mask=casual_mask
        )
    else:
        tt_output_sharded = ttl.operations.primary.softmax_in_place(in1_t_shard, program_config=program_config)

    tt_output = ttl.tensor.sharded_to_interleaved(tt_output_sharded, in0_mem_config)
    tt_output_tensor = tt_output.cpu().to_torch().float()
    tt_output_tensor = torch.Tensor(tt_output_tensor).reshape(input_shape)
    tt_output_tensor = untilize(tt_output_tensor)

    if casual_mask == False:
        attention_mask = attention_mask.reshape(batch, 1, 1, seq_len)

    if scale_mask:
        golden_output_tensor = input_tensor * scale + attention_mask
    else:
        golden_output_tensor = input_tensor
    golden_output_tensor = torch.softmax(golden_output_tensor, dim=-1)

    allclose, output = comp_pcc(
        tt_output_tensor,
        golden_output_tensor,
    )
    logger.info(output)
    assert allclose, f"FAILED: {output}"
