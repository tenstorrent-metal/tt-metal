# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import pytest
import torch
import math

import tt_lib as ttl

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from tests.tt_eager.python_api_testing.sweep_tests import (
    pytorch_ops,
    tt_lib_ops,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from models.utility_functions import skip_for_wormhole_b0
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


def get_block_subblock_dim(grid_size, M, K, N):
    in0_block_w = K // grid_size[1] // 32  # 16
    in0_block_h = M // grid_size[0] // 32
    out_block_h = M // grid_size[0] // 32
    out_block_w = N // grid_size[1] // 32

    if out_block_w <= 8:
        out_subblock_w = out_block_w
        out_subblock_h = 8 // out_subblock_w
    else:
        out_subblock_h = 1
        out_subblock_w = 8 // out_subblock_h
        while out_block_w % out_subblock_w != 0:
            out_subblock_w = out_block_w // 2

    return in0_block_w, in0_block_h, out_block_h, out_block_w, out_subblock_h, out_subblock_w


@skip_for_wormhole_b0()
def test_ff1_to_ff2(device):
    interleaved_mem_config_L1 = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    interleaved_mem_config_DRAM = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    torch.manual_seed(1234)
    # device shape
    grid_size = (12, 8)

    # FF1 + GELU
    M = 4608
    K = 1024
    N = 4096
    in0_block_w, in0_block_h, out_block_h, out_block_w, out_subblock_h, out_subblock_w = get_block_subblock_dim(
        grid_size, M, K, N
    )

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]

    FF1_in0 = torch.randn(in0_shape).bfloat16().float()
    FF1_in1 = torch.randn(in1_shape).bfloat16().float()
    FF1_bias = torch.randn(bias_shape).bfloat16().float()

    FF1_in0_t = torch2tt_tensor(
        FF1_in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )
    FF1_in1_t = torch2tt_tensor(
        FF1_in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )
    FF1_bias_t = pad_by_zero(
        FF1_bias, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )[0]

    FF1_in0_t_sharded = ttl.tensor.interleaved_to_sharded(
        FF1_in0_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=True,
        fused_activation=(ttl.tensor.FusibleActivation.GELU, True),
    )

    logger.info("FF1")
    out_ff1_t = ttl.operations.primary.matmul(
        FF1_in0_t_sharded,
        FF1_in1_t,
        bias=FF1_bias_t,
        program_config=program_config,
        output_mem_config=sharded_mem_config,
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
    )
    logger.info("FF1_done")

    # FF2
    M = 4608
    K = 4096
    N = 1024
    in0_block_w, in0_block_h, out_block_h, out_block_w, out_subblock_h, out_subblock_w = get_block_subblock_dim(
        grid_size, M, K, N
    )

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]

    FF2_in1 = torch.randn(in1_shape).bfloat16().float()
    FF2_bias = torch.randn(bias_shape).bfloat16().float()

    FF2_in1_t = torch2tt_tensor(
        FF2_in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )
    FF2_bias_t = pad_by_zero(
        FF2_bias, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )[0]

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=True,
        fused_activation=None,
    )

    logger.info("FF2")
    out_ff2_t = ttl.operations.primary.matmul(
        out_ff1_t,
        FF2_in1_t,
        bias=FF2_bias_t,
        program_config=program_config,
        output_mem_config=sharded_mem_config,
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
    )
    logger.info("FF2_done")

    output_t = ttl.tensor.sharded_to_interleaved(out_ff2_t, interleaved_mem_config_DRAM)
    output_t = tt2torch_tensor(output_t)

    # compare results
    ref_ff1 = torch.nn.functional.gelu(FF1_in0 @ FF1_in1 + FF1_bias)
    ref_ff2 = ref_ff1 @ FF2_in1 + FF2_bias
    passing, output = comp_pcc(output_t, ref_ff2, 0.99)
    logger.info(output)
    assert passing


@skip_for_wormhole_b0()
def test_softmax(device):
    interleaved_mem_config_L1 = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    interleaved_mem_config_DRAM = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    torch.manual_seed(0)
    sm_op = ttl.operations.primary.transformers.scale_mask_softmax_in_place

    grid_size = (12, 8)
    batch = grid_size[0]
    input_shape = (batch, 1, 1024, 384)
    M = input_shape[2]
    K = input_shape[3] * batch

    hidden_dim = 1024
    num_heads = 16
    # scale = 1.0
    scale = 1 / math.sqrt(hidden_dim // num_heads)
    # attention_mask = torch.zeros(1, 1, 1, 384 * batch)
    attention_mask = torch.rand(batch, 1, 1, 384)
    attention_mask = (attention_mask > 0.5).float()
    attention_mask = attention_mask.reshape(batch, 1, 12, 32)
    attention_mask32 = tilize_to_list(pad_weight(attention_mask))
    attention_mask_t = ttl.tensor.Tensor(
        attention_mask32,
        [batch, 1, 32, 32],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    )

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = torch2tt_tensor(
        input_tensor, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )
    in1_t_shard = ttl.tensor.interleaved_to_sharded(
        in1_t,
        grid_size,
        [M // grid_size[1], K // grid_size[0]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=6,
        block_h=4,
        block_w=12,
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        im_data_format=ttl.tensor.DataType.BFLOAT16,
    )

    tt_output_sharded = sm_op(in1_t_shard, scale, attention_mask_t, program_config=program_config)

    tt_output = ttl.tensor.sharded_to_interleaved(tt_output_sharded, interleaved_mem_config_DRAM)
    tt_output_tensor = tt_output.cpu().to_torch().float()
    tt_output_tensor = torch.Tensor(tt_output_tensor).reshape(input_shape)
    tt_output_tensor = untilize(tt_output_tensor)

    attention_mask = attention_mask.reshape(batch, 1, 1, 384)

    for i in range(batch):
        golden_output_tensor = input_tensor[i] * scale + attention_mask[i]
        golden_output_tensor = torch.softmax(golden_output_tensor, dim=-1)

        allclose, output = comp_pcc(
            tt_output_tensor[i],
            golden_output_tensor,
        )
        logger.info(output)
        assert allclose, f"FAILED: {output}"

    # LN
    epsf = 1e-2

    in0_shape = (batch, 1, 384, 1024)
    M = in0_shape[2] * batch
    K = in0_shape[3]

    in0 = tt_output_tensor.reshape([batch, 1, 384, 1024])
    in0_t = torch2tt_tensor(
        in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )
    in0_t_shard = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    in1 = torch.rand(in0_shape) * 2 - 0.8
    in1_t = torch2tt_tensor(
        in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )
    in1_t_shard = ttl.tensor.interleaved_to_sharded(
        in1_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )
    gamma = torch.rand(in0_shape[3]) * 2 - 1
    beta = torch.rand(in0_shape[3]) * 2.0 - 1.1

    gamma = gamma.reshape(1, 1, -1, 32)
    gamma_t = ttl.tensor.Tensor(
        gamma.reshape(-1).tolist(),
        gamma.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, interleaved_mem_config_DRAM)

    beta = beta.reshape(1, 1, -1, 32)
    beta_t = ttl.tensor.Tensor(
        beta.reshape(-1).tolist(),
        beta.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, interleaved_mem_config_DRAM)

    program_config = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=batch,
        block_w=4,
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        im_data_format=ttl.tensor.DataType.BFLOAT16,
        out_data_format=ttl.tensor.DataType.BFLOAT8_B,
        inplace=True,
    )

    logger.info("Running add_LN_GB")
    ttz = ttl.operations.primary.add_layernorm(
        in0_t_shard,
        in1_t_shard,
        epsf,
        gamma_t,
        beta_t,
        output_mem_config=sharded_mem_config,
        program_config=program_config,
    )
    logger.info("Done")

    ttz = ttl.tensor.sharded_to_interleaved(ttz, interleaved_mem_config_DRAM)
    t2_data = ttz.cpu().to_torch().float()
    tt_got_back = torch.Tensor(t2_data).reshape(in0_shape)
    tt_got_back = untilize(tt_got_back)

    ref_lnorm = torch.nn.functional.layer_norm(
        tt_output_tensor.reshape([batch, 1, 384, 1024]) + in1, in1.shape[-1:], gamma.flatten(), beta.flatten(), epsf
    )

    passing, output = comp_pcc(tt_got_back, ref_lnorm, 0.999)
    logger.info(output)
    assert passing
