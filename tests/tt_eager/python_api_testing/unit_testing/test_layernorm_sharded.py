# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import pytest
import torch

import tt_lib as ttl

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from models.utility_functions import is_wormhole_b0, skip_for_wormhole_b0
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1),),
    ids=["out_L1"],
)
@pytest.mark.parametrize(
    "gamma_beta_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
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
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2, 3),
    ids=["LN", "LN_G", "LN_GB", "add_LN_GB"],
)
def test_layernorm_sharded(test_id, dtype, in0_mem_config, gamma_beta_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    grid_size = (12, 8)
    fidelity = ttl.tensor.MathFidelity.HiFi4

    epsf = 1e-2

    in0_shape = (1, 1, 12 * 384, 1024)
    M = in0_shape[2]
    K = in0_shape[3]

    in0 = torch.rand(in0_shape) * 2 - 0.95
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=dtype)
    in0_t_shard = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    if test_id < 3:
        in1 = torch.zeros(in0_shape)
    else:
        in1 = torch.rand(in0_shape) * 2 - 0.8

    in1_t = torch2tt_tensor(in1, device, tt_memory_config=in0_mem_config, tt_dtype=dtype)
    in1_t_shard = ttl.tensor.interleaved_to_sharded(
        in1_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    if test_id == 0:
        gamma = torch.ones(1, 1, 32, in0_shape[3])
        beta = torch.zeros(1, 1, 32, in0_shape[3])
    if test_id == 1:
        gamma = torch.rand(1, 1, 32, in0_shape[3]) * 2 - 1
        beta = torch.zeros(1, 1, 32, in0_shape[3])
    if test_id >= 2:
        gamma = torch.rand(1, 1, 32, in0_shape[3]) * 2 - 1
        beta = torch.rand(1, 1, 32, in0_shape[3]) * 2.0 - 1.1

    gamma_t = torch2tt_tensor(gamma, device, tt_memory_config=gamma_beta_mem_config, tt_dtype=dtype)
    beta_t = torch2tt_tensor(beta, device, tt_memory_config=gamma_beta_mem_config, tt_dtype=dtype)

    program_config = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=12,
        block_w=4,
        math_fidelity=fidelity,
        data_format=dtype,
    )

    if test_id == 0:
        logger.info("Running LN_NOGB")
        ttz = ttl.operations.primary.layernorm(
            in0_t_shard,
            epsf,
            program_config=program_config,
            output_mem_config=out_mem_config,
        )
    if test_id == 1:
        logger.info("Running LN_G")
        ttz = ttl.operations.primary.layernorm(
            in0_t_shard,
            epsf,
            gamma_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
        )
    if test_id == 2:
        logger.info("Running LN_GB")
        ttz = ttl.operations.primary.layernorm(
            in0_t_shard,
            epsf,
            gamma_t,
            beta_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
        )
    if test_id == 3:
        logger.info("Running add_LN_GB")
        ttz = ttl.operations.primary.add_layernorm(
            in0_t_shard,
            in1_t_shard,
            epsf,
            gamma_t,
            beta_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )

    logger.info("Done")

    ttz = ttl.tensor.sharded_to_interleaved(ttz, in0_mem_config)
    t2_data = ttz.cpu().to_torch()
    tt_got_back = torch.Tensor(t2_data).reshape(in0_shape)
    tt_got_back = untilize(tt_got_back)

    ref_lnorm = torch.nn.functional.layer_norm(
        in0 + in1, in0.shape[-1:], gamma[:, :, 0:1, :].flatten(), beta[:, :, 0:1, :].flatten(), epsf
    )

    assert is_close(tt_got_back, ref_lnorm)
