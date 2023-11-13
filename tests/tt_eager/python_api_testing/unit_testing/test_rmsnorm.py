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
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0


def rmsnorm(x, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma + beta


def run_rmsnorm_tests(test_id, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    tensor = ttl.tensor
    dev = device

    epsf = 1e-2

    test_dims = ((1, 9, 384, 1024),)
    for N, C, H, W in test_dims:
        """
        test_id = 0  : rmsn(x)*1+0 path
        test_id = 1  : rmsn(x)*g+0 path
        test_id = 2  : rmsn(x)*gamma+beta path
        """
        if test_id >= 0:
            gamma = torch.ones(1, 1, 1, W)
            beta = torch.zeros(1, 1, 1, W)
        if test_id >= 1:
            gamma = torch.rand(1, 1, 1, W) * 2 - 1
            gammah32 = tilize_to_list(pad_weight(gamma))
            ttgamma = tensor.Tensor(
                gammah32,
                [1, 1, 32, W],
                dtype,
                tensor.Layout.TILE,
                dev,
                in0_mem_config,
            )
        if test_id >= 2:
            beta = torch.rand(1, 1, 1, W) * 2.0 - 1.1
            betah32 = tilize_to_list(pad_weight(beta))
            ttbeta = tensor.Tensor(
                betah32,
                [1, 1, 32, W],
                dtype,
                tensor.Layout.TILE,
                dev,
                in0_mem_config,
            )

        x = torch.rand((N, C, H, W)) * 2 - 0.95

        ttx = tensor.Tensor(
            tilize_to_list(x),
            [N, C, H, W],
            dtype,
            tensor.Layout.TILE,
            dev,
            in0_mem_config,
        )

        if test_id == 0:
            logger.info("Running RMSN_NOGB")
            ttz = tensor.rmsnorm(ttx, epsf, output_mem_config=out_mem_config)
        elif test_id == 1:
            logger.info("Running RMSN_G")
            ttz = tensor.rmsnorm(ttx, epsf, ttgamma, output_mem_config=out_mem_config)
        elif test_id == 2:
            logger.info("Running RMSN_GB")
            ttz = tensor.rmsnorm(ttx, epsf, ttgamma, ttbeta, out_mem_config)
        else:
            assert False
        logger.info("Done")

        assert ttx.memory_config().buffer_storage == in0_mem_config.buffer_storage
        assert ttz.memory_config().buffer_storage == out_mem_config.buffer_storage

        logger.debug(f"ttx is on: {ttx.memory_config().buffer_storage}")
        logger.debug(f"ttz is on: {ttz.memory_config().buffer_storage}")

        t2_data = ttz.cpu().to_torch()

        tt_got_back = torch.Tensor(t2_data).reshape((N, C, H, W))
        tt_got_back = untilize(tt_got_back)

        # ref_lnorm = ref_layernorm(x, epsf, gammaf, betaf, H, W)
        ref_rmsnorm = rmsnorm(x, gamma.flatten(), beta.flatten(), epsf)

        assert is_close(tt_got_back, ref_rmsnorm)


@skip_for_wormhole_b0
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
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2),
    ids=["RMSN", "RMSN_G", "RMSN_GB"],
)
def test_rmsnorm_test(test_id, dtype, in0_mem_config, out_mem_config, device):
    run_rmsnorm_tests(test_id, dtype, in0_mem_config, out_mem_config, device)
