# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch

import tt_lib as ttl

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)


def run_layernorm_tests(device, test_id, batch, dtype, in0_mem_config, out_mem_config):
    torch.manual_seed(1234)

    tensor = ttl.tensor

    epsf = 1e-2

    test_dims = ((batch, 1, 384, 1024),)
    for N, C, H, W in test_dims:
        """
        test_id = 0  : ln(x)*1+0 path
        test_id = 1  : ln(x)*g+0 path
        test_id = 2  : ln(x)*gamma+beta path
        test_id = 3  : ln(a+b)*gamma+beta path
        """
        for nrepeat in range(0, 1):
            if test_id >= 0:
                gamma = torch.ones(1, 1, 1, W)
                beta = torch.zeros(1, 1, 1, W)
            if test_id >= 1:
                gamma = torch.rand(1, 1, 1, W) * 2 - 1
                gammah32 = gamma.reshape([1, 1, -1, 32])
                ttgamma = tensor.Tensor(
                    gammah32.reshape(-1).tolist(),
                    gammah32.shape,
                    dtype,
                    tensor.Layout.ROW_MAJOR,
                    device,
                    in0_mem_config,
                )
            if test_id >= 2:
                beta = torch.rand(1, 1, 1, W) * 2.0 - 1.1
                betah32 = beta.reshape([1, 1, -1, 32])
                ttbeta = tensor.Tensor(
                    betah32.reshape(-1).tolist(),
                    betah32.shape,
                    dtype,
                    tensor.Layout.ROW_MAJOR,
                    device,
                    in0_mem_config,
                )

            x = torch.rand((N, C, H, W)) * 2 - 0.95
            y = torch.rand((N, C, H, W)) * 2 - 0.8

            if test_id < 3:
                y *= 0.0  # zero out the y to exclude x+y from reference calculation

            ttx = tensor.Tensor(
                tilize_to_list(x),
                [N, C, H, W],
                dtype,
                tensor.Layout.TILE,
                device,
                in0_mem_config,
            )
            tty = tensor.Tensor(
                tilize_to_list(y),
                [N, C, H, W],
                dtype,
                tensor.Layout.TILE,
                device,
                in0_mem_config,
            )

            if test_id == 0:
                logger.info("Running LN_NOGB")
                ttz = ttl.operations.primary.layernorm(ttx, epsf, None, None, output_mem_config=out_mem_config)
            elif test_id == 1:
                logger.info("Running LN_G")
                ttz = ttl.operations.primary.layernorm(ttx, epsf, ttgamma, None, output_mem_config=out_mem_config)
            elif test_id == 2:
                logger.info("Running LN_GB")
                ttz = ttl.operations.primary.layernorm(ttx, epsf, ttgamma, ttbeta, out_mem_config)
            elif test_id == 3:
                logger.info("Running add_LN_GB")
                ttz = ttl.operations.primary.add_layernorm(ttx, tty, epsf, ttgamma, ttbeta, out_mem_config)
            else:
                assert False
            logger.info("Done")

            assert ttx.memory_config().buffer_storage == in0_mem_config.buffer_storage
            assert tty.memory_config().buffer_storage == in0_mem_config.buffer_storage
            assert ttz.memory_config().buffer_storage == out_mem_config.buffer_storage

            logger.debug(f"ttx is on: {ttx.memory_config().buffer_storage}")
            logger.debug(f"tty is on: {tty.memory_config().buffer_storage}")
            logger.debug(f"ttz is on: {ttz.memory_config().buffer_storage}")

            tt_got_back = ttz.cpu().to_torch()
            tt_got_back = untilize(tt_got_back)

            ref_lnorm = torch.nn.functional.layer_norm(x + y, x.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

            assert is_close(tt_got_back, ref_lnorm)


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
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
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
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2, 3),
    ids=["LN", "LN_G", "LN_GB", "add_LN_GB"],
)
def test_layernorm_test(device, test_id, batch, dtype, in0_mem_config, out_mem_config, request):
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/BERT_large_fused_layernorm_{request.node.callspec.id}"
    )
    run_layernorm_tests(device, test_id, batch, dtype, in0_mem_config, out_mem_config)
