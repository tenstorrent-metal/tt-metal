# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from models.utility_functions import is_wormhole_b0
from loguru import logger
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32


@pytest.mark.parametrize(
    "input_shape, shard_scheme, shard_size",
    [
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64)),
        ([1, 1, 128, 50176], ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, (128, 512)),
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (2048, 32)),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation", [ttl.tensor.ShardOrientation.ROW_MAJOR, ttl.tensor.ShardOrientation.COL_MAJOR]
)
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_tile(device, input_shape, shard_size, shard_scheme, shard_orientation, dtype, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    input_size = torch.Size(input_shape)
    num_cores = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    x = torch.arange(input_size.numel()).reshape(input_size).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttl.tensor.BufferType.L1,
            ),
        )
    )

    yt = ttl.tensor.interleaved_to_sharded(xt, grid_size, shard_size, shard_scheme, shard_orientation)

    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_got_back = zt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    if dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(tt_og, tt_got_back)
    else:
        passing, output = comp_pcc(tt_og, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize(
    "input_shape, shard_scheme, shard_size",
    [
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64)),
        ([1, 1, 128, 50176], ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, (128, 512)),
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (2048, 32)),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttl.tensor.ShardOrientation.ROW_MAJOR, ttl.tensor.ShardOrientation.COL_MAJOR],
)
def test_sharded_rm(device, input_shape, shard_size, shard_scheme, shard_orientation, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    input_size = torch.Size(input_shape)
    num_cores = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    x = torch.arange(input_size.numel()).reshape(input_size).bfloat16().float()

    xt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(
        device,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    assert input_shape[0] == 1 and input_shape[1] == 1

    yt = ttl.tensor.interleaved_to_sharded(xt, grid_size, shard_size, shard_scheme, shard_orientation)

    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to_torch()

    tt_got_back = zt.cpu().to_torch()

    passing, output = comp_equal(tt_og, tt_got_back)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("H, num_cores", [[100352, 98], [25088, 98]])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_untilize(H, num_cores, in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    N = 1
    C = 1
    W = 64
    if out_sharded and not in_sharded and H == 100352:
        pytest.skip("Unsupported config for sharding")

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [H // num_cores, W],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    yt = ttl.tensor.untilize(
        xt,
        output_mem_config=out_mem_config,
        use_multicore=True,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    if dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(x, tt_got_back)
    else:
        passing, output = comp_pcc(x, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("H, num_cores", [[25088, 98]])
@pytest.mark.parametrize("output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_tilize(H, num_cores, output_dtype, device, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    N = 1
    C = 1
    W = 64

    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(
        device,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt,
        grid_size,
        [H // num_cores, W],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    yt_tilized = ttl.tensor.tilize(
        yt,
        output_mem_config=ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
        use_multicore=True,
        output_dtype=output_dtype,
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt_tilized,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_got_back = zt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    if output_dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(x, tt_got_back)
    else:
        passing, output = comp_pcc(x, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M, num_cores", [[25088, 98], [50176, 98]])
@pytest.mark.parametrize("K, N", [[64, 64], [64, 256], [256, 64], [256, 128]])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("weights_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_matmul_1d_in1(
    device, in0_sharded, out_sharded, M, K, N, num_cores, activations_dtype, weights_dtype, function_level_defaults
):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // num_cores, K],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(12, 9),
        in0_block_w=K // 32,
        out_subblock_h=8 // (N // 32),
        out_subblock_w=N // 32,
        per_core_M=M // 32 // num_cores,
        per_core_N=N // 32,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("in1_sharded", [True, False], ids=["in1_sharded", "in1_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("H, num_cores", [[25088, 98]])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_binary(
    device,
    in0_sharded,
    in1_sharded,
    out_sharded,
    H,
    num_cores,
    activations_dtype,
    output_dtype,
    function_level_defaults,
):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, H, 64]
    in1_shape = in0_shape
    W = in0_shape[-1]

    if out_sharded and not in0_sharded and not in1_sharded and H == 25088:
        pytest.skip("Unsupported sharding config")

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [H // num_cores, W],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    if in1_sharded:
        in1_t = ttl.tensor.interleaved_to_sharded(
            in1_t,
            grid_size,
            [H // num_cores, W],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    output_t = ttl.tensor.add(in0_t, in1_t, output_mem_config=output_mem_config, output_dtype=output_dtype)
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 + in1

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


def test_sharded_program_cache(device, use_program_cache, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    num_cores = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 25088
    W = 64
    x = torch.ones((N, C, H, W)).bfloat16().float()
    x2 = torch.zeros((N, C, H, W)).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttl.tensor.BufferType.L1,
            ),
        )
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt,
        grid_size,
        [H // num_cores, W],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    xt2 = (
        ttl.tensor.Tensor(
            x2.reshape(-1).tolist(),
            x2.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttl.tensor.BufferType.L1,
            ),
        )
    )

    yt2 = ttl.tensor.interleaved_to_sharded(
        xt2,
        grid_size,
        [H // num_cores, W],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    zt2 = ttl.tensor.sharded_to_interleaved(
        yt2,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )
    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    tt_og2 = xt2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_got_back = zt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    tt_got_back2 = zt2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    eq = torch.equal(tt_og, tt_got_back)
    assert eq
    eq = torch.equal(tt_og2, tt_got_back2)
    assert eq


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [1600])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("weights_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_matmul_2d(
    device, in0_sharded, out_sharded, M, N, activations_dtype, weights_dtype, function_level_defaults
):
    K = 256
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    grid_size = (8, 5)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[1], K // grid_size[0]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=10,
        per_core_N=4,
        transpose_mcast=False,
        fused_activation=None,
    )
    output_t = ttl.operations.primary.matmul(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [1600])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("weights_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_matmul_2d_transposed(
    device, in0_sharded, out_sharded, M, N, activations_dtype, weights_dtype, function_level_defaults
):
    K = 256
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    grid_size = (10, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[0], K // grid_size[1]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=5,
        per_core_N=4,
        transpose_mcast=True,
        fused_activation=None,
    )
    output_t = ttl.operations.primary.matmul(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


def test_resharded_binary_to_matmul(device, function_level_defaults):
    grid_size_binary = device.compute_with_storage_grid_size()
    num_cores_binary = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores_binary > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores_binary} cores to run this test but core grid is {compute_grid_size}")
    grid_size_matmul = (10, 8)
    if grid_size_matmul[0] > compute_grid_size.x or grid_size_matmul[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size_matmul} grid size to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, 6272, 512]
    in1_shape = in0_shape
    weight_shape = [1, 1, 512, 256]
    bias_shape = [1, 1, 1, 256]
    H = in0_shape[-2]
    W = in0_shape[-1]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    height_sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    block_sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    weight = torch.randn(weight_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config)
    weight_t = torch2tt_tensor(weight, device, tt_memory_config=interleaved_mem_config)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config)[0]

    in0_t = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size_binary,
        [H // num_cores_binary, W],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    in1_t = ttl.tensor.interleaved_to_sharded(
        in1_t,
        grid_size_binary,
        [H // num_cores_binary, W],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    output_binary_t = ttl.tensor.add(in0_t, in1_t, output_mem_config=interleaved_mem_config)
    output_binary_t = ttl.tensor.interleaved_to_sharded(
        output_binary_t,
        grid_size_matmul,
        [math.ceil((H // 32) / grid_size_matmul[0]) * 32, W // grid_size_matmul[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )
    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size_matmul,
        in0_block_w=2,
        out_subblock_h=5,
        out_subblock_w=1,
        per_core_M=20,
        per_core_N=1,
        transpose_mcast=True,
        fused_activation=None,
    )
    output_matmul_t = ttl.operations.primary.matmul(
        output_binary_t,
        weight_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=block_sharded_mem_config,
    )
    output_matmul_t = ttl.tensor.sharded_to_interleaved(output_matmul_t, interleaved_mem_config)

    tt_out = tt2torch_tensor(output_matmul_t)

    pt_out = (in0 + in1) @ weight

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_untilize_padded_shard(in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = (10, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 6272
    W = 256

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [math.ceil((xt.shape()[-2] // 32) / grid_size[0]) * 32, xt.shape()[-1] // grid_size[1]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    yt = ttl.tensor.untilize(
        xt,
        output_mem_config=out_mem_config,
        use_multicore=True,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    if dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(x, tt_got_back)
    else:
        passing, output = comp_pcc(x, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("in_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_binary_padded_shard(
    in_sharded, out_sharded, activations_dtype, output_dtype, device, function_level_defaults
):
    grid_size = (10, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 1568
    W = 1024

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.ones((N, C, H, W)).bfloat16()
    y = torch.ones((N, C, H, W)).bfloat16() * 2

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            activations_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    yt = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            activations_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [math.ceil((xt.shape()[-2] // 32) / grid_size[0]) * 32, xt.shape()[-1] // grid_size[1]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )
        yt = ttl.tensor.interleaved_to_sharded(
            yt,
            grid_size,
            [math.ceil((xt.shape()[-2] // 32) / grid_size[0]) * 32, xt.shape()[-1] // grid_size[1]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    zt = ttl.tensor.add(xt, yt, output_mem_config=out_mem_config, output_dtype=output_dtype)

    if out_sharded:
        zt = ttl.tensor.sharded_to_interleaved(
            zt,
            interleaved_mem_config,
        )

    tt_got_back = zt.cpu().to_torch()

    passing, output = comp_equal(x + y, tt_got_back)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("in_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_block_sharded_untilize_with_unpadding(in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = (7, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 416
    W = 512

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [math.ceil((xt.shape()[-2] // 32) / grid_size[0]) * 32, xt.shape()[-1] // grid_size[1]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    yt = ttl.tensor.untilize_with_unpadding(
        xt,
        [0, 0, 0, 0],
        [0, 0, 391, 511],
        output_mem_config=out_mem_config,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    y = x[..., :392, :512]

    if dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)

    logger.info(output)

    assert passing


@pytest.mark.parametrize("in_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize(
    "shape, output_H, out_sharded",
    [
        [(8, 1, 32, 2048), 1, True],
        [(1, 1, 32, 1024), 8, False],
        [(16, 1, 32, 2048), 1, True],
        [(1, 1, 32, 1024), 16, False],
    ],
    ids=[
        "batched_8_shape_out_sharded",
        "unbatched_8_shape_out_interleaved",
        "batched_16_shape_out_sharded",
        "unbatched_16_shape_out_interleaved",
    ],
)
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_width_sharded_untilize_with_unpadding(
    shape, output_H, in_sharded, out_sharded, dtype, device, function_level_defaults
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N, C, H, W = shape

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [N * C * H, W // (grid_size[0] * grid_size[1])],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    yt = ttl.tensor.untilize_with_unpadding(
        xt,
        [0, 0, 0, 0],
        [N - 1, C - 1, output_H - 1, W - 1],
        output_mem_config=out_mem_config,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    y = x[..., :output_H, :]
    if dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("input_shape", [[8, 1, 49, 2048], [1, 1, 8, 2048], [16, 1, 49, 2048], [1, 1, 16, 2048]])
@pytest.mark.parametrize("in_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize("output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_tilize_with_val_padding(
    input_shape, in_sharded, out_sharded, output_dtype, device, function_level_defaults
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N, C, H, W = input_shape

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(
        device,
        interleaved_mem_config,
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [N * C * H, W // (grid_size[0] * grid_size[1])],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    yt = ttl.tensor.tilize_with_val_padding(
        xt, [N, C, roundup32(H), W], [0, 0, 0, 0], 1.0, output_mem_config=out_mem_config, output_dtype=output_dtype
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    y = torch.nn.functional.pad(x, [0, 0, 0, roundup32(H) - H], "constant", 1.0)

    if output_dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("N", [8, 16])
@pytest.mark.parametrize("in_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_reduce_h(N, in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    C = 1
    H = 64
    W = 2048

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [N * C * H, W // (grid_size[0] * grid_size[1])],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    yt = ttl.tensor.reduce(
        xt,
        ttl.tensor.ReduceOpMath.MAX,
        ttl.tensor.ReduceOpDim.H,
        1.0,
        output_mem_config=out_mem_config,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()[:, :, :1, :]

    y = torch.max(x, 2, True)[0]

    if dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("weights_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_matmul_1d_in0(
    device, in0_sharded, out_sharded, M, N, activations_dtype, weights_dtype, function_level_defaults
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")
    num_cores = grid_size[0] * grid_size[1]
    K = 2048
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, K // num_cores],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing
