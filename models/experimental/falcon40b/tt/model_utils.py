# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl
import ttnn
import math


def convert_to_layout(tensor, input_memory_layout, output_memory_layout):
    if input_memory_layout == output_memory_layout:
        return tensor
    else:  # Convert layout
        if isinstance(
            tensor, list
        ):  # if input is a list of tensors call convert_to_layout for each tensor individually
            return [convert_to_layout(t, input_memory_layout, output_memory_layout) for t in tensor]
        else:
            if input_memory_layout.is_sharded() and not output_memory_layout.is_sharded():  # sharded_to_interleaved
                tensor = ttl.tensor.sharded_to_interleaved(tensor, output_mem_config=output_memory_layout)
            elif not input_memory_layout.is_sharded() and output_memory_layout.is_sharded():  # interleaved_to_sharded
                tensor = ttl.tensor.interleaved_to_sharded(tensor, sharded_mem_config=output_memory_layout)
            else:  # reshard
                tensor = ttl.tensor.sharded_to_interleaved(
                    tensor,
                    output_mem_config=ttl.tensor.MemoryConfig(
                        ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
                    ),
                )
                tensor = ttl.tensor.interleaved_to_sharded(tensor, sharded_mem_config=output_memory_layout)
            return tensor


def memcfg_1d_width_sharded_from_tensor_shape(shape, grid=ttnn.CoreGrid(x=8, y=8)):
    start_core_coord = ttl.tensor.CoreCoord(0, 0)
    end_core_coord = ttl.tensor.CoreCoord(grid.x - 1, grid.y - 1)
    assert shape[3] % (grid.x * grid.y) == 0, f"Tensor width must be divisible by the number of cores"
    shard_width = int(shape[3] / (grid.x * grid.y))
    shard_height = int(shape[0] * shape[1] * shape[2])
    return ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        start_core_coord,
                        end_core_coord,
                    ),
                }
            ),
            [
                shard_height,
                shard_width,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )


def matmul_1d_config_from_tensor_shapes(
    in0_shape, in1_shape, grid=ttnn.CoreGrid(x=8, y=8), act=None, is_fp32_accumulate=False
):
    m, k, n = in0_shape[0] * in0_shape[1] * in0_shape[2], in0_shape[3], in1_shape[3]
    return matmul_1d_config(m, k, n, grid, act, is_fp32_accumulate)


def matmul_1d_config(
    m, k, n, grid=ttnn.CoreGrid(x=8, y=8), act=None, is_fp32_accumulate=False, overwrite_per_core_k=None
):
    tile_width = 32
    tile_height = 32

    if n // tile_width // grid.num_cores < 1:  # use less number of cores in case we have more N num tiles than cores
        # assert (n // tile_width) % grid.x == 0
        grid_y = n // tile_width // grid.x
        grid = ttnn.CoreGrid(x=grid.x, y=grid_y)

    per_core_m = m // tile_height
    per_core_k = math.ceil(k / tile_width / grid.num_cores)
    per_core_n = math.ceil(n / tile_width / grid.num_cores)

    if overwrite_per_core_k is not None:
        per_core_k = overwrite_per_core_k

    if is_fp32_accumulate:
        max_subblock_w_h = 4
    else:
        max_subblock_w_h = 8

    # find the largest value between 1 and 8 that is a factor of per_core_n
    # e.g. if per_core_n is 14, then out_subblock_w = 7
    out_subblock_w = max([i for i in range(1, max_subblock_w_h + 1) if per_core_n % i == 0])

    # find the largest value that is a factor of per_core_m such that
    # out_subblock_w * out_subblock_h <= 8
    out_subblock_h = max(
        [i for i in range(1, max_subblock_w_h + 1) if per_core_m % i == 0 and i * out_subblock_w <= max_subblock_w_h]
    )

    return ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=per_core_k,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )


def matmul_2d_config_from_tensor_shapes(
    in0_shape, in1_shape, grid=ttnn.CoreGrid(x=8, y=8), act=None, is_fp32_accumulate=False
):
    m, k, n = in0_shape[0] * in0_shape[1] * in0_shape[2], in0_shape[3], in1_shape[3]
    return matmul_2d_config(m, k, n, grid, act, is_fp32_accumulate)


def matmul_2d_config(
    m,
    k,
    n,
    grid=ttnn.CoreGrid(x=8, y=8),
    act=None,
    is_fp32_accumulate=False,
    transpose_mcast=False,
    overwrite_per_core_k=None,
):
    tile_width = 32
    tile_height = 32

    if transpose_mcast:
        grid_x = grid.y
        grid_y = grid.x
    else:
        grid_x = grid.x
        grid_y = grid.y

    assert m % (tile_height * grid_y) == 0, f"m: {m} // 32 not devisible by grid.y: {grid_y}"
    # assert(k % (tile_height * grid_x) == 0), f"k: {k} // 32 not devisible by grid.x: {grid_x}"
    # assert(n % (tile_height * grid_x) == 0), f"n: {n} // 32 not devisible by grid.x: {grid_x}"

    per_core_m = m // tile_height // grid_y
    # per_core_k = k // tile_width // grid_x
    # per_core_n = n // tile_width // grid_x
    per_core_k = math.ceil(k / tile_width / grid_x)
    per_core_n = math.ceil(n / tile_width / grid_x)

    if is_fp32_accumulate:
        max_subblock_w_h = 4
    else:
        max_subblock_w_h = 8

    # find the largest value between 1 and 8 that is a factor of per_core_n
    # e.g. if per_core_n is 14, then out_subblock_w = 7
    out_subblock_w = max([i for i in range(1, max_subblock_w_h + 1) if per_core_n % i == 0])

    # find the largest value that is a factor of per_core_m such that
    # out_subblock_w * out_subblock_h <= 8
    out_subblock_h = max(
        [i for i in range(1, max_subblock_w_h + 1) if per_core_m % i == 0 and i * out_subblock_w <= max_subblock_w_h]
    )

    if per_core_m * per_core_n >= 512:
        max_per_core_k = 1
    elif per_core_m * per_core_n >= 128:
        max_per_core_k = 8
    else:
        max_per_core_k = 16

    if overwrite_per_core_k is not None:
        per_core_k = overwrite_per_core_k
    else:
        per_core_k = min(per_core_k, max_per_core_k)

    # print(
    #     f"per_core_m: {per_core_m}, per_core_k: {per_core_k}, per_core_n: {per_core_n}, out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}"
    # )

    return ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=per_core_k,  # how much inner dim you take each time
        out_subblock_h=out_subblock_h,  # Must be divisible by per_core_M
        out_subblock_w=out_subblock_w,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4 for is_fp32_accumulate otherwise <= 8
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=transpose_mcast,
        fused_activation=act,
    )


def falcon_prefill_matmul(
    in0,
    in1,
    compute_kernel_config,
    output_mem_config=ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    output_dtype=ttl.tensor.DataType.BFLOAT8_B,
    grid=ttnn.CoreGrid(x=8, y=8),
    act=None,
    transpose_mcast=False,
    overwrite_per_core_k=None,
):
    in0_shape = in0.shape
    in1_shape = in1.shape
    m, k, n = in0_shape[0] * in0_shape[1] * in0_shape[2], in0_shape[3], in1_shape[3]

    use_2d_mm = m >= 512  # select 2d matmul for S >= 512, otherwise fall back to matmul 1d

    is_fp32_accumulate = compute_kernel_config.fp32_dest_acc_en

    if use_2d_mm:
        # print("Selecting MM 2d")
        matmul_pgmcfg = matmul_2d_config(m, k, n, grid, act, is_fp32_accumulate, transpose_mcast, overwrite_per_core_k)
        # print(f"Program config: {matmul_pgmcfg}")
        return ttl.operations.primary.matmul(
            in0,
            in1,
            program_config=matmul_pgmcfg,
            output_mem_config=output_mem_config,
            output_dtype=output_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        # print("Selecting MM 1d")
        matmul_pgmcfg = matmul_1d_config(m, k, n, grid, act, is_fp32_accumulate, overwrite_per_core_k)
        # print(f"Program config: {matmul_pgmcfg}")
        return ttl.operations.primary.matmul_1d(
            in0,
            in1,
            program_config=matmul_pgmcfg,
            output_mem_config=output_mem_config,
            output_dtype=output_dtype,
            compute_kernel_config=compute_kernel_config,
        )
