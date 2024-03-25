# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl
import ttnn


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
    return matmul_1d_config(m, k, n, grid, act)


def matmul_1d_config(m, k, n, grid=ttnn.CoreGrid(x=8, y=8), act=None, is_fp32_accumulate=False):
    tile_width = 32
    tile_height = 32

    per_core_m = m // tile_height
    per_core_k = k // tile_width // grid.num_cores
    per_core_n = n // tile_width // grid.num_cores

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

    print(
        f"per_core_m: {per_core_m}, per_core_k: {per_core_k}, per_core_n: {per_core_n}, out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}"
    )

    return ttnn.ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttl.tensor.CoreCoord(grid.x, grid.y),
        in0_block_w=per_core_k,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )
