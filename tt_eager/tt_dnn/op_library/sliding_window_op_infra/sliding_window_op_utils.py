# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
from collections import namedtuple

import tt_lib as ttl
from tt_lib.utils import _nearest_y

SlidingWindowOpParams = namedtuple(
    "SlidingWindowOpParams", "stride_h stride_w pad_h pad_w window_h window_w batch_size input_h input_w"
)
SlidingWindowOpParamsWithParallelConfig = namedtuple(
    "SlidingWindowOpParamsWithParallelConfig",
    "stride_h stride_w pad_h pad_w window_h window_w batch_size input_h input_w num_cores_w num_cores_h num_cores_nhw",
)


def get_hash_from_sliding_window_op_params(sliding_window_op_params: SlidingWindowOpParamsWithParallelConfig):
    return f"{sliding_window_op_params.stride_h}_{sliding_window_op_params.stride_w}_{sliding_window_op_params.pad_h}_{sliding_window_op_params.pad_w}_{sliding_window_op_params.window_h}_{sliding_window_op_params.window_w}_{sliding_window_op_params.batch_size}_{sliding_window_op_params.input_h}_{sliding_window_op_params.input_w}_{sliding_window_op_params.num_cores_w}_{sliding_window_op_params.num_cores_h}_{sliding_window_op_params.num_cores_nhw}"


def get_sliding_window_op_output_nhw_shape(
    input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
):
    output_h = ((int)((input_h + (2 * pad_h) - window_h) / stride_h)) + 1
    output_w = ((int)((input_w + (2 * pad_w) - window_w) / stride_w)) + 1
    return [input_n, output_h, output_w]


def get_sliding_window_op_output_shard_nhw_size(
    num_cores_nhw, input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w, is_out_tiled=True
):
    output_nhw_shape = get_sliding_window_op_output_nhw_shape(
        input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
    )
    if is_out_tiled:
        output_nhw_size_to_shard_evenly = _nearest_y(np.prod(output_nhw_shape), num_cores_nhw * 32)
    else:
        output_nhw_size_to_shard_evenly = _nearest_y(np.prod(output_nhw_shape), num_cores_nhw)
    output_shard_nhw_size = (int)(output_nhw_size_to_shard_evenly / num_cores_nhw)
    return output_shard_nhw_size


class SWOParallelConfig:
    config_keys = ["num_cores", "grid_size", "shard_layout"]

    def __init__(
        self, num_cores=1, grid_size=(1, 1), shard_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    ) -> None:
        self.num_cores = num_cores
        self.grid_size = grid_size
        self.shard_layout = shard_layout
