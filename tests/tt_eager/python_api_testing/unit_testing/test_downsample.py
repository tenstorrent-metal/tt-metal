# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path
import sys
import math
import numpy as np

import tt_lib as ttl
from tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    _nearest_32,
    _nearest_y,
    convert_weights_2d_matrix,
)
from models.utility_functions import print_diff_argmax, is_close, comp_pcc, comp_allclose_and_pcc
from tests.tt_eager.python_api_testing.conv.conv_unit_test_utils import (
    create_conv_act_tensor,
    create_conv_weight_tensor,
    create_conv_bias_tensor,
    create_conv_weight_tensor_special_padding,
)
from tests.tt_eager.python_api_testing.sweep_tests.common import (
    skip_for_wormhole_b0,
)
import torch

@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, stride_h, stride_w, num_cores, grid_size, height_sharded",
    (
        #(10, 64, 64, 16, 16, 2, 2, 20, (10,2), False),
        #(10, 64, 64, 16, 16, 1, 1, 20, (10,2), False),
        #(8, 64, 64, 56, 56, 1, 1, 98, (12,9), True),
        (8, 64, 64, 56, 56, 2, 2, 98, (12,9), True),
        (8, 512, 512, 28, 28, 2, 2, 80, (10,8), False),
        (8, 1024, 1024, 14, 14, 2, 2, 56, (7,8), False),
    ),
)
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@skip_for_wormhole_b0
def test_run_downsample(
    use_program_cache,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    stride_h,
    stride_w,
    num_cores,
    grid_size,
    height_sharded,
    dtype,
    device,
):
    assert(input_channels % 32 == 0)
    assert(output_channels % 32 == 0)
    assert(stride_h == stride_w)

    torch.set_printoptions(
        precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32
    )

    torch.manual_seed(0)
    a_activation_shape = [batch_size, input_channels, input_height, input_width]
    A_pyt = torch.normal(mean=0, std=0.1, size=a_activation_shape)

    b_weights_shape = [output_channels, input_channels, 1, 1]
    B_pyt = torch.normal(mean=0, std=0.1, size=b_weights_shape)

    output_height = math.ceil(input_height / stride_h)
    output_width = math.ceil(input_width / stride_w)

    conv_output_shape = [batch_size, output_height, output_width, output_channels]

    # Convert NCHW to NHWC shape
    A_pyt_nhwc = torch.permute(A_pyt, (0, 2, 3, 1))
    A_pyt_nhwc = A_pyt_nhwc.reshape(1, 1, batch_size*input_height*input_width, input_channels)
    #for i in range(2):
    #    for j in range(32):
    #        print(f"A_pyt_nhwc_2d[{i}][{j}]={A_pyt_nhwc[0][0][i][j]}")
    #print("A_pyt_nhwc_2d[32][0]=", A_pyt_nhwc[0][0][32][0])
    a_activation_shape_nhwc = [batch_size, input_height, input_width, input_channels]
    A_cl_host = ttl.tensor.Tensor(A_pyt_nhwc, dtype).reshape(1, 1, batch_size*input_height*input_width, input_channels)
    num_cores_height_slices = num_cores if height_sharded else grid_size[0]
    input_shape = [1, 1, _nearest_y(batch_size*input_height*input_width, 32), input_channels]
    A_cl_host = A_cl_host.pad(input_shape, (0,0,0,0), 0.0)
    A_interleaved = A_cl_host.to(ttl.tensor.Layout.TILE).to(device, ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_storage=ttl.tensor.BufferStorage.L1,
        ))
    assert A_interleaved.shape()[0] == 1 and A_interleaved.shape()[1] == 1

    # image flattened params
    input_2d_height = A_interleaved.shape()[2]
    input_2d_width = A_interleaved.shape()[3]
    input_2d_height_padded = _nearest_y(input_2d_height, num_cores_height_slices * 32)
    input_shard_height = (int) (input_2d_height_padded / num_cores_height_slices)
    output_2d_height_padded = _nearest_y(batch_size * output_height * output_width, num_cores_height_slices * 32)
    output_shard_height = (int) (output_2d_height_padded / num_cores_height_slices)
    print("input_2d_height=", input_2d_height)
    print("input_2d_width=", input_2d_width)
    sharded_memory_layout = ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED if height_sharded else ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED
    sharded_memory_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR if height_sharded else  ttl.tensor.ShardOrientation.COL_MAJOR
    input_shard_width = input_2d_width if height_sharded else ((int) (input_2d_width / grid_size[1]))
    print("grid_size=", grid_size)
    print("shard_memory_layout", sharded_memory_layout)
    print(f"input_shard_height={input_shard_height}, input_shard_width={input_shard_width}")

    A_sharded = ttl.tensor.interleaved_to_sharded(
        A_interleaved, grid_size, [input_shard_height, input_shard_width], sharded_memory_layout, sharded_memory_orientation
    )
    # Prepare weights for simple matmul
    B_tiled_host = create_conv_weight_tensor(
        B_pyt, output_channels, input_channels, 1, 1, 1, 1
    )
    B_tiled = B_tiled_host.to(device)

    # downsample golden output using maxpool
    out_golden = torch.nn.functional.max_pool2d(A_pyt, 1, stride=stride_h)
    out_golden_2d_nhwc = torch.permute(out_golden, (0, 2, 3, 1)).reshape(1, 1, batch_size*output_height*output_width, input_channels)

    downsample_params = [batch_size, input_height, input_width, stride_h, stride_w]
    sharded_memory_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferStorage.L1
            )
    # Run downsample op
    A_downampled_sharded = ttl.tensor.downsample(A_sharded, downsample_params, output_dtype=dtype)
    A_downsampled = ttl.tensor.sharded_to_interleaved(A_downampled_sharded, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferStorage.L1))
    out = A_downsampled
    out_shape = [1, 1, _nearest_y(batch_size*output_height*output_width, 32), input_channels]
    assert out_shape == out.shape()
    out_shape_unpadded = [1, 1, batch_size*output_height*output_width, input_channels]
    assert out_shape_unpadded == out.shape_without_padding()
    out = ttl.tensor.format_output_tensor(out, out.shape_without_padding(), device, ttl.tensor.Layout.ROW_MAJOR)
    out = out.cpu()

    out_debug = out
    out_debug = out_debug.to_torch().float()

    # DEBUG
    # for i in range(16):
    #     for j in range(input_2d_width):
    #         print(f"out_golden_2d_nhwc[{i}][{j}]={out_golden_2d_nhwc[0][0][i][j]}")

    # for i in range(16):
    #     for j in range(input_2d_width):
    #         print(f"out_result_2d_nhwc[{i}][{j}]={out_debug[0][0][i][j]}")

    num_errors = 0
    core_idx = 0
    start_i = core_idx * output_shard_height
    end_i = start_i + output_shard_height
    for i in range(start_i, end_i):
        for j in range(input_shard_width):
            calculated = torch.tensor(out_golden_2d_nhwc[0][0][i][j])
            golden = torch.tensor(out_debug[0][0][i][j])
            atol_delta = torch.abs(golden - calculated).item()
            rtol_delta = torch.abs(golden - calculated) / torch.abs(calculated)
            if dtype == ttl.tensor.DataType.BFLOAT8_B:
                fail = atol_delta > 0.1
            else:
                fail = atol_delta > 0.1 or rtol_delta > 0.1
            if fail:
                if num_errors < 10:
                    print(f"Bad value at {i} (sharded index {i - start_i}), {j} with ATOL={atol_delta} and RTOL={rtol_delta}")
                    print(f"    result={calculated}, golden={golden}")
                num_errors += 1
                # if (num_errors >= 10):
                #     assert False
    print("Num errors: ", num_errors)

    out = out.reshape(batch_size, output_height, output_width, input_channels)
    assert out.layout() == ttl.tensor.Layout.ROW_MAJOR

    # Copy output to host and convert tt tensor to pytorch tensor
    out_result = out.to_torch().float()
    out_result = torch.transpose(out_result, 2, 3)
    out_result = torch.transpose(out_result, 1, 2)

    # print (f'OUTPUT: {out_result}')
    # print (f'GOLDEN: {out_golden}')

    passing_allclose_and_pcc, output_info = comp_allclose_and_pcc(out_golden, out_result, rtol=1e-1, atol=1e-3, pcc=0.9999)  # For LowFi we need 0.99976
    print("Passing=", passing_allclose_and_pcc)
    print("Output info=", output_info)
    passing_pcc_ds, _ = comp_pcc(out_golden, out_result, pcc=0.9998) # For LowFi we need 0.99976
    assert passing_pcc_ds
