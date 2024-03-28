# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import tt_lib as ttl
from loguru import logger
from tests.ttnn.utils_for_testing import update_process_id
from tt_lib.utils import (
    _nearest_y,
)

update_process_id()


@pytest.mark.parametrize(
    "batch_size, input_channels, input_height, input_width, num_cores, grid_size, height_sharded",
    ((1, 1, 32, 32, 98, (12, 9), True),),
)
def test_bcast(
    device,
    batch_size,
    input_channels,
    input_height,
    input_width,
    num_cores,
    grid_size,
    height_sharded,
):
    input_shape = [batch_size, input_channels, input_height, input_width]
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_bias = torch.randn(input_shape, dtype=torch.bfloat16)
    tt_input_test = ttl.tensor.Tensor(torch_input, ttl.tensor.DataType.BFLOAT16).to(device)
    mem_config = tt_input_test.memory_config()
    tt_input = ttl.tensor.Tensor(torch_input, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE)
    print(mem_config)
    tt_bias = ttl.tensor.Tensor(torch_bias, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE)

    print(tt_input)
    print(tt_bias)

    tt_output = ttl.tensor.bcast(
        tt_input, tt_bias, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.H, output_mem_config=mem_config
    )

    return
