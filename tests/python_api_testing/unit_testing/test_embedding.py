import math
from pathlib import Path
import sys
import time
import os
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../../..")

import torch

import tt_lib as ttl

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)


def run_embeddings_tests(
    num_embeddings, embedding_dim, num_rows, dtype, in0_mem_config, out_mem_config
):
    torch.manual_seed(1234)

    # Initialize the device
    tensor = ttl.tensor
    device = ttl.device
    dev = device.CreateDevice(device.Arch.GRAYSKULL, 0)
    device.InitializeDevice(dev)

    input_rows_torch = torch.IntTensor((num_rows)).uniform_(0, num_embeddings - 1)
    weights_shape = [num_embeddings, embedding_dim]
    weights_torch = torch.randn(weights_shape)
    input_tensor = tensor.Tensor(
        input_rows_torch.tolist(),
        [num_rows],
        dtype,
        tensor.Layout.ROW_MAJOR,
        dev,
        in0_mem_config,
    )
    weight_tensor = tensor.Tensor(
        weights_torch.flatten().tolist(),
        weights_shape,
        dtype,
        tensor.Layout.ROW_MAJOR,
        dev,
        in0_mem_config,
    )
    ttz = tensor.embeddings(
        num_embeddings, embedding_dim, input_tensor, weight_tensor, out_mem_config
    )
    tt_data = ttz.cpu().to(tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm_buff = torch.Tensor(tt_data.data()).reshape(tt_data.shape())
    print(pyt_got_back_rm_buff)
    device.CloseDevice(dev)


import pytest


@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "num_embeddings",
    (4),
    ids=["Num_Input_Rows_4"],
)
@pytest.mark.parametrize(
    "embedding_dim",
    (2),
    ids=["Num_Cols_2"],
)
@pytest.mark.parametrize(
    "num_rows",
    (1),
    ids=["Num_Output_Rows_1"],
)
def test_layernorm_test(
    num_embeddings, embedding_dim, num_rows, dtype, in0_mem_config, out_mem_config
):
    run_embeddings_tests(
        num_embeddings, embedding_dim, num_rows, dtype, in0_mem_config, out_mem_config
    )
