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

    input_rows_shape = [1, 1, num_rows, 1]
    print("Input_rows_shape ")
    print(input_rows_shape)
    input_rows_torch = torch.as_tensor([0, 2]).reshape((1,1,num_rows,1))

    #input_rows_torch = torch.randint(1, num_embeddings-1, input_rows_shape, dtype=torch.int32)
    print("Input_rows")
    print(input_rows_torch)
    weights_shape = [1,1,num_embeddings, embedding_dim]
    weights_torch = torch.randn(weights_shape)
    input_tensor = tensor.Tensor(input_rows_torch, ttl.tensor.DataType.UINT32).to(dev,in0_mem_config)
    weights_tensor = tensor.Tensor(weights_torch, dtype).to(dev, in0_mem_config)
    print("weights")
    print(weights_torch)

    ttz = tensor.embeddings(
        num_embeddings, embedding_dim, input_tensor, weights_tensor, out_mem_config
    )
    tt_data = ttz.cpu().to_torch()
    tt_got_back = torch.Tensor(tt_data).reshape((1, 1, num_rows, embedding_dim))
    print("output")
    print(tt_got_back)
    device.CloseDevice(dev)


import pytest


@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),),
    ids=["in0_DRAM"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "num_embeddings",
    (4,),
    ids=["Num_Input_Rows_4"],
)
@pytest.mark.parametrize(
    "embedding_dim",
    (2,),
    ids=["Num_Cols_2"],
)
@pytest.mark.parametrize(
    "num_rows",
    (2,),
    ids=["Num_Output_Rows_2"],
)
def test_embeddings(
    num_embeddings, embedding_dim, num_rows, dtype, in0_mem_config, out_mem_config
):
    run_embeddings_tests(
        num_embeddings, embedding_dim, num_rows, dtype, in0_mem_config, out_mem_config
    )


if __name__ == '__main__':
    run_embeddings_tests(
        4, 2, 2, ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
                ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM)
    )
