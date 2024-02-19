# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(input_shape, output_shape, dims, device):
    torch.manual_seed(2023)
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    # prod_cpu" not implemented for 'BFloat16'
    cpu_dtype = torch.float32
    npu_layout = ttl.tensor.Layout.TILE

    if dims[0] == 3:
        # print("dims ==> ", dims)
        torch_input = torch.randint(-10, 10, input_shape, dtype=cpu_dtype, requires_grad=True)
        torch_input_permute = torch_input.permute(3, 0, 1, 2)
        torch_output = torch.randint(-10, 10, output_shape, dtype=cpu_dtype).permute(3, 0, 1, 2)

        print("torch_input shape ==> ", torch_input_permute.shape)
        print("torch_output shape ==> ", torch_output.shape)
        tt_input = ttl.tensor.Tensor(torch_input_permute, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    else:
        torch_input = torch.randint(-100, 100, input_shape, dtype=cpu_dtype, requires_grad=True)
        torch_output = torch.randint(-100, 100, output_shape, dtype=cpu_dtype)

        tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


@pytest.mark.parametrize(
    "input_shape",
    (
        ([4, 5, 6, 2]),
        # ([1, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1]),
        # ([4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1]),
        # ([4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1]),
        # ([8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1]),
    ),
    ids=[
        "1, 1, TILE_HEIGHT-1,TILE_WIDTH - 1",
        # "1, 2, TILE_HEIGHT-1,TILE_WIDTH - 1",
        # "4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1",
        # "4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1",
        # "8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1",
    ],
)
@pytest.mark.parametrize(
    "dims",
    (
        [
            3,
        ],
        # [
        #     1,
        # ],
    ),
    ids=[
        "0",
    ],
)
def test_moreh_prod_dims(input_shape, dims, device):
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1  # [4, 5, 6, 1]

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, dims, device)

    torch_output = torch.prod(torch_input, dims[0], True).permute(3, 0, 1, 2)

    if dims[0] == 3:
        dims[0] = 0
        output_shape = [1, input_shape[0], input_shape[1], input_shape[2]]

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_cpu = (
        ttl.operations.primary.prod(tt_input, tt_output, dims=dims)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    print("torch_output  ===> ", torch_output, torch_output.shape)
    print("tt_output_cpu ===> ", tt_output_cpu, tt_output_cpu.shape)
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.info(f"Out passing={passing}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing
