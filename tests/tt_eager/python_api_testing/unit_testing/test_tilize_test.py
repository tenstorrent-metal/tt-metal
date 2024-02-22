# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import tilize, comp_pcc
from models.utility_functions import is_grayskull


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.FLOAT32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize(
    "nb, nc, nh, nw",
    (
        (5, 2, 4, 8),
        (5, 2, 4, 7),
        ## resnet shapes
        (1, 1, 784, 2),
        (8, 1, 2, 64),
        (1, 1, 1, 64),
    ),
)
@pytest.mark.parametrize(
    "multicore",
    (
        False,
        True,
    ),
)
def test_run_tilize_test(dtype, nb, nc, nh, nw, multicore, device):
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    nt = nb * nc * nh * nw
    shape = [nb, nc, 32 * nh, 32 * nw]

    if dtype == ttl.tensor.DataType.FLOAT32:
        inp = torch.rand(*shape).float() * 1000.0
    else:
        inp = torch.rand(*shape).bfloat16()

    a = ttl.tensor.Tensor(
        inp,
        dtype,
    ).to(device)
    b = ttl.tensor.tilize(a, use_multicore=multicore)
    c = b.cpu().to_torch()

    tilized_inp = tilize(inp)
    if dtype == ttl.tensor.DataType.FLOAT32:
        passing, output = comp_pcc(tilized_inp, c, 0.999999)
        logger.info(output)
    else:
        passing = torch.equal(tilized_inp, c)
    assert passing


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.FLOAT32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize(
    "nb, nc, nh, nw",
    (
        (5, 2, 4, 8),
        (5, 2, 4, 7),
        ## resnet shapes
        (1, 1, 784, 2),
        (8, 1, 2, 64),
        (1, 1, 1, 64),
    ),
)
@pytest.mark.parametrize(
    "multicore",
    (
        False,
        True,
    ),
)
def test_run_tilize_with_val_padding_test(dtype, nb, nc, nh, nw, multicore, device):
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    nt = nb * nc * nh * nw
    shape = [nb, nc, nh, 32 * nw]

    if dtype == ttl.tensor.DataType.FLOAT32:
        inp = torch.rand(*shape).float() * 1000.0
    else:
        inp = torch.rand(*shape).bfloat16()

    nh_new = ((nh + 31) // 32) * 32
    padded_shape = [nb, nc, nh_new, 32 * nw]
    pad_nh = nh_new - nh
    padded_inp = torch.nn.functional.pad(inp, (0, 0, 0, pad_nh), "constant", 0.123456789)

    a = ttl.tensor.Tensor(
        inp,
        dtype,
    ).to(device)
    b = ttl.tensor.tilize_with_val_padding(a, padded_shape, [0, 0, 0, 0], 0.123456789)
    c = b.cpu().to_torch()

    tilized_inp = tilize(padded_inp)

    passing, output = comp_pcc(tilized_inp, c, 0.999999)
    logger.info(output)
    assert passing
