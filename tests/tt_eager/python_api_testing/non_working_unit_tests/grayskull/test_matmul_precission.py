# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops, tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


f = open("pcc_results.txt", "a")


def run_matmul_test(input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, low, high, device):
    torch.manual_seed(0)

    x = gen_rand(size=input_shape_1, low=low, high=high).to(torch.bfloat16)
    y = gen_rand(size=input_shape_2, low=low, high=high).to(torch.bfloat16)

    factor = random.randrange(low, high)

    try:
        pt_result = pytorch_ops.eltwise_rpow(x, factor=factor)

        tt_result = tt_lib_ops.eltwise_rpow(
            x=x,
            factor=factor,
            device=device,
            dtype=dtype,
            layout=dlayout,
            input_mem_config=in_mem_config,
            output_mem_config=out_mem_config,
        )

        success, pcc_value = comp_pcc(pt_result, tt_result)
        logger.debug(pcc_value)

        # Max ATOL Delta: 0.24883651733398438, Max RTOL Delta: 0.007148577831685543, PCC: 0.9813426036602166, PCC check failed
        pcc = pcc_value.split()[9]
        pcc = pcc.rstrip(",")

        f.write(f"{pcc}\n")
    except Exception as e:
        msg = f"{e.args[0]}"

        if "rpow cannot be calcualted for non-positive numbers" in msg:
            f.write(f"rpow cannot be calcualted for non-positive number\n")
        else:
            f.write(f"{e.args[0]}\n")

    assert success, f"low: {low}; high: {high}; message: {pcc_value}; extracted pcc: {pcc}"


range_start = [0, 0, 50]
range_end = [100, 50, 100]

# 0-10, 10-20 ... 90-100
i = 0
while i < 100:
    range_start.append(i)
    range_end.append(i + 10)
    i += 10


# 0-1, 1-2 ... 99-100
i = 0
while i < 100:
    range_start.append(i)
    range_end.append(i + 1)
    i += 1

# -100-0
range_start.append(-100)
range_end.append(0)

range_start.append(-50)
range_end.append(0)

range_start.append(-100)
range_end.append(-50)

# -10 - 0, -20 - -10 ... -100 - 90
i = 0
while i > -100:
    range_start.append(i - 10)
    range_end.append(i)
    i -= 10

# -1 - 0, -2 - -1 ... -100 - 99
i = 0
while i > -100:
    range_start.append(i - 1)
    range_end.append(i)
    i -= 1

# -100 - 100, -99 - 99, ... -1 - 1
i = 100
while i > 0:
    range_start.append(-i)
    range_end.append(i)
    i -= 1

# 100-110, 1000-1010, ...
i = 100
while i <= 100000000:
    range_start.append(i)
    range_end.append(i + 10)
    i *= 10

# -110 - -100, -1010 - -1000, ...
i = 100
while i <= 100000000:
    range_start.append(-i - 10)
    range_end.append(-i)
    i *= 10

# 100-200, 1000-1100, ...
i = 100
while i <= 100000000:
    range_start.append(i)
    range_end.append(i + 100)
    i *= 10

# -200 - -100, -1100 - -1000, ...
i = 100
while i <= 100000000:
    range_start.append(-i - 100)
    range_end.append(-i)
    i *= 10

# 0-100, 0-1000, ...
i = 100
while i <= 10000000:
    range_start.append(0)
    range_end.append(i)
    i *= 10

# 1000-10000, 1000-100000, ...
i = 10000
while i <= 10000000:
    range_start.append(1000)
    range_end.append(i)
    i *= 10

# -100 - 0, -1000 - 0, ...
i = 100
while i <= 10000000:
    range_start.append(-i)
    range_end.append(0)
    i *= 10

# -10000 - -1000, -100000 - -1000, ...
i = 10000
while i <= 10000000:
    range_start.append(-i)
    range_end.append(-1000)
    i *= 10

# -100 - 100, -1000 - 1000, ...
i = 100
while i <= 10000000:
    range_start.append(-i)
    range_end.append(i)
    i *= 10

test_sweep_args = []

for i in range(len(range_start)):
    test_sweep_args.append(
        (
            (1, 4, 128, 128),
            (1, 4, 128, 128),
            [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
            [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
            [
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ],
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            range_start[i],
            range_end[i],
        )
    )

    logger.info(f"range: {range_start[i]} --> {range_end[i]}")


@pytest.mark.parametrize(
    "input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, low, high",
    (test_sweep_args),
)
def test_matmul_test(input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, low, high, device):
    random.seed(0)
    run_matmul_test(input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, low, high, device)
