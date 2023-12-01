# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl
import math
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops, tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand

random.seed(0)
torch.manual_seed(12345)


def closestNumber(n, m):
    # Find the quotient
    q = math.floor(n / m)

    # 1st possible closest number
    n1 = m * q

    # 2nd possible closest number
    if (n * m) > 0:
        n2 = m * (q - 1)
    else:
        n2 = m * (q - 2)

    # if true, then n1 is the required closest number
    if abs(n - n1) < abs(n - n2):
        return n1

    # else n2 is the required closest number
    return n2


def generate_rest_random_shapes(max_product):
    while True:
        x, y = random.sample(range(1, max_product), 2)
        z = random.choice([num for num in range(2, max_product) if num % 2 == 0])
        if x * y * z <= max_product:
            return [x, y, z]


def select_random_input_shapes(max_dim, max_volume, max_dim_position):
    shapes = []
    rest_max_volume = closestNumber(math.floor(max_volume / max_dim), 2)
    # print(rest_max_volume)
    # print(max_dim)

    for i in range(10):
        x, y, z = generate_rest_random_shapes(rest_max_volume)
        random_max_dim = random.randint(int(0.5 * max_dim), max_dim + 1)
        # print(int(0.8*max_dim))
        # print(random_max_dim)
        my_list = [x, y, z]
        my_list.insert(max_dim_position, random_max_dim)

        logger.info("Generated list:")
        logger.info(my_list)

        shapes.append(my_list)

    return shapes


f = open("pcc_results.txt", "a")


def run_op_test(input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, low, high, device):
    x = gen_rand(size=input_shape_1, low=low, high=high).to(torch.bfloat16)
    y = gen_rand(size=input_shape_2, low=low, high=high).to(torch.bfloat16)

    pt_result = pytorch_ops.add(
        x=x,
        y=y,
    )

    tt_result = tt_lib_ops.eltwise_add(
        x=x,
        y=y,
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

    assert success, f"low: {low}; high: {high}; message: {pcc_value}; extracted pcc: {pcc}"


test_sweep_args = []
input_shapes = select_random_input_shapes(1376576, 59651228, 0)
lowest = -100
highest = 100

for input_shape in input_shapes:
    test_sweep_args.append(
        (
            input_shape,
            input_shape,
            [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
            [ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR],
            [
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ],
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            lowest,
            highest,
        )
    )


@pytest.mark.parametrize(
    "input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, low, high",
    (test_sweep_args),
)
def test_matmul_test(input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, low, high, device):
    run_op_test(input_shape_1, input_shape_2, dtype, dlayout, in_mem_config, out_mem_config, low, high, device)
