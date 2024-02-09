# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
from loguru import logger
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops, tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand, gen_rand_infinite
from tests.tt_eager.python_api_testing.sweep_tests.common import set_slow_dispatch_mode


def run_eltwise_isfinite_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device
):
    random.seed(0)
    torch.manual_seed(data_seed)
    prev_dispatch_mode = set_slow_dispatch_mode(dispatch_mode)

    x = gen_rand_infinite(size=input_shape, low=-10, high=10).to(torch.bfloat16)
    x = torch.where(x.abs() > 1e-3, x, 1e-3)

    # compute ref value
    ref_value = pytorch_ops.isfinite(x=x)

    tt_result = tt_lib_ops.eltwise_isfinite(
        x=x,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    set_slow_dispatch_mode(prev_dispatch_mode)
    assert success


test_sweep_args = []
random.seed(0)


def make_in_mem_config(buffer_type):
    if buffer_type is None:
        return None

    return ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, buffer_type)


for memorylayout in [ttl.tensor.Layout.TILE, ttl.tensor.Layout.ROW_MAJOR]:
    for shape in [(4, 7, 32, 96), (6, 7, 192, 224)]:
        for bufertype in [ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.DRAM, None]:
            for dispatch_mode in ["", "1"]:
                test_sweep_args.append(
                    (
                        shape,
                        [ttl.tensor.DataType.BFLOAT16],
                        [memorylayout],
                        [make_in_mem_config(bufertype)],
                        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
                        random.randint(1000000, 10000000),
                        dispatch_mode,
                    )
                )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_eltwise_isfinite(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device):
    run_eltwise_isfinite_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode, device
    )
