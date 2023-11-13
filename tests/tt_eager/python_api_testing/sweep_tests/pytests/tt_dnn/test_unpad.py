# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
import torch
from pathlib import Path
from functools import partial
import tt_lib

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")



from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0
import random

# Seed here for fixed params
# TODO @tt-aho: Move random param generation into the function so it's seeded by fixture
random.seed(213919)
torch.manual_seed(213919)

params = [
    pytest.param([[5, 5, 50, 50]], unpad_args)
    for unpad_args in generation_funcs.gen_unpad_args(
        [[5, 5, 50, 50]],
        dtypes=[[tt_lib.tensor.DataType.BFLOAT16]],
        layouts=[[tt_lib.tensor.Layout.ROW_MAJOR]],
        mem_configs=[[tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.DRAM)]])
]

params += [
    pytest.param([[5, 5, 64, 96]], unpad_args)
    for unpad_args in generation_funcs.gen_unpad_args(
        [[5, 5, 64, 96]],
        dtypes=[[tt_lib.tensor.DataType.BFLOAT16]],
        layouts=[[tt_lib.tensor.Layout.TILE]],
        mem_configs=[[tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.DRAM)]])
]


@pytest.mark.parametrize("input_shapes, unpad_args", params)
def test_run_unpad_test(input_shapes, unpad_args, device):
    if is_wormhole_b0():
        if input_shapes == [[5,5,64,96]]:
            pytest.skip("skip this shape for Wormhole B0")
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        "unpad",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        unpad_args,
    )
