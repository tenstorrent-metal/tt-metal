# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
from itertools import product
from collections import defaultdict
from math import pi
import random
import numpy as np


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
from models.utility_functions import is_wormhole_b0


reference_pcc = defaultdict(lambda: 0.999)
reference_pcc["silu"] = 0.9714
reference_pcc["swish"] = reference_pcc["silu"]
reference_pcc["softplus"] = 0.9984


def custom_compare(*args, **kwargs):
    function = kwargs.pop("function")
    if function in [
        "logical_xor",
        "logical_ori",
        "logical_or",
        "logical_xori",
        "logical_noti",
        "logical_not",
        "logical_andi",
        "is_close",
    ]:
        comparison_func = comparison_funcs.comp_equal
    elif function in ["empty"]:
        comparison_func = comparison_funcs.comp_shape
    else:
        comparison_func = partial(comparison_funcs.comp_pcc, pcc=reference_pcc[function])
    result = comparison_func(*args, **kwargs)
    return result


shapes = ([[1, 1, 32, 32]], [[1, 3, 320, 64]])
if is_wormhole_b0():
    shapes = (shapes[0],)


# TODO: This function should be split apart instead of having all these if cases
@pytest.mark.parametrize(
    "fn, input_shapes",
    list(
        product(
            (
                "lerp_binary",
                "lerp_ternary",
                "addcmul",
                "addcdiv",
                "min",
                "max",
                "swish",
                "log1p",
                "softplus",
                "mish",
                "silu",
                "polyval",
                "mac",
                "cbrt",
                "threshold",
                "hypot",
                "hardswish",
                "hardsigmoid",
                "ones_like",
                "zeros_like",
                "full_like",
                "ones",
                "empty",
                "zeros",
                "full",
                "arange",
                "hardshrink",
                "softshrink",
                "sinh",
                "cosh",
                "tanhshrink",
                "xlogy",
                "asinh",
                "acosh",
                "atanh",
                "atan2",
                "subalpha",
                "bias_gelu_unary",
                "addalpha",
                "logit",
                "logical_ori",
                "logical_xor",
                "logical_xori",
                "logical_noti",
                "logical_andi",
                "isclose",
                "digamma",
                "lgamma",
                "multigammaln",
                "polygamma",
                "nextafter",
                "scatter",
                "prod",
            ),
            shapes,
        )
    ),  # Single core, and multi-core
)
def test_run_eltwise_composite_test(fn, input_shapes, device, function_level_defaults):
    options = defaultdict(lambda: (-1.0, 1.0))
    options["log1"] = (0.0, 1.0)
    options["polyval"] = (1, 100)
    options["logit"] = (0, 1)
    options["deg2rad"] = (-180, 180)
    options["bias_gelu_unary"] = (-1e10, 1e10)
    options["rad2deg"] = (0, 2 * pi)
    options["hypot"] = (1, 100)
    options["atan2"] = (-100, 100)
    options["cbrt"] = (-1000, 1000)
    options["prod"] = (1, 1.5)
    options["hardsigmoid"] = (-100, 100)
    options["hardswish"] = (-100, 100)
    options["hardshrink"] = (-100, 100)
    options["softshrink"] = (-100, 100)
    options["leaky_shrink"] = (-100, 100)
    options["softsign"] = (1, 100)
    options["digamma"] = (1, 1000)
    options["lgamma"] = (0.1, 1e32)
    options["multigammaln"] = (1.6, 1e32)
    options["polygamma"] = (1, 10)

    options["sinh"] = (-9, 9)
    options["tanhshrink"] = (-100, 100)
    options["atanh"] = (-1, 1)
    options["cosh"] = options["sinh"]
    options["asinh"] = (-100, 100)
    options["isclose"] = (-100, 100)
    options["acosh"] = (1, 100)
    options["logical_ori"] = (-100, 100)
    options["logical_andi"] = (-100, 100)
    options["logical_xori"] = (-100, 100)

    generator = generation_funcs.gen_rand

    if is_wormhole_b0():
        if fn in ["logit"]:
            pytest.skip("does not work for Wormhole -skipping")
    if fn in ["logical_xor", "logical_xori", "logical_ori", "logical_andi"]:
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generator, low=options[fn][0], high=options[fn][1]),
                torch.int32,
            )
        ]
    elif fn in ["prod"]:  # "prod_cpu" not implemented for 'BFloat16'
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generator, low=options[fn][0], high=options[fn][1]),
                torch.float32,
            )
        ]
    else:
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generator, low=options[fn][0], high=options[fn][1]),
                torch.bfloat16,
            )
        ]
    num_inputs = 1
    if fn in ["mac", "addcmul", "addcdiv", "lerp_ternary"]:
        num_inputs = 3
    elif fn in [
        "hypot",
        "scatter",
        "min",
        "max",
        "lerp_binary",
        "xlogy",
        "subalpha",
        "addalpha",
        "bias_gelu_unary",
        "atan2",
        "subalpha",
        "addalpha",
        "logit",
        "logical_xor",
        "isclose",
        "assign_binary",
        "nextafter",
    ]:
        num_inputs = 2

    input_shapes = input_shapes * num_inputs
    datagen_func = datagen_func * num_inputs
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update({"scalar": np.random.randint(-100, 100)})
    if fn == "arange":
        test_args.update({"start": -10, "end": 1024 - 10, "step": 1})
    elif fn == "polyval":
        test_args.update({"coeffs": [1.0, 2.0, 1.0, 2.0]})
    elif fn == "threshold":
        test_args.update({"threshold": 5.0, "value": 1.0})
    elif fn in ["softshrink", "hardshrink"]:
        test_args.update({"_lambda": np.random.randint(1, 100)})
    elif fn in ["addcmul", "addcdiv"]:
        test_args.update({"value": np.random.randint(1, 100)})
    elif fn in ["lerp_binary"]:
        test_args.update({"weight": np.random.randint(1, 100)})
    elif fn in ["subalpha"]:
        test_args.update({"alpha": np.random.randint(1, 100)})
    elif fn in ["addalpha"]:
        test_args.update({"alpha": np.random.randint(1, 100)})
    elif fn in ["bias_gelu_unary", "bias_gelu"]:
        test_args.update({"bias": np.random.randint(1, 100)})
    elif fn in ["logit"]:
        test_args.update({"eps": np.random.randint(-1e-6, 1e6)})
    elif fn in ["polygamma"]:
        test_args.update({"k": np.random.randint(1, 10)})
    elif fn in ["logical_ori", "logical_andi", "logical_xori", "logical_noti"]:
        test_args.update({"immediate": np.random.randint(0, 100)})
    elif fn in ["prod"]:
        test_args.update(
            {
                "all_dimensions": random.choice([False]),
                "dim": random.choice([-4, -3, -2, -1, 0, 1, 2, 3]),
            }
        )
    elif fn in ["isclose"]:
        test_args.update(
            {
                "rtol": random.choice([1e-3, 1e-5, 1e-7]),
                "atol": random.choice([1e-2, 1e-4, 1e-6]),
                "equal_nan": random.choice([False, True]),
            }
        )
    elif fn in ["softplus"]:
        test_args.update(
            {
                "beta": random.choice([0.5, -3, 1, 4]),
                "threshold": random.choice([-20, 10, 20, 5]),
            }
        )
    run_single_pytorch_test(
        "eltwise-%s" % (fn),
        input_shapes,
        datagen_func,
        partial(custom_compare, function=fn),
        device,
        test_args,
    )
