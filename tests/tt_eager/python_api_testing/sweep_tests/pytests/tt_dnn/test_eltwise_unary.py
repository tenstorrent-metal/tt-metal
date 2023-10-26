# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
from math import pi
import copy


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
from tests.tt_eager.python_api_testing.sweep_tests.common import (
    is_wormhole_b0,
)

shapes = [
    [[1, 1, 32, 32]],  # Single core
    [[1, 1, 320, 384]],  # Multi core
    [[1, 3, 320, 384]],  # Multi core
]
input_mem_cfgs = copy.copy(generation_funcs.supported_mem_configs)
output_mem_cfgs = copy.copy(generation_funcs.supported_mem_configs)
if is_wormhole_b0():
    shapes = [
        shapes[0],
    ]


@pytest.mark.parametrize(
    "input_shapes",
    shapes,
)
@pytest.mark.parametrize("input_mem_config", input_mem_cfgs)
@pytest.mark.parametrize("output_mem_config", output_mem_cfgs)
class TestEltwiseUnary:
    @pytest.mark.parametrize(
        "fn_kind",
        ["relu", "sigmoid", "square", "tanh", "relu6", "add1", "deg2rad", "rad2deg"],
    )
    def test_run_eltwise_unary_ops(
        self,
        input_shapes,
        fn_kind,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize(
        "fn_kind",
        ["atan"],
    )
    def test_run_eltwise_atan_op(
        self,
        fn_kind,
        device,
        input_shapes,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-1e6, high=1e6), torch.bfloat16)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize(
        "fn_kind",
        ["erfinv"],
    )
    def test_run_eltwise_erfinv_op(
        self,
        fn_kind,
        device,
        input_shapes,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        if is_wormhole_b0():
            datagen_func = [
                generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-1, high=1), torch.bfloat16)
            ]
        else:
            datagen_func = [
                generation_funcs.gen_func_with_cast(
                    partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
                )
            ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize(
        "fn_kind",
        ["logical_not_unary"],
    )
    def test_run_eltwise_not_op(
        self,
        fn_kind,
        device,
        input_shapes,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.int32)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("fast_and_approx", [True, False])
    def test_run_eltwise_gelu_op(
        self,
        input_shapes,
        fast_and_approx,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["fast_and_approx"] = fast_and_approx
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-gelu",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("fast_and_approx", [True, False])
    def test_run_eltwise_rsqrt_op(
        self,
        input_shapes,
        fast_and_approx,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=0, high=1e8), torch.bfloat16)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["fast_and_approx"] = fast_and_approx
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-rsqrt",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("fast_and_approx", [True, False])
    def test_run_eltwise_i0_op(
        self,
        input_shapes,
        fast_and_approx,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-10, high=10), torch.bfloat16)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["fast_and_approx"] = fast_and_approx
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-i0",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize(
        "exp_type, input_range, comparison_func",
        (
            ("exp", {"low": -10, "high": 10}, comparison_funcs.comp_pcc),
            (
                "exp",
                {"low": -150, "high": -10},
                partial(comparison_funcs.comp_allclose, atol=5e-6, rtol=0),
            ),
            (
                "exp",
                {"low": -5e6, "high": -0.85e6},
                partial(comparison_funcs.comp_allclose, atol=3e-29, rtol=0),
            ),
            (
                "exp",
                {"low": -5e6, "high": -0.85e6},
                partial(comparison_funcs.comp_allclose, atol=3e-29, rtol=0),
            ),
            (
                "exp",
                {"low": -70, "high": -66},
                partial(comparison_funcs.comp_allclose, atol=3e-29, rtol=0),
            ),
        ),
    )
    @pytest.mark.parametrize("fast_and_approx", [True, False])
    def test_run_eltwise_exp_ops(
        self,
        input_shapes,
        exp_type,
        input_range,
        fast_and_approx,
        comparison_func,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, **input_range),
                torch.float32,
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
                "fast_and_approx": fast_and_approx,
            }
        )
        run_single_pytorch_test(
            f"eltwise-{exp_type}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize(
        "exp_type, input_range, comparison_func",
        (
            ("exp2", {"low": -10, "high": 10}, comparison_funcs.comp_pcc),
            ("expm1", {"low": -10, "high": 10}, comparison_funcs.comp_pcc),
        ),
    )
    def test_run_eltwise_exp2m1_ops(
        self,
        input_shapes,
        exp_type,
        input_range,
        comparison_func,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, **input_range),
                torch.float32,
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        run_single_pytorch_test(
            f"eltwise-{exp_type}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    def test_run_eltwise_recip_op(
        self,
        input_shapes,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand_symmetric, low=1, high=100),
                torch.float32,
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-recip",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    def test_run_eltwise_sqrt_op(
        self,
        input_shapes,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=0, high=100), torch.float32)
        ]
        comparison_func = partial(comparison_funcs.comp_pcc)
        run_single_pytorch_test(
            "eltwise-sqrt",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
        )

    @pytest.mark.parametrize("relu_type, limit_type", [["min", "lower"], ["max", "upper"]])
    @pytest.mark.parametrize("input_value", [-2.0, -1.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("limit", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_run_eltwise_relu_limit_ops(
        self,
        input_shapes,
        relu_type,
        limit_type,
        input_value,
        limit,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_constant, constant=input_value),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_equal
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({f"{limit_type}_limit": limit})
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        run_single_pytorch_test(
            f"eltwise-relu_{relu_type}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize(
        "fn_kind",
        [
            "ltz",
            "gtz",
            "lez",
            "gez",
            "eqz",
            "nez",
        ],
    )
    def test_run_eltwise_cmp_ops(
        self,
        fn_kind,
        input_shapes,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        numel = torch.Size(input_shapes[0]).numel()
        low = 0 - numel // 2
        high = numel + low
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_linspace, low=low, high=high),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_equal
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
        )

    @pytest.mark.parametrize("log_kind", ["log", "log2", "log10"])
    @pytest.mark.parametrize("range", [{"low": 1e-6, "high": 1.0}, {"low": 1, "high": 100000}])
    def test_run_eltwise_log_ops(
        self,
        log_kind,
        range,
        input_shapes,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=range["low"], high=range["high"]),
                torch.float32,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{log_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
        )

    @pytest.mark.parametrize("trig_kind", ["sin", "cos"])
    def test_run_eltwise_periodic_trig_ops(
        self,
        input_shapes,
        trig_kind,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=0.0, high=2.0 * pi),
                torch.float32,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{trig_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
        )

    @pytest.mark.parametrize("input_value", [-1.0, 2.0, 3.0])
    @pytest.mark.parametrize("exponent", [0, 1, 2, 3])
    def test_run_eltwise_power_op(
        self,
        input_shapes,
        input_value,
        exponent,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_constant, constant=input_value),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_equal
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"exponent": exponent})
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        run_single_pytorch_test(
            "eltwise-power",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("fn_kind", ["abs", "sign", "neg", "signbit"])
    @pytest.mark.parametrize("fill_val", [-1.0, 0.0, 1.0])
    def test_run_eltwise_sign_ops(
        self,
        fn_kind,
        input_shapes,
        fill_val,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        if fn_kind == "signbit" and fill_val == 0.0:
            pytest.skip("Signbit fails for 0 value")
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_constant, constant=fill_val),
                torch.bfloat16,
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        comparison_func = comparison_funcs.comp_equal
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("scalar", [0.5])
    def test_run_eltwise_heaviside(
        self,
        input_shapes,
        scalar,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"scalar": scalar})
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        comparison_func = comparison_funcs.comp_equal
        run_single_pytorch_test(
            "eltwise-heaviside",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("unary_kind", ["add_unary", "sub_unary", "mul_unary", "div_unary"])
    @pytest.mark.parametrize("scalar", [-2.0, 1.0, 2.0, 8.0])
    def test_run_eltwise_binop_to_unary_ops(
        self,
        unary_kind,
        input_shapes,
        device,
        scalar,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
        ]
        comparison_func = comparison_funcs.comp_pcc
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"scalar": scalar})
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        run_single_pytorch_test(
            f"eltwise-{unary_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("clip_kind", ["clip", "hardtanh"])
    @pytest.mark.parametrize("clip_range", ({"low": -2.0, "high": 2.0}, {"low": -5.5, "high": 27.5}))
    def test_run_eltwise_clip_ops(
        self,
        clip_kind,
        clip_range,
        input_shapes,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-10, high=10), torch.float32)
        ]
        comparison_func = comparison_funcs.comp_pcc
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(clip_range)
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        run_single_pytorch_test(
            f"eltwise-{clip_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("alpha", [-0.5, 0, 0.5])
    def test_run_eltwise_elu_op(
        self,
        input_shapes,
        alpha,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"alpha": alpha})
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        comparison_func = comparison_funcs.comp_pcc

        run_single_pytorch_test(
            "eltwise-elu",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("negative_slope", [-0.5, 0, 0.5])
    def test_run_eltwise_leaky_relu_op(
        self,
        input_shapes,
        negative_slope,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"negative_slope": negative_slope})
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        if is_wormhole_b0():
            comparison_func = partial(comparison_funcs.comp_pcc, pcc=0.87)
        else:
            comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-leaky_relu",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    def test_run_eltwise_log_sigmoid_op(
        self,
        input_shapes,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-4, high=10), torch.float32)
        ]
        comparison_func = partial(comparison_funcs.comp_pcc)
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        run_single_pytorch_test(
            "eltwise-log_sigmoid",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
        )

    @pytest.mark.parametrize("arc_trig_kind", ["asin", "acos"])
    def test_run_eltwise_arc_trig_ops(
        self,
        input_shapes,
        arc_trig_kind,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-1, high=1),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        run_single_pytorch_test(
            f"eltwise-{arc_trig_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
        )

    @pytest.mark.parametrize("fn_kind", ["erf", "erfc"])
    @pytest.mark.parametrize("fast_and_approx", [True, False])
    def test_run_eltwise_erf_ops(
        self,
        input_shapes,
        fn_kind,
        fast_and_approx,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["fast_and_approx"] = fast_and_approx
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("fn_kind", ["isinf", "isposinf", "isneginf", "isnan", "isfinite"])
    def test_run_eltwise_inf_nan_ops(
        self,
        input_shapes,
        fn_kind,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_equal
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    def test_run_eltwise_tan_op(
        self,
        input_shapes,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-1.45, high=1.45),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-tan",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
        )

    @pytest.mark.parametrize("fn_kind", ["rpow", "rsub", "rdiv"])
    def test_run_eltwise_reverse_ops(
        self,
        input_shapes,
        fn_kind,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        values = {"rdiv": (1.0, 100.0), "rsub": (1.0, 50.0), "rpow": (5.0, 10.0)}
        low_v, high_v = values[fn_kind]
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=low_v, high=high_v),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
                "factor": 4.0,
            }
        )
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("fn_kind", ["rsub", "rdiv"])
    def test_run_eltwise_reverse_neg_ops(
        self,
        input_shapes,
        fn_kind,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        values = {"rdiv": (1.0, 100.0), "rsub": (1.0, 50.0), "rpow": (5.0, 10.0)}
        low_v, high_v = values[fn_kind]
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=low_v, high=high_v),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
                "factor": pi,
            }
        )
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("diag", [-2, -1, 0, 1, 2])
    @pytest.mark.parametrize("fn_kind", ["tril", "triu"])
    def test_run_eltwise_tri_ops(
        self,
        input_shapes,
        fn_kind,
        diag,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        low_v, high_v = 0, 10
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=low_v, high=high_v),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
                "diag": diag,
            }
        )
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )
