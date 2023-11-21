# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests import (
    pytorch_ops,
    tt_lib_ops,
)


op_map = {
    ################################################
    ################# Helper-Funcs #################
    ################################################
    "linear": {"tt_lib_op": tt_lib_ops.linear, "pytorch_op": pytorch_ops.linear},
    ################################################
    #################### TT-LIB ####################
    ################################################
    "clone": {
        "tt_lib_op": tt_lib_ops.clone,
        "pytorch_op": pytorch_ops.clone,
    },
    "copy": {
        "tt_lib_op": tt_lib_ops.copy,
        "pytorch_op": pytorch_ops.copy,
    },
    "move": {
        "tt_lib_op": tt_lib_ops.move,
        "pytorch_op": pytorch_ops.move,
    },
    "arange": {
        "tt_lib_op": tt_lib_ops.arange,
        "pytorch_op": pytorch_ops.arange,
    },
    # stats
    "stats-var_hw": {
        "tt_lib_op": tt_lib_ops.var_hw,
        "pytorch_op": pytorch_ops.var_hw,
    },
    "stats-std_hw": {
        "tt_lib_op": tt_lib_ops.std_hw,
        "pytorch_op": pytorch_ops.std_hw,
    },
    "stats-mean_hw": {
        "tt_lib_op": tt_lib_ops.mean_hw,
        "pytorch_op": pytorch_ops.mean_hw,
    },
    "stats-normalize_hw": {
        "tt_lib_op": tt_lib_ops.normalize_hw,
        "pytorch_op": pytorch_ops.normalize_hw,
    },
    "stats-var_global": {
        "tt_lib_op": None,  # tt_lib_ops.var_global,
        "pytorch_op": pytorch_ops.var_global,
    },
    "stats-std_global": {
        "tt_lib_op": None,  # tt_lib_ops.std_global,
        "pytorch_op": pytorch_ops.std_global,
    },
    "stats-mean_global": {
        "tt_lib_op": None,  # tt_lib_ops.mean_global,
        "pytorch_op": pytorch_ops.mean_global,
    },
    "stats-normalize_global": {
        "tt_lib_op": None,  # tt_lib_ops.normalize_global,
        "pytorch_op": pytorch_ops.normalize_global,
    },
    # Eltwise unary
    "eltwise-hardtanh": {
        "tt_lib_op": tt_lib_ops.eltwise_hardtanh,
        "pytorch_op": pytorch_ops.hardtanh,
    },
    "eltwise-clip": {
        "tt_lib_op": tt_lib_ops.clip,
        "pytorch_op": pytorch_ops.clip,
    },
    "eltwise-tril": {
        "tt_lib_op": tt_lib_ops.tril,
        "pytorch_op": pytorch_ops.tril,
    },
    "eltwise-triu": {
        "tt_lib_op": tt_lib_ops.triu,
        "pytorch_op": pytorch_ops.triu,
    },
    "eltwise-zeros": {
        "tt_lib_op": tt_lib_ops.zeros,
        "pytorch_op": pytorch_ops.zeros,
    },
    "eltwise-empty": {
        "tt_lib_op": tt_lib_ops.empty,
        "pytorch_op": pytorch_ops.empty,
    },
    "eltwise-ones": {
        "tt_lib_op": tt_lib_ops.ones,
        "pytorch_op": pytorch_ops.ones,
    },
    "fill-rm": {
        "tt_lib_op": tt_lib_ops.fill_rm,
        "pytorch_op": pytorch_ops.fill_rm,
    },
    "fill-ones-rm": {
        "tt_lib_op": tt_lib_ops.fill_ones_rm,
        "pytorch_op": pytorch_ops.fill_ones_rm,
    },
    "eltwise-full": {
        "tt_lib_op": tt_lib_ops.full,
        "pytorch_op": pytorch_ops.full,
    },
    "eltwise-zeros_like": {
        "tt_lib_op": tt_lib_ops.zeros_like,
        "pytorch_op": pytorch_ops.zeros_like,
    },
    "eltwise-ones_like": {
        "tt_lib_op": tt_lib_ops.ones_like,
        "pytorch_op": pytorch_ops.ones_like,
    },
    "eltwise-full_like": {
        "tt_lib_op": tt_lib_ops.full_like,
        "pytorch_op": pytorch_ops.full_like,
    },
    "eltwise-div_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_div_unary,
        "pytorch_op": pytorch_ops.div_unary,
    },
    "eltwise-mul_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_mul_unary,
        "pytorch_op": pytorch_ops.mul_unary,
    },
    "eltwise-sub_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_sub_unary,
        "pytorch_op": pytorch_ops.sub_unary,
    },
    "eltwise-add_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_add_unary,
        "pytorch_op": pytorch_ops.add_unary,
    },
    "eltwise-logical_not_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_not_unary,
        "pytorch_op": pytorch_ops.logical_not_unary,
    },
    "eltwise-i0": {
        "tt_lib_op": tt_lib_ops.eltwise_i0,
        "pytorch_op": pytorch_ops.i0,
    },
    "eltwise-lgamma": {
        "tt_lib_op": tt_lib_ops.eltwise_lgamma,
        "pytorch_op": pytorch_ops.lgamma,
    },
    "eltwise-logical_noti": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_noti,
        "pytorch_op": pytorch_ops.logical_noti,
    },
    "eltwise-bitwise_complement": {
        "tt_lib_op": None,  # tt_lib_ops.eltwise_bitwise_complement,
        "pytorch_op": None,  # pytorch_ops.bitwise_complement,
    },
    "eltwise-logical_xor": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_xor,
        "pytorch_op": pytorch_ops.logical_xor,
    },
    "eltwise-sinh": {
        "tt_lib_op": tt_lib_ops.eltwise_sinh,
        "pytorch_op": pytorch_ops.sinh,
    },
    "eltwise-cosh": {
        "tt_lib_op": tt_lib_ops.eltwise_cosh,
        "pytorch_op": pytorch_ops.cosh,
    },
    "eltwise-ltz": {
        "tt_lib_op": tt_lib_ops.eltwise_ltz,
        "pytorch_op": pytorch_ops.ltz,
    },
    "eltwise-gtz": {
        "tt_lib_op": tt_lib_ops.eltwise_gtz,
        "pytorch_op": pytorch_ops.gtz,
    },
    "eltwise-lez": {
        "tt_lib_op": tt_lib_ops.eltwise_lez,
        "pytorch_op": pytorch_ops.lez,
    },
    "eltwise-gez": {
        "tt_lib_op": tt_lib_ops.eltwise_gez,
        "pytorch_op": pytorch_ops.gez,
    },
    "eltwise-eqz": {
        "tt_lib_op": tt_lib_ops.eltwise_eqz,
        "pytorch_op": pytorch_ops.eqz,
    },
    "eltwise-nez": {
        "tt_lib_op": tt_lib_ops.eltwise_nez,
        "pytorch_op": pytorch_ops.nez,
    },
    "eltwise-abs": {
        "tt_lib_op": tt_lib_ops.eltwise_abs,
        "pytorch_op": pytorch_ops.abs,
    },
    "eltwise-digamma": {
        "tt_lib_op": tt_lib_ops.eltwise_digamma,
        "pytorch_op": pytorch_ops.digamma,
    },
    "eltwise-isfinite": {
        "tt_lib_op": tt_lib_ops.eltwise_isfinite,
        "pytorch_op": pytorch_ops.isfinite,
    },
    "eltwise-isinf": {
        "tt_lib_op": tt_lib_ops.eltwise_isinf,
        "pytorch_op": pytorch_ops.isinf,
    },
    "eltwise-isposinf": {
        "tt_lib_op": tt_lib_ops.eltwise_isposinf,
        "pytorch_op": pytorch_ops.isposinf,
    },
    "eltwise-isneginf": {
        "tt_lib_op": tt_lib_ops.eltwise_isneginf,
        "pytorch_op": pytorch_ops.isneginf,
    },
    "eltwise-isnan": {
        "tt_lib_op": tt_lib_ops.eltwise_isnan,
        "pytorch_op": pytorch_ops.isnan,
    },
    "eltwise-sign": {
        "tt_lib_op": tt_lib_ops.eltwise_sign,
        "pytorch_op": pytorch_ops.sign,
    },
    "eltwise-multigammaln": {
        "tt_lib_op": tt_lib_ops.eltwise_multigammaln,
        "pytorch_op": pytorch_ops.multigammaln,
    },
    "eltwise-silu": {
        "tt_lib_op": tt_lib_ops.eltwise_silu,
        "pytorch_op": pytorch_ops.silu,
    },
    "eltwise-elu": {
        "tt_lib_op": tt_lib_ops.eltwise_elu,
        "pytorch_op": pytorch_ops.elu,
    },
    "eltwise-square": {
        "tt_lib_op": tt_lib_ops.eltwise_square,
        "pytorch_op": pytorch_ops.square,
    },
    "eltwise-mish": {
        "tt_lib_op": tt_lib_ops.eltwise_mish,
        "pytorch_op": pytorch_ops.mish,
    },
    "eltwise-softplus": {
        "tt_lib_op": tt_lib_ops.eltwise_softplus,
        "pytorch_op": pytorch_ops.softplus,
    },
    "eltwise-log1p": {
        "tt_lib_op": tt_lib_ops.eltwise_log1p,
        "pytorch_op": pytorch_ops.log1p,
    },
    "eltwise-add1": {
        "tt_lib_op": tt_lib_ops.eltwise_add1,
        "pytorch_op": pytorch_ops.add1,
    },
    "eltwise-neg": {
        "tt_lib_op": tt_lib_ops.eltwise_neg,
        "pytorch_op": pytorch_ops.neg,
    },
    "eltwise-swish": {
        "tt_lib_op": tt_lib_ops.eltwise_swish,
        "pytorch_op": pytorch_ops.swish,
    },
    "eltwise-cos": {
        "tt_lib_op": tt_lib_ops.eltwise_cos,
        "pytorch_op": pytorch_ops.cos,
    },
    "eltwise-sin": {
        "tt_lib_op": tt_lib_ops.eltwise_sin,
        "pytorch_op": pytorch_ops.sin,
    },
    "eltwise-tan": {
        "tt_lib_op": tt_lib_ops.eltwise_tan,
        "pytorch_op": pytorch_ops.tan,
    },
    "eltwise-asin": {
        "tt_lib_op": tt_lib_ops.eltwise_asin,
        "pytorch_op": pytorch_ops.asin,
    },
    "eltwise-atan": {
        "tt_lib_op": tt_lib_ops.eltwise_atan,
        "pytorch_op": pytorch_ops.atan,
    },
    "eltwise-atanh": {
        "tt_lib_op": tt_lib_ops.eltwise_atanh,
        "pytorch_op": pytorch_ops.atanh,
    },
    "eltwise-acos": {
        "tt_lib_op": tt_lib_ops.eltwise_acos,
        "pytorch_op": pytorch_ops.acos,
    },
    "eltwise-asinh": {
        "tt_lib_op": tt_lib_ops.eltwise_asinh,
        "pytorch_op": pytorch_ops.asinh,
    },
    "eltwise-acosh": {
        "tt_lib_op": tt_lib_ops.eltwise_acosh,
        "pytorch_op": pytorch_ops.acosh,
    },
    "eltwise-exp": {
        "tt_lib_op": tt_lib_ops.eltwise_exp,
        "pytorch_op": pytorch_ops.exp,
    },
    "eltwise-exp2": {
        "tt_lib_op": tt_lib_ops.eltwise_exp2,
        "pytorch_op": pytorch_ops.exp2,
    },
    "eltwise-expm1": {
        "tt_lib_op": tt_lib_ops.eltwise_expm1,
        "pytorch_op": pytorch_ops.expm1,
    },
    "eltwise-recip": {
        "tt_lib_op": tt_lib_ops.eltwise_recip,
        "pytorch_op": pytorch_ops.recip,
    },
    "eltwise-sqrt": {
        "tt_lib_op": tt_lib_ops.eltwise_sqrt,
        "pytorch_op": pytorch_ops.sqrt,
    },
    "eltwise-gelu": {
        "tt_lib_op": tt_lib_ops.eltwise_gelu,
        "pytorch_op": pytorch_ops.gelu,
    },
    "eltwise-softmax_in_place": {
        "tt_lib_op": tt_lib_ops.eltwise_softmax_in_place,
        "pytorch_op": pytorch_ops.softmax_in_place,
    },
    "eltwise-scale_mask_softmax_in_place": {
        "tt_lib_op": tt_lib_ops.eltwise_scale_mask_softmax_in_place,
        "pytorch_op": pytorch_ops.scale_mask_softmax_in_place,
    },
    "eltwise-rsqrt": {
        "tt_lib_op": tt_lib_ops.eltwise_rsqrt,
        "pytorch_op": pytorch_ops.rsqrt,
    },
    "eltwise-xlogy": {
        "tt_lib_op": tt_lib_ops.eltwise_xlogy,
        "pytorch_op": pytorch_ops.xlogy,
    },
    "eltwise-logical_and": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_and,
        "pytorch_op": pytorch_ops.logical_and,
    },
    "eltwise-logical_andi": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_andi,
        "pytorch_op": pytorch_ops.logical_andi,
    },
    "eltwise-atan2": {
        "tt_lib_op": tt_lib_ops.eltwise_atan2,
        "pytorch_op": pytorch_ops.atan2,
    },
    "eltwise-lerp_binary": {
        "tt_lib_op": tt_lib_ops.eltwise_lerp_binary,
        "pytorch_op": pytorch_ops.lerp_binary,
    },
    "eltwise-lerp_ternary": {
        "tt_lib_op": tt_lib_ops.eltwise_lerp_ternary,
        "pytorch_op": pytorch_ops.lerp_ternary,
    },
    "eltwise-leaky_relu": {
        "tt_lib_op": tt_lib_ops.eltwise_leaky_relu,
        "pytorch_op": pytorch_ops.leaky_relu,
    },
    "eltwise-prelu": {
        "tt_lib_op": tt_lib_ops.eltwise_prelu,
        "pytorch_op": pytorch_ops.prelu,
    },
    "eltwise-hardshrink": {
        "tt_lib_op": tt_lib_ops.eltwise_hardshrink,
        "pytorch_op": pytorch_ops.hardshrink,
    },
    "eltwise-bias_gelu_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_bias_gelu_unary,
        "pytorch_op": pytorch_ops.bias_gelu_unary,
    },
    "eltwise-softshrink": {
        "tt_lib_op": tt_lib_ops.eltwise_softshrink,
        "pytorch_op": pytorch_ops.softshrink,
    },
    "eltwise-softsign": {
        "tt_lib_op": tt_lib_ops.eltwise_softsign,
        "pytorch_op": pytorch_ops.softsign,
    },
    "eltwise-relu": {
        "tt_lib_op": tt_lib_ops.eltwise_relu,
        "pytorch_op": pytorch_ops.relu,
    },
    "eltwise-power": {
        "tt_lib_op": tt_lib_ops.eltwise_power,
        "pytorch_op": pytorch_ops.power,
    },
    "eltwise-power_fp": {
        "tt_lib_op": tt_lib_ops.eltwise_power_fp,
        "pytorch_op": pytorch_ops.power,
    },
    "bert-large-fused-qkv-matmul": {
        "tt_lib_op": tt_lib_ops.bert_large_fused_qkv_matmul,
        "pytorch_op": pytorch_ops.bert_large_fused_qkv_matmul,
    },
    "eltwise-relu_max": {
        "tt_lib_op": tt_lib_ops.eltwise_relu_max,
        "pytorch_op": pytorch_ops.relu_max,
    },
    "eltwise-relu_min": {
        "tt_lib_op": tt_lib_ops.eltwise_relu_min,
        "pytorch_op": pytorch_ops.relu_min,
    },
    "eltwise-polyval": {
        "tt_lib_op": tt_lib_ops.eltwise_polyval,
        "pytorch_op": pytorch_ops.polyval,
    },
    "eltwise-mac": {
        "tt_lib_op": tt_lib_ops.eltwise_mac,
        "pytorch_op": pytorch_ops.mac,
    },
    "eltwise-addcmul": {
        "tt_lib_op": tt_lib_ops.eltwise_addcmul,
        "pytorch_op": pytorch_ops.addcmul,
    },
    "eltwise-addcdiv": {
        "tt_lib_op": tt_lib_ops.eltwise_addcdiv,
        "pytorch_op": pytorch_ops.addcdiv,
    },
    "eltwise-sigmoid": {
        "tt_lib_op": tt_lib_ops.eltwise_sigmoid,
        "pytorch_op": pytorch_ops.sigmoid,
    },
    "eltwise-log_sigmoid": {
        "tt_lib_op": tt_lib_ops.eltwise_log_sigmoid,
        "pytorch_op": pytorch_ops.log_sigmoid,
    },
    "eltwise-heaviside": {
        "tt_lib_op": tt_lib_ops.eltwise_heaviside,
        "pytorch_op": pytorch_ops.heaviside,
    },
    "eltwise-erf": {
        "tt_lib_op": tt_lib_ops.eltwise_erf,
        "pytorch_op": pytorch_ops.erf,
    },
    "eltwise-erfc": {
        "tt_lib_op": tt_lib_ops.eltwise_erfc,
        "pytorch_op": pytorch_ops.erfc,
    },
    "eltwise-erfinv": {
        "tt_lib_op": tt_lib_ops.eltwise_erfinv,
        "pytorch_op": pytorch_ops.erfinv,
    },
    "eltwise-nextafter": {
        "tt_lib_op": tt_lib_ops.eltwise_nextafter,
        "pytorch_op": pytorch_ops.nextafter,
    },
    "eltwise-subalpha": {
        "tt_lib_op": tt_lib_ops.eltwise_subalpha,
        "pytorch_op": pytorch_ops.subalpha,
    },
    "eltwise-addalpha": {
        "tt_lib_op": tt_lib_ops.eltwise_addalpha,
        "pytorch_op": pytorch_ops.addalpha,
    },
    "eltwise-logit": {
        "tt_lib_op": tt_lib_ops.eltwise_logit,
        "pytorch_op": pytorch_ops.logit,
    },
    "eltwise-polygamma": {
        "tt_lib_op": tt_lib_ops.eltwise_polygamma,
        "pytorch_op": pytorch_ops.polygamma,
    },
    "eltwise-logical_xori": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_xori,
        "pytorch_op": pytorch_ops.logical_xori,
    },
    "eltwise-hardsigmoid": {
        "tt_lib_op": tt_lib_ops.eltwise_hardsigmoid,
        "pytorch_op": pytorch_ops.hardsigmoid,
    },
    "eltwise-hardswish": {
        "tt_lib_op": tt_lib_ops.eltwise_hardswish,
        "pytorch_op": pytorch_ops.hardswish,
    },
    "eltwise-log": {
        "tt_lib_op": tt_lib_ops.eltwise_log,
        "pytorch_op": pytorch_ops.log,
    },
    "eltwise-log2": {
        "tt_lib_op": tt_lib_ops.eltwise_log2,
        "pytorch_op": pytorch_ops.log2,
    },
    "eltwise-log10": {
        "tt_lib_op": tt_lib_ops.eltwise_log10,
        "pytorch_op": pytorch_ops.log10,
    },
    "eltwise-tanh": {
        "tt_lib_op": tt_lib_ops.eltwise_tanh,
        "pytorch_op": pytorch_ops.tanh,
    },
    "eltwise-tanhshrink": {
        "tt_lib_op": tt_lib_ops.eltwise_tanhshrink,
        "pytorch_op": pytorch_ops.tanhshrink,
    },
    "eltwise-signbit": {
        "tt_lib_op": tt_lib_ops.eltwise_signbit,
        "pytorch_op": pytorch_ops.signbit,
    },
    "eltwise-rpow": {
        "tt_lib_op": tt_lib_ops.eltwise_rpow,
        "pytorch_op": pytorch_ops.eltwise_rpow,
    },
    "eltwise-rdiv": {
        "tt_lib_op": tt_lib_ops.eltwise_rdiv,
        "pytorch_op": pytorch_ops.eltwise_rdiv,
    },
    "eltwise-rsub": {
        "tt_lib_op": tt_lib_ops.eltwise_rsub,
        "pytorch_op": pytorch_ops.eltwise_rsub,
    },
    # Eltwise binary
    "eltwise-ne": {
        "tt_lib_op": tt_lib_ops.eltwise_ne,
        "pytorch_op": pytorch_ops.ne,
    },
    "eltwise-bias_gelu": {
        "tt_lib_op": tt_lib_ops.eltwise_bias_gelu,
        "pytorch_op": pytorch_ops.bias_gelu,
    },
    "eltwise-eq": {
        "tt_lib_op": tt_lib_ops.eltwise_eq,
        "pytorch_op": pytorch_ops.eq,
    },
    "eltwise-lt": {
        "tt_lib_op": tt_lib_ops.eltwise_lt,
        "pytorch_op": pytorch_ops.lt,
    },
    "eltwise-gt": {
        "tt_lib_op": tt_lib_ops.eltwise_gt,
        "pytorch_op": pytorch_ops.gt,
    },
    "eltwise-gte": {
        "tt_lib_op": tt_lib_ops.eltwise_gte,
        "pytorch_op": pytorch_ops.gte,
    },
    "eltwise-lte": {
        "tt_lib_op": tt_lib_ops.eltwise_lte,
        "pytorch_op": pytorch_ops.lte,
    },
    "eltwise-add": {
        "tt_lib_op": tt_lib_ops.eltwise_add,
        "pytorch_op": pytorch_ops.add,
    },
    "eltwise-sub": {
        "tt_lib_op": tt_lib_ops.eltwise_sub,
        "pytorch_op": pytorch_ops.sub,
    },
    "eltwise-mul": {
        "tt_lib_op": tt_lib_ops.eltwise_mul,
        "pytorch_op": pytorch_ops.mul,
    },
    "eltwise-min": {
        "tt_lib_op": tt_lib_ops.eltwise_min,
        "pytorch_op": pytorch_ops.min,
    },
    "eltwise-max": {
        "tt_lib_op": tt_lib_ops.eltwise_max,
        "pytorch_op": pytorch_ops.max,
    },
    "eltwise-squared_difference": {
        "tt_lib_op": tt_lib_ops.eltwise_squared_difference,
        "pytorch_op": pytorch_ops.squared_difference,
    },
    "eltwise-deg2rad": {
        "tt_lib_op": tt_lib_ops.eltwise_deg2rad,
        "pytorch_op": pytorch_ops.deg2rad,
    },
    "eltwise-rad2deg": {
        "tt_lib_op": tt_lib_ops.eltwise_rad2deg,
        "pytorch_op": pytorch_ops.rad2deg,
    },
    "eltwise-cbrt": {
        "tt_lib_op": tt_lib_ops.eltwise_cbrt,
        "pytorch_op": pytorch_ops.cbrt,
    },
    "eltwise-hypot": {
        "tt_lib_op": tt_lib_ops.eltwise_hypot,
        "pytorch_op": pytorch_ops.hypot,
    },
    "eltwise-scatter": {
        "tt_lib_op": tt_lib_ops.eltwise_scatter,
        "pytorch_op": pytorch_ops.scatter,
    },
    "eltwise-threshold": {
        "tt_lib_op": tt_lib_ops.eltwise_threshold,
        "pytorch_op": pytorch_ops.threshold,
    },
    "eltwise-relu6": {
        "tt_lib_op": tt_lib_ops.eltwise_relu6,
        "pytorch_op": pytorch_ops.relu6,
    },
    "eltwise-ldexp": {
        "tt_lib_op": tt_lib_ops.eltwise_ldexp,
        "pytorch_op": pytorch_ops.ldexp,
    },
    "eltwise-logaddexp": {
        "tt_lib_op": tt_lib_ops.eltwise_logaddexp,
        "pytorch_op": pytorch_ops.logaddexp,
    },
    "eltwise-logaddexp2": {
        "tt_lib_op": tt_lib_ops.eltwise_logaddexp2,
        "pytorch_op": pytorch_ops.logaddexp2,
    },
    "eltwise-assign_binary": {
        "tt_lib_op": tt_lib_ops.eltwise_assign_binary,
        "pytorch_op": pytorch_ops.assign_binary,
    },
    "eltwise-assign_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_assign_unary,
        "pytorch_op": pytorch_ops.assign_unary,
    },
    "eltwise-logical_or": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_or,
        "pytorch_op": pytorch_ops.logical_or,
    },
    "eltwise-logical_ori": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_ori,
        "pytorch_op": pytorch_ops.logical_ori,
    },
    "eltwise-isclose": {
        "tt_lib_op": tt_lib_ops.eltwise_isclose,
        "pytorch_op": pytorch_ops.isclose,
    },
    "eltwise-masked_fill": {
        "tt_lib_op": tt_lib_ops.masked_fill,
        "pytorch_op": pytorch_ops.masked_fill,
    },
    # Eltwise ternary
    "eltwise-arange": {
        "tt_lib_op": tt_lib_ops.arange,
        "pytorch_op": pytorch_ops.arange,
    },
    "eltwise-where": {
        "tt_lib_op": tt_lib_ops.where,
        "pytorch_op": pytorch_ops.where,
    },
    # Matmul
    "matmul": {
        "tt_lib_op": tt_lib_ops.matmul,
        "pytorch_op": pytorch_ops.matmul,
    },
    "outer": {
        "tt_lib_op": tt_lib_ops.outer,
        "pytorch_op": pytorch_ops.outer,
    },
    "bmm": {
        "tt_lib_op": tt_lib_ops.bmm,
        "pytorch_op": pytorch_ops.matmul,
    },
    # Broadcast
    "bcast-add-h": {
        "tt_lib_op": tt_lib_ops.bcast_add_h,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-w": {
        "tt_lib_op": tt_lib_ops.bcast_add_w,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-hw": {
        "tt_lib_op": tt_lib_ops.bcast_add_hw,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-sub-h": {
        "tt_lib_op": tt_lib_ops.bcast_sub_h,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-w": {
        "tt_lib_op": tt_lib_ops.bcast_sub_w,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-hw": {
        "tt_lib_op": tt_lib_ops.bcast_sub_hw,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-mul-h": {
        "tt_lib_op": tt_lib_ops.bcast_mul_h,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-w": {
        "tt_lib_op": tt_lib_ops.bcast_mul_w,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-hw": {
        "tt_lib_op": tt_lib_ops.bcast_mul_hw,
        "pytorch_op": pytorch_ops.mul,
    },
    # Reduce
    "reduce-max-h": {
        "tt_lib_op": tt_lib_ops.reduce_max_h,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2,)),
    },
    "reduce-max-w": {
        "tt_lib_op": tt_lib_ops.reduce_max_w,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-1,)),
    },
    "reduce-max-hw": {
        "tt_lib_op": tt_lib_ops.reduce_max_hw,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2, -1)),
    },
    "reduce-min-h": {
        "tt_lib_op": tt_lib_ops.reduce_min_h,
        "pytorch_op": partial(pytorch_ops.reduce_min, dims=(-2,)),
    },
    "reduce-min-w": {
        "tt_lib_op": tt_lib_ops.reduce_min_w,
        "pytorch_op": partial(pytorch_ops.reduce_min, dims=(-1,)),
    },
    "reduce-min-hw": {
        "tt_lib_op": tt_lib_ops.reduce_min_hw,
        "pytorch_op": partial(pytorch_ops.reduce_min, dims=(-2, -1)),
    },
    "reduce-sum-h": {
        "tt_lib_op": tt_lib_ops.reduce_sum_h,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2,)),
    },
    "reduce-sum-w": {
        "tt_lib_op": tt_lib_ops.reduce_sum_w,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-1,)),
    },
    "reduce-sum-hw": {
        "tt_lib_op": tt_lib_ops.reduce_sum_hw,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2, -1)),
    },
    # Transpose
    "transpose-wh": {
        "tt_lib_op": tt_lib_ops.transpose_wh,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-2, dim1=-1),
    },
    "transpose-hc": {
        "tt_lib_op": tt_lib_ops.transpose_hc,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-3, dim1=-2),
    },
    "transpose-cn": {
        "tt_lib_op": tt_lib_ops.transpose_cn,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=1),
    },
    "transpose-nh": {
        "tt_lib_op": tt_lib_ops.transpose_nh,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=-2),
    },
    "transpose-nw": {
        "tt_lib_op": tt_lib_ops.transpose_nw,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=-1),
    },
    "transpose-cw": {
        "tt_lib_op": tt_lib_ops.transpose_cw,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=1, dim1=-1),
    },
    "sum-0": {
        "tt_lib_op": partial(tt_lib_ops.sum, dim=0),
        "pytorch_op": partial(pytorch_ops.sum, dim=0),
    },
    "sum-1": {
        "tt_lib_op": partial(tt_lib_ops.sum, dim=1),
        "pytorch_op": partial(pytorch_ops.sum, dim=1),
    },
    "sum-2": {
        "tt_lib_op": partial(tt_lib_ops.sum, dim=2),
        "pytorch_op": partial(pytorch_ops.sum, dim=2),
    },
    "sum-3": {
        "tt_lib_op": partial(tt_lib_ops.sum, dim=3),
        "pytorch_op": partial(pytorch_ops.sum, dim=3),
    },
    "permute": {
        "tt_lib_op": tt_lib_ops.permute,
        "pytorch_op": pytorch_ops.permute,
    },
    "reshape": {
        "tt_lib_op": tt_lib_ops.reshape,
        "pytorch_op": pytorch_ops.reshape,
    },
    "split-last-dim-two-chunks-tiled": {
        "tt_lib_op": tt_lib_ops.split_last_dim_two_chunks_tiled,
        "pytorch_op": pytorch_ops.split_last_dim_two_chunks_tiled,
    },
    "tilize": {
        "tt_lib_op": tt_lib_ops.tilize,
        "pytorch_op": pytorch_ops.tilize,
    },
    "untilize": {
        "tt_lib_op": tt_lib_ops.untilize,
        "pytorch_op": pytorch_ops.untilize,
    },
    "tilize_with_zero_padding": {
        "tt_lib_op": tt_lib_ops.tilize_with_zero_padding,
        "pytorch_op": pytorch_ops.tilize_with_zero_padding,
    },
    "tilize_with_val_padding": {
        "tt_lib_op": tt_lib_ops.tilize_with_val_padding,
        "pytorch_op": pytorch_ops.tilize_with_val_padding,
    },
    "untilize_with_unpadding": {
        "tt_lib_op": tt_lib_ops.untilize_with_unpadding,
        "pytorch_op": pytorch_ops.untilize_with_unpadding,
    },
    "layernorm": {
        "tt_lib_op": tt_lib_ops.layernorm,
        "pytorch_op": pytorch_ops.layernorm,
    },
    "layernorm-noweights": {
        "tt_lib_op": tt_lib_ops.layernorm_noweights,
        "pytorch_op": pytorch_ops.layernorm_noweights,
    },
    "add-layernorm-noweights": {
        "tt_lib_op": tt_lib_ops.add_layernorm_noweights,
        "pytorch_op": pytorch_ops.add_layernorm_noweights,
    },
    "add-layernorm": {
        "tt_lib_op": tt_lib_ops.add_layernorm,
        "pytorch_op": pytorch_ops.add_layernorm,
    },
    "pad": {
        "tt_lib_op": tt_lib_ops.pad,
        "pytorch_op": pytorch_ops.pad,
    },
    "unpad": {
        "tt_lib_op": tt_lib_ops.unpad,
        "pytorch_op": pytorch_ops.unpad,
    },
    ################################################
    #################### Tensor ####################
    ################################################
    "datacopy": {
        "tt_lib_op": tt_lib_ops.datacopy,
        "pytorch_op": pytorch_ops.datacopy,
    },
    "tensor_pad": {
        "tt_lib_op": tt_lib_ops.tensor_pad,
        "pytorch_op": pytorch_ops.pad,
    },
    "tensor_unpad": {
        "tt_lib_op": tt_lib_ops.tensor_unpad,
        "pytorch_op": pytorch_ops.unpad,
    },
    "pad_to_tile": {
        "tt_lib_op": tt_lib_ops.pad_to_tile,
        "pytorch_op": pytorch_ops.pad_to_tile,
    },
    "unpad_from_tile": {
        "tt_lib_op": tt_lib_ops.unpad_from_tile,
        "pytorch_op": pytorch_ops.unpad_from_tile,
    },
    "conv": {
        "tt_lib_op": tt_lib_ops.conv,
        "pytorch_op": pytorch_ops.conv,
    },
    "repeat_interleave": {
        "tt_lib_op": tt_lib_ops.repeat_interleave,
        "pytorch_op": pytorch_ops.repeat_interleave,
    },
    "activation_glu": {
        "tt_lib_op": tt_lib_ops.activation_glu,
        "pytorch_op": pytorch_ops.activation_glu,
    },
    "activation_reglu": {
        "tt_lib_op": tt_lib_ops.activation_reglu,
        "pytorch_op": pytorch_ops.activation_reglu,
    },
    "activation_geglu": {
        "tt_lib_op": tt_lib_ops.activation_geglu,
        "pytorch_op": pytorch_ops.activation_geglu,
    },
    "activation_swiglu": {
        "tt_lib_op": tt_lib_ops.activation_swiglu,
        "pytorch_op": pytorch_ops.activation_swiglu,
    },
    "groupnorm-noweights": {
        "tt_lib_op": tt_lib_ops.groupnorm_noweights,
        "pytorch_op": pytorch_ops.groupnorm_noweights,
    },
    "bert-large-pre-softmax-bmm": {
        "tt_lib_op": tt_lib_ops.bert_large_pre_softmax_bmm,
        "pytorch_op": pytorch_ops.bert_large_pre_softmax_bmm,
    },
    "bert-large-post-softmax-bmm": {
        "tt_lib_op": tt_lib_ops.bert_large_post_softmax_bmm,
        "pytorch_op": pytorch_ops.bert_large_post_softmax_bmm,
    },
    "bert-large-ff1-matmul": {
        "tt_lib_op": tt_lib_ops.bert_large_ff1_matmul,
        "pytorch_op": pytorch_ops.bert_large_ff1_matmul,
    },
    "bert-large-selfout-matmul": {
        "tt_lib_op": tt_lib_ops.bert_large_selfout_matmul,
        "pytorch_op": pytorch_ops.bert_large_selfout_matmul,
    },
    "bert-large-ff2-matmul": {
        "tt_lib_op": tt_lib_ops.bert_large_ff2_matmul,
        "pytorch_op": pytorch_ops.bert_large_ff2_matmul,
    },
    "embeddings": {
        "tt_lib_op": tt_lib_ops.embeddings,
        "pytorch_op": pytorch_ops.embeddings,
    },
    "rmsnorm-noweights": {
        "tt_lib_op": tt_lib_ops.rmsnorm_noweights,
        "pytorch_op": pytorch_ops.rmsnorm_noweights,
    },
    "rmsnorm": {
        "tt_lib_op": tt_lib_ops.rmsnorm,
        "pytorch_op": pytorch_ops.rmsnorm,
    },
    "groupnorm": {
        "tt_lib_op": tt_lib_ops.groupnorm,
        "pytorch_op": pytorch_ops.groupnorm,
    },
    "complex-real": {
        "tt_lib_op": tt_lib_ops.complex_real,
        "pytorch_op": pytorch_ops.complex_real,
    },
    "complex-recip": {
        "tt_lib_op": tt_lib_ops.complex_recip,
        "pytorch_op": pytorch_ops.complex_recip,
    },
    "complex-div": {
        "tt_lib_op": tt_lib_ops.complex_div,
        "pytorch_op": pytorch_ops.complex_div,
    },
    "complex-mul": {
        "tt_lib_op": tt_lib_ops.complex_mul,
        "pytorch_op": pytorch_ops.complex_mul,
    },
    "complex-conj": {
        "tt_lib_op": tt_lib_ops.complex_conj,
        "pytorch_op": pytorch_ops.complex_conj,
    },
    "complex-abs": {
        "tt_lib_op": tt_lib_ops.complex_abs,
        "pytorch_op": pytorch_ops.complex_abs,
    },
    "complex-imag": {
        "tt_lib_op": tt_lib_ops.complex_imag,
        "pytorch_op": pytorch_ops.complex_imag,
    },
}
