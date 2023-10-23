# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib as ttl
from models.helper_funcs import Linear as tt_Linear
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


def setup_tt_tensor(x, device, layout, input_mem_config, dtype):
    if input_mem_config is None:
        device = None

    return torch2tt_tensor(x, device, layout, input_mem_config, dtype)


# pcie slot arg will eventually be fully deprecated in favour of pytest uplift
# and passing device from fixture
def setup_host_and_device(func):
    def wrap(*args, device, **kwargs):
        output = func(*args, device=device, **kwargs)
        ttl.device.DeallocateBuffers(device)
        return output

    return wrap


################################################
################## Helper-Funcs ################
################################################


@setup_host_and_device
def linear(
    x,
    weight,
    bias=None,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    tt_weight = setup_tt_tensor(weight, device, layout[1], input_mem_config[1], dtype[1])
    tt_bias = None

    if bias is not None:
        tt_bias = setup_tt_tensor(bias, device, layout[2], input_mem_config[2], dtype[2])

    _, __, out_features, in_features = tt_weight.shape()
    tt_linear = tt_Linear(in_features, out_features, tt_weight, tt_bias)

    t1 = tt_linear(t0)
    return tt2torch_tensor(t1)


################################################
#################### TT-DNN ####################
################################################
@setup_host_and_device
def copy(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttl.tensor.copy(t0, t1)

    return tt2torch_tensor(t1)


@setup_host_and_device
def clone(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.clone(t0, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def move(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.move(t0, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_exp(
    x,
    *args,
    fast_and_approx,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.exp(t0, fast_and_approx, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_erf(
    x,
    *args,
    fast_and_approx,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.erf(t0, fast_and_approx, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_erfc(
    x,
    *args,
    fast_and_approx,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.erfc(t0, fast_and_approx, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_threshold(
    x,
    *args,
    threshold,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.threshold(t0, threshold, value, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_hardtanh(
    x,
    *args,
    low,
    high,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.hardtanh(t0, low, high, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_leaky_relu(
    x,
    *args,
    negative_slope,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.leaky_relu(t0, negative_slope, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_logical_noti(
    x,
    *args,
    immediate,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.logical_noti(t0, immediate, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_hardshrink(
    x,
    *args,
    _lambda,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.hardshrink(t0, _lambda, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_softshrink(
    x,
    *args,
    _lambda,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.softshrink(t0, _lambda, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_elu(
    x,
    *args,
    alpha,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.elu(t0, alpha, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_gelu(
    x,
    *args,
    fast_and_approx,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.gelu(t0, fast_and_approx, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_softmax_in_place(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.operations.primary.softmax_in_place(t0)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_scale_mask_softmax_in_place(
    x,
    y,
    scale,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttl.operations.primary.transformers.scale_mask_softmax_in_place(t0, scale, t1)
    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_rsqrt(
    x,
    *args,
    fast_and_approx,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.rsqrt(t0, fast_and_approx, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_relu_min(
    x,
    *args,
    lower_limit,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.relu_min(t0, lower_limit, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_relu_max(
    x,
    *args,
    upper_limit,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.relu_max(t0, upper_limit, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


# stats ops
@setup_host_and_device
def std_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.std_hw(t0, output_mem_config=output_mem_config)

    output = tt2torch_tensor(t1)
    output = output.max(2, True)[0].max(3, True)[0]

    return output


@setup_host_and_device
def var_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.var_hw(t0, output_mem_config=output_mem_config)

    output = tt2torch_tensor(t1)
    output = output.max(2, True)[0].max(3, True)[0]

    return output


@setup_host_and_device
def mean_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.mean_hw(t0, output_mem_config=output_mem_config)

    output = tt2torch_tensor(t1)
    output = output.max(2, True)[0].max(3, True)[0]

    return output


@setup_host_and_device
def eltwise_polyval(
    x,
    *args,
    coeffs,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.polyval(t0, coeffs, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_mac(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttl.tensor.mac(t0, t1, t2, output_mem_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_addcmul(
    x,
    y,
    z,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttl.tensor.addcmul(t0, t1, t2, scalar, output_mem_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_addcdiv(
    x,
    y,
    z,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttl.tensor.addcdiv(t0, t1, t2, scalar, output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_lerp_binary(
    x,
    y,
    *args,
    weight,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttl.tensor.lerp(t0, t1, weight, output_mem_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def conv(
    x,
    y,
    conv_params,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttl.tensor.conv(t0, t1, conv_params, 0, 0, 0, 0, 0, conv_params[0])

    return tt2torch_tensor(t2)


@setup_host_and_device
def layernorm_noweights(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.operations.primary.layernorm(t0, 1e-5, None, None, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def groupnorm_noweights(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.groupnorm(t0, 1, 1e-5, None, None, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def add_layernorm_noweights(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttl.operations.primary.add_layernorm(t0, t1, 1e-5, None, None, output_mem_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def layernorm(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    if layout[1] == ttl.tensor.Layout.TILE:
        y = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))

    if layout[2] == ttl.tensor.Layout.TILE:
        z = torch.nn.functional.pad(z, (0, 0, 0, 32 - z.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttl.operations.primary.layernorm(t0, 1e-5, t1, t2, output_mem_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def add_layernorm(
    x,
    y,
    z,
    w,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = setup_tt_tensor(w, device, layout[2], input_mem_config[2], dtype[2])
    t4 = ttl.operations.primary.add_layernorm(t0, t1, 1e-5, t2, t3, output_mem_config=output_mem_config)

    return tt2torch_tensor(t4)


@setup_host_and_device
def eltwise_lerp_ternary(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttl.tensor.lerp(t0, t1, t2, output_mem_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_subalpha(
    x,
    y,
    *args,
    alpha,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttl.tensor.subalpha(t0, t1, alpha, output_mem_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_logit(x, *args, eps, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.logit(t0, eps, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_polygamma(x, *args, k, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.polygamma(t0, k, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_logical_xori(
    x,
    *args,
    immediate,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.logical_xori(t0, immediate, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_assign_binary(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config=ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttl.tensor.assign(t0, t1, output_mem_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_addalpha(
    x,
    y,
    *args,
    alpha,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttl.tensor.addalpha(t0, t1, alpha, output_mem_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def eltwise_heaviside(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.heaviside(t0, scalar, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_repeat_interleave(
    x,
    *args,
    repeat,
    dim,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.repeat_interleave(t0, repeat, dim, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_isclose(
    x,
    y,
    *args,
    rtol,
    atol,
    equal_nan,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttl.tensor.isclose(t0, t1, rtol, atol, equal_nan, output_mem_config=output_mem_config)

    return tt2torch_tensor(t2)


@setup_host_and_device
def full_like(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.full_like(t0, scalar, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def ones(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = ttl.tensor.ones(
        x.shape,
        layout=layout[0],
        device=device if input_mem_config[0] is not None else None,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def zeros(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = ttl.tensor.zeros(
        x.shape,
        layout=layout[0],
        device=device if input_mem_config[0] is not None else None,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def triu(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    tx = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    diag = kwargs.get("diag", 0)
    t1 = ttl.tensor.triu(tx, diag, output_mem_config)
    return tt2torch_tensor(t1)


@setup_host_and_device
def tril(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    tx = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    diag = kwargs.get("diag", 0)
    t1 = ttl.tensor.tril(tx, diag, output_mem_config)
    return tt2torch_tensor(t1)


@setup_host_and_device
def empty(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = ttl.tensor.empty(
        x.shape,
        layout=layout[0],
        device=device if input_mem_config[0] is not None else None,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def full(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t1 = ttl.tensor.full(
        x.shape,
        scalar,
        layout=layout[0],
        device=device if input_mem_config[0] is not None else None,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def fill_rm(
    x,
    *args,
    hOnes,
    wOnes,
    val_hi,
    val_lo,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.fill_rm(
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        hOnes,
        wOnes,
        t0,
        val_hi,
        val_lo,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def fill_ones_rm(
    x,
    *args,
    hOnes,
    wOnes,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.fill_ones_rm(
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        hOnes,
        wOnes,
        t0,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def arange(
    x,
    *args,
    start,
    end,
    step=1,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t1 = ttl.tensor.arange(
        start,
        end,
        step,
        device=device if input_mem_config[0] is not None else None,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_logical_andi(
    x,
    *args,
    immediate,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.logical_andi(t0, immediate, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_logical_ori(
    x,
    *args,
    immediate,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.logical_ori(t0, immediate, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def clip(
    x,
    *args,
    low,
    high,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.clip(t0, low, high, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def where(x, y, z, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = ttl.tensor.where(t0, t1, t2, output_mem_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_div_unary(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.div_unary(t0, scalar, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_mul_unary(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.mul_unary(t0, scalar, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_sub_unary(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.sub_unary(t0, scalar, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_add_unary(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.add_unary(t0, scalar, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def bcast_add_h(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(t1, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttl.tensor.bcast(
        t0,
        t1,
        ttl.tensor.BcastOpMath.ADD,
        ttl.tensor.BcastOpDim.H,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_add_w(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE or (input_mem_config[1] and layout[1] == ttl.tensor.Layout.ROW_MAJOR):
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(t1, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttl.tensor.bcast(
        t0,
        t1,
        ttl.tensor.BcastOpMath.ADD,
        ttl.tensor.BcastOpDim.W,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_add_hw(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3], 0, 32 - y.shape[2]))
    elif input_mem_config[1] and layout[1] == ttl.tensor.Layout.ROW_MAJOR:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(t1, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttl.tensor.bcast(
        t0,
        t1,
        ttl.tensor.BcastOpMath.ADD,
        ttl.tensor.BcastOpDim.HW,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_sub_h(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(t1, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttl.tensor.bcast(
        t0,
        t1,
        ttl.tensor.BcastOpMath.SUB,
        ttl.tensor.BcastOpDim.H,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_sub_w(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE or (input_mem_config[1] and layout[1] == ttl.tensor.Layout.ROW_MAJOR):
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(t1, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttl.tensor.bcast(
        t0,
        t1,
        ttl.tensor.BcastOpMath.SUB,
        ttl.tensor.BcastOpDim.W,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_sub_hw(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3], 0, 32 - y.shape[2]))
    elif input_mem_config[1] and layout[1] == ttl.tensor.Layout.ROW_MAJOR:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(t1, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttl.tensor.bcast(
        t0,
        t1,
        ttl.tensor.BcastOpMath.SUB,
        ttl.tensor.BcastOpDim.HW,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_mul_h(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(t1, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttl.tensor.bcast(
        t0,
        t1,
        ttl.tensor.BcastOpMath.MUL,
        ttl.tensor.BcastOpDim.H,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_mul_w(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE or (input_mem_config[1] and layout[1] == ttl.tensor.Layout.ROW_MAJOR):
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(t1, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttl.tensor.bcast(
        t0,
        t1,
        ttl.tensor.BcastOpMath.MUL,
        ttl.tensor.BcastOpDim.W,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def bcast_mul_hw(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3], 0, 32 - y.shape[2]))
    elif input_mem_config[1] and layout[1] == ttl.tensor.Layout.ROW_MAJOR:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(t1, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttl.tensor.bcast(
        t0,
        t1,
        ttl.tensor.BcastOpMath.MUL,
        ttl.tensor.BcastOpDim.HW,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t2)


@setup_host_and_device
def reduce_sum_h(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.reduce(
        t0,
        ttl.tensor.ReduceOpMath.SUM,
        ttl.tensor.ReduceOpDim.H,
        1.0,
        output_mem_config=output_mem_config,
    )

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :]


@setup_host_and_device
def reduce_sum_w(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.reduce(
        t0,
        ttl.tensor.ReduceOpMath.SUM,
        ttl.tensor.ReduceOpDim.W,
        1.0,
        output_mem_config=output_mem_config,
    )

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :, :1]


@setup_host_and_device
def reduce_sum_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.reduce(
        t0,
        ttl.tensor.ReduceOpMath.SUM,
        ttl.tensor.ReduceOpDim.HW,
        1.0,
        output_mem_config=output_mem_config,
    )

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :1]


@setup_host_and_device
def reduce_max_h(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.reduce(
        t0,
        ttl.tensor.ReduceOpMath.MAX,
        ttl.tensor.ReduceOpDim.H,
        1.0,
        output_mem_config=output_mem_config,
    )

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :]


@setup_host_and_device
def reduce_max_w(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.reduce(
        t0,
        ttl.tensor.ReduceOpMath.MAX,
        ttl.tensor.ReduceOpDim.W,
        1.0,
        output_mem_config=output_mem_config,
    )

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1]


@setup_host_and_device
def reduce_max_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.reduce(
        t0,
        ttl.tensor.ReduceOpMath.MAX,
        ttl.tensor.ReduceOpDim.HW,
        1.0,
        output_mem_config=output_mem_config,
    )

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :1]


@setup_host_and_device
def reduce_min_h(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.reduce(
        t0,
        ttl.tensor.ReduceOpMath.MIN,
        ttl.tensor.ReduceOpDim.H,
        1.0,
        output_mem_config=output_mem_config,
    )

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :]


@setup_host_and_device
def reduce_min_w(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.reduce(
        t0,
        ttl.tensor.ReduceOpMath.MIN,
        ttl.tensor.ReduceOpDim.W,
        1.0,
        output_mem_config=output_mem_config,
    )

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1]


@setup_host_and_device
def reduce_min_hw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.reduce(
        t0,
        ttl.tensor.ReduceOpMath.MIN,
        ttl.tensor.ReduceOpDim.HW,
        1.0,
        output_mem_config=output_mem_config,
    )

    output = tt2torch_tensor(t1)

    # Slice out the 0 values from reduction
    return output[..., :1, :1]


@setup_host_and_device
def transpose_nh(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.transpose(t0, 0, 2, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def transpose_nw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.transpose(t0, 0, 3, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def transpose_cw(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.transpose(t0, 1, 3, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def sum(x, *args, dim, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    assert dim >= 0 and dim <= 3
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.sum(t0, dim, output_mem_config=output_mem_config)

    output = tt2torch_tensor(t1)

    if dim == 2:
        output = output[:, :, :1, :]
    elif dim == 3:
        output = output[:, :, :, :1]
    return output


@setup_host_and_device
def permute(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    permute_dims,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.permute(t0, permute_dims, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def reshape(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    reshape_dims,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.reshape(t0, *reshape_dims, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def split_last_dim_two_chunks_tiled(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.split_last_dim_two_chunks_tiled(t0, output_mem_config=output_mem_config)

    output0 = tt2torch_tensor(t1[0])
    output1 = tt2torch_tensor(t1[1])

    return [output0, output1]


@setup_host_and_device
def tilize(x, *args, device, dtype, layout, input_mem_config, output_mem_config, use_multicore, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.tilize(t0, output_mem_config=output_mem_config, use_multicore=use_multicore)

    return t1.cpu().to_torch()


@setup_host_and_device
def tilize_with_zero_padding(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.tilize_with_zero_padding(t0, output_mem_config=output_mem_config)

    return t1.cpu().to_torch()


@setup_host_and_device
def tilize_with_val_padding(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_shape,
    input_tensor_start,
    pad_value,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.tilize_with_val_padding(
        t0,
        output_tensor_shape,
        input_tensor_start,
        pad_value,
        output_mem_config=output_mem_config,
    )

    return t1.cpu().to_torch()


@setup_host_and_device
def untilize(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    if input_mem_config[0] is None:
        t0 = ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype[0],
            ttl.tensor.Layout.TILE,
        )
    else:
        t0 = ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype[0],
            ttl.tensor.Layout.TILE,
            device,
            input_mem_config[0],
        )

    t1 = ttl.tensor.untilize(t0, output_mem_config=output_mem_config)
    return t1.cpu().to_torch()


@setup_host_and_device
def untilize_with_unpadding(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_start,
    output_tensor_end,
    **kwargs,
):
    if input_mem_config[0] is None:
        t0 = ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype[0],
            ttl.tensor.Layout.TILE,
        )
    else:
        t0 = ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype[0],
            ttl.tensor.Layout.TILE,
            device,
            input_mem_config[0],
        )

    t1 = ttl.tensor.untilize_with_unpadding(
        t0, output_tensor_start, output_tensor_end, output_mem_config=output_mem_config
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def pad(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_shape,
    input_tensor_start,
    pad_value,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.pad(
        t0,
        output_tensor_shape,
        input_tensor_start,
        pad_value,
        output_mem_config=output_mem_config,
    )

    return tt2torch_tensor(t1)


@setup_host_and_device
def unpad(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_start,
    output_tensor_end,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.unpad(t0, output_tensor_start, output_tensor_end, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_rdiv(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    factor = kwargs["factor"]
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.rdiv(t0, factor, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_rsub(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    factor = kwargs["factor"]
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.rsub(t0, factor, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_rpow(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    factor = kwargs["factor"]
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.rpow(t0, factor, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def eltwise_power(
    x,
    *args,
    exponent,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.power(t0, exponent, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


def bert_large_fused_qkv_matmul(
    x,
    y,
    z,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    a_t = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    b_t = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    bias_t = (
        ttl.tensor.Tensor(
            z.flatten().tolist(),
            z.shape,
            dtype[2],
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .pad([1, 1, 32, 3072], [0, 0, 0, 0], 0)
        .to(layout[2])
        .to(device, input_mem_config[2])
    )

    t3 = ttl.tensor.bert_large_fused_qkv_matmul(a_t, b_t, bias_t, output_mem_config=output_mem_config)
    return tt2torch_tensor(t3)


@setup_host_and_device
def eltwise_bias_gelu_unary(x, *args, bias, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.bias_gelu_unary(t0, bias, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


def make_unary_op(ttl_tensor_unop):
    @setup_host_and_device
    def unary_op(
        x,
        *args,
        device,
        dtype,
        layout,
        input_mem_config,
        output_mem_config=ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
        ),
        **kwargs,
    ):
        t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
        t1 = ttl_tensor_unop(t0, output_mem_config=output_mem_config)

        return tt2torch_tensor(t1)

    return unary_op


# mean_global = make_unary_op(ttl.tensor.global_mean)
# var_global = make_unary_op(ttl.tensor.global_var)
# std_global = make_unary_op(ttl.tensor.global_std)
# normalize_global = make_unary_op(ttl.tensor.global_normalize)
# eltwise_softmax_in_place = make_unary_op(ttl.tensor.softmax_in_place)
eltwise_cos = make_unary_op(ttl.tensor.cos)
eltwise_sin = make_unary_op(ttl.tensor.sin)
eltwise_tan = make_unary_op(ttl.tensor.tan)
eltwise_acos = make_unary_op(ttl.tensor.acos)
eltwise_asin = make_unary_op(ttl.tensor.asin)
eltwise_atan = make_unary_op(ttl.tensor.atan)
eltwise_atanh = make_unary_op(ttl.tensor.atanh)
eltwise_cosh = make_unary_op(ttl.tensor.cosh)
eltwise_sinh = make_unary_op(ttl.tensor.sinh)
eltwise_tanh = make_unary_op(ttl.tensor.tanh)
eltwise_asinh = make_unary_op(ttl.tensor.asinh)
eltwise_acosh = make_unary_op(ttl.tensor.acosh)
eltwise_tanhshrink = make_unary_op(ttl.tensor.tanhshrink)
eltwise_lgamma = make_unary_op(ttl.tensor.lgamma)
eltwise_multigammaln = make_unary_op(ttl.tensor.multigammaln)
eltwise_softsign = make_unary_op(ttl.tensor.softsign)
eltwise_relu = make_unary_op(ttl.tensor.relu)
eltwise_relu6 = make_unary_op(ttl.tensor.relu6)
eltwise_sqrt = make_unary_op(ttl.tensor.sqrt)
eltwise_cbrt = make_unary_op(ttl.tensor.cbrt)
eltwise_rad2deg = make_unary_op(ttl.tensor.rad2deg)
eltwise_deg2rad = make_unary_op(ttl.tensor.deg2rad)
eltwise_sign = make_unary_op(ttl.tensor.sign)
eltwise_signbit = make_unary_op(ttl.tensor.signbit)
eltwise_abs = make_unary_op(ttl.tensor.abs)
eltwise_exp2 = make_unary_op(ttl.tensor.exp2)
eltwise_expm1 = make_unary_op(ttl.tensor.expm1)
eltwise_neg = make_unary_op(ttl.tensor.neg)
eltwise_recip = make_unary_op(ttl.tensor.recip)
eltwise_sigmoid = make_unary_op(ttl.tensor.sigmoid)
eltwise_log_sigmoid = make_unary_op(ttl.tensor.log_sigmoid)
eltwise_log = make_unary_op(ttl.tensor.log)
eltwise_log2 = make_unary_op(ttl.tensor.log2)
eltwise_log10 = make_unary_op(ttl.tensor.log10)
eltwise_swish = make_unary_op(ttl.tensor.swish)
eltwise_add1 = make_unary_op(ttl.tensor.add1)
eltwise_log1p = make_unary_op(ttl.tensor.log1p)
eltwise_softplus = make_unary_op(ttl.tensor.softplus)
eltwise_erfinv = make_unary_op(ttl.tensor.erfinv)
eltwise_mish = make_unary_op(ttl.tensor.mish)
eltwise_hardswish = make_unary_op(ttl.tensor.hardswish)
eltwise_hardsigmoid = make_unary_op(ttl.tensor.hardsigmoid)
eltwise_digamma = make_unary_op(ttl.tensor.digamma)
eltwise_silu = make_unary_op(ttl.tensor.silu)
eltwise_square = make_unary_op(ttl.tensor.square)
eltwise_ltz = make_unary_op(ttl.tensor.ltz)
eltwise_gtz = make_unary_op(ttl.tensor.gtz)
eltwise_lez = make_unary_op(ttl.tensor.lez)
eltwise_gez = make_unary_op(ttl.tensor.gez)
eltwise_nez = make_unary_op(ttl.tensor.nez)
eltwise_eqz = make_unary_op(ttl.tensor.eqz)
eltwise_assign_unary = make_unary_op(ttl.tensor.assign)
zeros_like = make_unary_op(ttl.tensor.zeros_like)
ones_like = make_unary_op(ttl.tensor.ones_like)
# eltwise_logical_not = make_unary_op(ttl.tensor.logical_not)
normalize_hw = make_unary_op(ttl.tensor.normalize_hw)
transpose_wh = make_unary_op(ttl.tensor.transpose)
transpose_hc = make_unary_op(ttl.tensor.transpose_hc)
transpose_cn = make_unary_op(ttl.tensor.transpose_cn)


def make_binary_op(ttl_tensor_binop):
    @setup_host_and_device
    def binary_op(
        x,
        y,
        *args,
        device,
        dtype,
        layout,
        input_mem_config,
        output_mem_config,
        **kwargs,
    ):
        t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
        t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
        t2 = ttl_tensor_binop(t0, t1, output_mem_config=output_mem_config)

        return tt2torch_tensor(t2)

    return binary_op


eltwise_add = make_binary_op(ttl.tensor.add)
eltwise_sub = make_binary_op(ttl.tensor.sub)
eltwise_mul = make_binary_op(ttl.tensor.mul)
eltwise_bias_gelu = make_binary_op(ttl.tensor.bias_gelu)
eltwise_squared_difference = make_binary_op(ttl.tensor.squared_difference)
eltwise_hypot = make_binary_op(ttl.tensor.hypot)
eltwise_atan2 = make_binary_op(ttl.tensor.atan2)
eltwise_min = make_binary_op(ttl.tensor.min)
eltwise_max = make_binary_op(ttl.tensor.max)
eltwise_ne = make_binary_op(ttl.tensor.ne)
eltwise_eq = make_binary_op(ttl.tensor.eq)
eltwise_gt = make_binary_op(ttl.tensor.gt)
eltwise_lt = make_binary_op(ttl.tensor.lt)
eltwise_gte = make_binary_op(ttl.tensor.gte)
eltwise_lte = make_binary_op(ttl.tensor.lte)
eltwise_xlogy = make_binary_op(ttl.tensor.xlogy)
eltwise_ldexp = make_binary_op(ttl.tensor.ldexp)
eltwise_logaddexp = make_binary_op(ttl.tensor.logaddexp)
eltwise_logaddexp2 = make_binary_op(ttl.tensor.logaddexp2)
eltwise_logical_xor = make_binary_op(ttl.tensor.logical_xor)
eltwise_logical_and = make_binary_op(ttl.tensor.logical_and)
eltwise_logical_or = make_binary_op(ttl.tensor.logical_or)
matmul = make_binary_op(ttl.tensor.matmul)
outer = make_binary_op(ttl.tensor.outer)
bmm = make_binary_op(ttl.tensor.bmm)
bert_large_pre_softmax_bmm = make_binary_op(ttl.tensor.bert_large_pre_softmax_bmm)
bert_large_post_softmax_bmm = make_binary_op(ttl.tensor.bert_large_post_softmax_bmm)
eltwise_bias_gelu = make_binary_op(ttl.tensor.bias_gelu)
eltwise_nextafter = make_binary_op(ttl.tensor.nextafter)
eltwise_isfinite = make_unary_op(ttl.tensor.isfinite)
eltwise_isinf = make_unary_op(ttl.tensor.isinf)
eltwise_isposinf = make_unary_op(ttl.tensor.isposinf)
eltwise_isneginf = make_unary_op(ttl.tensor.isneginf)
eltwise_isnan = make_unary_op(ttl.tensor.isnan)
eltwise_logical_not_unary = make_unary_op(ttl.tensor.logical_not_unary)
eltwise_i0 = make_unary_op(ttl.tensor.i0)

################################################
#################### Tensor ####################
################################################


@setup_host_and_device
def datacopy(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    device_tensor = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    host_tensor = tt2torch_tensor(device_tensor)

    return host_tensor


@setup_host_and_device
def tensor_pad(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_shape,
    input_tensor_start,
    pad_value,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = t0.pad(output_tensor_shape, input_tensor_start, pad_value)

    return tt2torch_tensor(t1)


@setup_host_and_device
def tensor_unpad(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_start,
    output_tensor_end,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])

    if (device is not None) and (input_mem_config[0] is not None):
        t0 = t0.to(device, input_mem_config[0])

    t1 = t0.unpad(output_tensor_start, output_tensor_end)
    return tt2torch_tensor(t1)


@setup_host_and_device
def pad_to_tile(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    pad_value,
    **kwargs,
):
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = t0.pad_to_tile(pad_value)

    return tt2torch_tensor(t1)


@setup_host_and_device
def unpad_from_tile(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    output_tensor_shape,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])

    if (device is not None) and (input_mem_config[0] is not None):
        t0 = t0.to(device, input_mem_config[0])

    t1 = t0.unpad_from_tile(output_tensor_shape)
    return tt2torch_tensor(t1)


@setup_host_and_device
def activation_glu(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    dim = kwargs.get("dim", -1)
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.glu(t0, dim, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def activation_geglu(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    dim = kwargs.get("dim", -1)
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.geglu(t0, dim, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def activation_reglu(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    dim = kwargs.get("dim", -1)
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.reglu(t0, dim, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def activation_swiglu(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    dim = kwargs.get("dim", -1)
    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttl.tensor.swiglu(t0, dim, output_mem_config=output_mem_config)

    return tt2torch_tensor(t1)


@setup_host_and_device
def bert_large_selfout_matmul(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    if layout[2] == ttl.tensor.Layout.TILE:
        z = torch.nn.functional.pad(z, (0, 0, 0, 32 - z.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttl.tensor.bert_large_selfout_matmul(t0, t1, t2, output_mem_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def bert_large_ff2_matmul(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    if layout[2] == ttl.tensor.Layout.TILE:
        z = torch.nn.functional.pad(z, (0, 0, 0, 32 - z.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttl.tensor.bert_large_ff2_matmul(t0, t1, t2, output_mem_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def bert_large_ff1_matmul(x, y, z, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    if layout[2] == ttl.tensor.Layout.TILE:
        z = torch.nn.functional.pad(z, (0, 0, 0, 32 - z.shape[2]))

    t0 = setup_tt_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_tt_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_tt_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttl.tensor.bert_large_ff1_matmul(t0, t1, t2, output_mem_config=output_mem_config)

    return tt2torch_tensor(t3)


@setup_host_and_device
def embeddings(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    x = x.int()
    x_shape = x.shape
    y_shape = y.shape

    batch_size = x_shape[0]
    num_rows = x_shape[2]
    embedding_dim = y_shape[3]

    t0 = ttl.tensor.Tensor(x, dtype[0]).to(device, input_mem_config[0])

    t1 = ttl.tensor.Tensor(y, dtype[1]).to(device, input_mem_config[1])

    t2 = ttl.tensor.embeddings(t0, t1, False, False, output_mem_config=output_mem_config)

    tt_data = t2.cpu().to_torch()

    tt_got_back = torch.Tensor(tt_data).reshape((batch_size, 1, num_rows, embedding_dim))

    return tt_got_back
