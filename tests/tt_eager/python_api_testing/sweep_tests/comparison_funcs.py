# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from loguru import logger

import os


def custom_breakpoint():
    print("Current PID = ", os.getpid())
    breakpoint()


def get_atol_rtol_pcc(golden, calculated):
    if golden.is_complex() and calculated.is_complex():
        golden = torch.view_as_real(golden.clone())
        calculated = torch.view_as_real(calculated.clone())

    if not (golden.is_floating_point() or calculated.is_floating_point()):
        golden = golden.to(torch.float)
        calculated = calculated.to(torch.float)

    # Calculate atol and rtol
    cal_atol = torch.max(torch.abs(golden - calculated)).item()
    cal_rtol = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()

    # Calculate PCC
    def get_pcc(golden, calculated):
        # Both tensors are nan
        if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
            logger.warning("Both tensors are 'nan'")
            return 1.0

        # One tensor is all nan, the other is not
        if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
            logger.error("One tensor is all nan, the other is not.")
            return 0.0

        # One tensor is all zero, the other is not
        if torch.any(golden.bool()) != torch.any(calculated.bool()):
            logger.warning("One tensor is all zero")
            return 0.0

        # if torch.any(torch.isinf(golden)) or torch.any(torch.isinf(calculated)):
        #    raise RuntimeError(f"Tensor overflow to infinity: \n{golden}\n{calculated}")

        # if torch.any(torch.isneginf(golden)) or torch.any(torch.isneginf(calculated)):
        #    raise RuntimeError(f"Tensor overflow to negative infinity: \n{golden}\n{calculated}")

        else:
            # For now, mask all infs and nans so that we check the rest... TODO
            golden = golden.clone()
            golden[
                torch.logical_or(
                    torch.isnan(golden),
                    torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
                )
            ] = 0
            calculated = calculated.clone()
            calculated[
                torch.logical_or(
                    torch.isnan(calculated),
                    torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
                )
            ] = 0

            if torch.equal(golden, calculated):
                return 1.0

            if golden.dtype == torch.bfloat16:
                golden = golden.type(torch.float32)
                calculated = calculated.type(torch.float32)

            # Single element case
            if golden.numel() == 1:
                return float(torch.equal(golden, calculated))

            # one tensor is constant
            if torch.max(golden) == torch.min(golden) or torch.max(calculated) == torch.min(calculated):
                return float(torch.equal(golden, calculated))

            cal_pcc = np.ma.corrcoef(
                np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
                np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
            )
            # Remove correlation coefficient with self (typically always 1.0)
            mask = np.ones(cal_pcc.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            cal_pcc = np.min(cal_pcc[mask])

            if isinstance(cal_pcc, np.ma.core.MaskedConstant):
                return 1.0

            return cal_pcc

    cal_pcc = get_pcc(golden, calculated)

    return (
        cal_atol,
        cal_rtol,
        cal_pcc,
        f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}, PCC: {cal_pcc}",
    )


def comp_equal(golden, calculated):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    while len(golden.shape) < len(calculated.shape):
        golden = torch.unsqueeze(golden, 0)

    _, _, _, output_str = get_atol_rtol_pcc(golden, calculated)
    equal = torch.equal(golden, calculated)

    if not equal:
        output_str += ", Equal check failed"

    return equal, output_str


def comp_shape(golden, calculated):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)
    output_str = "compare shape"
    equal = golden.shape == calculated.shape
    return equal, output_str


def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    _, _, _, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = torch.allclose(golden, calculated, rtol, atol, True)
    if not passing:
        output_str += ", Allclose check failed"
    return passing, output_str


def replace_inf_with_0(t: torch.Tensor):
    t = torch.where(t >= 2.65e38, 0, t)
    t = torch.where(t <= -2.64e38, 0, t)
    return t


def comp_pcc_skip_inf(golden, calculated, *args, **kwargs):
    golden = replace_inf_with_0(golden)
    calculated = replace_inf_with_0(calculated)
    return comp_pcc(golden, calculated, *args, **kwargs)


def comp_pcc(golden, calculated, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)
    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = cal_pcc >= pcc
    if not passing:
        output_str += ", PCC check failed"
    return passing, output_str


def comp_pcc_list(golden, calculated, pcc=0.99):
    total_str = ""
    min_pcc = 1

    for i in range(len(golden)):
        if golden[i].dtype != calculated[i].dtype:
            calculated[i] = calculated[i].type(golden[i].dtype)
        _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden[i], calculated[i])

        total_str = f"{total_str}Tensor {i}: {output_str} "

        if cal_pcc < min_pcc:
            min_pcc = cal_pcc

    passing = min_pcc >= pcc
    if not passing:
        total_str += ", PCC check failed"
    return passing, total_str


def comp_allclose_and_pcc(golden, calculated, rtol=1e-05, atol=1e-08, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = True
    allclose_passing = torch.allclose(golden, calculated, rtol, atol, True)
    passing &= allclose_passing
    if not allclose_passing:
        output_str += ", Allclose check failed"
    pcc_passing = cal_pcc >= pcc
    passing &= pcc_passing
    if not pcc_passing:
        output_str += ", PCC check failed"
    return passing, output_str


def comp_using_plot(tname, input, golden, calculated):
    import matplotlib.pyplot as plt

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
        input = input.type(torch.float32)
    shape = "x".join(list(map(str, list(input.size()))))
    plot_name = "plot_" + tname + "_" + shape + ".png"
    input = input.flatten()
    golden = golden.flatten()
    calculated = calculated.flatten()
    plt.plot(input, golden, "+r", label="CPU (golden)")
    plt.plot(input, calculated, "-b", label="On device (calculated)")
    plt.legend(loc="upper center")
    plt.savefig(plot_name)
    plt.close()
