# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
import pytest
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger

import ttnn
from models.utility_functions import is_wormhole_b0

torch.set_printoptions(threshold=1000000)


@pytest.mark.parametrize(
    "shape_dim",
    (((1, 2, 32, 32), 1),),
)
def test_softmax_compute_kernel_config(shape_dim, device):
    device.enable_program_cache()

    shape, dim = shape_dim
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    if is_wormhole_b0():
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    else:
        # Grayskull doesn't support fp32 but test passing a GS config is ok
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
        )

    x = torch.ones((N * C * H * W,)).reshape((N, C, H, W)).to(torch.bfloat16)
    x[0][0] = 0
    x[0][1] = 0

    dev_x = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)

    tt_cpu = torch.softmax(x, dim)

    print("is_wormhole_b0", is_wormhole_b0())
    tt_npu = ttl.operations.primary.moreh_softmax(dev_x, dim, compute_kernel_config=compute_kernel_config)

    assert list(tt_npu.get_legacy_shape()) == list(tt_cpu.shape)
    tt_dev = tt_npu.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.debug(out)
    # print("tt_cpu", tt_cpu)
    print("tt_dev", tt_dev)
    # assert passing
