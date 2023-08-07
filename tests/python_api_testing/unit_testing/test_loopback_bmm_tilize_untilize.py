import sys
import pytest
import itertools
import argparse

from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np
import torch

from libs import tt_lib as ttl
from python_api_testing.models.utility_functions import (
    tilize_to_list,
    tilize,
    untilize,
    comp_pcc,
)

TILE_HEIGHT = TILE_WIDTH = 32

## parameters
# matrix sizes as number of blocks along h and w:
# a_height_nblocks = [1, 5, 8]  ## various
# a_width_nblocks = [1, 8]   ## various
# b_width_nblocks = [1, 8]   ## various
# a_height_nblocks = [1, 2]  ## various
# a_width_nblocks = [1]   ## various
# b_width_nblocks = [1]   ## various
# # block sizes as number of tiles along h and w:
# a_block_height_ntiles = [1] ## various
# a_block_width_ntiles = [1]  ## various
# b_block_width_ntiles = [1]  ## various
# # output sublobcking per block:
# out_subblock_height_ntiles = [1]    ## == a_block_height_ntiles, <= 8 (various)
# out_subblock_width_ntiles = [1]     ## == b_block_width_ntiles, <= 8 (various)


# @pytest.mark.parametrize(
#     'a_height_nblocks, a_width_nblocks, b_width_nblocks,\
#      a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,\
#      out_subblock_height_ntiles, out_subblock_width_ntiles',
#     itertools.product(a_height_nblocks, a_width_nblocks, b_width_nblocks,
#                       a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,
#                       out_subblock_height_ntiles, out_subblock_width_ntiles)
# )
def test_run_bmm_single_core_loopback_tilize_untilize(a_height_nblocks,
                                             a_width_nblocks,
                                             b_width_nblocks,
                                             a_block_height_ntiles,
                                             a_block_width_ntiles,
                                             b_block_width_ntiles,
                                             out_subblock_height_ntiles,
                                             out_subblock_width_ntiles):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    a_batch = b_batch = 1
    a_channel = b_channel = 1
    a_height = a_height_nblocks * a_block_height_ntiles * TILE_HEIGHT
    a_width = a_width_nblocks * a_block_width_ntiles * TILE_WIDTH   # == b_height
    b_width = b_width_nblocks * b_block_width_ntiles * TILE_WIDTH
    a_shape = [a_batch, a_channel, a_height, a_width]
    b_shape = [b_batch, b_channel, a_width, b_width]
    out_shape = [a_batch, a_channel, a_height, b_width]

    print("a_shape ", a_shape, "b_shape ", b_shape, "out_shape ", out_shape)
    print("out_subblock_height_ntiles: ", out_subblock_height_ntiles, "\nout_subblock_width_ntiles: ", out_subblock_width_ntiles)

    torch.manual_seed(0)
    a = torch.randn(a_shape, dtype=torch.bfloat16).float()
    b = torch.randn(b_shape, dtype=torch.bfloat16).float()

    # a = torch.arange(0, a_batch * a_channel * a_height * a_width).reshape(a_shape).to(torch.bfloat16).float()
    # b = torch.arange(0, b_batch * b_channel * a_width * b_width).reshape(b_shape).to(torch.bfloat16).float()
    # print(a)
    # print(b)

    ## a in row-major
    tta = ttl.tensor.Tensor(
        a.flatten().tolist(),
        a_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        device)
    ## b in tile major
    b_list = tilize_to_list(b)
    ttb = ttl.tensor.Tensor(
        b_list,
        b_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device)

    torch.set_printoptions(
       precision=2, threshold=10000,
       sci_mode=False, edgeitems=80, linewidth=480)

    # np.set_printoptions(precision=2, threshold=10000, edgeitems=80, linewidth=480, suppress=True)

    # ttb_pytorch = untilize(torch.tensor(ttb.to(host).data()).reshape([1, 1, 32, 32 * b_width_nblocks]))
    # print("b full:\n", ttb_pytorch)
    # print("b slice:\n", ttb_pytorch[0, 0, 1:2*a_width_nblocks:1, 0:32*b_width_nblocks:1])

    ## compute out
    out = ttl.tensor.loopback_bmm_tilize_untilize(tta, ttb,
                                         a_height_nblocks, a_width_nblocks, b_width_nblocks,
                                         a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,
                                         out_subblock_height_ntiles, out_subblock_width_ntiles)
    out = out.to(host)
    # out.pretty_print()
    # print(out.to(ttl.tensor.Layout.TILE).data())

    out_pytorch = torch.tensor(out.data()).reshape(out_shape)
    ttl.device.CloseDevice(device)

    ## reference
    golden_pytorch = torch.matmul(a, b)

    # print(torch.eq(out_pytorch, golden_pytorch))

    ## test for equivalance
    assert(out_pytorch.shape == golden_pytorch.shape)
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch)
    print(f'Passing PCC = {passing_pcc}')
    print(f'Output PCC = {output_pcc}')

    # assert(passing_pcc)

def get_args():
    parser = argparse.ArgumentParser('bmm_tilize_untilize parameters')
    parser.add_argument("--a-height-nblocks", default=1, type=int, help="a_height_nblocks")
    parser.add_argument("--a-width-nblocks", default=1, type=int, help="a_width_nblocks")
    parser.add_argument("--b-width-nblocks", default=1, type=int, help="b_width_nblocks")
    parser.add_argument("--a-block-height-ntiles", default=1, type=int, help="a_block_height_ntiles")
    parser.add_argument("--a-block-width-ntiles", default=1, type=int, help="a_block_width_ntiles")
    parser.add_argument("--b-block-width-ntiles", default=1, type=int, help="b_block_width_ntiles")
    parser.add_argument("--out-subblock-height-ntiles", default=1, type=int, help="out_subblock_height_ntiles")
    parser.add_argument("--out-subblock-width-ntiles", default=1, type=int, help="out_subblock_width_ntiles")
    args = parser.parse_args()
    return args

args = get_args()

test_run_bmm_single_core_loopback_tilize_untilize(args.a_height_nblocks,
                                             args.a_width_nblocks,
                                             args.b_width_nblocks,
                                             args.a_block_height_ntiles,
                                             args.a_block_width_ntiles,
                                             args.b_block_width_ntiles,
                                             args.out_subblock_height_ntiles,
                                             args.out_subblock_width_ntiles)
