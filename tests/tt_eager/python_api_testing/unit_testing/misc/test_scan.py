# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tt_lib as ttl
from tt_lib import tensor as tt

from models.utility_functions import skip_for_wormhole_b0, torch2tt_tensor, tt2torch_tensor

@skip_for_wormhole_b0()
def test_scan(device):
    torch.manual_seed(0)

    shape = (128, 1024)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    shard_grid = tt.CoreRangeSet({
        tt.CoreRange(
                tt.CoreCoord(0, 0),
                tt.CoreCoord(1, 0),
            )
    })
    n_cores = 2

    shard_spec = tt.ShardSpec(shard_grid, [64, 1024], tt.ShardOrientation.ROW_MAJOR, False)

    tt_input = torch2tt_tensor(
        torch_input,
        device,
        tt.Layout.TILE,
        tt_memory_config=tt.MemoryConfig(tt.TensorMemoryLayout.HEIGHT_SHARDED, tt.BufferType.L1, shard_spec),
    )

    tt.scan(tt_input)
