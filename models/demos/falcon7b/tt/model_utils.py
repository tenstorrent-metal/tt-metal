# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from models.utility_functions import pad_by_zero, torch2tt_tensor


def get_weights_cached(
    devices,
    model_config,
    tt_cache_path,
    weight_cache_str,
    weight_config_str,
    weights_to_cache,
    overwrite=False,
    padzero=False,
    custom_output_shape=None,
):
    """Load cached weights and duplicate per device. Store if not cached."""
    if custom_output_shape is not None:
        custom_output_shape_str = f"{custom_output_shape[-2]}"
    else:
        custom_output_shape_str = ""

    weights_posix_path = (
        tt_cache_path
        / f"{weight_cache_str}_{model_config[f'{weight_config_str}_DTYPE'].name}_{custom_output_shape_str}.bin"
    )

    if not overwrite and weights_posix_path.exists():
        # Load cached weights
        weights_host = tt_lib.tensor.load_tensor(str(weights_posix_path))
        # Duplicate weights on all devices
        weights = [weights_host.to(device, model_config[f"{weight_config_str}_MEMCFG"]) for device in devices]
    else:
        # Duplicate weights on all devices
        if custom_output_shape:
            # pad torch tensor weights_to_cache for optimal matmul performance
            # padding is inversed for torch tensors from last to first dim
            padding = (
                0,
                custom_output_shape[-1] - weights_to_cache.shape[-1],
                0,
                custom_output_shape[-2] - weights_to_cache.shape[-2],
            )
            weights_to_cache = torch.functional.F.pad(weights_to_cache, padding, "constant", 0.0)

        if padzero:
            weights = [
                pad_by_zero(
                    weights_to_cache,
                    device,
                    tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                    tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
                )[0]
                for device in devices
            ]
        else:
            weights = [
                torch2tt_tensor(
                    weights_to_cache,
                    device,
                    tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                    tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
                )
                for device in devices
            ]
        # Store weights (from first device)
        tt_lib.tensor.dump_tensor(str(weights_posix_path), weights[0].cpu())
    return weights
