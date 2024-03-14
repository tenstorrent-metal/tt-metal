import torch
import tt_lib

from models.utility_functions import torch2tt_tensor


def get_weights_cached(devices, model_config, state_dict, tt_cache_path, weight_cache_str, weight_config_str):
    """Load cached weights and duplicate per device. Store if not cached."""
    if (tt_cache_path / f"{weight_cache_str}_{model_config[f'{weight_config_str}_DTYPE'].name}.bin").exists():
        # Load cached weights
        weights_host = tt_lib.tensor.load_tensor(
            str(tt_cache_path / f"{weight_cache_str}_{model_config[f'{weight_config_str}_DTYPE'].name}.bin")
        )
        # Duplicate weights on all devices
        weights = [weights_host.to(device, model_config[f"{weight_config_str}_MEMCFG"]) for device in devices]
    else:
        weights_host = torch.transpose(
            state_dict[weight_cache_str],
            -2,
            -1,
        )
        # Duplicate weights on all devices
        weights = [
            torch2tt_tensor(
                weights_host,
                device,
                tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
            )
            for device in devices
        ]
        # Store weights (from first device)
        tt_lib.tensor.dump_tensor(
            str(tt_cache_path / f"{weight_cache_str}_{model_config[f'{weight_config_str}_DTYPE'].name}.bin"),
            weights[0].cpu(),
        )
    return weights
