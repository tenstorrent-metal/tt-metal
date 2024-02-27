# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl
from loguru import logger
from pathlib import Path
from dataclasses import dataclass


OP_KEYS = (
    # MLP
    "FF1_MM_WEIGHTS",  # gate_proj
    "FF1_MM_OUTPUT",
    "FF2_MM_WEIGHTS",  # down proj
    "FF2_MM_OUTPUT",
    "FF3_MM_WEIGHTS",  # up proj
    "FF3_MM_OUTPUT",
    "FF1_FF3_MUL_OUTPUT",
    # LayerNorm
    "LAYERNORM_WEIGHTS",
    "LAYERNORM_BIAS",
    "LAYERNORM_OUTPUT",
    # Rotary
    "SIN_CACHED_WEIGHTS",
    "COS_CACHED_WEIGHTS",
    # Attention
    "WO_MM_WEIGHTS",
    "FUSED_QKV_MM_WEIGHTS",
    "PAST_K",
    "PAST_V",
    # "CREATE_QKV_HEADS_OUTPUT",
    # "PRE_SOFTMAX_MM_OUTPUT",
    # "POST_SOFTMAX_MM_OUTPUT",
    "KEYS_OUTPUT",
    "VALUES_OUTPUT",
    "K_TRANSPOSED_OUTPUT",
    "KV_CACHE",
    "ATTN_ACT",
    "ATTN_MASK",
    "OUTPUT_MM_WEIGHTS",
)

NO_MEMCFG = ()

NO_DTYPE = (
    # "CONCAT_HEADS_OUTPUT",
)

ACCEPTABLE_MODEL_CONFIG_STRS = ("BFLOAT16-DRAM", "BFLOAT16-L1", "BFLOAT8-DRAM", "BFLOAT8-L1")


def pretty_print_model_config(model_config):
    print_str = []
    for key, val in model_config.items():
        if key.endswith("MEMCFG"):
            print_str.append(f"{key}: {val.buffer_type}")
        elif key.endswith("DTYPE") or key.endswith("BOOL") or key.endswith("PROGCFG"):
            print_str.append(f"{key}: {val}")
        elif key.endswith("CACHE_PATH"):
            continue
        else:
            raise NotImplementedError(f"Unknown key: {key}")

    return "\n".join(print_str)


def get_model_config(model_config_str, num_devices=1):
    assert model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS
    DRAM_MEMCFG = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    L1_MEMCFG = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    BFP8_DTYPE = ttl.tensor.DataType.BFLOAT8_B

    # Set default dtype and mem_config based on model_config_str
    if model_config_str in ("BFLOAT16-DRAM", "BFLOAT16-L1", "BFLOAT8-DRAM", "BFLOAT8-L1"):
        dtype_str, mem_config_str = model_config_str.split("-")
        mem_config = DRAM_MEMCFG if mem_config_str == "DRAM" else L1_MEMCFG
        dtype = ttl.tensor.DataType.BFLOAT16 if dtype_str == "BFLOAT16" else ttl.tensor.DataType.BFLOAT8_B
    else:
        raise NotImplementedError(f"Model config {model_config_str} is not supported!")

    # Set defaults for dtype and mem_config for all ops
    model_config = {
        "DEFAULT_DTYPE": dtype,
        "DEFAULT_MEMCFG": mem_config,
        "MOVE_DECODER_OUTPUT_BOOL": False,
        "DEFAULT_CACHE_PATH": "/proj_sw/user_dev/hf_data/mistral/mistral-7B-v0.1",
    }  # DEFAULT_MEMCFG also used to determine banking for ttl.device.InitializeDevice
    model_config.update({f"{key}_MEMCFG": mem_config for key in OP_KEYS if key not in NO_MEMCFG})
    model_config.update({f"{key}_DTYPE": dtype for key in OP_KEYS if key not in NO_DTYPE})

    # Matmul Weights must always be BFP8_B
    # Override defaults for certain configs
    for key in model_config.keys():
        if "MM_WEIGHTS_DTYPE" in key:
            model_config[key] = BFP8_DTYPE

    # Force the following to L1
    # if model_config_str in ("BFLOAT16-L1", "BFLOAT8-L1"):
    model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"] = L1_MEMCFG
    model_config["KEYS_OUTPUT_MEMCFG"] = L1_MEMCFG
    model_config["VALUES_OUTPUT_MEMCFG"] = L1_MEMCFG
    # model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"] = L1_MEMCFG
    # model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"] = L1_MEMCFG
    model_config["K_TRANSPOSED_OUTPUT_MEMCFG"] = L1_MEMCFG
    model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"] = L1_MEMCFG
    model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"] = L1_MEMCFG
    model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"] = L1_MEMCFG
    model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"] = L1_MEMCFG
    model_config["FF1_MM_OUTPUT_MEMCFG"] = L1_MEMCFG
    model_config["FF3_MM_OUTPUT_MEMCFG"] = L1_MEMCFG
    model_config["FF2_MM_OUTPUT_MEMCFG"] = L1_MEMCFG
    model_config["FF1_FF3_MUL_OUTPUT_MEMCFG"] = L1_MEMCFG
    # model_config["KV_CACHE_MEMCFG"] = L1_MEMCFG
    model_config["ATTN_ACT_MEMCFG"] = L1_MEMCFG
    model_config["ATTN_MASK_MEMCFG"] = L1_MEMCFG
    # model_config["PAST_K_MEMCFG"] = L1_MEMCFG
    # model_config["PAST_V_MEMCFG"] = L1_MEMCFG

    # FF1 & FF3 Matmul Config Variables
    # TODO: N300 currently would only have 8x7 available until dispatch moved to eth core
    compute_with_storage_grid_size = {
        1: (8, 8),  # - For 1 device, use full 8x8 grid
        4: (8, 4),  # - For 4 devices weight_dim sharded, GCF for 8192 and 7168 is 32, so use 8x4 grid.
        8: (8, 2),  # - For 8 devices, GCF for 8192 and 3584 is 16, so use 8x2 grid.
    }[num_devices]

    in0_block_w = {1: 4, 4: 8, 8: 16}[num_devices]

    per_core_N = 14 if num_devices == 1 else 7  # Same for 4 or 8 devices to balance workload across more cores.

    model_config["FF1_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=compute_with_storage_grid_size,
        in0_block_w=in0_block_w,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=7,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
        per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=per_core_N,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        fuse_batch=True,
        fused_activation=ttl.tensor.FusibleActivation.SILU,
        mcast_in0=True,
    )

    model_config["FF2_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=compute_with_storage_grid_size,
        in0_block_w=1,  # 14336 / TILE_WIDTH=32 / Grid_Size
        out_subblock_h=1,
        out_subblock_w=2,  # 4096 / TILE_WIDTH=32 / Grid_Size
        per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=2,  # N / Tile_Width / Grid_Size
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    model_config["FF3_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=compute_with_storage_grid_size,
        in0_block_w=in0_block_w,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=7,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
        per_core_M=1,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    model_config["QKV_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(7, 1),
        in0_block_w=32,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=1,
        per_core_N=9,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, 0),
                        ttl.tensor.CoreCoord(7, 0),
                    ),
                }
            ),
            [
                32,
                288,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    HEIGHT_SHARDED_MEMCFG = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1
    )

    model_config["QHEADS_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, 0),
                        ttl.tensor.CoreCoord(7, 3),
                    ),
                }
            ),
            [
                32,
                128,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG

    model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG

    model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG

    # uncomment if need to see all the configs
    logger.debug(f"Falcon model config: \n{pretty_print_model_config(model_config)}")

    return model_config


# TODO check if needed to remove this class and use the config below from HF instead
@dataclass
class TtModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    sliding_window: int
    norm_eps: float
    vocab_size: int

    max_batch_size: int = 32
    max_seq_len: int = 4096

    FALLBACK_SOFTMAX: bool = False
    FALLBACK_EMPTY: bool = False
    FALLBACK_SCATTER: bool = True
    FALLBACK_DRAM: bool = True
    WEIGHTS_DTYPE = ttl.tensor.DataType.BFLOAT16

    if FALLBACK_DRAM:
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    else:
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
