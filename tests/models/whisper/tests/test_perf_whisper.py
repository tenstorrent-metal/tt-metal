# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import WhisperModel, AutoFeatureExtractor
import torch
from datasets import load_dataset
from loguru import logger
import pytest

import tt_lib

from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.utility_functions import prep_report, Profiler
from models.whisper.tt.whisper_model import TtWhisperModel

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)

BATCH_SIZE = 1


def run_perf_whisper(expected_inference_time, expected_compile_time, device):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    comments = "tiny"
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    pytorch_model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    configuration = pytorch_model.config

    pytorch_model.eval()
    state_dict = pytorch_model.state_dict()

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    # original from HF example should be: seq_len = 3000, when max_source_positions=1500
    input_features = inputs.input_features

    dec_seq_len = 32
    decoder_input_ids = (
        torch.tensor(
            [
                [
                    1,
                ]
                * dec_seq_len
            ]
        )
        * pytorch_model.config.decoder_start_token_id
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        pytorch_output = pytorch_model(
            input_features=input_features, decoder_input_ids=decoder_input_ids
        )
        profiler.end(cpu_key)

    tt_whisper = TtWhisperModel(
        state_dict=state_dict, device=device, config=pytorch_model.config
    )
    tt_whisper.eval()

    with torch.no_grad():
        input_features = torch2tt_tensor(
            input_features, device, tt_lib.tensor.Layout.ROW_MAJOR
        )
        input_features = input_features.to(
            device, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.L1)
        )
        profiler.start(first_key)
        ttm_output = tt_whisper(
            input_features=input_features, decoder_input_ids=decoder_input_ids
        )
        tt_lib.device.Synchronize()
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        ttm_output = tt_whisper(
            input_features=input_features, decoder_input_ids=decoder_input_ids
        )
        tt_lib.device.Synchronize()
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    prep_report(
        model_name="whisper",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"whisper tiny inference time: {second_iter_time}")
    logger.info(f"whisper compile time: {compile_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            4.15,
            25,
        ),
    ),
)
def test_perf_bare_metal(
    use_program_cache, expected_inference_time, expected_compile_time, device
):
    run_perf_whisper(expected_inference_time, expected_compile_time, device)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            5,
            27,
        ),
    ),
)
def test_perf_virtual_machine(
    use_program_cache, expected_inference_time, expected_compile_time, device
):
    run_perf_whisper(expected_inference_time, expected_compile_time, device)
