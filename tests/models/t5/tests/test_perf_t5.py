# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer, T5Model
import torch
import pytest
import tt_lib
from loguru import logger

from models.utility_functions import (
    Profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    prep_report,
)
from models.t5.tt.t5_model import TtT5Model

BATCH_SIZE = 1


def run_perf_t5(expected_inference_time, expected_compile_time, device, model_name):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    comments = f"{model_name}"
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    use_attention_mask = True

    tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=32)
    hf_reference_model = T5Model.from_pretrained("t5-small")
    hf_reference_model.eval()

    config = hf_reference_model.config
    tt_model = TtT5Model(config, hf_reference_model.state_dict(), device)

    # Prepare input
    input_sentance = "Studies have been shown that owning a dog is good for you"
    tokenized = tokenizer(input_sentance, padding="max_length", max_length=32, return_tensors="pt")  # Batch size 1

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask if use_attention_mask else None

    decoder_input_sentence = "Studies show that"
    tokenized = tokenizer(
        decoder_input_sentence, padding="max_length", max_length=32, return_tensors="pt"
    )  # Batch size 1

    decoder_input_ids = tokenized.input_ids
    decoder_attention_mask = tokenized.attention_mask if use_attention_mask else None

    # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
    # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
    decoder_input_ids = hf_reference_model._shift_right(decoder_input_ids)

    with torch.no_grad():
        # PyTorch forward pass
        profiler.start(cpu_key)
        pt_out = hf_reference_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_model_outputs = tt_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        tt_lib.device.Synchronize()
        profiler.end(first_key)
        del tt_model_outputs

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_model_outputs = tt_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        tt_lib.device.Synchronize()
        profiler.end(second_key)
        del tt_model_outputs

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    model_name = {"t5-base": "t5-base", "google/flan-t5-small": "flan-t5-small"}[model_name]
    prep_report(
        model_name=f"{model_name}",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"{model_name} inference time: {second_iter_time}")
    logger.info(f"{model_name} compile time: {compile_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, model_name",
    (
        (
            0.07,
            6.5,
            "t5-base",
        ),
    ),
)
def test_perf_bare_metal_t5_base(
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    device,
    model_name,
):
    run_perf_t5(expected_inference_time, expected_compile_time, device, model_name)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, model_name",
    (
        (
            0.12,
            7,
            "t5-base",
        ),
    ),
)
def test_perf_virtual_machine_t5_base(
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    device,
    model_name,
):
    run_perf_t5(expected_inference_time, expected_compile_time, device, model_name)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, model_name",
    (
        (
            0.17,
            6.5,
            "google/flan-t5-small",
        ),
    ),
)
def test_perf_bare_metal_flan_t5_small(
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    device,
    model_name,
):
    run_perf_t5(expected_inference_time, expected_compile_time, device, model_name)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, model_name",
    (
        (
            0.18,
            7,
            "google/flan-t5-small",
        ),
    ),
)
def test_perf_virtual_machine_flan_t5_small(
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    device,
    model_name,
):
    run_perf_t5(expected_inference_time, expected_compile_time, device, model_name)
