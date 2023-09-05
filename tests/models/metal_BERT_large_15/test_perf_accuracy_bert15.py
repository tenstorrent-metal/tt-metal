from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from datasets import load_dataset
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
import pytest
import tt_lib as ttl
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from test_bert_batch_dram import TtBertBatchDram

from tests.models.metal_BERT_large_15.model_config import get_model_config

from models.utility_functions import (
    enable_persistent_kernel_cache,
    enable_compilation_reports,
    enable_memory_reports,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
    disable_persistent_kernel_cache,
    profiler,
)
from models.utility_functions import prep_report
import pytest
from loguru import logger

from tests.models.dataset_squadv2 import squadv2_1K_samples_input, squadv2_answer_decode_batch
import evaluate


BATCH_SIZE = 8
model_version = "phiyodr/bert-large-finetuned-squad2"
comments = "Large"
seq_len = 384
real_input = True
attention_mask = True
token_type_ids = True
model_config_str = "MIXED_PRECISION_BATCH8"


def run_perf_bert15(expected_inference_time, expected_compile_time, model_location_generator):
    model_config = get_model_config(model_config_str)
    model_name = str(model_location_generator(model_version, model_subdir = "Bert"))
    tokenizer_name = str(model_location_generator(model_version, model_subdir = "Bert"))

    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "accuracy_loop"
    cpu_key = "ref_key"

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)


    HF_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    HF_model.eval()
    tt_model = TtBertBatchDram(HF_model.config, HF_model, device, model_config)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    nlp = pipeline(
        "question-answering",
        model=HF_model,
        tokenizer=tokenizer,
    )

    context = BATCH_SIZE * [
        "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
    ]
    question = BATCH_SIZE * ["What discipline did Winkelmann create?"]

    inputs = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=seq_len,
        padding="max_length",
        #truncation=True,
        truncation= 'only_second',
        return_attention_mask=attention_mask,
        return_token_type_ids=token_type_ids,
        return_tensors="pt",
    )
    tt_input = tt_model.model_preprocessing(**inputs)

    #SQUaD-v2 - 1000 samples
    inputs_squadv2  = squadv2_1K_samples_input(tokenizer, seq_len, attention_mask, token_type_ids, BATCH_SIZE)
    squad_metric = evaluate.load("squad_v2")

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_out = HF_model(**inputs)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(1, *tt_input)
        ttl.device.Synchronize()
        profiler.end(first_key, force_enable=True)
        del tt_output
        tt_input = tt_model.model_preprocessing(**inputs)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(1, *tt_input)
        ttl.device.Synchronize()
        profiler.end(second_key, force_enable=True)
        del tt_output

        profiler.start(third_key)
        pred_labels = []
        cpu_pred_labels = []
        true_labels = []
        i = 0
        for batch in inputs_squadv2:
            # Limiting batch count < 42 to avoid the hanging issue
            loop_count = 40
            if (i < loop_count):
                logger.info(f"BATCH: {i}")
                batch_data = batch[0]
                cpu_output = HF_model(**batch_data)
                tt_batch = tt_model.model_preprocessing(**batch_data)
                tt_output = tt_model(1, *tt_batch)
                tt_untilized_output = tt_output.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().reshape(BATCH_SIZE, 1, seq_len, -1).to(torch.float32)
                references = batch[1]
                question = batch[2]
                context = batch[3]
                cpu_predictions, tt_predictions = squadv2_answer_decode_batch(HF_model, tokenizer, nlp, references, cpu_output, tt_untilized_output, BATCH_SIZE, question, context)
                pred_labels.extend(tt_predictions)
                cpu_pred_labels.extend(cpu_predictions)
                true_labels.extend(references)
            i += 1
        eval_score = squad_metric.compute( predictions=pred_labels, references=true_labels)
        cpu_eval_score = squad_metric.compute(predictions=cpu_pred_labels, references=true_labels)
        logger.info(f"\tTT_Eval: exact: {eval_score['exact']} --  F1: {eval_score['f1']}")
        logger.info(f"\tCPU_Eval: exact: {cpu_eval_score['exact']} -- F1:  {cpu_eval_score['f1']}")

        ttl.device.Synchronize()
        profiler.end(third_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)
    ttl.device.CloseDevice(device)

    prep_report(
        model_name="bert15",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time
    )
    compile_time = first_iter_time - second_iter_time
    logger.info(f"bert15 inference time: {second_iter_time}")
    logger.info(f"bert15 compile time: {compile_time}")
    logger.info(f"bert15 inference time Avg. (Loop 1K): {round(third_iter_time/(loop_count*BATCH_SIZE), 2)}")
    assert second_iter_time < expected_inference_time, "bert15 is too slow"
    assert compile_time < expected_compile_time, "bert15 compile time is too slow"


@pytest.mark.models_accuracy_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    ([0.15, 10],),
)
def test_perf_accuracy_virtual_machine(use_program_cache, expected_inference_time, expected_compile_time, model_location_generator):
    run_perf_bert15(expected_inference_time, expected_compile_time, model_location_generator)


@pytest.mark.models_accuracy_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    ([0.08, 8.5],),
)
def test_perf_accuracy_bare_metal(use_program_cache, expected_inference_time, expected_compile_time, model_location_generator):
    run_perf_bert15(expected_inference_time, expected_compile_time, model_location_generator)
