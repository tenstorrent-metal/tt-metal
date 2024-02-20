# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import torch
from loguru import logger

import transformers
import ttnn
import tt_lib
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)
from models.experimental.functional_bert.tt import ttnn_functional_bert
from models.experimental.functional_bert.tt import ttnn_optimized_functional_bert

from models.datasets.dataset_squadv2 import squadv2_1K_samples_input, squadv2_answer_decode_batch
from ttnn.model_converter import (
    from_torch_model,
)

from ttnn.model_converter import *
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline

import evaluate


def load_inputs(input_path, batch):
    with open(input_path) as f:
        input_data = json.load(f)
        assert len(input_data) >= batch, f"Input data needs to have at least {batch} (batch size) entries."

        context = []
        question = []
        for i in range(batch):
            context.append(input_data[i]["context"])
            question.append(input_data[i]["question"])

        return context, question


def run_bert_question_and_answering_inference(
    device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    functional_bert,
    model_location_generator,
    input_path,
):
    disable_persistent_kernel_cache()

    model = str(model_location_generator(model_name, model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model, torchscript=False)
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer_name = str(model_location_generator(model_name, model_subdir="Bert"))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    config = hugging_face_reference_model.config
    nlp = pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)

    if functional_bert == ttnn_functional_bert:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_bert == ttnn_optimized_functional_bert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_bert: {functional_bert}")

    profiler.start(f"preprocessing_parameter")
    parameters = ttnn.model_converter.from_torch_model(
        model_name=tt_model_name,
        model=lambda: transformers.BertForQuestionAnswering.from_pretrained(model_name, torchscript=False).eval(),
        converter=functional_bert.converter,
        device=device,
    )
    profiler.end(f"preprocessing_parameter")

    context, question = load_inputs(input_path, batch_size)

    preprocess_params, _, postprocess_params = nlp._sanitize_parameters()
    preprocess_params["max_seq_len"] = sequence_size
    inputs = nlp._args_parser({"context": context, "question": question})
    preprocessed_inputs = []
    for i in range(batch_size):
        model_input = next(nlp.preprocess(inputs[0][i], **preprocess_params))
        single_input = {
            "example": model_input["example"],
            "inputs": model_input,
        }
        preprocessed_inputs.append(single_input)

    bert_input = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=sequence_size,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )
    profiler.start(f"preprocessing_input")
    ttnn_bert_inputs = functional_bert.preprocess_inputs(
        bert_input["input_ids"],
        bert_input["token_type_ids"],
        torch.zeros(1, sequence_size) if functional_bert == ttnn_optimized_functional_bert else None,
        device=device,
    )
    profiler.end(f"preprocessing_input")

    profiler.start(f"inference_time")
    tt_output = functional_bert.bert_for_question_answering(
        config,
        *ttnn_bert_inputs,
        parameters=parameters,
    )
    profiler.end(f"inference_time")

    tt_output = ttnn.to_torch(ttnn.from_device(tt_output)).reshape(batch_size, 1, sequence_size, -1).to(torch.float32)

    tt_start_logits = tt_output[..., :, 0].squeeze(1)
    tt_end_logits = tt_output[..., :, 1].squeeze(1)

    model_answers = {}
    profiler.start("post_processing_output_to_string")
    for i in range(batch_size):
        tt_res = {
            "start": tt_start_logits[i],
            "end": tt_end_logits[i],
            "example": preprocessed_inputs[i]["example"],
            **preprocessed_inputs[i]["inputs"],
        }

        tt_answer = nlp.postprocess([tt_res], **postprocess_params)

        logger.info(f"answer: {tt_answer['answer']}\n")
        model_answers[i] = tt_answer["answer"]

    profiler.end("post_processing_output_to_string")

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "preprocessing_input": profiler.get("preprocessing_input"),
        "inference_time": profiler.get("inference_time"),
        "post_processing": profiler.get("post_processing_output_to_string"),
    }
    logger.info(f"preprocessing_parameter: {measurements['preprocessing_parameter']} s")
    logger.info(f"preprocessing_input: {measurements['preprocessing_input']} s")
    logger.info(f"inference_time: {measurements['inference_time']} s")
    logger.info(f"post_processing : {measurements['post_processing']} s")

    return measurements


def run_bert_question_and_answering_inference_squad_v2(
    device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    functional_bert,
    model_location_generator,
    n_iterations,
):
    disable_persistent_kernel_cache()

    model = str(model_location_generator(model_name, model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model, torchscript=False)
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer_name = str(model_location_generator(model_name, model_subdir="Bert"))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    config = hugging_face_reference_model.config

    if functional_bert == ttnn_functional_bert:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_bert == ttnn_optimized_functional_bert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_bert: {functional_bert}")

    parameters = ttnn.model_converter.from_torch_model(
        model_name=tt_model_name,
        model=lambda: transformers.BertForQuestionAnswering.from_pretrained(model_name, torchscript=False).eval(),
        converter=functional_bert.converter,
        device=device,
    )

    nlp = pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)

    attention_mask = True
    token_type_ids = True
    inputs_squadv2 = squadv2_1K_samples_input(tokenizer, sequence_size, attention_mask, token_type_ids, batch_size)
    squad_metric = evaluate.load("squad_v2")

    with torch.no_grad():
        pred_labels = []
        cpu_pred_labels = []
        true_labels = []
        i = 0
        for batch in inputs_squadv2:
            if i < n_iterations:
                batch_data = batch[0]
                curr_batch_size = batch_data["input_ids"].shape[0]
                ttnn_bert_inputs = functional_bert.preprocess_inputs(
                    batch_data["input_ids"],
                    batch_data["token_type_ids"],
                    torch.zeros(1, sequence_size) if functional_bert == ttnn_optimized_functional_bert else None,
                    device=device,
                )

                tt_output = functional_bert.bert_for_question_answering(
                    config,
                    *ttnn_bert_inputs,
                    parameters=parameters,
                )
                tt_output = (
                    ttnn.to_torch(ttnn.from_device(tt_output))
                    .reshape(batch_size, 1, sequence_size, -1)
                    .to(torch.float32)
                )
                cpu_output = hugging_face_reference_model(**batch_data)
                references = batch[1]
                question = batch[2]
                context = batch[3]

                cpu_predictions, tt_predictions = squadv2_answer_decode_batch(
                    hugging_face_reference_model,
                    tokenizer,
                    nlp,
                    references,
                    cpu_output,
                    tt_output,
                    curr_batch_size,
                    question,
                    context,
                )
                pred_labels.extend(tt_predictions)
                cpu_pred_labels.extend(cpu_predictions)
                true_labels.extend(references)

                del tt_output
            i += 1
        eval_score = squad_metric.compute(predictions=pred_labels, references=true_labels)
        cpu_eval_score = squad_metric.compute(predictions=cpu_pred_labels, references=true_labels)
        logger.info(f"\tTT_Eval: exact: {eval_score['exact']} --  F1: {eval_score['f1']}")
        logger.info(f"\tCPU_Eval: exact: {cpu_eval_score['exact']} -- F1:  {cpu_eval_score['f1']}")


@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("functional_bert", [ttnn_functional_bert, ttnn_optimized_functional_bert])
def test_demo(
    input_path,
    model_name,
    functional_bert,
    model_location_generator,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(f"tt_metal/tools/profiler/logs/functional_bert")
    return run_bert_question_and_answering_inference(
        device=device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        functional_bert=functional_bert,
        model_location_generator=model_location_generator,
        input_path=input_path,
    )


@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("functional_bert", [ttnn_functional_bert, ttnn_optimized_functional_bert])
@pytest.mark.parametrize(
    "n_iterations",
    ((3),),
)
def test_demo_squadv2(
    model_name,
    functional_bert,
    n_iterations,
    model_location_generator,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_bert_question_and_answering_inference_squad_v2(
        device=device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        functional_bert=functional_bert,
        model_location_generator=model_location_generator,
        n_iterations=n_iterations,
    )
