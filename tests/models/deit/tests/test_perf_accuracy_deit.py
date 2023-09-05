from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher
import torch
from loguru import logger
import pytest
import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor, profiler
from models.utility_functions import disable_persistent_kernel_cache, enable_persistent_kernel_cache
from models.utility_functions import prep_report
from tt.deit_for_image_classification_with_teacher import deit_for_image_classification_with_teacher
from tests.models.dataset_imagenet import imagenet_1K_samples_input
import evaluate

BATCH_SIZE = 1


def run_perf_deit(expected_inference_time, expected_compile_time, hf_cat_image_sample_input):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "accuracy_loop"
    cpu_key = "ref_key"
    comments = "distilled-patch16-wteacher"

    image = hf_cat_image_sample_input

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    HF_model = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")
    inputs = image_processor(image, return_tensors="pt")

    tt_inputs = torch_to_tt_tensor_rm(
        inputs["pixel_values"], device, put_on_device=False
    )
    tt_model = deit_for_image_classification_with_teacher(device)

    #ImageNet - default = 1000 samples
    sample_count = 300
    input_data = imagenet_1K_samples_input(image_processor, "vit", BATCH_SIZE, sample_count)

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = HF_model(**inputs).logits
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(tt_inputs)[0]
        tt_lib.device.Synchronize()
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(tt_inputs)[0]
        tt_lib.device.Synchronize()
        profiler.end(second_key)


        profiler.start(third_key)
        pred_labels = []
        cpu_pred_labels = []
        true_labels = []
        for batch, labels in input_data:
            cpu_output = HF_model(batch).logits
            tt_inputs = torch_to_tt_tensor_rm(batch, device, put_on_device=False)
            tt_output = tt_to_torch_tensor(tt_model(tt_inputs)[0])[0][0]
            pred_labels.extend(torch.argmax(tt_output, axis=-1))
            cpu_pred_labels.extend(torch.argmax(cpu_output, axis=-1))
            true_labels.extend(labels[0])
            #logger.info(f"ImageNet inference TT | CPU | REF: {pred_labels[-1]} | {cpu_pred_labels[-1]} | {true_labels[-1]}")
        accuracy_metric = evaluate.load("accuracy")
        eval_score = accuracy_metric.compute(references=true_labels, predictions=pred_labels)
        cpu_eval_score = accuracy_metric.compute(references=true_labels, predictions=cpu_pred_labels)
        cross_eval_score = accuracy_metric.compute(references=cpu_pred_labels, predictions=pred_labels)
        logger.info(f"\tTT_Eval: {eval_score}")
        logger.info(f"\tCPU_Eval: {cpu_eval_score}")
        logger.info(f"\tCross_Eval: {cross_eval_score}")
        tt_lib.device.Synchronize()
        profiler.end(third_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    tt_lib.device.CloseDevice(device)

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    prep_report(
        model_name="deit",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time
    )

    logger.info(f"deit {comments} inference time: {second_iter_time}")
    logger.info(f"deit {comments} compile time: {compile_time}")
    logger.info(f"deit inference time avg. (Real Data loop {sample_count}): {round(third_iter_time/sample_count, 2)}")

    assert second_iter_time < expected_inference_time, f"deit {comments} is too slow"
    assert compile_time < expected_compile_time, "deit compile time is too slow"


@pytest.mark.models_accuracy_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (2.1,
         19,
        ),
    ),
)
def test_perf_accuracy_bare_metal(use_program_cache, expected_inference_time, expected_compile_time, hf_cat_image_sample_input):
    run_perf_deit(expected_inference_time, expected_compile_time, hf_cat_image_sample_input)


@pytest.mark.models_accuracy_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (2.6,
         19.5,
        ),
    ),
)
def test_perf_accuracy_virtual_machine(use_program_cache, expected_inference_time, expected_compile_time, hf_cat_image_sample_input):
    run_perf_deit(expected_inference_time, expected_compile_time, hf_cat_image_sample_input)
