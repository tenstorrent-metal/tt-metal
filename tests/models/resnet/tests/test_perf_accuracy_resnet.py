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
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
import pytest
import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor, profiler
from models.utility_functions import disable_persistent_kernel_cache, enable_persistent_kernel_cache
from models.utility_functions import prep_report

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from loguru import logger
from tests.models.resnet.metalResnetBlock50 import ResNet, Bottleneck
from tests.models.dataset_imagenet import imagenet_1K_samples_input
import evaluate

BATCH_SIZE = 1


def run_perf_resnet(expected_inference_time, expected_compile_time, hf_cat_image_sample_input):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "accuracy_loop"
    cpu_key = "ref_key"
    model_name = "microsoft/resnet-50"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"]
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}"

    #ImageNet - 1000 samples
    sample_count = 1000
    input_data = imagenet_1K_samples_input(image_processor, "resnet", BATCH_SIZE,  sample_count)

    torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()

    tt_resnet50 = ResNet(Bottleneck, [3, 4, 6, 3],
                    device=device,
                    state_dict=state_dict,
                    base_address="",
                    fold_batchnorm=True,
                    storage_in_dram=False)


    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_resnet50(inputs)
        tt_lib.device.Synchronize()
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_resnet50(inputs)
        tt_lib.device.Synchronize()
        profiler.end(second_key)

        profiler.start(third_key)
        pred_labels = []
        cpu_pred_labels = []
        true_labels = []
        for batch, labels in input_data:
            tt_output = tt_resnet50(batch)
            cpu_output = torch_resnet50(batch)
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
        model_name="resnet50",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time
    )

    logger.info(f"resnet50 {comments} inference time: {second_iter_time}")
    logger.info(f"resnet50 compile time: {compile_time}")
    logger.info(f"resnet50 inference time avg. (Real Data loop  {sample_count}): {round(third_iter_time/sample_count, 2)}")

    assert second_iter_time < expected_inference_time, f"resnet50 {comments} is too slow"
    assert compile_time < expected_compile_time, "resnet50 compile time is too slow"


@pytest.mark.models_accuracy_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (0.225,
         34,
        ),
    ),
)

def test_perf_accuracy_bare_metal(use_program_cache, expected_inference_time, expected_compile_time, hf_cat_image_sample_input):
    run_perf_resnet(expected_inference_time, expected_compile_time, hf_cat_image_sample_input)


@pytest.mark.models_accuracy_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (0.3,
         36,
        ),
    ),
)

def test_perf_accuracy_virtual_machine(use_program_cache, expected_inference_time, expected_compile_time, hf_cat_image_sample_input):
    run_perf_resnet(expected_inference_time, expected_compile_time, hf_cat_image_sample_input)
