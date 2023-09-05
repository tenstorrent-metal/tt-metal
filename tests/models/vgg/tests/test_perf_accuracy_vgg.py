from torchvision import models
import torch
from datasets import load_dataset
from loguru import logger
import pytest
import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.utility_functions import disable_persistent_kernel_cache, enable_persistent_kernel_cache
from models.utility_functions import prep_report, Profiler

from tests.models.vgg.tt.vgg import *
from tests.models.dataset_imagenet import imagenet_1K_samples_input
import evaluate

BATCH_SIZE = 1

def run_perf_vgg(imagenet_sample_input, expected_inference_time, expected_compile_time):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "accuracy_loop"
    cpu_key = "ref_key"
    comments = "16"

    image = imagenet_sample_input

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    tt_image = tt_lib.tensor.Tensor(
        image.reshape(-1).tolist(),
        get_shape(image.shape),
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )

    tt_vgg = vgg16(device, disable_conv_on_tt_device=True)

    torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch_vgg.eval()

    #ImageNet - default = 1000 samples
    sample_count = 100
    input_data = imagenet_1K_samples_input(None, "vgg", BATCH_SIZE, sample_count)

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_vgg(tt_image)
        tt_lib.device.Synchronize()
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_vgg(tt_image)
        tt_lib.device.Synchronize()
        profiler.end(second_key)

        profiler.start(third_key)
        pred_labels = []
        cpu_pred_labels = []
        true_labels = []
        for batch, labels in input_data:
            ## VGG model is not supporting images with one color channel
            if batch.shape[1] == 3:
                torch_output = torch_vgg(batch)
                batch_tt = tt_lib.tensor.Tensor(
                            batch.reshape(-1).tolist(),
                            get_shape(batch.shape),
                            tt_lib.tensor.DataType.BFLOAT16,
                            tt_lib.tensor.Layout.ROW_MAJOR)
                tt_output = tt_to_torch_tensor(tt_vgg(batch_tt))[0][0]
                pred_labels.extend(torch.argmax(tt_output, axis=-1))
                cpu_pred_labels.extend(torch.argmax(torch_output, axis=-1))
                true_labels.extend(labels[0])
                #logger.info(f"ImageNet inference TT | CPU | REF: {pred_labels[-1]} | {cpu_pred_labels[-1]} | {true_labels[-1]}")
            #else:
            #    logger.info(f"Skipped Image: {batch.shape}")
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
        model_name="VGG",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time
    )

    logger.info(f"vgg inference time: {second_iter_time}")
    logger.info(f"vgg compile time: {compile_time}")
    logger.info(f"vgg inference time avg. (Real Data loop {sample_count}): {round(third_iter_time/sample_count, 2)}")

    assert second_iter_time < expected_inference_time, f"vgg {comments} is too slow"
    assert compile_time < expected_compile_time, f"vgg {comments} compile time is too slow"


@pytest.mark.models_accuracy_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (5.2,
         14,
        ),
    ),
)
def test_perf_accuracy_bare_metal(use_program_cache, imagenet_sample_input, expected_inference_time, expected_compile_time):
    run_perf_vgg(imagenet_sample_input, expected_inference_time, expected_compile_time)


@pytest.mark.models_accuracy_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (4.5,
         15,
        ),
    ),
)
def test_perf_accuracy_virtual_machine(use_program_cache, imagenet_sample_input, expected_inference_time, expected_compile_time):
    run_perf_vgg(imagenet_sample_input, expected_inference_time, expected_compile_time)
