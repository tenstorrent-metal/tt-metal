from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib
import pytest

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.codegen.tt.codegen_model as codegen_model
from transformers import CodeGenConfig, CodeGenModel


from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

def run_codegen_model_test(device2, pcc):

    model_hf = CodeGenModel.from_pretrained('Salesforce/codegen-350M-mono')
    sd = model_hf.state_dict()
    model_hf.eval()
    block = 0
    base_address = f"h.{block}"

    torch.manual_seed(0)

    test_in = torch.rand(1,1,1024)

    tt_test_in = torch2tt_tensor(test_in, device2, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)


    config = CodeGenConfig('Salesforce/codegen-350M-mono')


    #pt_attn = model_hf.h[block].attn
    #pt_out = pt_attn.forward(test_in)

    tt_attn = codegen_attention.TtCodeGenModel(base_address, config, sd, device2)

    tt_out = tt_attn.forward(
        device2,
        tt_test_in
    )


    tt_out_converted = tt2torch_tensor(tt_out[0])

    """
    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("codegen_attention: Passed!")
    else:
        logger.warning("codegen_attention: Failed!")

    assert does_pass
    """
    print('DONE')
@pytest.mark.parametrize(
    "pcc",
    (
        (
            0.99,
        ),
    ),
)
def test_codegen_model(pcc):
    device2 = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device2)
    tt_lib.device.SetDefaultDevice(device2)
    run_codegen_model_test(device2, pcc)
    tt_lib.device.CloseDevice(device2)
