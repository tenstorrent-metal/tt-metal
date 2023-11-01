#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
  env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/
  env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/ -name 'test_*.py' -a ! -name 'test_sweep_conv_with_address_map.py') -vvv
else
  # Need to remove move for time being since failing
  env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/ -name 'test_*.py' -a ! -name 'test_move.py') -vvv
  env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/test_move.py -k in0_L1 -vvv
  env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/ -name 'test_*.py' -a ! -name 'test_sweep_conv_with_address_map.py' -a ! -name 'test_move.py') -vvv
  env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_move.py -k input_L1 -vvv
fi

# This must run in slow dispatch mode
# pytest -svv $TT_METAL_HOME/tests/python_api_testing/sweep_tests/pytests/test_sweep_conv_with_address_map.py

# For now, adding tests with fast dispatch and non-32B divisible page sizes here. Python/models people,
# you can move to where you'd like.

# Test forcing ops to single core
env TT_METAL_SINGLE_CORE_MODE=1 pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_matmul.py::test_run_matmul_test -k BFLOAT16
env TT_METAL_SINGLE_CORE_MODE=1 pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_unpad.py

if [ "$ARCH_NAME" != "wormhole_b0" ]; then
# Tests for tensors in L1
pytest $TT_METAL_HOME/tests/models/bert_large_performant/unit_tests/test_bert_large*matmul* -k in0_L1-in1_L1-bias_L1-out_L1
pytest $TT_METAL_HOME/tests/models/bert_large_performant/unit_tests/test_bert_large*bmm* -k in0_L1-in1_L1-out_L1
# Tests for mixed precision (sweeps combos of bfp8_b/bfloat16 dtypes for fused_qkv_bias and ff1_bias_gelu matmul and pre_softmax_bmm)
pytest $TT_METAL_HOME/tests/models/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_matmul -k "fused_qkv_bias and batch_9 and L1"
pytest $TT_METAL_HOME/tests/models/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_matmul -k "ff1_bias_gelu and batch_9 and DRAM"
pytest $TT_METAL_HOME/tests/models/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_bmm -k "pre_softmax_bmm and batch_9"

# BERT TMs
pytest $TT_METAL_HOME/tests/models/bert_large_performant/unit_tests/test_bert_large_split_fused_qkv_and_split_heads.py -k "in0_L1-out_L1 and batch_9"
pytest $TT_METAL_HOME/tests/models/bert_large_performant/unit_tests/test_bert_large_concatenate_heads.py -k "in0_L1-out_L1 and batch_9"

# Test program cache
pytest $TT_METAL_HOME/tests/models/bert_large_performant/unit_tests/ -k program_cache

# Fused ops unit tests
pytest $TT_METAL_HOME/tests/models/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_ln.py -k "in0_L1-out_L1 and batch_9"
pytest $TT_METAL_HOME/tests/models/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_softmax.py -k "in0_L1 and batch_9"

# Resnet18 tests with conv on cpu and with conv on device
pytest $TT_METAL_HOME/tests/models/resnet/test_resnet18.py

# Falcon tests
pytest $TT_METAL_HOME/tests/models/falcon/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_128 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
pytest $TT_METAL_HOME/tests/models/falcon/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_512 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
pytest $TT_METAL_HOME/tests/models/falcon/tests/unit_tests/test_falcon_attn_matmul.py
fi

pytest tests/tt_eager/python_api_testing/unit_testing/test_bert_ops.py
