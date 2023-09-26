#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

# This must run in slow dispatch mode
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/

# To ensure we don't accidentally re-introduce a WH hang on BERT
pytest ${TT_METAL_HOME}/tests/models/bert_large_perf/test_bert_batch_dram.py::test_bert_batch_dram[phiyodr/bert-large-finetuned-squad2-9-384-True-True-True-0.98]
