#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
  echo "Only Slow Dispatch mode allowed - Must have TT_METAL_SLOW_DISPATCH_MODE set" 1>&2
  exit 1
fi

cd $TT_METAL_HOME
export PYTHONPATH=$TT_METAL_HOME

./tests/scripts/run_python_api_unit_tests.sh

env python tests/scripts/run_tt_metal.py --dispatch-mode slow
env python tests/scripts/run_tt_eager.py --dispatch-mode slow
./build/test/tt_metal/unit_tests
./build/test/tt_metal/unit_tests --gtest_filter=ConcurrentFixture.* --gtest_repeat=10

echo "Checking docs build..."

cd $TT_METAL_HOME/docs
python -m pip install -r requirements-docs.txt
make clean
make html
