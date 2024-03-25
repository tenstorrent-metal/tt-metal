# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

#!/bin/bash

for ((i=1; i<=100; i++))
do
    echo "============ Running command for iteration $i ============"
    if ! TT_METAL_LOGGER_TYPES=Op TT_METAL_LOGGER_LEVEL=DEBUG timeout 200s pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b/demo/input_data3.json' models/demos/falcon7b/demo/demo.py; then
        echo "Command hung at iteration $i"
        break
    fi
done
