# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import os
import sys
import time
import subprocess
import click
from glob import glob
from loguru import logger

from tt_metal.tools.profiler.profile_this import profile_command, post_process
from tt_metal.tools.profiler.common import (
    TT_METAL_HOME,
    PROFILER_OUTPUT_DIR,
    get_log_locations,
    test_profiler_build,
    clear_profiler_runtime_artifacts,
)


@click.command()
@click.option("-d", "--directory", required=True, type=str, help="Directory with test files")
@click.option("-r", "--result", default=PROFILER_OUTPUT_DIR, type=str, help="Directory to save results")
def main(directory, result):
    if test_profiler_build():
        logger.info(f"Profiler build flag is set")
    else:
        logger.error(f"Need to build with the profiler flag enabled. i.e. make build ENABLE_PROFILER=1")
        sys.exit(1)

    txt_files = glob(os.path.join(directory, "*.yaml"))
    txt_files.sort()
    do_run = True

    for txt_file in txt_files:
        basename = os.path.splitext(os.path.basename(txt_file))[0]
        outFolder = f"{result}/{basename}"
        command = (
            f"python tests/tt_eager/python_api_testing/sweep_tests/run_pytorch_test.py -i {txt_file} -o {outFolder}"
        )
        profile_output_folder = f"{outFolder}/profile"

        if do_run:
            print(command)

            clear_profiler_runtime_artifacts()

            start = time.time()
            profile_command(command)
            duration = time.time() - start

            post_process(f"{outFolder}/profile")

            with open(f"{outFolder}/total_time.txt", "w") as file:
                file.write(f"{duration:.2f}")


if __name__ == "__main__":
    main()
