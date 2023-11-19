// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/bfloat16.hpp"
#include "test_tiles.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/llrt/tt_debug_print_server.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

using namespace tt;

void measure_latency(string kernel_name) {
    const int device_id = 0;
    const tt_metal::Device& device = tt_metal::CreateDevice(device_id);

    auto dispatch_cores = device.dispatch_cores().begin();
    CoreCoord producer_logical_core = *dispatch_cores++;
    CoreCoord consumer_logical_core = *dispatch_cores;

    auto first_worker_physical_core = device.worker_core_from_logical_core({0, 0});

    std::map<string, string> defines = {
        {"WORKER_NOC_X", std::to_string(first_worker_physical_core.x)},
        {"WORKER_NOC_Y", std::to_string(first_worker_physical_core.y)},
    };

    tt_metal::Program program = tt_metal::CreateProgram();
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/noc/kernels/" + kernel_name + ".cpp",
        consumer_logical_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt::tt_metal::detail::SetProfilerDir(kernel_name + "_microbenchmark");
    tt::tt_metal::detail::FreshProfilerDeviceLog();
    detail::CompileProgram(device, program);
    tt_metal::detail::LaunchProgram(device, program);
    tt_metal::CloseDevice(device);
}

int main(int argc, char **argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        TT_THROW("Test not supported w/ fast dispatch, exiting");
    }

    measure_latency("multicast_to_single_core");
    measure_latency("unicast_to_single_core");
}
