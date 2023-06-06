#include <algorithm>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/tt_gdb/tt_gdb.hpp"
#include "tt_metal/llrt/tt_debug_print_server.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

void write_simple_program_to_core(tt_metal::Device *device) {

    try {
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};

        auto test_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/deassert_test_kernel.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);

        tt_metal::CompileProgram(device, program);

        tt_metal::ConfigureDeviceWithProgram(device, program);

    } catch (const std::exception &e) {
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }
}

void setup_and_launch_deassert_core(tt_metal::Device* device) {
    tt_metal::Program program = tt_metal::Program();
    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/deassert.cpp",
        {9, 0},
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    // Using tt-metal compile program flow since simpler, but then using llrt after since I need to run
    // another program on device
    tt_metal::CompileProgram(device, program);

    llrt::test_load_write_read_risc_binary(device->cluster(), "deassert_blank/8609066131763170589/brisc/brisc.hex", 0, {1, 11}, 0);
    tt::llrt::internal_::setup_riscs_on_specified_cores(device->cluster(), 0, tt::llrt::TensixRiscsOptions::BRISC_ONLY, {{1, 11}, {1, 1}});
    tt::llrt::internal_::run_riscs_on_specified_cores(device->cluster(), 0, tt::llrt::TensixRiscsOptions::BRISC_ONLY, {{1, 11}});
}

int main(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    string arch_name = "";
    try {
        std::tie(arch_name, input_args) =
            test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
    } catch (const std::exception &e) {
        log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
    }
    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int pci_express_slot = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(arch, pci_express_slot);
    tt_metal::InitializeDevice(device);

    tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}, {1, 11}});

    // This just writes the binary and data, but doesn't launch
    // any of the kernels
    write_simple_program_to_core(device);

    setup_and_launch_deassert_core(device);

    tt_metal::CloseDevice(device);
}
