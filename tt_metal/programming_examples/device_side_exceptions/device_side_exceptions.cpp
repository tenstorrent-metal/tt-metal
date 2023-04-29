#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

// We need the debug print server in order to see whether we hit any bad addresses in our kernel
#include "tt_metal/llrt/tt_debug_print_server.hpp"

using namespace tt::tt_metal;

int main(int argc, char **argv) {
    bool pass = true;

    std::cout << "Running " << argv[1] << std::endl;

    string kernel_name = argv[1];
    try {
        constexpr int pci_express_slot = 0;
        Device *device =
            CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= InitializeDevice(device);

        Program *program = new Program();

        constexpr tt_xy_pair core = {0, 0};
        int chip_id = 0;

        DataMovementKernel *bad_kernel = CreateDataMovementKernel(
            program,
            "tt_metal/programming_examples/device_side_exceptions/kernels_with_exceptions/" + kernel_name + ".cpp",
            core,
            DataMovementProcessor::RISCV_0,
            NOC::RISCV_0_default
        );
        // Enable the runtime address monitor
        bad_kernel->add_define("CHECK_VALID_ADDR", "1");

        pass &= CompileProgram(device, program);

        pass &= ConfigureDeviceWithProgram(device, program);

        // Need to start the server in order for us to monitor for a bad address
        tt_xy_pair debug_core = {1, 1};
        tt_start_debug_print_server(device->cluster(), {chip_id}, {debug_core});

        pass &= LaunchKernels(device, program);
        pass &= CloseDevice(device);

    } catch (const std::runtime_error& e) {
    }


    return 0;
}
