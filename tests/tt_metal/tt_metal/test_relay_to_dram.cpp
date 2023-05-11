#include "tt_metal/host_api.hpp"
#include "frameworks/tt_dispatch/impl/command_queue.hpp"
// #include "llrt/tt_debug_print_server.hpp"

using namespace tt;

bool test_enqueue_write_dram_buffer(Device* device) {

    bool pass = true;
    // Create a few buffers
    Buffer bufa(device, 2048, 0, 2048, BufferType::DRAM);
    // Buffer bufb(device, 0, 0, 0, BufferType::DRAM);
    // Buffer bufc(device, 0, 0, 0, BufferType::DRAM);

    CommandQueue cq(device);

    uint src[512];
    for (uint i = 0; i < 512; i++) {
        src[i] = i;
    }

    tt_start_debug_print_server(device->cluster(), {0}, {{1, 11}});
    EnqueueWriteBuffer(device, cq, bufa, (void*)(src), false);
    // EnqueueReadBuffer(device, cq, bufa, (void*)(src), false);
    Flush(cq);
    return pass;
}


int main(int argc, char **argv) {
    int pci_express_slot = 0;
    bool pass;
    tt_metal::Device *device =
        tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);


    pass = tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::SYSMEM);
    test_enqueue_write_dram_buffer(device);
}
