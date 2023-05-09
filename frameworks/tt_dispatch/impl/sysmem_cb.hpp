#include "frameworks/tt_dispatch/impl/command.hpp"

struct SystemMemoryCBWriteInterface {
    uint write_ptr;
};

class SystemMemoryWriter {
    Device* device;
    SystemMemoryCBWriteInterface cb_write_interface;

   public:
    SystemMemoryWriter(Device* device) { this->device = device; }

    // Ensure that there is enough space to push to the queue first
    void cb_reserve_back() {

    }

    void cb_push_back(const DeviceCommand& command) {
        const array<uint, SIZE>& desc = command.get_desc();
        vector<uint32_t> command_vector(desc.begin(), desc.end());
        this->device->cluster()->write_sysmem_vec(command_vector, cb_write_interface.write_ptr, 0);

        // Notify dispatch core
    }
};
