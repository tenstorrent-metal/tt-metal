#include "frameworks/tt_dispatch/impl/command.hpp"
#include "tt_metal/common/base.hpp"

struct SystemMemoryCBWriteInterface {
    uint fifo_wr_ptr;
    uint fifo_limit;
    uint fifo_size;
};

class SystemMemoryWriter {
    Device* device;
    SystemMemoryCBWriteInterface cb_write_interface;

   public:
    SystemMemoryWriter(Device* device) { this->device = device; }

    // Ensure that there is enough space to push to the queue first
    void cb_reserve_back() { TT_THROW("cb_reserve_back not implemented yet"); }

    void noc_write(const DeviceCommand& command) {
        const array<uint, SIZE>& desc = command.get_desc();
        vector<uint32_t> command_vector(desc.begin(), desc.end());
        this->device->cluster()->write_sysmem_vec(command_vector, cb_write_interface.fifo_wr_ptr, 0);
    }

    void cb_push_back() {
        // Notify dispatch core
        this->cb_write_interface.fifo_wr_ptr += DeviceCommand::size() * sizeof(uint);

        if (this->cb_write_interface.fifo_wr_ptr > this->cb_write_interface.fifo_limit) {
            this->cb_write_interface.fifo_wr_ptr -= this->cb_write_interface.fifo_size;
        }
    }
};
