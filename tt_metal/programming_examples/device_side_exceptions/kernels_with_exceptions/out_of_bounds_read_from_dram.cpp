#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Trying to read past 1GB in DRAM, which is out of bounds
    constexpr uint32_t bad_address = 1024 * 1024 * 1024;
    uint64_t src_noc_addr = get_noc_addr(1, 0, bad_address);
}
