#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // We are reading out of bounds (can only read up to 1MB for grayskull)
    volatile uint32_t bad_address = 1024 * 1024;
    uint64_t src_noc_addr = get_noc_addr(1, 1, bad_address);
}
