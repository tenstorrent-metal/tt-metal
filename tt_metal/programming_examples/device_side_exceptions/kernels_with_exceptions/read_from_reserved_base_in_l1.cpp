#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // This address is part of the firmware, no reason we should be reading from here.
    // Only doing this to show how we hit an assert on the runtime address monitor.
    constexpr uint32_t bad_address = 0;
    uint64_t src_noc_addr = get_noc_addr(1, 1, bad_address);
}
