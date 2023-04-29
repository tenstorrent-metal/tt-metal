#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Here we are trying to read from an invalid NOC coordinate
    uint64_t src_noc_addr = get_noc_addr(0, 0, 1024 * 500);
}
