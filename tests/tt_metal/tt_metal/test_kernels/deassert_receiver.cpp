#include "dataflow_api.h"
#include "debug_print.h"

void kernel_main() {
    DPRINT << 'Q' << ENDL(); // Should see this on the other side

    // Write semaphore
    u64 remote_notify_address = get_noc_addr(1, 11, 600 * 1024);

    u32 local_addr = 900 * 1024;
    *reinterpret_cast<volatile u32*>(local_addr) = 1;

    noc_semaphore_set_remote(local_addr, remote_notify_address);
}
