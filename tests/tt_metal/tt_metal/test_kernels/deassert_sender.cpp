#include "dataflow_api.h"
#include "debug_print.h"

void kernel_main() {
    // Prepare the deassert reset flag at address 500KB
    noc_prepare_deassert_reset_flag(500 * 1024);
    u32 wait_address = 600 * 1024;

    *reinterpret_cast<volatile u32*>(wait_address) = 0;

    u64 dst_noc_addr = get_noc_addr(1, 1, TENSIX_SOFT_RESET_ADDR);
    // Write to remote core
    noc_semaphore_set_remote(500 * 1024, dst_noc_addr); // Deassert worker core
    while (*reinterpret_cast<volatile u32*>(wait_address) != 1);

    noc_prepare_assert_reset_flag(500 * 1024);
    noc_semaphore_set_remote(500 * 1024, dst_noc_addr); // Assert reset of worker core
    noc_async_write_barrier();
}
