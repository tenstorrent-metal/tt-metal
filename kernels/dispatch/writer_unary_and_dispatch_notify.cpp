#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr      = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_x     = get_arg_val<uint32_t>(1);
    uint32_t dst_noc_y     = get_arg_val<uint32_t>(2);
    uint32_t num_tiles     = get_arg_val<uint32_t>(3);
    uint32_t dispatch_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t ublock_size_tiles = 1;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr);

        cb_wait_front(cb_id_out0, ublock_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

        noc_async_write_barrier();

        cb_pop_front(cb_id_out0, ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }

    // Notify dispatch kernel that work is done
    // TODO(agrebenisan): Make this generic
    constexpr uint32_t dispatch_x = 0;
    constexpr uint32_t dispatch_y = 11;
    noc_semaphore_inc(get_noc_addr(dispatch_x, dispatch_y, dispatch_addr), 1);
}
