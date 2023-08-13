#include <stdint.h>
#include "dataflow_api.h"
#include "debug_print.h"


template <bool DRAM>
inline __attribute__((always_inline))
void read_and_push_to_cb(const uint32_t cb_id, uint32_t num_pages, uint32_t page_size,
                               const InterleavedAddrGenFast<DRAM> s) {
    // read a ublock of tiles at the time from DRAM to L1 buffer, and push a ublock at the time to unpacker
    DPRINT << "NUM TILES IN CB " << num_pages <<  ENDL();
    for (uint32_t i = 0; i<num_pages ; i++) {
        // DRAM NOC src address
        auto dram_buffer_src_noc_addr = get_noc_addr(i, s);
        cb_reserve_back(cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);

        noc_async_read(dram_buffer_src_noc_addr, l1_write_addr, page_size);
        noc_async_read_barrier();

        cb_push_back(cb_id, 1);


    }
}

void kernel_main() {
    std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_pages = get_compile_time_arg_val(4);

    #define in_is_dram get_compile_time_arg_val(0) == 1
    #define tile_dtype_is_bfloat16 get_compile_time_arg_val(1) == 1

    #if (tile_dtype_is_bfloat16)
        const InterleavedAddrGenFast<in_is_dram> s0 = {
            .bank_base_address = dram_buffer_src_addr, .page_size = page_size, .data_format = DataFormat::Float16};
    #else
        const InterleavedAddrGenFast<in_is_dram> s0 = {
            .bank_base_address = dram_buffer_src_addr, .page_size = page_size, .data_format = DataFormat::Bfp8_b};
    #endif

    read_and_push_to_cb<in_is_dram>(cb_id, num_pages, page_size,
                              s0);
}
