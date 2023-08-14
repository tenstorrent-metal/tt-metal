#include "dataflow_api.h"

#include "debug_print.h"
#define MAX_SEQ_LENGTH 512

template <bool DRAM>
inline __attribute__((always_inline))
void read_input_rows(const uint32_t num_rows,
                    bool * embeddings_boolean_mask,
                    uint32_t dst_l1_addr,
                    const InterleavedAddrGenFast<DRAM>& s

) {


    for (uint32_t i = 0; i < num_rows; i++) {
        auto noc_addr = get_noc_addr(i, s);
        noc_async_read(noc_addr, dst_l1_addr, sizeof(uint32_t));
        noc_async_read_barrier();
        uint32_t row = ((uint32_t *)dst_l1_addr)[0];
        embeddings_boolean_mask[row] = true;
    }

}



template <bool weights_DRAM, bool out_DRAM>
inline __attribute__((always_inline))
void write_output(
                    const uint32_t num_input_rows,
                    const uint32_t page_size,
                    bool * embeddings_boolean_mask,
                    uint32_t l1_addr,
                    const InterleavedAddrGenFast<weights_DRAM>& weights,
                    const InterleavedAddrGenFast<out_DRAM>& out

) {


    uint32_t output_index=0;
    for (uint32_t i = 0; i < num_input_rows; i++) {
        if(embeddings_boolean_mask[i]){
            DPRINT << "SELECTING ROW " << i <<  ENDL();
            auto noc_src_addr = get_noc_addr(i, weights);
            auto noc_dst_addr = get_noc_addr(output_index, out);
            noc_async_read(noc_src_addr, l1_addr, page_size);
            noc_async_read_barrier();
            noc_async_write(l1_addr, noc_dst_addr, page_size);
            noc_async_write_barrier();
            output_index++;
        }
    }

}

void kernel_main() {
    std::uint32_t input_dram_buffer_src_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t weights_dram_buffer_src_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t dst_l1_input_addr      = get_arg_val<uint32_t>(2);
    std::uint32_t dst_l1_weights_addr      = get_arg_val<uint32_t>(3);
    std::uint32_t output_dram_buffer_dst_addr      = get_arg_val<uint32_t>(4);


    #define in_is_dram get_compile_time_arg_val(0) == 1
    #define weights_is_dram get_compile_time_arg_val(1) == 1
    #define out_is_dram get_compile_time_arg_val(2) == 1
    #define tile_dtype_is_bfloat16 get_compile_time_arg_val(3) == 1
    constexpr uint32_t page_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_output_rows      = get_compile_time_arg_val(5);
    constexpr uint32_t num_input_rows      = get_compile_time_arg_val(6);

    //DPRINT << "NUM ROWS ARE " << num_output_rows <<  ENDL();

    //initialize no rows to be selected
    bool embeddings_boolean_mask[MAX_SEQ_LENGTH] = {false};

    const InterleavedAddrGenFast<in_is_dram> s0 = {
        .bank_base_address = input_dram_buffer_src_addr, .page_size = sizeof(uint32_t), .data_format = DataFormat::UInt32};


    #if (tile_dtype_is_bfloat16)
        const InterleavedAddrGenFast<weights_is_dram> weights_0 = {
            .bank_base_address = weights_dram_buffer_src_addr , .page_size = page_size, .data_format = DataFormat::Float16};
        const InterleavedAddrGenFast<out_is_dram> out_0 = {
            .bank_base_address = output_dram_buffer_dst_addr, .page_size = page_size, .data_format = DataFormat::Float16};
    #else
        const InterleavedAddrGenFast<weights_is_dram> weights_0 = {
            .bank_base_address = weights_dram_buffer_src_addr , .page_size = page_size, .data_format = DataFormat::Bfp8_b};
        const InterleavedAddrGenFast<out_is_dram> out_0 = {
            .bank_base_address = output_dram_buffer_dst_addr , .page_size = page_size, .data_format = DataFormat::Bfp8_b};
    #endif

    read_input_rows<in_is_dram>(num_output_rows,
                    embeddings_boolean_mask,
                    dst_l1_input_addr,
                    s0
    );

    write_output<weights_is_dram, out_is_dram>(
                            num_input_rows,
                            page_size,
                            embeddings_boolean_mask,
                            dst_l1_weights_addr,
                            weights_0,
                            out_0
    );


    //DPRINT << "AFTER READING" <<  ENDL();

}
