//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "debug/dprint.h"

void kernel_main() {
	constexpr uint32_t shard_cb = get_compile_time_arg_val(0);
	const uint32_t input_shard_addr  = get_arg_val<uint32_t>(0);
	const uint32_t num_output_pages = get_arg_val<uint32_t>(1);
	const uint32_t num_ranges = get_arg_val<uint32_t>(2);
	uint32_t arg_index = 3;

	cb_reserve_back(shard_cb, num_output_pages);
	uint32_t l1_write_addr = get_write_ptr(shard_cb);
	uint32_t mask_data = 0x0ff; //8 bits

	for(uint32_t range_id = 0; range_id <num_ranges; range_id++) {
		//uint32_t core_start_stride = get_arg_val<uint32_t>(arg_index++);
		//uint32_t start_x = (core_start_stride >> 24);
		//uint32_t start_y = (core_start_stride >> 16) & mask_data;
		//uint32_t stride_x = (core_start_stride >> 8) & mask_data;
		//uint32_t stride_y = (core_start_stride) & mask_data;
		uint32_t start_x = get_arg_val<uint32_t>(arg_index++);
		uint32_t start_y = get_arg_val<uint32_t>(arg_index++);
		uint32_t stride_x = get_arg_val<uint32_t>(arg_index++);
		uint32_t stride_y = get_arg_val<uint32_t>(arg_index++);
		uint32_t stride_data = get_arg_val<uint32_t>(arg_index++);
		uint32_t offset = get_arg_val<uint32_t>(arg_index++);
		uint32_t stride_size = get_arg_val<uint32_t>(arg_index++);
		uint32_t num_strides = get_arg_val<uint32_t>(arg_index++);

        DPRINT << "START_X " << start_x << ENDL();
        DPRINT << "START_Y " << start_y << ENDL();
        DPRINT << "STRIDE_X " << stride_x << ENDL();
        DPRINT << "STRIDE_Y " << stride_y << ENDL();
        DPRINT << "STRIDE_DATA " << stride_y << ENDL();
        DPRINT << "OFFSET " << stride_y << ENDL();
        DPRINT << "STRIDE_SIZE " << stride_y << ENDL();
        DPRINT << "NUM_STRIDES " << stride_y << ENDL();


		if((stride_data == stride_size) and (stride_x == 0) and (stride_y == 0)) {
			uint32_t size = num_strides * stride_data;
			uint64_t noc_address = get_noc_addr(start_x, start_y,
					input_shard_addr + offset);
			noc_async_read(noc_address, l1_write_addr, size);
			l1_write_addr+=size;
		}
		else {
			uint32_t addr_offset = offset;
			uint32_t core_id_x = start_x;
			uint32_t core_id_y = start_y;
			for(uint32_t stride_idx = 0; stride_idx < num_strides; stride_idx++) {
				uint64_t noc_address = get_noc_addr(core_id_x, core_id_y,
						input_shard_addr + addr_offset);
				noc_async_read(noc_address, l1_write_addr, stride_size);
				l1_write_addr+=stride_size;
				addr_offset += stride_size;
				core_id_x += stride_x;
				core_id_y += stride_y;
			}
		}


	}
	noc_async_read_barrier();
	cb_push_back(shard_cb, num_output_pages);

}
