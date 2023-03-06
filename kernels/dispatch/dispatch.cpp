#include <stdint.h>
#include "dataflow_api.h"

inline void send_kernel_args_to_core(uint32_t local_kernel_args_addr, uint32_t core_x, uint32_t core_y) {
    uint32_t num_args;
    constexpr uint32_t num_dataflow_riscs = 2;

    num_args = *reinterpret_cast<volatile uint32_t*>(local_kernel_args_addr);
    noc_async_write(local_kernel_args_addr + sizeof(uint32_t), get_noc_addr(core_x, core_y, NCRISC_L1_ARG_BASE), num_args * sizeof(uint32_t));
    local_kernel_args_addr += num_args * sizeof(uint32_t);

    num_args = *reinterpret_cast<volatile uint32_t*>(local_kernel_args_addr);
    noc_async_write(local_kernel_args_addr + sizeof(uint32_t), get_noc_addr(core_x, core_y, BRISC_L1_ARG_BASE), num_args * sizeof(uint32_t));
}

// For time being, no multicasting
inline void send_kernels_to_core(uint32_t kernel_src_addr, uint32_t core_x, uint32_t core_y) {
    // Send NCRISC kernel
    noc_async_write(
        kernel_src_addr,
        get_noc_addr(core_x, core_y, l1_mem::address_map::NCRISC_L1_CODE_BASE),
        l1_mem::address_map::NCRISC_L1_CODE_SIZE);
    kernel_src_addr += l1_mem::address_map::NCRISC_L1_CODE_SIZE;


    // Send TRISC0 kernel
    noc_async_write(
        kernel_src_addr,
        get_noc_addr(core_x, core_y, l1_mem::address_map::TRISC0_BASE),
        l1_mem::address_map::TRISC0_SIZE);
    kernel_src_addr += l1_mem::address_map::TRISC0_SIZE;


    // Send TRISC1 kernel
    noc_async_write(
        kernel_src_addr,
        get_noc_addr(core_x, core_y, l1_mem::address_map::TRISC1_BASE),
        l1_mem::address_map::TRISC1_SIZE);
    kernel_src_addr += l1_mem::address_map::TRISC1_SIZE;


    // Send TRISC2 kernel
    noc_async_write(
        kernel_src_addr,
        get_noc_addr(core_x, core_y, l1_mem::address_map::TRISC2_BASE),
        l1_mem::address_map::TRISC2_SIZE);
    kernel_src_addr += l1_mem::address_map::TRISC2_SIZE;


    // Send BRISC kernel
    noc_async_write(
        kernel_src_addr,
        get_noc_addr(core_x, core_y, l1_mem::address_map::FIRMWARE_BASE),
        l1_mem::address_map::BRISC_FIRMWARE_SIZE);
}

/*
    The overall goal of this kernel is to eventually be a pseudo-RTOS that can
    schedule work on the chip
*/
void kernel_main() {
    uint32_t arg_idx = 0;
    uint32_t kernel_src_addr      = get_arg_val<uint32_t>(arg_idx++);
    uint32_t kernel_dst_addr      = get_arg_val<uint32_t>(arg_idx++);
    uint32_t kernel_args_src_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t kernel_args_dst_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t src_noc_x            = get_arg_val<uint32_t>(arg_idx++);
    uint32_t src_noc_y            = get_arg_val<uint32_t>(arg_idx++);
    uint32_t column               = get_arg_val<uint32_t>(arg_idx++);
    uint32_t dispatch_addr        = get_arg_val<uint32_t>(arg_idx++);

    bool cached = false; // for now, but this should just be read from L1

    constexpr uint32_t num_compute_cores_in_col = 11;
    constexpr uint32_t hexes_size =
        l1_mem::address_map::NCRISC_L1_CODE_SIZE +
        l1_mem::address_map::TRISC0_SIZE +
        l1_mem::address_map::TRISC1_SIZE +
        l1_mem::address_map::TRISC2_SIZE +
        l1_mem::address_map::BRISC_FIRMWARE_SIZE;

    // In one shot, read all of the kernels for the column
    if (not cached) {
        constexpr uint32_t total_hexes_size = hexes_size * num_compute_cores_in_col;
        uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, kernel_src_addr);

        uint32_t dst_addr = 0;
        noc_async_read(src_noc_addr, dst_addr, total_hexes_size);
        noc_async_read_barrier();
    }

    // This specifies how many args we're going to be sending in total
    uint32_t total_num_args_to_send = 0;
    for (uint32_t i = 0; i < num_compute_cores_in_col; i++) {
        total_num_args_to_send += get_arg_val<uint32_t>(arg_idx + i);
    }

    // Read kernel args from DRAM into L1. These kernel args are only used by
    // dataflow kernels
    uint64_t kernel_args_src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, kernel_args_src_addr);
    noc_async_read(kernel_args_src_noc_addr, kernel_args_dst_addr, total_num_args_to_send * sizeof(uint32_t));
    noc_async_read_barrier();

    // ... and then send them to remote core
    #pragma unroll(num_compute_cores_in_col)
    for (uint32_t i = 0; i < num_compute_cores_in_col; i++) {
        send_kernel_args_to_core(kernel_args_dst_addr, column, i);
    }
    noc_async_write_barrier();

    // Send kernels to cores
    kernel_src_addr = kernel_dst_addr;
    #pragma unroll(num_compute_cores_in_col)
    for (uint32_t i = 0; i < num_compute_cores_in_col; i++) {
        send_kernels_to_core(kernel_src_addr, column, i);
        kernel_src_addr += hexes_size;
    }
    noc_async_write_barrier();

    // Wait for compute to fully finish... we poll a flag afterwards to make sure all the receivers
    noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(dispatch_addr), num_compute_cores_in_col);
    noc_semaphore_set(reinterpret_cast<volatile uint32_t*>(dispatch_addr), 0);
}
