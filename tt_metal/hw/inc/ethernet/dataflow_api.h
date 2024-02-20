// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_msgs.h"
#include "eth_l1_address_map.h"
#include "noc_nonblocking_api.h"
#include "risc_common.h"
#include "tt_eth_api.h"

#include "../dataflow_api.h"

#define FORCE_INLINE inline __attribute__((always_inline))

inline void RISC_POST_STATUS(uint32_t status) {
    volatile uint32_t *ptr = (volatile uint32_t *)(NOC_CFG(ROUTER_CFG_2));
    ptr[0] = status;
}

struct eth_word_t {
    volatile uint32_t bytes_sent;
    uint32_t reserved_0;
    uint32_t reserved_1;
    uint32_t reserved_2;
};

struct erisc_info_t {
    volatile uint32_t launch_user_kernel;
    volatile uint32_t unused_arg0;
    volatile uint32_t unused_arg1;
    volatile uint32_t unused_arg2;
    eth_word_t per_channel_user_bytes_send[eth_l1_mem::address_map::MAX_NUM_CHANNELS];
    volatile uint32_t fast_dispatch_buffer_msgs_sent;
    uint32_t reserved_3_;
    uint32_t reserved_4_;
    uint32_t reserved_5_;
};

// Routing info
uint32_t relay_src_noc_encoding;
uint32_t relay_dst_noc_encoding;
uint32_t eth_router_noc_encoding;
EthRouterMode my_routing_mode;

tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);

erisc_info_t *erisc_info = (erisc_info_t *)(eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
routing_info_t *routing_info = (routing_info_t *)(eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_BASE);
volatile uint32_t *flag_disable = (uint32_t *)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);

extern uint32_t __erisc_jump_table;
volatile uint32_t *RtosTable =
    (volatile uint32_t *)&__erisc_jump_table;  // Rtos Jump Table. Runtime application needs rtos function handles.;

void (*rtos_context_switch_ptr)();

FORCE_INLINE
void reset_erisc_info() {
    #pragma GCC unroll eth_l1_mem::address_map::MAX_NUM_CHANNELS
    for (uint32_t i = 0; i < eth_l1_mem::address_map::MAX_NUM_CHANNELS; ++i) {
        erisc_info->per_channel_user_bytes_send[i].bytes_sent = 0;
    }
}

namespace internal_ {
FORCE_INLINE
void __attribute__((section("code_l1"))) risc_context_switch() {
    ncrisc_noc_full_sync();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init();
}

FORCE_INLINE
void eth_send_packet(uint32_t q_num, uint32_t src_word_addr, uint32_t dest_word_addr, uint32_t num_words) {
    while (eth_txq_reg_read(q_num, ETH_TXQ_CMD) != 0) {
        risc_context_switch();
    }
    eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_START_ADDR, src_word_addr << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_DEST_ADDR, dest_word_addr << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_SIZE_BYTES, num_words << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_CMD, ETH_TXQ_CMD_START_DATA);
}

FORCE_INLINE
void eth_write_remote_reg(uint32_t q_num, uint32_t reg_addr, uint32_t val) {
    while (eth_txq_reg_read(q_num, ETH_TXQ_CMD) != 0) {
        risc_context_switch();
    }
    eth_txq_reg_write(q_num, ETH_TXQ_DEST_ADDR, reg_addr);
    eth_txq_reg_write(q_num, ETH_TXQ_REMOTE_REG_DATA, val);
    eth_txq_reg_write(q_num, ETH_TXQ_CMD, ETH_TXQ_CMD_START_REG);
}

void check_and_context_switch() {
    uint32_t start_time = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t end_time = start_time;
    while (end_time - start_time < 100000) {
        RISC_POST_STATUS(0xdeadCAFE);
        internal_::risc_context_switch();
        RISC_POST_STATUS(0xdeadFEAD);
        end_time = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    }
    // proceed
}

FORCE_INLINE
void notify_dispatch_core_done(uint64_t dispatch_addr) {
    //  flush both nocs because ethernet kernels could be using different nocs to try to atomically increment semaphore
    //  in dispatch core
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        while (!noc_cmd_buf_ready(n, NCRISC_AT_CMD_BUF))
            ;
    }
    noc_fast_atomic_increment_l1(noc_index, NCRISC_AT_CMD_BUF, dispatch_addr, 1, 31 /*wrap*/, false /*linked*/);
}


FORCE_INLINE
void disable_erisc_app() { flag_disable[0] = 0; }

FORCE_INLINE
void send_fd_packets() {
    internal_::eth_send_packet(
        0,
        (eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE) >> 4,
        ((eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE)) >> 4,
        (eth_l1_mem::address_map::ERISC_APP_RESERVED_SIZE) >> 4);
    routing_info->fd_buffer_msgs_sent = 1;
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(routing_info->fd_buffer_msgs_sent))) >> 4,
        ((uint32_t)(&(routing_info->fd_buffer_msgs_sent))) >> 4,
        1);
    // There should always be a valid cmd here, since eth_db_acquire completed
    while (routing_info->fd_buffer_msgs_sent != 0) {
        // TODO: add timer to restrict this
        risc_context_switch();
    }
}

FORCE_INLINE
void wait_for_fd_packet() {
    // There may not be a valid cmd here, since DST router is always polling
    // This should only happen on cluster close
    while (routing_info->fd_buffer_msgs_sent != 1 && routing_info->routing_enabled) {
        // TODO: add timer to restrict this
        risc_context_switch();
    }
}

FORCE_INLINE
void ack_fd_packet() {
    routing_info->fd_buffer_msgs_sent = 0;
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(routing_info->fd_buffer_msgs_sent))) >> 4,
        ((uint32_t)(&(routing_info->fd_buffer_msgs_sent))) >> 4,
        1);
}

}  // namespace internal_

void run_routing() {
    // TODO: maybe split into two FWs? or this may be better to sometimes allow each eth core to do both send and
    // receive of fd packets
    if (my_routing_mode == EthRouterMode::FD_SRC) {
        // TODO: port changes from erisc to here
        internal_::risc_context_switch();
    } else if (my_routing_mode == EthRouterMode::FD_DST) {
        // TODO: port changes from erisc to here
        internal_::risc_context_switch();
    } else if (my_routing_mode == EthRouterMode::SD) {
        // slow dispatch mode
        internal_::risc_context_switch();
    } else {
        internal_::risc_context_switch();
    }
}
/**
 * A blocking call that waits until the value of a local L1 memory address on
 * the Tensix core executing this function becomes equal to a target value.
 * This L1 memory address is used as a semaphore of size 4 Bytes, as a
 * synchronization mechanism. Also, see *noc_semaphore_set*.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range        | Required |
 * |-----------|----------------------------------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True     |
 * | val       | The target value of the semaphore                              | uint32_t | Any uint32_t value | True     |
 */
FORCE_INLINE
void eth_noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    while ((*sem_addr) != val) {
        run_routing();
    }
}

/**
 * A blocking call that waits until the value of a local L1 memory address on
 * the Tensix core executing this function becomes equal to or greater than a target value.
 * This L1 memory address is used as a semaphore of size 4 Bytes, as a
 * synchronization mechanism. Also, see *noc_semaphore_set*.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range        | Required |
 * |-----------|----------------------------------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True     |
 * | val       | The target value of the semaphore                              | uint32_t | Any uint32_t value | True     |
 */
FORCE_INLINE
void eth_noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    while ((*sem_addr) < val) {
        run_routing();
    }
}
/**
 * This blocking call waits for all the outstanding enqueued *noc_async_read*
 * calls issued on the current Tensix core to complete. After returning from
 * this call the *noc_async_read* queue will be empty for the current Tensix
 * core.
 *
 * Return value: None
 */
FORCE_INLINE
void eth_noc_async_read_barrier() {
    while (!ncrisc_noc_reads_flushed(noc_index)) {
        run_routing();
    }
}

/**
 * This blocking call waits for all the outstanding enqueued *noc_async_write*
 * calls issued on the current Tensix core to complete. After returning from
 * this call the *noc_async_write* queue will be empty for the current Tensix
 * core.
 *
 * Return value: None
 */
FORCE_INLINE
void eth_noc_async_write_barrier() {
    while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
        run_routing();
    }
}

/**
 * Initiates an asynchronous write from a source address in L1 memory on the local ethernet core to L1 of the connected
 * remote ethernet core. Also, see \a eth_wait_for_receiver_done and \a eth_wait_for_bytes.
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|-------------|----------|
 * | src_addr          | Source address in local eth core L1 memory              | uint32_t | 0..256kB | True     |
 * | dst_addr          | Destination address in remote eth core L1 memory        | uint32_t | 0..256kB | True     |
 * | num_bytes         | Size of data transfer in bytes, must be multiple of 16  | uint32_t | 0..256kB | True     |
 */
FORCE_INLINE
void eth_send_bytes(
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t num_bytes,
    uint32_t num_bytes_per_send = 16,
    uint32_t num_bytes_per_send_word_size = 1,
    uint8_t channel = 0) {
    uint32_t num_bytes_sent = 0;
    while (num_bytes_sent < num_bytes) {
        internal_::eth_send_packet(
            0, ((num_bytes_sent + src_addr) >> 4), ((num_bytes_sent + dst_addr) >> 4), num_bytes_per_send_word_size);
        num_bytes_sent += num_bytes_per_send;
    }
    erisc_info->per_channel_user_bytes_send[channel].bytes_sent += num_bytes;
}

/**
 * A blocking call that waits for receiver to acknowledge that all data sent with eth_send_bytes since the last
 * reset_erisc_info call is no longer being used. Also, see \a eth_receiver_done().
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|-------------|----------|
 */
FORCE_INLINE
void eth_wait_for_receiver_done(uint8_t channel = 0) {
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(erisc_info->per_channel_user_bytes_send[channel].bytes_sent))) >> 4,
        ((uint32_t)(&(erisc_info->per_channel_user_bytes_send[channel].bytes_sent))) >> 4,
        1);
    while (erisc_info->per_channel_user_bytes_send[channel].bytes_sent != 0) {
        run_routing();
    }
}

FORCE_INLINE
void eth_send_done(uint8_t channel = 0) {
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(erisc_info->per_channel_user_bytes_send[channel].bytes_sent))) >> 4,
        ((uint32_t)(&(erisc_info->per_channel_user_bytes_send[channel].bytes_sent))) >> 4,
        1);
}

FORCE_INLINE
void eth_wait_receiver_done(uint8_t channel = 0) {
    while (erisc_info->per_channel_user_bytes_send[channel].bytes_sent != 0) {
        run_routing();
    }
}

/**
 * A blocking call that waits for num_bytes of data to be sent from the remote sender ethernet core using any number of
 * eth_send_byte. User must ensure that num_bytes is equal to the total number of bytes sent. Example 1:
 * eth_send_bytes(32), eth_wait_for_bytes(32). Example 2: eth_send_bytes(16), eth_send_bytes(32),
 * eth_wait_for_bytes(48).
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|-------------|----------|
 * | num_bytes         | Size of data transfer in bytes, must be multiple of 16  | uint32_t | 0..256kB | True     |
 */
FORCE_INLINE
void eth_wait_for_bytes(uint32_t num_bytes, uint8_t channel = 0) {
    while (erisc_info->per_channel_user_bytes_send[channel].bytes_sent != num_bytes) {
        run_routing();
    }
}

/**
 * Initiates an asynchronous call from receiver ethernet core to tell remote sender ethernet core that data sent
 * via eth_send_bytes is no longer being used. Also, see \a eth_wait_for_receiver_done
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|-------------|----------|
 */
FORCE_INLINE
void eth_receiver_done(uint8_t channel = 0) {
    erisc_info->per_channel_user_bytes_send[channel].bytes_sent = 0;
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(erisc_info->per_channel_user_bytes_send[channel].bytes_sent))) >> 4,
        ((uint32_t)(&(erisc_info->per_channel_user_bytes_send[channel].bytes_sent))) >> 4,
        1);
}
