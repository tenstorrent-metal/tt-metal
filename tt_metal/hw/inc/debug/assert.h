// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "watcher_common.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT)

void assert_and_hang(uint32_t line_num) {
    // Write the line number into the memory mailbox for host to read.
    debug_assert_msg_t tt_l1_ptr *v = GET_MAILBOX_ADDRESS_DEV(assert_status);
    if (v->tripped == DebugAssertOK) {
        v->line_num = line_num;
        v->tripped = DebugAssertTripped;
        v->which = debug_get_which_riscv();
    }

    // Hang, or in the case of erisc, early exit.
#if defined(COMPILE_FOR_ERISC)
    internal_::disable_erisc_app();
    erisc_early_exit(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_STACK_SAVE);
#endif

    while(1) { ; }
}

// The do... while(0) in this macro allows for it to be called more flexibly, e.g. in an if-else
// without {}s.
#define ASSERT(condition) do{ if (not (condition)) assert_and_hang(__LINE__); } while(0)

#else // !WATCHER_ENABLED

#define ASSERT(condition)

#endif // WATCHER_ENABLED
