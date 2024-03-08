// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);

    cb_wait_front(cb_id_out, num_units);

    volatile tt_l1_ptr uint16_t* rptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_id_out));

    DPRINT << "num_units " <<num_units << ENDL();

    for (uint32_t i=0; i<num_units*32;++i) {
        DPRINT << rptr[i] << ENDL();
    }
}
