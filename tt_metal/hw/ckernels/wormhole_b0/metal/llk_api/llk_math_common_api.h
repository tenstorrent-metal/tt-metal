// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_math_common.h"
#include "llk_operands.h"
#include "llk_param_structs.h"
#include "debug/status.h"

// Need to revisit why we even need this
#define EPS 1.19209e-07  // std::numeric_limits::epsilon() for FP32

/*************************************************************************
 * LLK MATH COMMON
 *************************************************************************/

template <DstSync Dst>
inline void llk_math_wait_for_dest_available() {
    DEBUG_STATUS('M', 'W', 'D', 'W');
    _llk_math_wait_for_dest_available_<Dst>();
    DEBUG_STATUS('M', 'W', 'D', 'D');
}

template <DstSync Dst = SyncFull, bool is_fp32_dest_acc_en = false>
inline void llk_math_dest_section_done() {
    _llk_math_dest_section_done_<Dst, is_fp32_dest_acc_en>();
}

template <DstSync Dst, bool is_fp32_dest_acc_en = false>
inline void llk_math_pack_sync_init() {
    _llk_math_pack_sync_init_<Dst, is_fp32_dest_acc_en>();
}

template <bool mail2math = true, bool mail2pack = true>
inline void llk_math_get_tile(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t *p_tile) {
    _llk_math_get_tile_<mail2math, mail2pack>(tile_index, p_tile);
}

template <bool mail2math = true, bool mail2pack = true>
inline void llk_math_release_tile(std::uint32_t operand) {
    _llk_math_release_tile_<mail2math, mail2pack>();
}

inline void llk_math_debug_dump(std::uint8_t *data, std::uint32_t byte_size) { _llk_math_debug_dump_(data, byte_size); }

inline void llk_math_debug_dump_seek(std::uint8_t offset) { _llk_math_debug_dump_seek_(offset); }

inline void llk_math_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);
    _llk_math_reconfig_data_format_srca_(unpack_dst_format[new_srca_operand_id]);
}

inline void llk_math_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);
    _llk_math_reconfig_data_format_srcb_(unpack_dst_format[new_srcb_operand_id]);
}

inline void llk_math_reconfig_data_format(const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    _llk_math_reconfig_data_format_(unpack_dst_format[new_srca_operand_id], unpack_dst_format[new_srcb_operand_id]);
}

inline void llk_math_reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if ((unpack_dst_format[old_srca_operand_id] != unpack_dst_format[new_srca_operand_id]) &&
        (unpack_dst_format[old_srcb_operand_id] != unpack_dst_format[new_srcb_operand_id])) {
        llk_math_reconfig_data_format(srca_new_operand, srcb_new_operand);
    } else if ((unpack_dst_format[old_srca_operand_id] != unpack_dst_format[new_srca_operand_id])) {
        llk_math_reconfig_data_format_srca(srca_new_operand);
    } else if ((unpack_dst_format[old_srcb_operand_id] != unpack_dst_format[new_srcb_operand_id])) {
        llk_math_reconfig_data_format_srcb(srcb_new_operand);
    }
}

inline void llk_math_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);

    if ((unpack_dst_format[old_srca_operand_id] != unpack_dst_format[new_srca_operand_id])) {
        llk_math_reconfig_data_format_srca(srca_new_operand);
    }
}

inline void llk_math_reconfig_data_format_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if ((unpack_dst_format[old_srcb_operand_id] != unpack_dst_format[new_srcb_operand_id])) {
        llk_math_reconfig_data_format_srcb(srcb_new_operand);
    }
}
