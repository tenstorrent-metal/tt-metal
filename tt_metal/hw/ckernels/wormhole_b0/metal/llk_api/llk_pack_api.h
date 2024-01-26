// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cpack_common.h"
#include "ckernel_globals.h"
#include "circular_buffer.h"

#include "llk_io.h"
#include "llk_defs.h"
#include "llk_outputs.h"
#include "llk_param_structs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_untilize.h"

/*************************************************************************
* LLK PACK
*************************************************************************/

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void llk_pack_mop_config(const uint32_t output) {

    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const bool partial_face = get_output_partial_face(output_id) && IS_BFP_FORMAT((uint)pack_dst_format[output_id]);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_pack_mop_config_<untilize, zero_output, FaceLayout, false>(
        pack_dst_format[output_id],
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile
    );
}

template <bool untilize = false, bool is_fp32_dest_acc_en = false>
inline void llk_pack_hw_configure(const llk_pack_params_t *pack_params) {

    const std::uint32_t output_id = get_output_id(pack_params->pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    const std::uint32_t tile_size = cb_interface[output_id].fifo_page_size;

    _llk_pack_hw_configure_<untilize, is_fp32_dest_acc_en>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        tile_size,
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile,
        pack_params->relu_config.val
    );
}

template <bool untilize = false, bool is_fp32_dest_acc_en = false, ReluType relu_type = ReluType::NO_RELU, std::uint32_t relu_threshold = 0>
inline void llk_pack_hw_configure_disaggregated(std::uint32_t pack_output) {
    llk_pack_params_t llk_pack_params = {
        .pack_output = pack_output, .relu_config = {.f = {.ApplyRelu = (std::uint32_t)relu_type, .Threshold = relu_threshold,}}};
    llk_pack_hw_configure<untilize, is_fp32_dest_acc_en>(&llk_pack_params);
}

template <bool untilize = false, PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en = false>
inline void llk_pack_reduce_hw_configure(const llk_pack_params_t *pack_params) {
    const std::uint32_t output_id = get_output_id(pack_params->pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    const std::uint32_t tile_size = cb_interface[output_id].fifo_page_size;

    _llk_pack_reduce_hw_configure_<untilize, type, dim, is_fp32_dest_acc_en>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        tile_size,
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile,
        pack_params->relu_config.val
    );
}

template <bool untilize = false, PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en = false, ReluType relu_type = ReluType::NO_RELU, std::uint32_t relu_threshold = 0>
inline void llk_pack_reduce_hw_configure_disaggregated(std::uint32_t pack_output) {
    llk_pack_params_t llk_pack_params = {
        .pack_output = pack_output, .relu_config = {.f = {.ApplyRelu = (std::uint32_t)relu_type, .Threshold = relu_threshold}}};
    llk_pack_reduce_hw_configure<untilize, type, dim, is_fp32_dest_acc_en>(&llk_pack_params);
}

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void llk_pack_init(const std::uint32_t pack_output = 16) {

    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_pack_init_<untilize, zero_output, FaceLayout, false>(
        pack_dst_format[output_id],
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile
    );
}

template <bool out_of_order_output, bool untilize>
inline std::uint32_t get_output_tile_address(std::uint8_t output_id, std::uint32_t output_tile_index) {

    std::uint32_t pack_tile_addr;
    if constexpr (out_of_order_output) {
        pack_tile_addr = cb_interface[output_id].fifo_wr_ptr +
                        (std::uint32_t)(cb_interface[output_id].fifo_page_size)*output_tile_index - 1;
    } else {
        if constexpr (untilize) {
            // FIXME: Need to support pack-untilize?
            // std::uint16_t out_tile_index = (cb_interface[output_id].ublock_tile_cnt/cb_interface[output_id].ublock_ct)*cb_interface[output_id].row_tile_dim +
            //                                 cb_interface[output_id].ublock_tile_cnt%cb_interface[output_id].ublock_ct; //FIXME: optimize perf
            // pack_tile_addr = cb_interface[output_id].fifo_wr_ptr + cb_interface[output_id].fifo_wr_tile_ptr - 1;
            // pack_tile_addr += out_tile_index*(std::uint32_t)(cb_interface[output_id].fifo_page_size);

            // cb_interface[output_id].ublock_tile_cnt++;

            // if (cb_interface[output_id].ublock_tile_cnt == cb_interface[output_id].ublock_tile_dim) {
            //    cb_interface[output_id].ublock_tile_cnt=0;
            //    cb_interface[output_id].fifo_wr_tile_ptr += (std::uint32_t)(cb_interface[output_id].fifo_page_size)*cb_interface[output_id].ublock_ct;
            // }
        } else {
            pack_tile_addr = cb_interface[output_id].fifo_wr_ptr + cb_interface[output_id].fifo_wr_tile_ptr - 1;
            cb_interface[output_id].fifo_wr_tile_ptr += cb_interface[output_id].fifo_page_size;
        }
    }
    return pack_tile_addr;
}

template <bool out_of_order_output = false, DstSync Dst = SyncFull, bool untilize = false, bool is_fp32_dest_acc_en = false>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);

    static_assert((!(untilize && out_of_order_output)) && "untilize out of order packing is not supported!");

    std::uint32_t pack_tile_addr = get_output_tile_address<out_of_order_output, untilize>(output_id, output_tile_index);

    _llk_pack_<Dst, untilize, is_fp32_dest_acc_en>(
        tile_index,
        pack_tile_addr
    );
}

template <bool out_of_order_output = false, DstSync Dst = SyncFull, bool untilize = false, bool is_fp32_dest_acc_en = false>
inline void llk_matmul_pack(std::uint32_t start_tile_index, std::uint32_t output, uint32_t ntiles, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);

    static_assert((!(untilize && out_of_order_output)) && "untilize out of order packing is not supported!");

    for (uint32_t tile_index=start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        std::uint32_t pack_tile_addr = get_output_tile_address<out_of_order_output, untilize>(output_id, output_tile_index);

        _llk_pack_<Dst, untilize, is_fp32_dest_acc_en>(
            tile_index,
            pack_tile_addr
        );
    }
}

/*************************************************************************
* LLK PACK COMMON
*************************************************************************/


inline void llk_packer_wait_for_math_done() {
    _llk_packer_wait_for_math_done_();
}

template <uint WaitRes = p_stall::NONE>
inline void llk_packer_set_math_semaphore() {
    _llk_packer_set_math_semaphore_<WaitRes>();
}

template <DstSync Dst, bool is_fp32_dest_acc_en = false>
inline void llk_pack_dest_section_done() {
    _llk_pack_dest_section_done_<Dst, is_fp32_dest_acc_en>();
}

template <DstSync Dst, DstTileFaceLayout FaceLayout, bool untilize = false>
inline void llk_init_packer_dest_offset_registers(const std::uint32_t pack_output = 16) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_init_packer_dest_offset_registers_<Dst, FaceLayout, untilize>(
        face_r_dim,
        narrow_tile
    );
}

template <DstSync Dst, DstTileFaceLayout FaceLayout = RowMajor, bool untilize = false, bool is_fp32_dest_acc_en = false>
inline void llk_pack_dest_init(const std::uint32_t pack_output = 16) {

    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_pack_dest_init_<Dst, FaceLayout, untilize, is_fp32_dest_acc_en>(
        face_r_dim,
        narrow_tile
    );
}

template <bool mail2math=true, bool mail2pack=true>
inline void llk_pack_get_tile(std::uint32_t output, std::uint32_t tile_index, std::uint32_t *p_tile) {
    _llk_pack_get_tile_<mail2math, mail2pack>(tile_index, p_tile);
}

template <bool mail2math=true, bool mail2pack=true>
inline void llk_pack_release_tile(std::uint32_t output) {
    _llk_pack_release_tile_<mail2math, mail2pack>();
}

inline void llk_pack_debug_dump(std::uint8_t *data, std::uint32_t byte_size) {
    _llk_pack_debug_dump_(data, byte_size);
}

inline void llk_pack_debug_dump_seek(std::uint8_t offset) {
    _llk_pack_debug_dump_seek_(offset);
}

template <bool is_fp32_dest_acc_en = false, bool is_tile_dim_reconfig_en = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void llk_pack_reconfig_data_format(const std::uint32_t new_output) {

    const std::uint32_t output_id = get_output_id(new_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, is_tile_dim_reconfig_en, FaceLayout>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        cb_interface[output_id].fifo_page_size,
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile
    );
}

template <bool is_fp32_dest_acc_en = false, bool is_tile_dim_reconfig_en = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void llk_pack_reconfig_data_format(const std::uint32_t old_output, const std::uint32_t new_output) {
    std::uint32_t old_output_id = get_output_id(old_output);
    std::uint32_t new_output_id = get_output_id(new_output);

    if((pack_dst_format[old_output_id] != pack_dst_format[new_output_id])
       && (pack_dst_format[old_output_id] != (uint)DataFormat::Invalid)
       && (pack_dst_format[new_output_id] != (uint)DataFormat::Invalid)) {
        llk_pack_reconfig_data_format<is_fp32_dest_acc_en, is_tile_dim_reconfig_en, FaceLayout>(new_output);
    } else if constexpr (is_tile_dim_reconfig_en) {
        // Same format but different tile dims
        llk_pack_mop_config<false, false, FaceLayout, false>(new_output);
    }
}

TT_ALWAYS_INLINE void llk_pack_relu_config(const std::uint32_t config) {
    _llk_pack_relu_config_(config);
}

inline void llk_pack_reconfig_l1_acc(const std::uint32_t enable) {
    _llk_pack_reconfig_l1_acc_(enable);
}

template <bool untilize = false, ReduceDim dim>
inline void llk_pack_reduce_mask_config() {
    _llk_pack_reduce_mask_config_<untilize, dim>();
}

inline void llk_pack_reduce_mask_clear() {
    _llk_pack_reduce_mask_clear_();
}

// FIXME-WH-UPLIFT
template <ReduceDim dim, bool at_kernel_start = false, bool revert=false, bool is_fp32_dest_acc_en = false>
inline void llk_pack_reduce_config_v2(uint32_t icb_out) {

    const bool untilize = false;
    if constexpr (at_kernel_start) {

        const std::uint32_t output_id = get_output_id(icb_out);
        const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
        const std::uint32_t num_faces = get_output_num_faces(output_id);
        const bool partial_face = get_output_partial_face(output_id);
        const bool narrow_tile = get_output_narrow_tile(output_id);
        const std::uint32_t tile_size = cb_interface[output_id].fifo_page_size;
        const llk_relu_config_u relu_config = {.f = {.ApplyRelu = (std::uint32_t)ReluType::NO_RELU, .Threshold = 0,}};

        _llk_pack_hw_configure_<untilize, is_fp32_dest_acc_en>(
            pack_src_format[output_id],
            pack_dst_format[output_id],
            tile_size,
            face_r_dim,
            num_faces,
            partial_face,
            narrow_tile,
            relu_config.val
        );
    }

    if constexpr (revert) {
        _llk_pack_reduce_mask_clear_();
    } else {
        _llk_pack_reduce_mask_config_<untilize, dim>();
    }
}
