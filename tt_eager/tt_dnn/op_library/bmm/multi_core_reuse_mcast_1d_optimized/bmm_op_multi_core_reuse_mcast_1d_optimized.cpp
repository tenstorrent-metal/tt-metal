// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include <algorithm>
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt;

namespace reuse_mcast_1d_optimized_helpers {
using namespace tt::constants;
using namespace tt;
using namespace tt_metal;

operation::ProgramWithCallbacks create_program_mcast_in0(
    tt_metal::Device *device,
    MathFidelity math_fidelity,
    CoreCoord core_range,
    uint32_t B, uint32_t M, uint32_t N, uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h, uint32_t out_subblock_w,
    uint32_t per_core_M, uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    tt_metal::Buffer* in0_buffer, tt_metal::Buffer* in1_buffer, tt_metal::Buffer* bias_buffer, tt_metal::Buffer* out_buffer,
    tt::DataFormat in0_data_format, tt::DataFormat in1_data_format, tt::DataFormat bias_data_format, tt::DataFormat output_data_format
) {

    tt_metal::Program program{};

    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t bias_single_tile_size = tt_metal::detail::TileSize(bias_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2; // double buffer
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * 2; // double buffer
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles; // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;

    uint32_t in3_block_tiles = per_core_N;
    uint32_t in3_CB_tiles = in3_block_tiles; // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    uint32_t interm1_block_tiles = out_subblock_h * out_subblock_w;
    uint32_t interm1_CB_tiles = interm1_block_tiles; // No double buffer
    uint32_t interm1_CB_size = interm1_CB_tiles * output_single_tile_size;


    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    uint32_t num_cores = num_blocks_total;
    uint32_t num_cores_in_mcast_grid = num_cores_c * num_cores_r - 1; // Exclude Sender

    CoreRangeSet all_cores = num_cores_to_corerange_set(num_cores, core_range, true);


    CoreRange mcast_sender{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x, (std::size_t) start_core_y}};

    // TODO: Optimize difference of corerangesets
    std::set<CoreRange> mcast_receivers_set;
    for(uint32_t i = 0; i < num_cores; i++) {
        uint32_t core_idx_x = i % num_cores_c;
        uint32_t core_idx_y = i / num_cores_c;
        if (!(core_idx_x == 0 && core_idx_y == 0)) {
            CoreRange core = {
                .start={(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y},
                .end={(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y}
            };
            mcast_receivers_set.insert(core);
        }
    }
    CoreRangeSet mcast_receivers(mcast_receivers_set);

    // Mcast args
    auto in0_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    CoreCoord top_left_core = {(std::size_t) start_core_x, (std::size_t) start_core_y};
    CoreCoord bottom_right_core = {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1};
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in3_is_dram = true;
    if (bias_buffer != nullptr) {
        in3_is_dram = bias_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    }
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> in0_sender_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) in0_is_dram,

            // in0 tensor args
            (std::uint32_t)  1, // in0_tensor_stride_w
            (std::uint32_t)  K, // in0_tensor_stride_h
            (std::uint32_t)  in0_block_w, // in0_tensor_next_block_stride
            // in0 block args
            (std::uint32_t)  in0_block_w, // in0_block_w
            (std::uint32_t)  per_core_M, // in0_block_h
            (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)  K / in0_block_w, // num_blocks
            // in0 mcast args
            (std::uint32_t)  top_left_core_physical.x, // in0_mcast_dest_noc_start_x
            (std::uint32_t)  bottom_right_core_physical.x, // in0_mcast_dest_noc_end_x
            (std::uint32_t)  in0_mcast_sender_semaphore,
            (std::uint32_t)  in0_mcast_receiver_semaphore,
            (std::uint32_t)  num_cores - 1, // in0_mcast_num_dests
            (std::uint32_t)  num_cores_in_mcast_grid, // in0_mcast_num_cores
            // batch args
            (std::uint32_t)  M * K, // MtKt
            (std::uint32_t)  B // batch
    };
    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) in1_is_dram,
            (std::uint32_t) out_is_dram,

            // READER
            // in1 tensor args
            (std::uint32_t)  1, // in1_tensor_stride_w
            (std::uint32_t)  N, // in1_tensor_stride_h
            (std::uint32_t)  in0_block_w * N, //in1_tensor_next_block_stride
            // in1 block args
            (std::uint32_t)  per_core_N, // in1_block_w
            (std::uint32_t)  in0_block_w, //in1_block_h
            (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)  K / in0_block_w, // num_blocks
            // in1 mcast args
            (std::uint32_t)  0, // in1_mcast_dest_noc_start_y
            (std::uint32_t)  0, // in1_mcast_dest_noc_end_y
            (std::uint32_t)  0,
            (std::uint32_t)  0,
            (std::uint32_t)  0, // in1_mcast_num_dests
            (std::uint32_t)  0, // in1_mcast_num_cores
            // batch args
            (std::uint32_t)  K * N, // KtNt
            (std::uint32_t)  B, // batch
            (std::uint32_t)  bcast_batch, // bcast_B

            // WRITER
            // out tensor args
            (std::uint32_t)  1, // out_tensor_stride_w
            (std::uint32_t)  N,  // out_tensor_stride_h
            (std::uint32_t)  out_subblock_w, // out_tensor_next_subblock_stride_w
            (std::uint32_t)  out_subblock_h * N, // out_tensor_next_subblock_stride_h
            // out subblock args
            (std::uint32_t)  out_subblock_w, // out_subblock_w
            (std::uint32_t)  out_subblock_h, // out_subblock_h
            (std::uint32_t)  (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
            // batch args
            (std::uint32_t)  M * N // MtNt
    };
    if (bias_buffer != nullptr) {
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)  in3_is_dram);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)  1);
    }
    std::vector<uint32_t> in0_receiver_compile_time_args = {
            // in0 block args
            (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)  K / in0_block_w, // num_blocks
            // in0 mcast args
            (std::uint32_t)  top_left_core_physical.x, // in0_mcast_sender_noc_x
            (std::uint32_t)  in0_mcast_sender_semaphore,
            (std::uint32_t)  in0_mcast_receiver_semaphore,
            // batch args
            (std::uint32_t)  B // batch
    };

    std::map<string, string> mm_kernel_defines;
    std::map<string, string> mm_kernel_in1_sender_writer_defines;
    if (bias_buffer != nullptr) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        mm_kernel_defines.merge(eltwise_unary_op_utils::get_defines(fused_activation.value().op_type, fused_activation.value().param, "ACTIVATION", "i"));
    }

    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";
    auto mm_kernel_in0_sender_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        mcast_sender,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = in0_sender_compile_time_args});

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = in1_sender_writer_compile_time_args, .defines = mm_kernel_in1_sender_writer_defines});


    auto mm_kernel_in0_receiver_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
        /* in0_receiver_in1_sender, // If not using half-half noc setup */
        mcast_receivers,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = in0_receiver_compile_time_args});

    // Compute kernel compile time args
    uint32_t num_blocks = (K/in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M/out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N/out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w, // in0_block_w
        in0_num_subblocks, // in0_num_subblocks
        in0_block_num_tiles, // in0_block_num_tiles
        in0_subblock_num_tiles, // in0_subblock_num_tiles

        in1_num_subblocks, // in1_num_subblocks
        in1_block_num_tiles, // in1_block_num_tiles
        in1_per_core_w, // in1_per_core_w

        num_blocks, // num_blocks

        out_subblock_h, // out_subblock_h
        out_subblock_w, // out_subblock_w
        out_subblock_num_tiles, // out_subblock_num_tiles
        B // batch
    };

    // Create compute kernel
    bool fp32_dest_acc_en = false;
    // Gelu currently has better accuracy when run in approx mode
    bool math_approx_mode = false;
    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compute_kernel_args, .defines = mm_kernel_defines}
    );

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
		.set_page_size(src0_cb_index, in0_single_tile_size);
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config = tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
		.set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);


    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
        {output_cb_index, output_data_format},
        {interm0_cb_index, output_data_format}
    };
    tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
		.set_page_size(output_cb_index, output_single_tile_size)
        .set_page_size(interm0_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = 3;
        tt_metal::CircularBufferConfig cb_src3_config = tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
		    .set_page_size(src3_cb_index, bias_single_tile_size);
        auto cb_src3 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src3_config);

        uint32_t interm1_cb_index = 25;
        tt_metal::CircularBufferConfig cb_interm1_config = tt_metal::CircularBufferConfig(interm1_CB_size, {{interm1_cb_index, output_data_format}})
		    .set_page_size(interm1_cb_index, output_single_tile_size);
        auto cb_interm1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_interm1_config);
    }

    // Parameters for last row, col, or block
    uint32_t last_block_h = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_block_w = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_block_num_nonzero_subblocks_h = (last_block_h  - 1) / out_subblock_h + 1;
    uint32_t last_block_num_nonzero_subblocks_w = (last_block_w  - 1) / out_subblock_w + 1;
    uint32_t last_subblock_of_last_block_h = last_block_h % out_subblock_h == 0 ? out_subblock_h : last_block_h % out_subblock_h;
    uint32_t last_subblock_of_last_block_w = last_block_w % out_subblock_w == 0 ? out_subblock_w : last_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip = output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =  (out_subblock_w * out_subblock_h) * (per_core_N / out_subblock_w - last_block_num_nonzero_subblocks_w);
    uint32_t last_block_padded_block_tiles_h_skip = (per_core_M / out_subblock_h - last_block_num_nonzero_subblocks_h) * (per_core_N * out_subblock_h);

    std::vector<KernelID> reader_kernel_ids;
    std::vector<KernelID> writer_kernel_ids;
    for(uint32_t i = 0; i < num_cores; i++) {
        uint32_t core_idx_x = i % num_cores_c;
        uint32_t core_idx_y = i / num_cores_c;
        uint32_t output_idx_x = i % num_blocks_x;
        uint32_t output_idx_y = i / num_blocks_x;
        CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

        // in0 sender and in1 sender
        if (core_idx_x == 0 and core_idx_y == 0) {
            std::vector<uint32_t> mm_in0_sender_args =  {
                // in0 tensor args
                (std::uint32_t)  in0_buffer->address(),
                (std::uint32_t)  K * per_core_M * output_idx_y, // in0_tensor_start_tile_id
                // in0 mcast args
                (std::uint32_t)  top_left_core_physical.y, // in0_mcast_dest_noc_start_y
                (std::uint32_t)  bottom_right_core_physical.y, // in0_mcast_dest_noc_end_y

                // padding args
                (std::uint32_t) per_core_M // last_block_h
            };
            std::vector<uint32_t> mm_in1_sender_writer_args = {
                // READER
                // in1 tensor args
                (std::uint32_t)  in1_buffer->address(),
                (std::uint32_t)  per_core_N * output_idx_x, //in1_tensor_start_tile_id
                // in1 mcast args
                (std::uint32_t)  0, // in1_mcast_dest_noc_start_x
                (std::uint32_t)  0, // in1_mcast_dest_noc_end_x

                // WRITER
                // out tensor args
                (std::uint32_t)  out_buffer->address(),
                (std::uint32_t)  output_idx_x * per_core_N + output_idx_y * per_core_M * N, // out_tensor_start_tile_id

                // padding args (READER)
                (std::uint32_t)  per_core_N, // last_block_w
                // padding args (WRITER)
                (std::uint32_t)  per_core_M / out_subblock_h,
                (std::uint32_t)  out_subblock_h,
                (std::uint32_t)  0,
                (std::uint32_t)  per_core_N / out_subblock_w,
                (std::uint32_t)  out_subblock_w,
                (std::uint32_t)  0,
                (std::uint32_t)  0
            };

            if (bias_buffer != nullptr) {
                mm_in1_sender_writer_args.push_back((std::uint32_t)  bias_buffer->address());
                mm_in1_sender_writer_args.push_back((std::uint32_t)  per_core_N * output_idx_x); //in3_tensor_start_tile_id
            }

            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args); // RISCV_0_default
            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args); // RISCV_1_default
            reader_kernel_ids.push_back(mm_kernel_in0_sender_id);
            writer_kernel_ids.push_back(mm_kernel_in1_sender_writer_id);
        }
        // in0 receiver and in 1 sender
        else if (!(core_idx_x == 0 and core_idx_y == 0)) {
            std::vector<uint32_t> mm_in0_receiver_args = {
                // in0 mcast args
                (std::uint32_t)  top_left_core_physical.y // in0_mcast_sender_noc_y
            };
            std::vector<uint32_t> mm_in1_sender_writer_args = {
                // READER
                // in1 tensor args
                (std::uint32_t)  in1_buffer->address(),
                (std::uint32_t)  per_core_N * output_idx_x, //in1_tensor_start_tile_id
                // in1 mcast args
                (std::uint32_t)  0, // in1_mcast_dest_noc_start_x
                (std::uint32_t)  0, // in1_mcast_dest_noc_end_x

                // WRITER
                // out tensor args
                (std::uint32_t)  out_buffer->address(),
                (std::uint32_t)  output_idx_x * per_core_N + output_idx_y * per_core_M * N // out_tensor_start_tile_id
            };

            if (output_idx_x == num_blocks_x - 1) {
                // padding args (READER)
                mm_in1_sender_writer_args.push_back(last_block_w);

                // padding args (WRITER)
                mm_in1_sender_writer_args.push_back(per_core_M /out_subblock_h);
                mm_in1_sender_writer_args.push_back(out_subblock_h);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                mm_in1_sender_writer_args.push_back(last_subblock_of_last_block_w);
                mm_in1_sender_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                mm_in1_sender_writer_args.push_back(last_block_padded_block_tiles_w_skip);
            } else {
                // padding args (READER)
                mm_in1_sender_writer_args.push_back(per_core_N);

                // padding args (WRITER)
                mm_in1_sender_writer_args.push_back(per_core_M /out_subblock_h);
                mm_in1_sender_writer_args.push_back(out_subblock_h);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(per_core_N / out_subblock_w);
                mm_in1_sender_writer_args.push_back(out_subblock_w);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(0);
            }

            if (bias_buffer != nullptr) {
                mm_in1_sender_writer_args.push_back((std::uint32_t)  bias_buffer->address());
                mm_in1_sender_writer_args.push_back((std::uint32_t)  per_core_N * output_idx_x); //in3_tensor_start_tile_id
            }
            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_receiver_id, core, mm_in0_receiver_args); // RISCV_1_default
            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args); // RISCV_0_default
            reader_kernel_ids.push_back(mm_kernel_in0_receiver_id);
            writer_kernel_ids.push_back(mm_kernel_in1_sender_writer_id);
        }
    }

    auto override_runtime_args_callback = [
            reader_kernel_ids,
            writer_kernel_ids,
            num_cores_r,
            num_cores_c,
            num_cores,
            start_core_x,
            start_core_y
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {
        TT_ASSERT(input_buffers.size() == 3);
        TT_ASSERT(output_buffers.size() == 1);

        auto src_dram_buffer_a = input_buffers.at(0);
        auto src_dram_buffer_b = input_buffers.at(1);
        auto bias_dram_buffer = input_buffers.at(2);

        auto dst_dram_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < num_cores; i++) {
            uint32_t core_idx_x = i % num_cores_c;
            uint32_t core_idx_y = i / num_cores_c;
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

            // in0 sender and in1 sender
            if (core_idx_x == 0 and core_idx_y == 0) {
                {
                    auto reader_kernel_id = reader_kernel_ids.at(i);
                    auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = src_dram_buffer_a->address();
                    SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
                }

                {
                    auto writer_kernel_id = writer_kernel_ids.at(i);
                    auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = src_dram_buffer_b->address();
                    runtime_args[4] = dst_dram_buffer->address();
                    if (bias_dram_buffer != nullptr) {
                        runtime_args[14] = bias_dram_buffer->address();
                    }
                    SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
                }
            }
            // in0 receiver and in1 sender
            else if (!(core_idx_x == 0 and core_idx_y == 0)) {
                {
                    auto writer_kernel_id = writer_kernel_ids.at(i);
                    auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = src_dram_buffer_b->address();
                    runtime_args[4] = dst_dram_buffer->address();
                    if (bias_dram_buffer != nullptr) {
                        runtime_args[14] = bias_dram_buffer->address();
                    }
                    SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
                }
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks create_program_mcast_in1(
    tt_metal::Device *device,
    MathFidelity math_fidelity,
    CoreCoord core_range,
    uint32_t B, uint32_t M, uint32_t N, uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h, uint32_t out_subblock_w,
    uint32_t per_core_M, uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    tt_metal::Buffer* in0_buffer, tt_metal::Buffer* in1_buffer, tt_metal::Buffer* bias_buffer, tt_metal::Buffer* out_buffer,
    tt::DataFormat in0_data_format, tt::DataFormat in1_data_format, tt::DataFormat bias_data_format, tt::DataFormat output_data_format,
    std::optional<uint32_t> in0_address, std::optional<uint32_t> output_address
) {

    tt_metal::Program program{};

    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t bias_single_tile_size = tt_metal::detail::TileSize(bias_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles;
    if (in0_address.has_value()) {
        uint32_t in0_num_blocks = K / in0_block_w;
        in0_CB_tiles = in0_num_blocks * in0_block_tiles * B;
    } else {
        in0_CB_tiles = in0_block_tiles * 2; // double buffer
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * 2; // double buffer
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles; // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;


    uint32_t in3_block_tiles = per_core_N;
    uint32_t in3_CB_tiles = in3_block_tiles; // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    uint32_t interm1_block_tiles = out_subblock_h * out_subblock_w;
    uint32_t interm1_CB_tiles = interm1_block_tiles; // No double buffer
    uint32_t interm1_CB_size = interm1_CB_tiles * output_single_tile_size;


    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    uint32_t num_cores = num_blocks_total;
    uint32_t num_cores_in_mcast_grid = num_cores_c * num_cores_r - 1; // Exclude Sender

    CoreRangeSet all_cores = num_cores_to_corerange_set(num_cores, core_range, true);


    CoreRange mcast_sender{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x, (std::size_t) start_core_y}};

    // TODO: Optimize difference of corerangesets
    std::set<CoreRange> mcast_receivers_set;
    for(uint32_t i = 0; i < num_cores; i++) {
        uint32_t core_idx_x = i % num_cores_c;
        uint32_t core_idx_y = i / num_cores_c;
        if (!(core_idx_x == 0 && core_idx_y == 0)) {
            CoreRange core = {
                .start={(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y},
                .end={(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y}
            };
            mcast_receivers_set.insert(core);
        }
    }
    CoreRangeSet mcast_receivers(mcast_receivers_set);

    // Mcast args
    auto in1_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    uint32_t in3_mcast_sender_semaphore = 0;
    uint32_t in3_mcast_receiver_semaphore = 0;

    CoreCoord top_left_core = {(std::size_t) start_core_x, (std::size_t) start_core_y};
    CoreCoord bottom_right_core = {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1};
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in3_is_dram = true;
    if (bias_buffer != nullptr) {
        in3_is_dram = bias_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    }
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> in0_sender_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) in0_is_dram,

            // in0 tensor args
            (std::uint32_t)  1, // in0_tensor_stride_w
            (std::uint32_t)  K, // in0_tensor_stride_h
            (std::uint32_t)  in0_block_w, // in0_tensor_next_block_stride
            // in0 block args
            (std::uint32_t)  in0_block_w, // in0_block_w
            (std::uint32_t)  per_core_M, // in0_block_h
            (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)  K / in0_block_w, // num_blocks
            // in0 mcast args
            (std::uint32_t)  0, // in0_mcast_dest_noc_start_x
            (std::uint32_t)  0, // in0_mcast_dest_noc_end_x
            (std::uint32_t)  0,
            (std::uint32_t)  0,
            (std::uint32_t)  0, // in0_mcast_num_dests
            (std::uint32_t)  0, // in0_mcast_num_cores
            // batch args
            (std::uint32_t)  M * K, // MtKt
            (std::uint32_t)  B // batch
    };
    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) in1_is_dram,
            (std::uint32_t) out_is_dram,

            // READER
            // in1 tensor args
            (std::uint32_t)  1, // in1_tensor_stride_w
            (std::uint32_t)  N, // in1_tensor_stride_h
            (std::uint32_t)  in0_block_w * N, //in1_tensor_next_block_stride
            // in1 block args
            (std::uint32_t)  per_core_N, // in1_block_w
            (std::uint32_t)  in0_block_w, //in1_block_h
            (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)  K / in0_block_w, // num_blocks
            // in1 mcast args
            (std::uint32_t)  bottom_right_core_physical.y, // in1_mcast_dest_noc_start_y
            (std::uint32_t)  top_left_core_physical.y, // in1_mcast_dest_noc_end_y
            (std::uint32_t)  in1_mcast_sender_semaphore,
            (std::uint32_t)  in1_mcast_receiver_semaphore,
            (std::uint32_t)  num_cores - 1, // in1_mcast_num_dests
            (std::uint32_t)  num_cores_in_mcast_grid, // in1_mcast_num_cores
            // batch args
            (std::uint32_t)  K * N, // KtNt
            (std::uint32_t)  B, // batch
            (std::uint32_t)  bcast_batch, // bcast_B

            // WRITER
            // out tensor args
            (std::uint32_t)  1, // out_tensor_stride_w
            (std::uint32_t)  N,  // out_tensor_stride_h
            (std::uint32_t)  out_subblock_w, // out_tensor_next_subblock_stride_w
            (std::uint32_t)  out_subblock_h * N, // out_tensor_next_subblock_stride_h
            // out subblock args
            (std::uint32_t)  out_subblock_w, // out_subblock_w
            (std::uint32_t)  out_subblock_h, // out_subblock_h
            (std::uint32_t)  (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
            // batch args
            (std::uint32_t)  M * N // MtNt
    };
    if (bias_buffer != nullptr) {
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)  in3_is_dram);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)  1);
    }
    std::vector<uint32_t> in1_receiver_writer_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) out_is_dram,

            // READER
            // in1 block args
            (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)  K / in0_block_w, // num_blocks
            // in1 mcast args
            (std::uint32_t)  top_left_core_physical.y, // in1_mcast_sender_noc_y
            (std::uint32_t)  in1_mcast_sender_semaphore,
            (std::uint32_t)  in1_mcast_receiver_semaphore,
            // batch args
            (std::uint32_t)  B, // batch

            // WRITER
            // out tensor args
            (std::uint32_t)  1, // out_tensor_stride_w
            (std::uint32_t)  N,  // out_tensor_stride_h
            (std::uint32_t)  out_subblock_w, // out_tensor_next_subblock_stride_w
            (std::uint32_t)  out_subblock_h * N, // out_tensor_next_subblock_stride_h
            // out subblock args
            (std::uint32_t)  out_subblock_w, // out_subblock_w
            (std::uint32_t)  out_subblock_h, // out_subblock_h
            (std::uint32_t)  (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
            // batch args
            (std::uint32_t)  M * N // MtNt
    };

    if (bias_buffer != nullptr) {
        in1_receiver_writer_compile_time_args.push_back((std::uint32_t)  per_core_N);
    }

    std::map<string, string> mm_kernel_defines;
    std::map<string, string> mm_kernel_in0_sender_defines;
    std::map<string, string> mm_kernel_in1_sender_writer_defines;
    std::map<string, string> mm_kernel_in1_receiver_writer_defines;
    if (bias_buffer != nullptr) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        mm_kernel_defines.merge(eltwise_unary_op_utils::get_defines(fused_activation.value().op_type, fused_activation.value().param, "ACTIVATION", "i"));
    }
    if (in0_address.has_value()) {
        mm_kernel_in0_sender_defines["IN0_SHARDED"] = "1";
    }
    if (output_address.has_value()) {
        mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
        mm_kernel_in1_receiver_writer_defines["OUT_SHARDED"] = "1";
    }

    mm_kernel_in0_sender_defines["SKIP_MCAST"] = "1";
    auto mm_kernel_in0_sender_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = in0_sender_compile_time_args, .defines = mm_kernel_in0_sender_defines});

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        mcast_sender,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = in1_sender_writer_compile_time_args, .defines = mm_kernel_in1_sender_writer_defines});


    auto mm_kernel_in1_receiver_writer_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
        mcast_receivers,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = in1_receiver_writer_compile_time_args, .defines = mm_kernel_in1_receiver_writer_defines});

    // Compute kernel compile time args
    uint32_t num_blocks = (K/in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M/out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N/out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w, // in0_block_w
        in0_num_subblocks, // in0_num_subblocks
        in0_block_num_tiles, // in0_block_num_tiles
        in0_subblock_num_tiles, // in0_subblock_num_tiles

        in1_num_subblocks, // in1_num_subblocks
        in1_block_num_tiles, // in1_block_num_tiles
        in1_per_core_w, // in1_per_core_w

        num_blocks, // num_blocks

        out_subblock_h, // out_subblock_h
        out_subblock_w, // out_subblock_w
        out_subblock_num_tiles, // out_subblock_num_tiles
        B // batch
    };

    // Create compute kernel
    bool fp32_dest_acc_en = false;
    // Gelu currently has better accuracy when run in approx mode
    bool math_approx_mode = false;
    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compute_kernel_args, .defines = mm_kernel_defines}
    );

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
		.set_page_size(src0_cb_index, in0_single_tile_size);
    if (in0_address.has_value()) {
        src0_cb_config = src0_cb_config.set_globally_allocated_address(in0_address.value());
    }
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config = tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
		.set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
        {output_cb_index, output_data_format},
        {interm0_cb_index, output_data_format}
    };
    tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
		.set_page_size(output_cb_index, output_single_tile_size)
        .set_page_size(interm0_cb_index, output_single_tile_size);
    if (output_address.has_value()) {
        output_cb_config = output_cb_config.set_globally_allocated_address(output_address.value());
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = 3;
        tt_metal::CircularBufferConfig cb_src3_config = tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
		    .set_page_size(src3_cb_index, bias_single_tile_size);
        auto cb_src3 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src3_config);

        uint32_t interm1_cb_index = 25;
        tt_metal::CircularBufferConfig cb_interm1_config = tt_metal::CircularBufferConfig(interm1_CB_size, {{interm1_cb_index, output_data_format}})
		    .set_page_size(interm1_cb_index, output_single_tile_size);
        auto cb_interm1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_interm1_config);
    }

    // Parameters for last row, col, or block
    uint32_t last_block_h = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_block_w = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_block_num_nonzero_subblocks_h = (last_block_h  - 1) / out_subblock_h + 1;
    uint32_t last_block_num_nonzero_subblocks_w = (last_block_w  - 1) / out_subblock_w + 1;
    uint32_t last_subblock_of_last_block_h = last_block_h % out_subblock_h == 0 ? out_subblock_h : last_block_h % out_subblock_h;
    uint32_t last_subblock_of_last_block_w = last_block_w % out_subblock_w == 0 ? out_subblock_w : last_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip = output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =  (out_subblock_w * out_subblock_h) * (per_core_N / out_subblock_w - last_block_num_nonzero_subblocks_w);
    uint32_t last_block_padded_block_tiles_h_skip = (per_core_M / out_subblock_h - last_block_num_nonzero_subblocks_h) * (per_core_N * out_subblock_h);

    std::vector<KernelID> reader_kernel_ids;
    std::vector<KernelID> writer_kernel_ids;
    for(uint32_t i = 0; i < num_cores; i++) {
        uint32_t core_idx_x = i % num_cores_c;
        uint32_t core_idx_y = i / num_cores_c;
        uint32_t output_idx_x = i / num_blocks_y;
        uint32_t output_idx_y = i % num_blocks_y;
        CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

        // in0 sender and in1 sender
        if (core_idx_x == 0 and core_idx_y == 0) {
            std::vector<uint32_t> mm_in0_sender_args =  {
                // in0 tensor args
                (std::uint32_t)  in0_buffer->address(),
                (std::uint32_t)  K * per_core_M * output_idx_y, // in0_tensor_start_tile_id
                // in0 mcast args
                (std::uint32_t)  0, // in0_mcast_dest_noc_start_y
                (std::uint32_t)  0, // in0_mcast_dest_noc_end_y

                // padding args
                (std::uint32_t) per_core_M // last_block_h
            };
            std::vector<uint32_t> mm_in1_sender_writer_args = {
                // READER
                // in1 tensor args
                (std::uint32_t)  in1_buffer->address(),
                (std::uint32_t)  per_core_N * output_idx_x, //in1_tensor_start_tile_id
                // in1 mcast args
                (std::uint32_t)  bottom_right_core_physical.x, // in1_mcast_dest_noc_start_x
                (std::uint32_t)  top_left_core_physical.x, // in1_mcast_dest_noc_end_x

                // WRITER
                // out tensor args
                (std::uint32_t)  out_buffer->address(),
                (std::uint32_t)  output_idx_x * per_core_N + output_idx_y * per_core_M * N, // out_tensor_start_tile_id

                // padding args (READER)
                (std::uint32_t)  per_core_N, // last_block_w
                // padding args (WRITER)
                (std::uint32_t)  per_core_M / out_subblock_h,
                (std::uint32_t)  out_subblock_h,
                (std::uint32_t)  0,
                (std::uint32_t)  per_core_N / out_subblock_w,
                (std::uint32_t)  out_subblock_w,
                (std::uint32_t)  0,
                (std::uint32_t)  0
            };

            if (bias_buffer != nullptr) {
                mm_in1_sender_writer_args.push_back((std::uint32_t)  bias_buffer->address());
                mm_in1_sender_writer_args.push_back((std::uint32_t)  per_core_N * output_idx_x); //in3_tensor_start_tile_id
            }

            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args); // RISCV_0_default
            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args); // RISCV_1_default
            reader_kernel_ids.push_back(mm_kernel_in0_sender_id);
            writer_kernel_ids.push_back(mm_kernel_in1_sender_writer_id);
        }
        // in0 sender and in1 receiver
        else if (!(core_idx_x == 0 and core_idx_y == 0)) {
            std::vector<uint32_t> mm_in0_sender_args =  {
                // in0 tensor args
                (std::uint32_t)  in0_buffer->address(),
                (std::uint32_t)  K * per_core_M * output_idx_y, // in0_tensor_start_tile_id
                // in0 mcast args
                (std::uint32_t)  0, // in0_mcast_dest_noc_start_y
                (std::uint32_t)  0, // in0_mcast_dest_noc_end_y

                // padding args
                (std::uint32_t) per_core_M // last_block_h
            };
            std::vector<uint32_t> mm_in1_receiver_writer_args = {
                    // READER
                    // in1 mcast args
                    (std::uint32_t)  top_left_core_physical.x, // in1_mcast_sender_noc_x

                    // WRITER
                    // out tensor args
                    (std::uint32_t)  out_buffer->address(), // out_tensor_addr
                    (std::uint32_t)  output_idx_x * per_core_N + output_idx_y * per_core_M * N // out_tensor_start_tile_id
                };

            if (output_idx_x == num_blocks_x - 1 and output_idx_y == num_blocks_y - 1) {
                // padding args (WRITER)
                mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
                mm_in1_receiver_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_w_skip);
            } else if (output_idx_y == num_blocks_y - 1) {
                // padding args (WRITER)
                mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                mm_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
                mm_in1_receiver_writer_args.push_back(out_subblock_w);
                mm_in1_receiver_writer_args.push_back(0);
                mm_in1_receiver_writer_args.push_back(0);
            } else if (output_idx_x == num_blocks_x - 1) {
                // padding args (WRITER)
                mm_in1_receiver_writer_args.push_back(per_core_M / out_subblock_h);
                mm_in1_receiver_writer_args.push_back(out_subblock_h);
                mm_in1_receiver_writer_args.push_back(0);
                mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
                mm_in1_receiver_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_w_skip);
            } else {
                // padding args (WRITER)
                mm_in1_receiver_writer_args.push_back(per_core_M / out_subblock_h);
                mm_in1_receiver_writer_args.push_back(out_subblock_h);
                mm_in1_receiver_writer_args.push_back(0);
                mm_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
                mm_in1_receiver_writer_args.push_back(out_subblock_w);
                mm_in1_receiver_writer_args.push_back(0);
                mm_in1_receiver_writer_args.push_back(0);
            }

            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args); // RISCV_1_default
            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_receiver_writer_id, core, mm_in1_receiver_writer_args); // RISCV_0_default
            reader_kernel_ids.push_back(mm_kernel_in0_sender_id);
            writer_kernel_ids.push_back(mm_kernel_in1_receiver_writer_id);
        }
    }

    auto override_runtime_arguments_callback = [
            reader_kernel_ids,
            writer_kernel_ids,
            cb_src0,
            cb_output,
            num_cores_r,
            num_cores_c,
            num_cores,
            start_core_x,
            start_core_y
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        TT_ASSERT(input_tensors.size() + optional_input_tensors.size() == 3);
        TT_ASSERT(output_tensors.size() == 1);

        auto src_buffer_a = input_tensors.at(0).buffer();
        auto src_buffer_b = input_tensors.at(1).buffer();
        auto bias_tensor = optional_input_tensors.at(0);

        auto dst_buffer = output_tensors.at(0).buffer();

        bool src0_sharded = input_tensors.at(0).memory_config().is_sharded();
        bool out_sharded = output_tensors.at(0).memory_config().is_sharded();

        for (uint32_t i = 0; i < num_cores; i++) {
            uint32_t core_idx_x = i % num_cores_c;
            uint32_t core_idx_y = i / num_cores_c;
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

            // in0 sender and in1 sender
            if (core_idx_x == 0 and core_idx_y == 0) {
                {
                    auto reader_kernel_id = reader_kernel_ids.at(i);
                    auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = src_buffer_a->address();
                    SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
                }

                {
                    auto writer_kernel_id = writer_kernel_ids.at(i);
                    auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = src_buffer_b->address();
                    runtime_args[4] = dst_buffer->address();
                    if (bias_tensor.has_value()) {
                        runtime_args[14] = bias_tensor.value().buffer()->address();
                    }
                    SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
                }
            }
            // in0 receiver and in1 sender
            else if (!(core_idx_x == 0 and core_idx_y == 0)) {
                {
                    auto reader_kernel_id = reader_kernel_ids.at(i);
                    auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = src_buffer_a->address();
                    SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
                }

                {
                    auto writer_kernel_id = writer_kernel_ids.at(i);
                    auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[1] = dst_buffer->address();
                    SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
                }
            }
        }

        if (src0_sharded) {
            auto& src0_cb_config = GetCircularBufferConfig(program, cb_src0);
            src0_cb_config.set_globally_allocated_address(src_buffer_a->address());
        }

        if (out_sharded) {
            auto& output_cb_config = GetCircularBufferConfig(program, cb_output);
            output_cb_config.set_globally_allocated_address(dst_buffer->address());
        }
    };
    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}

namespace tt {

namespace tt_metal {


operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized_(const Tensor &a, const Tensor &b, const std::optional<const Tensor> bias, Tensor& output, bool bcast_batch, CoreCoord compute_with_storage_grid_size, tt::tt_metal::DataType output_dtype, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, std::optional<UnaryWithParam> fused_activation, bool mcast_in0) {

    const auto& ashape = a.shape(), bshape = b.shape();

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype()); // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype()); // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output_dtype); // output

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b; // bias; doesn't matter if bias=nullptr
    if (bias.has_value()) {
        auto& c = bias.value();
        TT_ASSERT(c.storage_type() == StorageType::DEVICE);
        TT_ASSERT(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_ASSERT(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

        bias_buffer = c.buffer();

        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    tt_metal::Device *device = a.device();

    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    tt_metal::Buffer *in0_buffer = a.buffer();
    tt_metal::Buffer *in1_buffer = b.buffer();
    if (bcast_batch)
        TT_ASSERT(bshape[0]*bshape[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    else {
        // same condition as above, different message
        TT_ASSERT(ashape[1] == bshape[1] && ashape[0] == bshape[0]
            && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    }
    TT_ASSERT(in0_buffer->size() % in0_single_tile_size == 0);
    TT_ASSERT(in1_buffer->size() % in1_single_tile_size == 0);

    TT_ASSERT(ashape[3] == bshape[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(ashape[2] % TILE_HEIGHT == 0);
    TT_ASSERT(ashape[3] % TILE_WIDTH == 0);
    TT_ASSERT(bshape[2] % TILE_HEIGHT == 0);
    TT_ASSERT(bshape[3] % TILE_WIDTH == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Pads matmul input dims to 512 x 512 multiples (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = ashape[0]*ashape[1];
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = bshape[3]/TILE_WIDTH;


    if (fuse_batch) {
        Mt = B * Mt;
        B = 1;
    }
    TT_ASSERT(Kt % in0_block_w == 0);

    // This should allocate a DRAM buffer on the device
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    uint32_t num_blocks_y = (Mt - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (Nt - 1) / per_core_N + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Buffer *out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    std::optional<uint32_t> a_addr = std::nullopt;
    std::optional<uint32_t> out_addr = std::nullopt;
    if (a.memory_config().is_sharded()) {
        a_addr = a.buffer()->address();
    }
    if (output.memory_config().is_sharded()) {
        out_addr = output.buffer()->address();
    }

    if (mcast_in0) {
        return reuse_mcast_1d_optimized_helpers::create_program_mcast_in0(
            device,
            math_fidelity,
            compute_with_storage_grid_size,
            B, Mt, Nt, Kt,
            bcast_batch,
            in0_block_w,
            out_subblock_h, out_subblock_w,
            per_core_M, per_core_N,
            fused_activation,
            in0_buffer, in1_buffer, bias_buffer, out_buffer,
            in0_data_format, in1_data_format, bias_data_format, output_data_format
        );
    } else {
        return reuse_mcast_1d_optimized_helpers::create_program_mcast_in1(
            device,
            math_fidelity,
            compute_with_storage_grid_size,
            B, Mt, Nt, Kt,
            bcast_batch,
            in0_block_w,
            out_subblock_h, out_subblock_w,
            per_core_M, per_core_N,
            fused_activation,
            in0_buffer, in1_buffer, bias_buffer, out_buffer,
            in0_data_format, in1_data_format, bias_data_format, output_data_format,
            a_addr, out_addr
        );
    }
}

operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized(const Tensor& a, const Tensor& b, const std::optional<const Tensor> bias, Tensor& output_tensor, CoreCoord compute_with_storage_grid_size, tt::tt_metal::DataType output_dtype, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, std::optional<UnaryWithParam> fused_activation, bool mcast_in0) {
    return matmul_multi_core_reuse_mcast_1d_optimized_(a, b, bias, output_tensor, true, compute_with_storage_grid_size, output_dtype, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch, fused_activation, mcast_in0);
}

}  // namespace tt_metal

}  // namespace tt
