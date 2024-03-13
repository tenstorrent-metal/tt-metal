// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"
#include <algorithm>
#include "tt_metal/common/tilize_untilize.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;


void golden_matmul(vector<bfloat16>& a, vector<bfloat16>& b, vector<bfloat16>& output,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;

    float c_f;
    float float_tmp;
    vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            idx_c = j+ (i * N);
            idx_a = i * K;
            idx_b = j;
            c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                float_tmp = a[idx_a].to_float() * b[idx_b].to_float();
                c_f += float_tmp;
                idx_a += 1;
                idx_b += K;
            }
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}

void matmul_multicore_reuse_mcast(vector<bfloat16>& a, vector<bfloat16>& b, vector<bfloat16>& output, bool bcast_batch,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B, Device* device) {

    /*
    * Setup program to execute along with its buffers and kernels to use
    * Core range is just single core
    */
    CommandQueue& cq = device->command_queue();
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);
    //uint32_t single_tile_size = 2 * 1024;

    ////////// NEW STUFF //////////
                // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
                bool fp32_dest_acc_en = true;
                bool packer_l1_acc = true;

                tt::DataFormat in0_data_format = tt::DataFormat::Float32;
                tt::DataFormat in1_data_format = tt::DataFormat::Float32;
                tt::DataFormat bias_data_format = tt::DataFormat::Float32;
                tt::DataFormat output_data_format = tt::DataFormat::Float32;
                tt::DataFormat interm0_data_format = tt::DataFormat::Float32;

                interm0_data_format = packer_l1_acc ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b) : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

                uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
                uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
                uint32_t bias_single_tile_size = tt_metal::detail::TileSize(bias_data_format);
                uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);
                uint32_t interm0_single_tile_size = tt_metal::detail::TileSize(interm0_data_format);
    ////////// END //////////

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    /*
    * EXtracting Matrix dimensions from input/output vectors
    */
    // C = A*B
    // MN = MK*KN
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])2
    uint32_t in0_block_w = 2;
    //uint32_t out_subblock_h = 4;
    //uint32_t out_subblock_w = 2;
    //uint32_t per_core_M = 16;
    //uint32_t per_core_N = 16;

    uint32_t num_blocks = (Kt/in0_block_w);

    // DEFINE B HERE, TO HELP DEFINE BUFFERS
    uint32_t B_size = 1;

    // Get large matmul params
    auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
    uint32_t per_core_M = std::get<0>(matmul_params);
    uint32_t per_core_N = std::get<1>(matmul_params);
    uint32_t out_subblock_h = std::get<2>(matmul_params);
    uint32_t out_subblock_w = std::get<3>(matmul_params);

    log_info(tt::LogVerif, " -- Metalium Core Sizing --");
    log_info(tt::LogVerif, " -- per_core_M= {} -- per_core_N= {} -- out_subblock_h= {} -- out_subblock_w= {} --", per_core_M, per_core_N, out_subblock_h, out_subblock_w);

    TT_ASSERT(Mt % per_core_M == 0);
    TT_ASSERT(Nt % per_core_N == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    // uint32_t in0_block_tiles = per_core_M * in0_block_w;
    // uint32_t in0_CB_tiles = in0_block_tiles;
    // if (B * num_blocks > 1) {
    //     in0_CB_tiles = in0_CB_tiles * 2; // double buffer
    // }
    // uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
    // uint32_t in1_block_tiles = per_core_N * in0_block_w;
    // uint32_t in1_CB_tiles = in1_block_tiles * 2; // double buffer
    // uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
    // uint32_t out_block_tiles = per_core_M * per_core_N;
    // uint32_t out_CB_tiles = out_block_tiles; // No double buffer
    // uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    ////////// NEW STUFF //////////
                    uint32_t in0_block_tiles = per_core_M * in0_block_w;
                    uint32_t in0_CB_tiles = in0_block_tiles;
                    if (B_size * num_blocks > 1) {
                        in0_CB_tiles = in0_CB_tiles * 2; // double buffer
                    }
                    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
                    uint32_t in1_block_tiles = per_core_N * in0_block_w;
                    uint32_t in1_CB_tiles = in1_block_tiles;
                    if (B_size * num_blocks > 1) {
                        in1_CB_tiles = in1_CB_tiles * 2; // double buffer
                    }
                    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

                    uint32_t out_block_tiles = per_core_M * per_core_N;
                    uint32_t out_CB_tiles = out_block_tiles; // No double buffer
                    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
                    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

                    uint32_t in2_block_tiles = per_core_M * in0_block_w;
                    uint32_t in2_CB_tiles = in2_block_tiles;
                    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

                    uint32_t in3_block_tiles = per_core_N;
                    uint32_t in3_CB_tiles = in3_block_tiles; // No double buffer
                    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    ////////// END //////////

    // Compute kernel compile time args
    // MOVED THIS ABOVE, TO HELP DEFINE BUFFERS
    // uint32_t num_blocks = (Kt/in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M/out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N/out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    // USER DEFINED
    bool in0_is_sharded = false;
    bool output_is_sharded = false;

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
        B, // batch
        out_block_tiles // out_block_num_tiles
    };

    // NEW STUFF
    log_info(tt::LogVerif, " -- out_block_tiles= {} --", out_block_tiles);

    /*
    * Multi-Core prep
    */

     ////////// NEW STUFF //////////
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    uint32_t num_cores = num_blocks_total;

    log_info(tt::LogVerif, " -- num_blocks_total= {} --", num_blocks_total);
    log_info(tt::LogVerif, " -- num_blocks_y = {} --", num_blocks_y);
    log_info(tt::LogVerif, " -- num_blocks_x= {} --", num_blocks_x);
    log_info(tt::LogVerif, " -- num_cores_x= {} --", num_cores_x);
    log_info(tt::LogVerif, " -- num_cores_y)= {} --", num_cores_y);

    TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);

    CoreCoord start_core = {0, 0};
    CoreCoord core_range = bmm_op_utils::get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);

    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;
    uint32_t num_mcast_cores = num_cores_c * num_cores_r; // Exclude Sender

    CoreRangeSet all_cores = num_cores_to_corerange_set(num_cores, core_range, true);

    ////////// END //////////

    // CoreRange all_cores(
    //     {(std::size_t)start_core_x, (std::size_t)start_core_y},
    //     {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    // CoreRange left_column(
    //     {(std::size_t)start_core_x, (std::size_t)start_core_y},
    //     {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

    // CoreRange all_except_left_column(
    //     {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
    //     {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    // CoreRange in0_sender_in1_sender(
    //     {(std::size_t)start_core_x, (std::size_t)start_core_y}, {(std::size_t)start_core_x, (std::size_t)start_core_y});

    // CoreRange in0_sender_in1_receiver(
    //     {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
    //     {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

    // CoreRange in0_receiver_in1_sender(
    //     {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
    //     {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y});

    // CoreRange in0_receiver_in1_receiver(
    //     {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
    //     {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});


    CoreRange mcast_sender(
        {(std::size_t) start_core_x, (std::size_t) start_core_y},
        {(std::size_t) start_core_x, (std::size_t) start_core_y});

    // TODO: Optimize difference of corerangesets
    std::set<CoreRange> mcast_receivers_set;
    for(uint32_t i = 0; i < num_cores; i++) {
        uint32_t core_idx_x = i % num_cores_c;
        uint32_t core_idx_y = i / num_cores_c;
        if (!(core_idx_x == 0 && core_idx_y == 0)) {
            CoreRange core(
                {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y},
                {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y}
            );
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

    //////////////////////////////////////////////////
    /*
    * Create DRAM Buffers for input and output vectors
    * Writing data from input vectors to source buffers
    */

    uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    tt_metal::InterleavedBufferConfig dram_config_A{
                    .device= device,
                    .size = dram_buffer_A_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    tt_metal::InterleavedBufferConfig dram_config_B{
                    .device= device,
                    .size = dram_buffer_B_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    tt_metal::InterleavedBufferConfig dram_config_C{
                    .device= device,
                    .size = dram_buffer_B_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    auto src0_dram_buffer = CreateBuffer(dram_config_A);
    auto src1_dram_buffer = CreateBuffer(dram_config_B);
    auto dst_dram_buffer = CreateBuffer(dram_config_C);
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();

    // /*
    // * Config of Circular Buffer in the device L1
    // * input tiles count is = 2 because it's single tile process, and double-buffer
    // */
    // uint32_t src0_cb_index = CB::c_in0; //0
    // CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
	// 	.set_page_size(src0_cb_index, single_tile_size);
    // auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // uint32_t src1_cb_index = CB::c_in1; // 1
    // CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
	// 	.set_page_size(src1_cb_index, single_tile_size);
    // auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    // uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    // uint32_t interm0_cb_index = 24;
    // std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
    //     {output_cb_index, cb_data_format},
    //     {interm0_cb_index, cb_data_format}
    // };
    // CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
	// 	.set_page_size(output_cb_index, single_tile_size)
    //     .set_page_size(interm0_cb_index, single_tile_size);
    // auto cb_output = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), cb_output_config);


    /////////////

    ////////////////////////////
    /*
    * Compile time arguments
    */
    bool in0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    // HAD TO ADD THIS FOR 1D_SYS
    bool in3_is_dram = true;
    // if (bias_buffer != nullptr) {
    //     in3_is_dram = bias_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    // }
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)in0_is_dram, (uint32_t)in1_is_dram};

    bool out_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    //std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t) output_cb_index, (uint32_t)dst_is_dram};
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)out_is_dram};


    // NEW STUFF.  1D_SYS IMPLEMENTATION

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
            (std::uint32_t)  num_blocks, // num_blocks
            // in0 mcast args
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
            (std::uint32_t)  num_blocks, // num_blocks
            // in1 mcast args
            (std::uint32_t)  in1_mcast_sender_semaphore,
            (std::uint32_t)  in1_mcast_receiver_semaphore,
            (std::uint32_t)  num_cores - 1, // in1_mcast_num_dests
            (std::uint32_t)  num_mcast_cores - 1, // in1_mcast_num_cores
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
    // if (bias_buffer != nullptr) {
    //     in1_sender_writer_compile_time_args.push_back((std::uint32_t)  in3_is_dram);
    //     in1_sender_writer_compile_time_args.push_back((std::uint32_t)  1);
    // }
    std::vector<uint32_t> in1_receiver_writer_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) out_is_dram,

            // READER
            // in1 block args
            (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles
            // in0/in1 common args
            (std::uint32_t)  num_blocks, // num_blocks
            // in1 mcast args
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

    // if (bias_buffer != nullptr) {
    //     in1_receiver_writer_compile_time_args.push_back((std::uint32_t)  per_core_N);
    // }

    std::map<string, string> mm_kernel_defines;
    std::map<string, string> mm_kernel_in0_sender_defines;
    std::map<string, string> mm_kernel_in1_sender_writer_defines;
    std::map<string, string> mm_kernel_in1_receiver_writer_defines;
    // if (bias_buffer != nullptr) {
    //     mm_kernel_defines["FUSE_BIAS"] = "1";
    //     mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
    //     mm_kernel_in1_receiver_writer_defines["FUSE_BIAS"] = "1";
    // }
    // if (fused_activation.has_value()) {
    //     if (fused_activation.value().op_type == UnaryOpType::RELU) {
    //         mm_kernel_defines["PACK_RELU"] = "1";
    //     } else {
    //         mm_kernel_defines.merge(eltwise_unary_op_utils::get_defines(fused_activation.value().op_type, fused_activation.value().param, "ACTIVATION", "i"));
    //     }
    // }
    if (packer_l1_acc) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    if (in0_is_sharded) {
        mm_kernel_in0_sender_defines["IN0_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
        mm_kernel_in1_receiver_writer_defines["OUT_SHARDED"] = "1";
    }

    mm_kernel_in0_sender_defines["SKIP_MCAST"] = "1";

    // // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC in1_noc = detail::GetPreferredNOCForDRAMRead(device->arch());

    auto mm_kernel_in0_sender_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/bmm/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc, .compile_args = in0_sender_compile_time_args, .defines = mm_kernel_in0_sender_defines});

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/bmm/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        mcast_sender,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc, .compile_args = in1_sender_writer_compile_time_args, .defines = mm_kernel_in1_sender_writer_defines});


    auto mm_kernel_in1_receiver_writer_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/bmm/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
        mcast_receivers,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc, .compile_args = in1_receiver_writer_compile_time_args, .defines = mm_kernel_in1_receiver_writer_defines});

    //////////// END /////////////

    /*
    * Create Kernels (Reader, Writer, Compute)
    */
    // Create reader and writer kernels per core group

    // Create compute kernel
    // bool fp32_dest_acc_en = false;
    // Gelu currently has better accuracy when run in approx mode
    // USER DEFINED MATH_APPROX_MODE
    bool math_approx_mode = false;
    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/bmm/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compute_kernel_args, .defines = mm_kernel_defines}
    );

    auto in0_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    //auto in1_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    //auto in1_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    //////////// NEW STUFF /////////////

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
        .set_page_size(src0_cb_index, in0_single_tile_size);
    if (in0_is_sharded) {
        src0_cb_config = src0_cb_config.set_globally_allocated_address(*src0_dram_buffer);
    }
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);
    log_debug(LogOp, "CB {} :: PS = {}, NP = {}, TOTAL = {}", src0_cb_index, in0_single_tile_size, in0_CB_size / in0_single_tile_size, in0_CB_size);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config = tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
        .set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    log_debug(LogOp, "CB {} :: PS = {}, NP = {}, TOTAL = {}", src1_cb_index, in1_single_tile_size, in1_CB_size / in1_single_tile_size, in1_CB_size);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    tt_metal::CircularBufferConfig interm0_cb_config = tt_metal::CircularBufferConfig(0, {{interm0_cb_index, interm0_data_format}});
    tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});

    if (interm0_data_format != output_data_format) {
        // output
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
            {output_cb_index, output_data_format},
        };
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
            .set_page_size(output_cb_index, output_single_tile_size);
        // interm0
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec {
            {interm0_cb_index, interm0_data_format},
        };
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
            .set_page_size(interm0_cb_index, interm0_single_tile_size);

        auto cb_interm0 = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), interm0_cb_config);
        log_debug(LogOp, "CB {} :: PS = {}, NP = {}, TOTAL = {}", interm0_cb_index, interm0_single_tile_size, interm0_CB_size / interm0_single_tile_size, interm0_CB_size);
    } else {
        // share buffer
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
            {output_cb_index, output_data_format},
            {interm0_cb_index, interm0_data_format}
        };
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
            .set_page_size(output_cb_index, output_single_tile_size)
            .set_page_size(interm0_cb_index, interm0_single_tile_size);
    }

    if (output_is_sharded) {
        output_cb_config = output_cb_config.set_globally_allocated_address(*dst_dram_buffer);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    log_debug(LogOp, "CB {} :: PS = {}, NP = {}, TOTAL = {}", output_cb_index, output_single_tile_size, out_CB_size / output_single_tile_size, out_CB_size);

    //////////// END /////////////

    //////////// NEW STUFF /////////////
    // if (bias_buffer != nullptr) {
    //         uint32_t src3_cb_index = 3;
    //         tt_metal::CircularBufferConfig cb_src3_config = tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
    //             .set_page_size(src3_cb_index, bias_single_tile_size);
    //         auto cb_src3 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src3_config);
    //         log_debug(LogOp, "CB {} :: PS = {}, NP = {}, TOTAL = {}", src3_cb_index, bias_single_tile_size, in3_CB_size / bias_single_tile_size, in3_CB_size);
    //     }

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc


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

    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;

    CoreCoord start_core_noc = bottom_right_core_physical;
    CoreCoord end_core_noc = top_left_core_physical;
    if (in1_noc == NOC::NOC_0) {
        std::swap(start_core_noc, end_core_noc);
    }

    for(uint32_t i = 0; i < num_cores; i++) {
        uint32_t core_idx_x = i % num_cores_c;
        uint32_t core_idx_y = i / num_cores_c;
        uint32_t output_idx_x = i / num_blocks_y;
        uint32_t output_idx_y = i % num_blocks_y;
        CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

        // in0 sender and in1 sender
        if (core_idx_x == 0 and core_idx_y == 0) {
            std::vector<uint32_t> mm_in1_sender_writer_args = {
                // READER
                // in1 tensor args
                (std::uint32_t)  src1_dram_buffer->address(),
                (std::uint32_t)  per_core_N * output_idx_x, //in1_tensor_start_tile_id
                // in1 mcast args
                (std::uint32_t)  start_core_noc.x, // in1_mcast_dest_noc_start_x
                (std::uint32_t)  start_core_noc.y, // in1_mcast_dest_noc_start_y
                (std::uint32_t)  end_core_noc.x, // in1_mcast_dest_noc_end_x
                (std::uint32_t)  end_core_noc.y, // in1_mcast_dest_noc_end_y

                // WRITER
                // out tensor args
                (std::uint32_t)  dst_dram_buffer->address(),
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

            // if (bias_buffer != nullptr) {
            //     mm_in1_sender_writer_args.push_back((std::uint32_t)  bias_buffer->address());
            //     mm_in1_sender_writer_args.push_back((std::uint32_t)  per_core_N * output_idx_x); //in3_tensor_start_tile_id
            // }

            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args); // RISCV_1_default
            writer_kernel_ids.push_back(mm_kernel_in1_sender_writer_id);
        }
        // in0 sender and in1 receiver
        else if (!(core_idx_x == 0 and core_idx_y == 0)) {
            std::vector<uint32_t> mm_in1_receiver_writer_args = {
                // READER
                // in1 mcast args
                (std::uint32_t)  top_left_core_physical.x, // in1_mcast_sender_noc_x
                (std::uint32_t)  top_left_core_physical.y, // in1_mcast_sender_noc_y

                // WRITER
                // out tensor args
                (std::uint32_t)  dst_dram_buffer->address(), // out_tensor_addr
                (std::uint32_t)  output_idx_x * per_core_N + output_idx_y * per_core_M * N // out_tensor_start_tile_id
            };

            if (output_idx_y == num_blocks_y - 1) {
                // padding args (WRITER)
                mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                mm_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
                mm_in1_receiver_writer_args.push_back(out_subblock_w);
                mm_in1_receiver_writer_args.push_back(0);
                mm_in1_receiver_writer_args.push_back(0);
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

            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_receiver_writer_id, core, mm_in1_receiver_writer_args); // RISCV_0_default
            writer_kernel_ids.push_back(mm_kernel_in1_receiver_writer_id);
        }
        std::vector<uint32_t> mm_in0_sender_args =  {
            // in0 tensor args
            (std::uint32_t)  src0_dram_buffer->address(),
            (std::uint32_t)  K * per_core_M * output_idx_y, // in0_tensor_start_tile_id
            // in0 mcast args
            (std::uint32_t)  0, // in0_mcast_dest_noc_start_x
            (std::uint32_t)  0, // in0_mcast_dest_noc_start_y
            (std::uint32_t)  0, // in0_mcast_dest_noc_end_x
            (std::uint32_t)  0, // in0_mcast_dest_noc_end_y

            // padding args
            (std::uint32_t) per_core_M // last_block_h
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args); // RISCV_1_default
        reader_kernel_ids.push_back(mm_kernel_in0_sender_id);
    }

    /*
    * Kernels - Runtime arguments
    */
    // std::vector<KernelHandle> reader_kernel_ids;
    // std::vector<KernelHandle> writer_kernel_ids;
    // for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
    //     for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
    //         CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

    //         CoreCoord left_core    = {(std::size_t) start_core_x, (std::size_t) core.y};
    //         CoreCoord left_core_plus_one    = {(std::size_t) start_core_x + 1, (std::size_t) core.y};
    //         CoreCoord right_core   = {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) core.y};
    //         CoreCoord top_core     = {(std::size_t) core.x, (std::size_t) start_core_y};
    //         CoreCoord top_core_plus_one     = {(std::size_t) core.x, (std::size_t) start_core_y + 1};
    //         CoreCoord bottom_core  = {(std::size_t) core.x, (std::size_t) start_core_y + num_cores_r - 1};

    //         auto left_core_physical = device->worker_core_from_logical_core(left_core);
    //         auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
    //         auto right_core_physical = device->worker_core_from_logical_core(right_core);
    //         auto top_core_physical = device->worker_core_from_logical_core(top_core);
    //         auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
    //         auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);

    //         std::vector<uint32_t> mm_reader_args = {
    //             (std::uint32_t)  src0_dram_buffer->address(), // in0_buffer_addr
    //             (std::uint32_t)  Kt * per_core_M * core_idx_y, // in0_buffer_start_tile_id
    //             (std::uint32_t)  1, // in0_buffer_stride_w
    //             (std::uint32_t)  Kt, // in0_buffer_stride_h
    //             (std::uint32_t)  in0_block_w, // in0_buffer_next_block_stride

    //             (std::uint32_t)  in0_block_w, // in0_block_w
    //             (std::uint32_t)  per_core_M, // in0_block_h
    //             (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles

    //             (std::uint32_t)  src1_dram_buffer->address(), // in1_buffer_addr
    //             (std::uint32_t)  per_core_N * core_idx_x, //in1_buffer_start_tile_id
    //             (std::uint32_t)  1, // in1_buffer_stride_w
    //             (std::uint32_t)  Nt, // in1_buffer_stride_h
    //             (std::uint32_t)  in0_block_w * Nt, //in1_buffer_next_block_stride

    //             (std::uint32_t)  per_core_N, // in1_block_w
    //             (std::uint32_t)  in0_block_w, //in1_block_h
    //             (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles

    //             (std::uint32_t)  Kt / in0_block_w, // num_blocks

    //             (std::uint32_t)  right_core_physical.x, // in0_mcast_dest_noc_start_x
    //             (std::uint32_t)  right_core_physical.y, // in0_mcast_dest_noc_start_y
    //             (std::uint32_t)  left_core_plus_one_physical.x, // in0_mcast_dest_noc_end_x
    //             (std::uint32_t)  left_core_plus_one_physical.y, // in0_mcast_dest_noc_end_y
    //             (std::uint32_t)  (num_cores_c - 1), // in0_mcast_num_dests
    //             (std::uint32_t)  left_core_physical.x, // in0_mcast_sender_noc_x
    //             (std::uint32_t)  left_core_physical.y, // in0_mcast_sender_noc_y
    //             (std::uint32_t)  in0_mcast_sender_semaphore,
    //             (std::uint32_t)  in0_mcast_receiver_semaphore,

    //             (std::uint32_t)  bottom_core_physical.x, // in0_mcast_dest_noc_start_x
    //             (std::uint32_t)  bottom_core_physical.y, // in0_mcast_dest_noc_start_y
    //             (std::uint32_t)  top_core_plus_one_physical.x, // in0_mcast_dest_noc_end_x
    //             (std::uint32_t)  top_core_plus_one_physical.y, // in0_mcast_dest_noc_end_y
    //             (std::uint32_t)  (num_cores_r - 1), // in0_mcast_num_dests
    //             (std::uint32_t)  top_core_physical.x, // in0_mcast_sender_noc_x
    //             (std::uint32_t)  top_core_physical.y, // in0_mcast_sender_noc_y
    //             (std::uint32_t)  in1_mcast_sender_semaphore,
    //             (std::uint32_t)  in1_mcast_receiver_semaphore,

    //             (std::uint32_t)  Mt * Kt, // MtKt
    //             (std::uint32_t)  Kt * Nt, // KtNt
    //             (std::uint32_t)  B, // batch
    //             (std::uint32_t)  bcast_batch // bcast_B
    //         };

    //         std::vector<uint32_t> writer_args = {
    //             (std::uint32_t) dst_dram_buffer->address(), // out_buffer_addr
    //             (std::uint32_t) core_idx_x * per_core_N + core_idx_y * per_core_M * Nt, // out_buffer_start_tile_id
    //             (std::uint32_t) 1, // out_buffer_stride_w
    //             (std::uint32_t) Nt,  // out_buffer_stride_h
    //             (std::uint32_t) out_subblock_w, // out_buffer_next_subblock_stride_w
    //             (std::uint32_t) out_subblock_h * Nt, // out_buffer_next_subblock_stride_h

    //             (std::uint32_t) out_subblock_w, // out_subblock_w
    //             (std::uint32_t) out_subblock_h, // out_subblock_h
    //             (std::uint32_t) (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
    //             (std::uint32_t) (per_core_N / out_subblock_w), // out_num_subblocks_w
    //             (std::uint32_t) (per_core_M / out_subblock_h), // out_num_subblocks_h

    //             (std::uint32_t) Mt * Nt, // MtNt
    //             (std::uint32_t) B // batch
    //         };

    //         if(core_idx_x == 0 and core_idx_y == 0) {
    //             tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_sender_in1_sender_id, core, mm_reader_args); // RISCV_0_default
    //             tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args); // RISCV_1_default
    //             reader_kernel_ids.push_back(mm_reader_kernel_in0_sender_in1_sender_id);
    //             writer_kernel_ids.push_back(unary_writer_kernel_noc1_id);
    //         } else if (core_idx_x == 0 and core_idx_y != 0) {
    //             tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_sender_in1_receiver_id, core, mm_reader_args); // RISCV_0_default
    //             tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args); // RISCV_1_default
    //             reader_kernel_ids.push_back(mm_reader_kernel_in0_sender_in1_receiver_id);
    //             writer_kernel_ids.push_back(unary_writer_kernel_noc1_id);
    //         } else if (core_idx_x != 0 and core_idx_y == 0) {
    //             tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_receiver_in1_sender_id, core, mm_reader_args); // RISCV_1_default
    //             tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args); // RISCV_0_default
    //             reader_kernel_ids.push_back(mm_reader_kernel_in0_receiver_in1_sender_id);
    //             writer_kernel_ids.push_back(unary_writer_kernel_noc0_id);
    //         } else {
    //             tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_receiver_in1_receiver_id, core, mm_reader_args); // RISCV_1_default
    //             tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args); // RISCV_0_default
    //             reader_kernel_ids.push_back(mm_reader_kernel_in0_receiver_in1_receiver_id);
    //             writer_kernel_ids.push_back(unary_writer_kernel_noc0_id);
    //         }

    //     }
    // }

    /* Launch program & read in output buffer result into the host vector */

    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
}


///////////////////////////////////////



int main(int argc, char **argv) {
    bool pass = true;

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    try {
        /* Silicon accelerator setup */
        constexpr int device_id = 0;
        Device *device = CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Matmul Parameters Setup
        ////////////////////////////////////////////////////////////////////////////
        // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
        // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])

        /* Create source data */
        constexpr uint32_t M = 832;  // user-defined
        constexpr uint32_t N = 832;  // user-defined
        constexpr uint32_t K = 832;  // user-defined
        constexpr uint32_t B = 1;  // user-defined

        // constexpr uint32_t M = 1600;  // user-defined
        // constexpr uint32_t N = 1600;  // user-defined
        // constexpr uint32_t K = 1600;  // user-defined
        // constexpr uint32_t B = 1;  // user-defined

        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Nt = N / TILE_WIDTH;

        constexpr uint32_t single_tile_size = 2 * 1024;
        uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B

        /* input vectors */
        std::vector<bfloat16> src0_vec = create_random_vector_of_bfloat16_native(dram_buffer_A_size, 1, 123, -0.4);
        std::vector<bfloat16> src1_vec = create_random_vector_of_bfloat16_native(dram_buffer_B_size, 1, 12522, -0.3);

        /* Golden Matmul running on CPU (Float)*/
        vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K, B);

        /* Input vector tilizing */
        tilize(src0_vec, M, K);
        tilize(src1_vec, K, N);

        /* Calling the MatMul host program. Read in result into a host vector */
        vector<bfloat16> result_vec(dram_buffer_C_size/sizeof(bfloat16));
        matmul_multicore_reuse_mcast(src0_vec, src1_vec, result_vec, false, M, N, K, B, device);
        untilize(result_vec, M, N);

        log_info(tt::LogVerif, "Output vector of size {}", result_vec.size());

        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        log_info(tt::LogVerif, "Metalium vs Golden -- PCC = {}", pearson);
        TT_FATAL(pearson > 0.98, "PCC not high enough. Result PCC: {}, Expected PCC: 0.98", pearson);

        pass &= CloseDevice(device);

    } catch (const std::exception &e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
