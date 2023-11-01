// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//#include "tt_metal/include/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "common/bfloat16.hpp"
#include "common/test_tiles.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/common/work_split.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;


template <typename T>
std::vector<T> tilize(std::vector<T> data, int rows, int cols) {
    TT_ASSERT(rows % 32 == 0);
    TT_ASSERT(cols % 32 == 0);
    int num_tiles_r = rows / 32;
    int num_tiles_c = cols / 32;
    std::vector<T> result;
    for(auto r = 0; r < num_tiles_r; r++) {
        for(auto c = 0; c < num_tiles_c; c++) {
            for(auto j = 0; j < 32; j++) { // tile rows
                for(auto i = 0; i < 32; i++) { // tile cols
                    // each row of tiles is 32x32 * num_tiles_c
                    // each row within the row of tiles is cols
                    // each col of tiles is 32
                    // pick row of tiles, pick the row within the tile, pick col tile
                    int index = r * 32 * 32 * num_tiles_c + j * cols + c * 32 + i;
                    result.push_back(data.at(index));
                }
            }
        }
    }
    return convert_to_tile_layout(result);
}


template <typename T>
std::vector<T> untilize(std::vector<T> data, int rows, int cols) {
    int TileWidth = 32;
    int TileHeight = 32;
    TT_ASSERT(rows % TileHeight == 0);
    TT_ASSERT(cols % TileWidth == 0);
    int elements_in_tile = TileHeight*TileWidth;
    int num_tiles_r = rows / TileHeight;
    int num_tiles_c = cols / TileWidth;
    std::vector<T> result;
    for(auto r = 0; r < num_tiles_r; r++) {
        for(auto i = 0; i < TileHeight; i++) {
            for(auto c = 0; c < num_tiles_c; c++) {
                int offset = r * elements_in_tile * num_tiles_c + c * elements_in_tile + i * TileWidth;
                for(auto j = 0; j < TileWidth; j++) {
                    result.push_back(data.at(offset + j));
                }
            }
        }
    }
    return convert_to_flat_layout(result);
}


void golden_matmul(vector<bfloat16> a, vector<bfloat16> b, vector<uint32_t>& output,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;

    //vector<float> c_f(M * N, 0);
    float c_f;
    float float_tmp;
    vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            idx_c = j+ (i * N);
            //c_f.at(idx_c) = 0;
            idx_a = i * K;
            idx_b = j;
            c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                //c_f.at(idx_c) += a[idx_a].to_float() * b[idx_b].to_float();
                float_tmp = a[idx_a].to_float() * b[idx_b].to_float();
                // uint32_t* int_tmp = (uint32_t*) &float_tmp;
                // *int_tmp &= 0xffff0000 ;
                c_f += float_tmp;
                idx_a += 1;
                idx_b += K;
            }
            c_bf.at(idx_c) = bfloat16(c_f);
            //if (idx_c < 128) {
            //    cout << "GG " << c_f << " .. " << c_bf.at(idx_c) << endl;
            //}
            //output[idx_c] = (uint32_t)c_bf.to_uint16() | ((uint32_t)c_bf.to_uint16() << 16);
        }
    }
    output = pack_bfloat16_vec_into_uint32_vec(c_bf);
}

void matmul_multi_core(vector<uint32_t>& a, vector<uint32_t>& b, vector<uint32_t>& output, bool bcast_batch,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B, Device* device) {

    /*
    * Setup program to execute along with its buffers and kernels to use
    */
    CommandQueue& cq = *detail::GLOBAL_CQ;
    Program program{};

    /*
    * Multi-Core prep
    */
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_output_tiles_total = (M * N) / TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

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

    /*
    * Create DRAM Buffers for input and output vectors
    * Writing data from input vectors to source buffers
    */
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    //uint32_t single_tile_size = detail::TileSize(cb_data_format);
    uint32_t single_tile_size = 2 * 1024;

    uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    Buffer src0_dram_buffer = CreateBuffer(device, dram_buffer_A_size, single_tile_size, BufferType::DRAM);
    Buffer src1_dram_buffer = CreateBuffer(device, dram_buffer_B_size, single_tile_size, BufferType::DRAM);
    Buffer dst_dram_buffer = CreateBuffer(device, dram_buffer_C_size, single_tile_size, BufferType::DRAM);
    uint32_t src0_addr = src0_dram_buffer.address();
    uint32_t src1_addr = src1_dram_buffer.address();
    uint32_t dst_addr = dst_dram_buffer.address();

    /*
    * Config of Circular Buffer in the device L1
    * input tiles count is = 2 because it's single tile process, and double-buffer
    */
    uint32_t src0_cb_index = CB::c_in0; //0
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1; // 1
    CircularBufferConfig cb_src1_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    /*
    * Compile time arguments
    */
    bool src0_is_dram = src0_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t) output_cb_index, (uint32_t)dst_is_dram};

    /*
    * Create Kernels (Reader, Writer, Compute)
    */
    auto reader_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    auto writer_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args_group_1 = {
        1, // B
        1, // Mt
        Kt, // Kt
        num_output_tiles_per_core_group_1 // Nt
    }; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

    auto matmul_multi_core_kernel_group_1_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm.cpp",
        core_group_1,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_1}
    );

    if (!core_group_2.ranges().empty()) {
        vector<uint32_t> compute_args_group_2 = {
            1, // B
            1, // Mt
            Kt, // Kt
            num_output_tiles_per_core_group_2 // Nt
        }; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

        auto matmul_multi_core_kernel_group_2_id = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/bmm.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_2}
        );
    }

    /*
    * Kernels - Runtime arguments
    */
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){

        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core;
		if (core_group_1.core_coord_in_core_ranges(core)) {
			num_output_tiles_per_core = num_output_tiles_per_core_group_1;
		} else if (core_group_2.core_coord_in_core_ranges(core)) {
			num_output_tiles_per_core = num_output_tiles_per_core_group_2;
		} else {
			TT_ASSERT(false, "Core not in specified core ranges");
		}
        tt_metal::SetRuntimeArgs(
            program, reader_id, core,
            {src0_addr,
            src1_addr,
            Mt,
            Kt,
            Nt,
            MtKt,
            KtNt,
            B,
            uint32_t(bcast_batch),
            num_tiles_written,
            num_output_tiles_per_core,
            MtNt }
        );
        tt_metal::SetRuntimeArgs(
            program,
            writer_id,
            core,
            {dst_addr,
            num_output_tiles_per_core,
            num_tiles_written }
        );
        num_tiles_written += num_output_tiles_per_core;
    }

    /* Launch program & read in output buffer result into the host vector */
    //LaunchProgram(device, program);
    //ReadFromBuffer(dst_dram_buffer, output);
    //ReadFromBuffer(src0_dram_buffer, output);

    EnqueueWriteBuffer(cq, src0_dram_buffer, a, false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b, false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, output, true);
}


///////////////////////////////////////



int main(int argc, char **argv) {
    bool pass = true;

    try {
        /* Silicon accelerator setup */
        constexpr int device_id = 0;
        Device *device = CreateDevice(device_id);

        /* Create source data */
        constexpr uint32_t M = 640;  // user-defined
        constexpr uint32_t N = 640;  // user-defined
        constexpr uint32_t K = 640;  // user-defined
        constexpr uint32_t B = 1;  // user-defined

        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Nt = N / TILE_WIDTH;

        constexpr uint32_t single_tile_size = 2 * 1024;
        uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B

        /* input vectors */
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_A_size, 1, 123, -0.4);
        std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(dram_buffer_B_size, 1, 12522, -0.2);

        //std::vector<uint32_t> src0_vec = create_arange_vector_of_bfloat16(dram_buffer_A_size, false);
        //std::vector<uint32_t> src1_vec = pack_bfloat16_vec_into_uint32_vec(create_identity_matrix(K, N, K));

        /* Input vector tilizing */
        std::vector<uint32_t> tilized_src0_vec = pack_bfloat16_vec_into_uint32_vec(tilize(unpack_uint32_vec_into_bfloat16_vec(src0_vec), M, K));
        std::vector<uint32_t> tilized_src1_vec = pack_bfloat16_vec_into_uint32_vec(tilize(unpack_uint32_vec_into_bfloat16_vec(src1_vec), K, N));

        cout << "-input size- " << src0_vec.size() << " -- " << src1_vec.size() << endl;


        cout << "----orig input 0--" << endl;
        for (int i = 0; i < src0_vec.size(); i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(src0_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();

            if (i % 1280 == 0){
                cout << "-- " << i << " -- " << a1<< "  " << a2  << "---" << src0_vec.at(i) << endl;
            }
        }
        /*
        cout << "----orig input 1--" << endl;
        for (int i = 0; i < 512; i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(src1_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            cout << "-- " << i << " -- " << a1<< "  " << a2  << "---" << src1_vec.at(i) << endl;
        }
        */

        /*
        cout << "----tiled input--" << endl;
        for (int i = 0; i < src0_vec.size(); i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(tilized_src0_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            if (i % 1280 == 0){
                cout << "-- " << i << " -- " << a1<< "  " << a2  << "---" << tilized_src0_vec.at(i) << endl;
            }
        }
        */

        /* Calling the MatMul host program. Read in result into a host vector */
        vector<uint32_t> result_vec;
        matmul_multi_core(tilized_src0_vec, tilized_src1_vec, result_vec, false, M, N, K, B, device);

        cout << "----metal--" << endl;
        cout << result_vec.size() << endl;
        for (int i = 0; i < result_vec.size(); i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(result_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            if (i % 1280 == 0){
                cout << "-- " << i << " -- " << a1<< "  " << a2  << "---" << result_vec.at(i) << endl;
            }
        }

        /*
        vector<uint32_t> result_vec_untilized = pack_bfloat16_vec_into_uint32_vec(untilize(unpack_uint32_vec_into_bfloat16_vec(result_vec), M, N));
        cout << "----metal_untilized--" << endl;
        cout << result_vec.size() << endl;
        for (int i = 0; i < result_vec.size(); i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(result_vec_untilized.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            if (i % 1280 == 0){
                cout << "-- " << i << " -- " << a1<< "  " << a2  << "---" << result_vec_untilized.at(i) << endl;
            }
        }
        */

        /* Golden Matmul running on CPU (Float)*/
        vector<uint32_t> golden_vec;
        golden_matmul(unpack_uint32_vec_into_bfloat16_vec(src0_vec), unpack_uint32_vec_into_bfloat16_vec(src1_vec), golden_vec, M, N, K, B);
        vector<uint32_t>  golden_vec_tilized = pack_bfloat16_vec_into_uint32_vec(tilize(unpack_uint32_vec_into_bfloat16_vec(golden_vec), M, N));

        cout << "----golden--" << endl;
        cout << golden_vec.size() << endl;
        for (int i = 0; i < golden_vec.size(); i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(golden_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            if (i % 1280 == 0){
                cout << "-- " << i << " -- " << a1 << "  " << a2 << "---" << golden_vec.at(i) << endl;
            }
        }

        /* Comparison: Golden vs. METAL Matmul*/
        constexpr float abs_tolerance = 0.01f;
        constexpr float rel_tolerance = 0.001f;
        std::function<bool(const float, const float)> comparison_function = [](const float a, const float b) {
            return is_close(a, b, rel_tolerance, abs_tolerance);
        };

        float calc_pcc = packed_uint32_t_vector_pcc(golden_vec_tilized, result_vec);
        cout << "PCC= " << calc_pcc << endl;

        float pearson = packed_uint32_t_vector_pcc_v2(golden_vec_tilized, result_vec);
        cout << "PCC_v2= " << pearson << endl;

        //pass &= packed_uint32_t_vector_comparison(golden_vec, result_vec_untilized, comparison_function);
        pass &= packed_uint32_t_vector_comparison(golden_vec_tilized, result_vec, comparison_function);

        pass &= CloseDevice(device);

    } catch (const std::exception &e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        tt::log_fatal(tt::LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
