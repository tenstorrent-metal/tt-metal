// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt;
using namespace tt::tt_metal;

/*
* 1. Host creates two vectors of data.
* 2. Device eltwise adds them together.
* 3. Intermediate result read back to host.
* 4. Create another vector and send vectors to input DRAMs again.
* 5. Device eltwise muls them together.
* 6. Read result back and compare to golden.
* */

/*
 * We need to copy the types of the compute kernel arguments to use them host-
 * side.
 */

struct BinaryOpType {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const auto all() { return magic_enum::enum_values<Enum>(); }
};

std::map<string, string> get_defines(BinaryOpType::Enum op_type){
    std::map<string, string> defines;
    // TODO(AP): remove duplication
    string op_name, op_code;
    switch (op_type) {
        case BinaryOpType::ADD: op_name = "add_tiles"; op_code = "0"; break;
        case BinaryOpType::SUB: op_name = "sub_tiles"; op_code = "1"; break;
        case BinaryOpType::MUL: op_name = "mul_tiles"; op_code = "2"; break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    defines["ELTWISE_OP"] = op_name.c_str();
    defines["ELTWISE_OP_CODE"] = op_code.c_str();
    return defines;
}

int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    tt::log_assert(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {
        /*
        * Silicon accelerator setup
        */
        constexpr int device_id = 0;
        Device *device =
            CreateDevice(device_id);



        /*
        * Setup program to execute along with its buffers and kernels to use
        */
        Program program = Program();

        constexpr CoreCoord core = {0, 0};

        constexpr uint32_t single_tile_size = 2 * 1024;
        constexpr uint32_t num_tiles = 2048;
        constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        Buffer src0_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, BufferType::DRAM);
        const uint32_t dram_buffer_src0_addr = src0_dram_buffer.address();
        Buffer src1_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, BufferType::DRAM);
        const uint32_t dram_buffer_src1_addr = src1_dram_buffer.address();
        Buffer dst_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, BufferType::DRAM);
        const uint32_t dram_buffer_dst_addr = dst_dram_buffer.address();

        /*
         * Use circular buffers to set input and output buffers that the
         * compute engine will use.
         */
        constexpr uint32_t src0_cb_index = CB::c_in0;
        constexpr uint32_t num_input_tiles = 2;
        CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, single_tile_size);
        CircularBufferID cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        constexpr uint32_t src1_cb_index = CB::c_in1;
        CircularBufferConfig cb_src1_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, single_tile_size);
        CircularBufferID cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        constexpr uint32_t output_cb_index = CB::c_out0;
        constexpr uint32_t num_output_tiles = 2;
        CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}}).set_page_size(output_cb_index, single_tile_size);
        CircularBufferID cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        /*
         * Specify data movement kernels for reading/writing data to/from
         * DRAM.
         */
        KernelID binary_reader_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        KernelID unary_writer_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        /*
         * Set the parameters that the compute kernel will use.
         */
        vector<uint32_t> compute_kernel_args = {
        };

        constexpr bool fp32_dest_acc_en = false;
        constexpr bool math_approx_mode = false;

        /*
         * Use the add_tiles operation available in the eltwise_binary
         * compute kernel.
         */
        KernelID eltwise_binary_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args,
                .defines = get_defines(BinaryOpType::ADD)
            }
        );

        /*
         * Create source data and write to DRAM.
         */
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 1, std::chrono::system_clock::now().time_since_epoch().count());

        WriteToBuffer(src0_dram_buffer, src0_vec);

        constexpr float val_to_add = -1.0f;
        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, val_to_add);

        WriteToBuffer(src1_dram_buffer, src1_vec);

        /*
         * Configure program and runtime kernel arguments, then execute.
         */


        SetRuntimeArgs(
            program,
            binary_reader_kernel_id,
            core,
            {
                src0_dram_buffer.address(),
                static_cast<uint32_t>(src0_dram_buffer.noc_coordinates().x),
                static_cast<uint32_t>(src0_dram_buffer.noc_coordinates().y),
                num_tiles,
                src1_dram_buffer.address(),
                static_cast<uint32_t>(src1_dram_buffer.noc_coordinates().x),
                static_cast<uint32_t>(src1_dram_buffer.noc_coordinates().y),
                num_tiles,
                0
            }
        );


        SetRuntimeArgs(
            program,
            eltwise_binary_kernel_id,
            core,
            {
                num_tiles, 1
            }
        );

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_dram_buffer.address(),
                static_cast<uint32_t>(dst_dram_buffer.noc_coordinates().x),
                static_cast<uint32_t>(dst_dram_buffer.noc_coordinates().y),
                num_tiles
            }
        );



        LaunchProgram(device, program);
        /*
         * Read in result into a host vector.
         */
        std::vector<uint32_t> result_vec;
        ReadFromBuffer(dst_dram_buffer, result_vec);

        /*
         * Move src data back into DRAM src buffer 0 to do another eltwise calculation
         */
        Program program_mul = Program();

        /*
         * Because we're using a new program, we must redeclare all the
         * circular buffers and kernels.
         */
        cb_src0 = tt_metal::CreateCircularBuffer(program_mul, core, cb_src0_config);
        cb_src1 = tt_metal::CreateCircularBuffer(program_mul, core, cb_src1_config);
        cb_output = tt_metal::CreateCircularBuffer(program_mul, core, cb_output_config);

        binary_reader_kernel_id = CreateKernel(
            program_mul,
            "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        unary_writer_kernel_id = CreateKernel(
            program_mul,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        /*
         * But now let's do an eltwise mul!
         */
        eltwise_binary_kernel_id = CreateKernel(
            program_mul,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args,
                .defines = get_defines(BinaryOpType::MUL)
            }
        );

        /*
         * Send new input data.
         */
        WriteToBuffer(src0_dram_buffer, result_vec);

        constexpr float val_to_mul = 2.0f;
        src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, val_to_mul);

        WriteToBuffer(src1_dram_buffer, src1_vec);



        /*
         * Configure program and runtime kernel arguments.
         */
        SetRuntimeArgs(
            program_mul,
            binary_reader_kernel_id,
            core,
            {
                src0_dram_buffer.address(),
                static_cast<uint32_t>(src0_dram_buffer.noc_coordinates().x),
                static_cast<uint32_t>(src0_dram_buffer.noc_coordinates().y),
                num_tiles,
                src1_dram_buffer.address(),
                static_cast<uint32_t>(src1_dram_buffer.noc_coordinates().x),
                static_cast<uint32_t>(src1_dram_buffer.noc_coordinates().y),
                num_tiles,
                0
            }
        );

        SetRuntimeArgs(
            program_mul,
            eltwise_binary_kernel_id,
            core,
            {
                num_tiles, 1
            }
        );

        SetRuntimeArgs(
            program_mul,
            unary_writer_kernel_id,
            core,
            {
                dst_dram_buffer.address(),
                static_cast<uint32_t>(dst_dram_buffer.noc_coordinates().x),
                static_cast<uint32_t>(dst_dram_buffer.noc_coordinates().y),
                num_tiles
            }
        );


        /*
         * Execute.
         */

        LaunchProgram(device, program_mul);

        /*
         * Read the result and compare to a golden result. Record pass/fail
         * and teardown.
         */
        ReadFromBuffer(dst_dram_buffer, result_vec);

        std::function<bfloat16(const bfloat16 &)> transform_to_golden = [](const bfloat16 &a) {
            return bfloat16((a.to_float() + val_to_add) * val_to_mul);
        };
        std::vector<uint32_t> golden_vec = pack_bfloat16_vec_into_uint32_vec(unpack_uint32_vec_into_bfloat16_vec(src0_vec, transform_to_golden));

        constexpr float abs_tolerance = 0.01f;
        constexpr float rel_tolerance = 0.001f;
        std::function<bool(const float, const float)> comparison_function = [](const float a, const float b) {
            return is_close(a, b, rel_tolerance, abs_tolerance);
        };

        pass &= packed_uint32_t_vector_comparison(golden_vec, result_vec, comparison_function);

        CloseDevice(device);

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
