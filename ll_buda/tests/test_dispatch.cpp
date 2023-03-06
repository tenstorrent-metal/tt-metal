#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
// #include "tt_gdb/tt_gdb.hpp"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

namespace unary_datacopy {
//#include "hlks/eltwise_copy.cpp"
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_tile_cnt;
};
}

void ConfigureDeviceWithDispatchCore(ll_buda::Device *device, ll_buda::Program *program, tt_xy_pair dispatch_core, int chip_id) {
    /*
        We only want to send the dispatch kernel, the dispatch will itself be
        responsible for sending the rest, including the blanks
    */

    // Send blanks to all columns except first, since only testing single col
    // dispatch
    vector<tt_xy_pair> blank_cores_not_in_first_col;
    for (tt_xy_pair p: device->cluster()->get_soc_desc(chip_id).workers) {
        if (p.y != 0) {
            blank_cores_not_in_first_col.push_back(p);
        }
    }
    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(
        device->cluster(), chip_id, llrt::TensixRiscsOptions::ALL_RISCS,
        blank_cores_not_in_first_col);

    // Send dispatch kernel to last core of first column
    ll_buda::KernelGroup dispatch_kg = program->core_to_kernel_group().at(dispatch_core);
    auto dispatch_physical_core = device->worker_core_from_logical_core(dispatch_core);
    llrt::disable_ncrisc(device->cluster(), chip_id, dispatch_physical_core);
    llrt::disable_triscs(device->cluster(), chip_id, dispatch_physical_core);

    constexpr static uint32_t INVALID = 0x4321; // PROF_BEGIN("WRITE_HEX")
    uint32_t stream_register_address = STREAM_REG_ADDR(0, 24);
    llrt::write_hex_vec_to_core(device->cluster(), chip_id, dispatch_physical_core, {INVALID}, stream_register_address); // PROF_END("WRITE_HEX")

    dispatch_kg.riscv_1->configure(device, dispatch_core);
    dispatch_kg.riscv_0->configure(device, dispatch_core);
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        tt_xy_pair data_copy_core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src_addr = 0;
        int dram_src_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src_channel_id, dram_buffer_size, dram_buffer_src_addr);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);

        auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = 8;
        auto cb_src0 = ll_buda::CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            data_copy_core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 300 * 1024;
        uint32_t num_output_tiles = 1;
        auto cb_output = ll_buda::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            data_copy_core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        tt_xy_pair dispatch_core = {11, 0};
        // Dispatch kernels
        auto dispatch_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dispatch/dispatch.cpp",
            dispatch_core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto blank_brisc_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/blank.cpp",
            dispatch_core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        // Data copy kernels
        auto unary_reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_unary_push_4.cpp",
            data_copy_core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_unary.cpp",
            data_copy_core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        void *hlk_args = new unary_datacopy::hlk_args_t{
            .per_core_tile_cnt = (int) num_tiles,
        };
        ll_buda::ComputeKernelArgs *eltwise_unary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(data_copy_core, hlk_args, sizeof(unary_datacopy::hlk_args_t));

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_unary_kernel = ll_buda::CreateComputeKernel(
            program,
            "kernels/compute/eltwise_copy.cpp",
            data_copy_core,
            eltwise_unary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);
        return 1;

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        pass &= ll_buda::WriteToDeviceDRAM(src_dram_buffer, src_vec);

        ConfigureDeviceWithDispatchCore(device, program, dispatch_core, pci_express_slot);


        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        pass &= (src_vec == result_vec);

        pass &= ll_buda::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
