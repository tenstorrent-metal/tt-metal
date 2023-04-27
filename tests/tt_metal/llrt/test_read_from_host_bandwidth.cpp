#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>
#include <iomanip>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"
// #include "tt_gdb/tt_gdb.hpp"

#include "llrt.hpp"
#include "common/bfloat16.hpp"
#include "test_libs/tiles.hpp"

#include "tt_metal/hostdevcommon/profiler_common.h"


using tt::llrt::CircularBufferConfigVec;

void dumpDeviceResultToFile(
        int chip_id,
        int core_x,
        int core_y,
        std::string hart_name,
        uint64_t timestamp,
        uint32_t timer_id,
        bool device_new_log){

    #define DEVICE_SIDE_LOG "profile_log_device.csv"

    std::filesystem::path output_dir = std::filesystem::path("tt_metal/tools/profiler/logs");
    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::ofstream log_file;
    if (device_new_log)
    {
        log_file.open(log_path);
        log_file << "Chip clock is at 1.2 GHz" << std::endl;
        log_file << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset]" << std::endl;
        device_new_log = false;
    }
    else
    {
        log_file.open(log_path, std::ios_base::app);
    }

    constexpr int DRAM_ROW = 6;
    if (core_y > DRAM_ROW){
       core_y = core_y - 2;
    }
    else{
       core_y--;
    }
    core_x--;

    log_file << chip_id << ", " << core_x << ", " << core_y << ", " << hart_name << ", ";
    log_file << timer_id << ", ";
    log_file << timestamp;
    log_file << std::endl;
    log_file.close();
}

void readRiscProfilerResults(
        tt_cluster* cluster,
        int pcie_slot,
        const tt_xy_pair &worker_core,
        std::string risc_name,
        int risc_print_buffer_addr){

    vector<std::uint32_t> profile_buffer;
    uint32_t end_index;
    uint32_t dropped_marker_counter;

    profile_buffer = tt::llrt::read_hex_vec_from_core(
            cluster,
            pcie_slot,
            worker_core,
            risc_print_buffer_addr,
            PRINT_BUFFER_SIZE);

    end_index = profile_buffer[kernel_profiler::BUFFER_END_INDEX];
    TT_ASSERT (end_index < (PRINT_BUFFER_SIZE/sizeof(uint32_t)));

    bool new_log = true;
    for (int i = kernel_profiler::MARKER_DATA_START; i < end_index; i+=kernel_profiler::TIMER_DATA_UINT32_SIZE) {
        dumpDeviceResultToFile(
                pcie_slot,
                worker_core.x,
                worker_core.y,
                risc_name,
                (uint64_t(profile_buffer[i+kernel_profiler::TIMER_VAL_H]) << 32) | profile_buffer[i+kernel_profiler::TIMER_VAL_L],
                profile_buffer[i+kernel_profiler::TIMER_ID],
                new_log);
        new_log = false;
    }
}

bool run_data_copy_multi_tile(tt_cluster* cluster, int chip_id, const tt_xy_pair& core, int num_tiles) {

    std::uint32_t single_tile_size = 2 * 1024;
    std::uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    std::uint32_t host_buffer_src_addr = 0;
    std::uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
    int dram_dst_channel_id = 0;
    log_info(tt::LogVerif, "num_tiles = {}", num_tiles);
    log_info(tt::LogVerif, "single_tile_size = {} B", single_tile_size);
    log_info(tt::LogVerif, "dram_bufer_size = {} B", dram_buffer_size);

    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});

    tt_xy_pair pcie_core_coordinates = {0, 4};

    // BufferConfigVec -- common across all kernels, so written once to the core
    CircularBufferConfigVec circular_buffer_config_vec = tt::llrt::create_circular_buffer_config_vector();

    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 0, 200*1024, 384*single_tile_size, 384);
    // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 16, 990*1024, 1*single_tile_size, 1);

    // buffer_config_vec written in one-shot
    tt::llrt::write_circular_buffer_config_vector_to_core(cluster, chip_id, core, circular_buffer_config_vec);

    tt_xy_pair dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_dst_channel_id);


    // NCRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        { host_buffer_src_addr, (std::uint32_t)pcie_core_coordinates.x, (std::uint32_t)pcie_core_coordinates.y, (std::uint32_t)num_tiles },
        NCRISC_L1_ARG_BASE);


    // BRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        { dram_buffer_dst_addr, (std::uint32_t)dram_dst_noc_xy.x, (std::uint32_t)dram_dst_noc_xy.y, (std::uint32_t)num_tiles},
        BRISC_L1_ARG_BASE);

    // Note: TRISC 0/1/2 kernel args are hard-coded

    TT_ASSERT(dram_buffer_size % sizeof(std::uint32_t) == 0);

    // Write tiles sequentially to DRAM
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100, tt::tiles_test::get_seed_from_systime());

    // Instead of writing DRAM vec, we write to host memory and have the device pull from host
    cluster->write_sysmem_vec(src_vec, 0, 0);

    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});

    readRiscProfilerResults(cluster, 0, core, "NCRISC", PRINT_BUFFER_NC);

    std::vector<std::uint32_t> dst_vec;
    cluster->read_dram_vec(dst_vec, tt_target_dram{chip_id, dram_dst_channel_id, 0}, dram_buffer_dst_addr, dram_buffer_size);

    bool pass = (dst_vec == src_vec);

    return pass;
}

int main(int argc, char** argv)
{
    bool pass = true;

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
    const std::string sdesc_file = get_soc_description_file(arch, target_type);

    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        string op = "datacopy_op";
        string op_path = "built_kernels/" + op;

        int chip_id = 0;
        const tt_xy_pair core = {1, 1};

        pass = tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/brisc/brisc.hex", chip_id, core, 0); // brisc
        pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/ncrisc/ncrisc.hex", chip_id, core, 1); // ncrisc

        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread0/tensix_thread0.hex", chip_id, core, 0); // trisc0
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread1/tensix_thread1.hex", chip_id, core, 1); // trisc1
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread2/tensix_thread2.hex", chip_id, core, 2); // trisc2

        if (pass) {
            const vector<tt_xy_pair> cores = {core};
            const vector<string> ops = {op};

            // tt_gdb::tt_gdb(cluster, chip_id, cores, ops);
            pass &= run_data_copy_multi_tile(cluster, chip_id, core, 384); // must match the value in test_compile_datacopy!
        }

        cluster->close_device();
        delete cluster;

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(tt::LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(tt::LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        log_fatal(tt::LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
