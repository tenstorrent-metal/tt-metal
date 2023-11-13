// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// 1. Host writes data to buffer in DRAM
// 2. Host generates DTX transformation node -> Producer : 1D vector sized 64 -> Consumer : 1D vector of same size but flips first 32 elements with last 32 elements
// 2. dram_to_l1_copy_with_address_map kernel on logical core {0, 0} BRISC copies data from buffer in step 1. to buffer in L1
// 3. Host reads from buffer written to in step 2.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    tt::log_assert(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        TransformationNode * node0 = new TransformationNode("producer", 1);
        TransformationNode * node1 = new TransformationNode("tx", 1);
        node0->groups[0]->shape = {64};
        node1->groups[0]->shape = {64};
        node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0}, {31}),  0, new DTXTensor({32},  {63}))  );
        node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({32}, {63}),  0, new DTXTensor({0},  {31}))  );
        DataTransformations * dtx = new DataTransformations();
        dtx->transformations.push_back(node0);
        dtx->transformations.push_back(node1);
        dtx->print();
        pass &= generate_transfer_addresses(dtx);

        // copy transfer addresses into a vector
        std::vector<uint32_t> address_map;
        for(auto transfer : node1->groups[0]->transfers){
            address_map.push_back(transfer->src_address*2); // 2 for bfloat16
            address_map.push_back(transfer->dst_address*2);
            address_map.push_back(transfer->size*2);
        }
        std::cout << "Address Map - " << std::endl;
        for(auto i = 0; i < address_map.size(); i+=3) {
            std::cout << "Source address - " << address_map[i];
            std::cout << ", Destination address - " << address_map[i+1];
            std::cout << ", Size to transfer in bytes - " << address_map[i+2] << std::endl;
        }

        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};

        uint32_t dram_buffer_size = 2 * 64;
        uint32_t address_map_l1_addr = 500 * 1024;

        auto input_dram_buffer = CreateBuffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferStorage::DRAM);
        uint32_t input_dram_buffer_addr = input_dram_buffer.address();

        auto l1_b0 = CreateBuffer(device, dram_buffer_size, dram_buffer_size, tt_metal::BufferStorage::L1);
        uint32_t l1_buffer_addr = l1_b0.address();

        auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();

        auto dram_to_l1_copy_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/dram_to_l1_copy_with_address_map.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto input_vector = unpack_uint32_vec_into_bfloat16_vec(input_vec);
        tt_metal::WriteToBuffer(input_dram_buffer, input_vec);



        tt_metal::detail::WriteToDeviceL1(device, core, address_map_l1_addr, address_map);

        tt_metal::SetRuntimeArgs(
            program,
            dram_to_l1_copy_kernel,
            core,
            {input_dram_buffer_addr,
            (std::uint32_t)input_dram_noc_xy.x,
            (std::uint32_t)input_dram_noc_xy.y,
            l1_buffer_addr,
            address_map_l1_addr,
            (std::uint32_t) address_map.size()});


        tt_metal::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromDeviceL1(device, core, l1_buffer_addr, dram_buffer_size, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto result_vector = unpack_uint32_vec_into_bfloat16_vec(result_vec);

        for(uint32_t i = 0; i < 32; i++) {
            pass &= (input_vector[i] == result_vector[i+32]);
        }
        for(uint32_t i = 32; i < 64; i++) {
            pass &= (input_vector[i] == result_vector[i-32]);
        }

        pass &= tt_metal::CloseDevice(device);;

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
