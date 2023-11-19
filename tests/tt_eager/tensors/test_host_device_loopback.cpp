// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "common/constants.hpp"
#include "tt_numpy/functions.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;


bool test_single_tile_single_dram_bank_loopback(const Device& device) {
    bool pass = true;
    Shape single_tile_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};

    Tensor host_a = tt::numpy::random::random(single_tile_shape).to(Layout::TILE);
    Tensor device_a = host_a.to(device);
    Tensor loopbacked_a = device_a.cpu();
    auto host_a_data = owned_buffer::get_as<bfloat16>(host_a);
    auto loopbacked_a_data = owned_buffer::get_as<bfloat16>(loopbacked_a);
    pass &= host_a_data == loopbacked_a_data;

    return pass;
}

bool test_multi_tile_multi_dram_bank_loopback(const Device& device) {
    bool pass = true;
    Shape multi_tile_shape = {1, 1, 4*TILE_HEIGHT, 3*TILE_WIDTH};

    Tensor host_a = tt::numpy::random::random(multi_tile_shape).to(Layout::TILE);
    Tensor device_a = host_a.to(device);
    Tensor loopbacked_a = device_a.cpu();
    auto host_a_data = owned_buffer::get_as<bfloat16>(host_a);
    auto loopbacked_a_data = owned_buffer::get_as<bfloat16>(loopbacked_a);
    pass &= host_a_data == loopbacked_a_data;
    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        const tt_metal::Device& device = tt_metal::CreateDevice(device_id);



        pass &= test_single_tile_single_dram_bank_loopback(device);

        pass &= test_multi_tile_multi_dram_bank_loopback(device);

        pass &= tt_metal::CloseDevice(device);

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
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}
