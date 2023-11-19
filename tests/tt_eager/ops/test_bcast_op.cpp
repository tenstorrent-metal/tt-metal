// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "common/constants.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include <tt_numpy/functions.hpp>

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        const tt_metal::Device& device = tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto shapes = std::vector<Shape>{{1, 1, TILE_HEIGHT, TILE_WIDTH}, {1, 1, TILE_HEIGHT * 2, TILE_WIDTH * 2}, {1, 1, TILE_HEIGHT * 3, TILE_WIDTH * 4}};

        auto run_operations = [&shapes, &device] {
            for (const auto shape : shapes) {
                for (auto bcast_dim: magic_enum::enum_values<BcastOpDim>()) {
                    auto input_shape_a = shape;
                    if (bcast_dim == BcastOpDim::H) {
                        input_shape_a[-1] = 32;
                    }
                    else if (bcast_dim == BcastOpDim::W) {
                        input_shape_a[-2] = 32;
                    }
                    else if (bcast_dim == BcastOpDim::HW) {
                        // do nothing
                    } else {
                        throw std::runtime_error("Unsupported Dim!");
                    }

                    Tensor a = tt::numpy::random::random(input_shape_a).to(Layout::TILE).to(device);
                    Tensor b = tt::numpy::zeros({1, 1, TILE_HEIGHT, TILE_WIDTH}, DataType::BFLOAT16).to(Layout::TILE).to(device);

                    for (auto bcast_math: magic_enum::enum_values<BcastOpMath>()) {
                        Tensor c = bcast(a, b, bcast_math, bcast_dim);
                        Tensor d = c.cpu();

                        ////////////////////////////////////////////////////////////////////////////
                        //                      Validation & Teardown
                        ////////////////////////////////////////////////////////////////////////////
                        Tensor host_a = a.cpu(); // Move tensor a to host to validate
                        //pass &= (host_a.data() == d.data()); // src1 is all 0's
                    }
                }
            }

            {
                Tensor a = tt::numpy::random::random({1, 1, 32, 4544}).to(Layout::TILE).to(device);
                Tensor b = tt::numpy::zeros({1, 1, 32, 4544}, DataType::BFLOAT16).to(Layout::TILE).to(device);
                Tensor c = bcast(a, b, BcastOpMath::MUL, BcastOpDim::H);
                Tensor d = c.cpu();
            }

            {
                Tensor a = tt::numpy::random::random({1, 1, 32, 4544}).to(Layout::TILE).to(device);
                Tensor b = tt::numpy::zeros({1, 1, 32, 4544}, DataType::BFLOAT16).to(Layout::TILE).to(device);
                Tensor c = bcast(a, b, BcastOpMath::ADD, BcastOpDim::H);
                Tensor d = c.cpu();
            }

            {
                Tensor a = tt::numpy::random::random({1, 71, 32, 32}).to(Layout::TILE).to(device);
                Tensor b = tt::numpy::zeros({1, 1, 32, 32}, DataType::BFLOAT16).to(Layout::TILE).to(device);
                Tensor c = bcast(a, b, BcastOpMath::MUL, BcastOpDim::HW);
                Tensor d = c.cpu();
            }

            {
                Tensor a = tt::numpy::random::random({1, 71, 32, 64}).to(Layout::TILE).to(device);
                Tensor b = tt::numpy::zeros({1, 1, 32, 32}, DataType::BFLOAT16).to(Layout::TILE).to(device);
                Tensor c = bcast(a, b, BcastOpMath::MUL, BcastOpDim::HW);
                Tensor d = c.cpu();
            }
        };
        run_operations();

        program_cache::enable();
        run_operations();
        run_operations();
        run_operations();
        program_cache::disable_and_clear();

        pass &= CloseDevice(device);

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
