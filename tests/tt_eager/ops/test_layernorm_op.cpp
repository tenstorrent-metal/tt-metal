// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include <tt_numpy/functions.hpp>

#include <algorithm>
#include <functional>
#include <random>
#include <optional>

using namespace tt;
using namespace tt::tt_metal;
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
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);
        Shape shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
        Tensor a = tt::numpy::random::random(shape).to(Layout::TILE).to(device);;
        Tensor c = layernorm(a, 1e-4f);
        Tensor d = c.cpu();
        Tensor host_a = a.cpu(); // Move tensor a to host to validate
        CloseDevice(device);
    } catch (const std::exception &e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
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
