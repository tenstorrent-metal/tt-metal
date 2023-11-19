/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_metal/impl/program.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/kernels/kernel_types.hpp"

using namespace tt::tt_metal;

namespace tt::tt_metal::detail{

    inline KernelID AddKernel ( Program & program, Kernel * kernel)
    {
        return program.add_kernel(kernel);
    }

    inline Kernel *GetKernel(const Program &program, KernelID kernel_id) {
        return program.kernels_.at(kernel_id);
    }

    inline std::shared_ptr<CircularBuffer> GetCircularBuffer(const Program &program, CircularBufferID id) {
        return program.get_circular_buffer(id);
    }

    // Checks that circular buffers do not grow into L1 buffer space
    inline void ValidateCircularBufferRegion(const Program &program, const Device& device) {
        program.validate_circular_buffer_region(device);
    }

}  // namespace tt::tt_metal::detail
