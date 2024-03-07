// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include "command_queue_fixture.hpp"
#include "common/env_lib.hpp"
#include "gtest/gtest.h"
#include "impl/program/program.hpp"
#include "logger.hpp"
#include "tt_metal/common/scoped_timer.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt::tt_metal;

struct TestBufferConfig {
    uint32_t num_pages;
    uint32_t page_size;
    BufferType buftype;
};

Program create_simple_unary_program(Buffer& input, Buffer& output) {
    Program program = CreateProgram();
    Device* device = input.device();
    CoreCoord worker = {0, 0};
    auto reader_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        worker,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        worker,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto sfpu_kernel = CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        worker,
        ComputeConfig{
            .math_approx_mode = true,
            .compile_args = {1, 1},
            .defines = {{"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}});

    CircularBufferConfig input_cb_config = CircularBufferConfig(2048, {{0, tt::DataFormat::Float16_b}})
            .set_page_size(0, 2048);

    CoreRange core_range({0, 0});
    CreateCircularBuffer(program, core_range, input_cb_config);
    std::shared_ptr<RuntimeArgs> writer_runtime_args = std::make_shared<RuntimeArgs>();
    std::shared_ptr<RuntimeArgs> reader_runtime_args = std::make_shared<RuntimeArgs>();

    *writer_runtime_args = {
        &output,
        (uint32_t)output.noc_coordinates().x,
        (uint32_t)output.noc_coordinates().y,
        output.num_pages()
    };

    *reader_runtime_args = {
        &input,
        (uint32_t)input.noc_coordinates().x,
        (uint32_t)input.noc_coordinates().y,
        input.num_pages()
    };

    SetRuntimeArgs(device, detail::GetKernel(program, writer_kernel), worker, writer_runtime_args);
    SetRuntimeArgs(device, detail::GetKernel(program, reader_kernel), worker, reader_runtime_args);

    CircularBufferConfig output_cb_config = CircularBufferConfig(2048, {{16, tt::DataFormat::Float16_b}})
            .set_page_size(16, 2048);

    CreateCircularBuffer(program, core_range, output_cb_config);
    return program;
}

// All basic trace tests just assert that the replayed result exactly matches
// the eager mode results
namespace basic_tests {

constexpr bool kBlocking = true;
constexpr bool kNonBlocking = false;
vector<bool> blocking_flags = {kBlocking, kNonBlocking};

TEST_F(CommandQueueFixture, InstantiateTraceSanity) {
    CommandQueue command_queue(this->device_, 0, CommandQueue::CommandQueueMode::PASSTHROUGH);

    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer interm(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);

    Program op0 = create_simple_unary_program(input, interm);
    Program op1 = create_simple_unary_program(interm, output);
    vector<uint32_t> input_data(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Trace mode output
    vector<uint32_t> trace_output;
    trace_output.resize(input_data.size());

    // Eager mode
    vector<uint32_t> expected_output_data;
    vector<uint32_t> eager_output_data;
    expected_output_data.resize(input_data.size());
    eager_output_data.resize(input_data.size());

    // Capture trace on a trace queue
    Trace trace;
    CommandQueue& trace_queue = BeginTrace(trace);
    EnqueueProgram(trace_queue, op0, kNonBlocking);
    EnqueueProgram(trace_queue, op1, kNonBlocking);
    EndTrace(trace);
    log_info(LogTest, "Captured trace!");

    // Instantiate a trace on a device bound command queue
    uint32_t trace_id = InstantiateTrace(trace, command_queue);
    log_info(LogTest, "Instantiated trace id {} completed!", trace_id);
}

TEST_F(CommandQueueFixture, EnqueueTwoProgramTrace) {
    // Get command queue from device for this test, since its running in async mode
    CommandQueue& command_queue = this->device_->command_queue();
    auto current_mode = CommandQueue::default_mode();
    command_queue.set_mode(CommandQueue::CommandQueueMode::ASYNC);

    Buffer input(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer interm(this->device_, 2048, 2048, BufferType::DRAM);
    Buffer output(this->device_, 2048, 2048, BufferType::DRAM);

    Program op0 = create_simple_unary_program(input, interm);
    Program op1 = create_simple_unary_program(interm, output);
    vector<uint32_t> input_data(input.size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Trace mode output
    uint32_t num_loops = parse_env<int>("TT_METAL_TRACE_LOOPS", 5);
    vector<vector<uint32_t>> trace_outputs;

    for (auto i = 0; i < num_loops; i++) {
        trace_outputs.push_back({});
        trace_outputs[i].resize(input_data.size());
    }

    // Eager mode
    vector<uint32_t> expected_output_data;
    vector<uint32_t> eager_output_data;
    expected_output_data.resize(input_data.size());
    eager_output_data.resize(input_data.size());

    // Warm up and use the eager blocking run as the expected output
    EnqueueWriteBuffer(command_queue, input, input_data.data(), kBlocking);
    EnqueueProgram(command_queue, op0, kBlocking);
    EnqueueProgram(command_queue, op1, kBlocking);
    EnqueueReadBuffer(command_queue, output, expected_output_data.data(), kBlocking);
    Finish(command_queue);

    for (bool blocking : blocking_flags) {
        std::string mode = blocking ? "Eager-B" : "Eager-NB";
        for (auto i = 0; i < num_loops; i++) {
            ScopedTimer timer(mode + " loop " + std::to_string(i));
            EnqueueWriteBuffer(command_queue, input, input_data.data(), blocking);
            EnqueueProgram(command_queue, op0, blocking);
            EnqueueProgram(command_queue, op1, blocking);
            EnqueueReadBuffer(command_queue, output, eager_output_data.data(), blocking);
        }
        if (not blocking) {
            // (Optional) wait for the last non-blocking command to finish
            Finish(command_queue);
        }
        EXPECT_TRUE(eager_output_data == expected_output_data);
    }

    // Capture trace on a trace queue
    Trace trace;
    CommandQueue& trace_queue = BeginTrace(trace);
    EnqueueProgram(trace_queue, op0, kNonBlocking);
    EnqueueProgram(trace_queue, op1, kNonBlocking);
    EndTrace(trace);

    // Instantiate a trace on a device bound command queue
    uint32_t trace_id = InstantiateTrace(trace, command_queue);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        ScopedTimer timer("Trace loop " + std::to_string(i));
        EnqueueWriteBuffer(command_queue, input, input_data.data(), kNonBlocking);
        EnqueueTrace(command_queue, trace_id, kNonBlocking);
        EnqueueReadBuffer(command_queue, output, trace_outputs[i].data(), kNonBlocking);
    }
    Finish(command_queue);

    // Expect same output across all loops
    for (auto i = 0; i < num_loops; i++) {
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }
    command_queue.set_mode(current_mode);
}

TEST_F(CommandQueueFixture, EnqueueMultiProgramTraceBenchmark) {
    CommandQueue& command_queue = this->device_->command_queue();
    auto current_mode = CommandQueue::default_mode();
    command_queue.set_mode(CommandQueue::CommandQueueMode::ASYNC);

    std::shared_ptr<Buffer> input = std::make_shared<Buffer>(this->device_, 2048, 2048, BufferType::DRAM);
    std::shared_ptr<Buffer> output = std::make_shared<Buffer>(this->device_, 2048, 2048, BufferType::DRAM);

    uint32_t num_loops = parse_env<int>("TT_METAL_TRACE_LOOPS", 4);
    uint32_t num_programs = parse_env<int>("TT_METAL_TRACE_PROGRAMS", 4);
    vector<std::shared_ptr<Buffer>> interm_buffers;
    vector<Program> programs;

    vector<uint32_t> input_data(input->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    for (int i = 0; i < num_programs; i++) {
        interm_buffers.push_back(std::make_shared<Buffer>(this->device_, 2048, 2048, BufferType::DRAM));
        if (i == 0) {
            programs.push_back(create_simple_unary_program(*input, *(interm_buffers[i])));
        } else if (i == (num_programs - 1)) {
            programs.push_back(create_simple_unary_program(*(interm_buffers[i - 1]), *output));
        } else {
            programs.push_back(create_simple_unary_program(*(interm_buffers[i - 1]), *(interm_buffers[i])));
        }
    }

    // Eager mode
    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());

    // Trace mode output
    vector<vector<uint32_t>> trace_outputs;

    for (uint32_t i = 0; i < num_loops; i++) {
        trace_outputs.push_back({});
        trace_outputs[i].resize(input_data.size());
    }

    for (bool blocking : blocking_flags) {
        std::string mode = blocking ? "Eager-B" : "Eager-NB";
        log_info(LogTest, "Starting {} profiling with {} programs", mode, num_programs);
        for (uint32_t iter = 0; iter < num_loops; iter++) {
            ScopedTimer timer(mode + " loop " + std::to_string(iter));
            EnqueueWriteBuffer(command_queue, input, input_data.data(), blocking);
            for (uint32_t i = 0; i < num_programs; i++) {
                EnqueueProgram(command_queue, programs[i], blocking);
            }
            EnqueueReadBuffer(command_queue, output, eager_output_data.data(), blocking);
        }
        if (not blocking) {
            // (Optional) wait for the last non-blocking command to finish
            Finish(command_queue);
        }
    }

    // Capture trace on a trace queue
    Trace trace;
    CommandQueue& trace_queue = BeginTrace(trace);
    for (uint32_t i = 0; i < num_programs; i++) {
        EnqueueProgram(trace_queue, programs[i], kNonBlocking);
    }
    EndTrace(trace);

    // Instantiate a trace on a device bound command queue
    uint32_t trace_id = InstantiateTrace(trace, command_queue);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        ScopedTimer timer("Trace loop " + std::to_string(i));
        EnqueueWriteBuffer(command_queue, input, input_data.data(), kNonBlocking);
        EnqueueTrace(command_queue, trace_id, kNonBlocking);
        EnqueueReadBuffer(command_queue, output, trace_outputs[i].data(), kNonBlocking);
    }
    Finish(command_queue);
    command_queue.set_mode(current_mode);
}

} // end namespace basic_tests
