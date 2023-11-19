// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "tensor/tensor.hpp"
#include "tensor/owned_buffer.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_numpy/functions.hpp"

#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/program_cache.hpp"

#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/transformer_tms/transformer_tms.hpp"
#include "tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"

#include <chrono>

using Parameters = std::map<std::string, Tensor>;

constexpr auto l1_memory_config = tt::tt_metal::MemoryConfig{.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED,.buffer_type=tt::tt_metal::BufferType::L1};
constexpr auto dram_memory_config = tt::tt_metal::MemoryConfig{.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED,.buffer_type=tt::tt_metal::BufferType::DRAM};

Tensor encoder(Tensor&& hidden_states, const Tensor& attention_mask, const Parameters& parameters, std::size_t encoder_index, const std::uint32_t head_size) {

    auto batch_size = hidden_states.shape()[0];

    auto fused_qkv_matmul_program_config = tt::operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .per_core_M = 12,
        .per_core_N = 8,
        .transpose_mcast = false,
        .fused_activation = std::nullopt,
    };
    auto fused_qkv_matmul_output = tt::operations::primary::matmul(
        hidden_states,
        parameters.at(fmt::format("fused_qkv_weight_{}", encoder_index)),
        parameters.at(fmt::format("fused_qkv_bias_{}", encoder_index)),
        fused_qkv_matmul_program_config,
        l1_memory_config
    );


    auto&& [query, key, value] = tt::operations::primary::transformers::split_fused_qkv_and_split_heads(fused_qkv_matmul_output, CoreCoord{12, batch_size}, l1_memory_config);
    fused_qkv_matmul_output.deallocate();


    auto pre_softmax_bmm_program_config = tt::operations::primary::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 1,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .per_core_M = 12,
        .per_core_N = 12,
    };
    auto pre_softmax_bmm_matmul = tt::operations::primary::matmul(query, key, pre_softmax_bmm_program_config, dram_memory_config);
    query.deallocate();
    key.deallocate();


    pre_softmax_bmm_matmul = tt::operations::primary::transformers::scale_mask_softmax_in_place(pre_softmax_bmm_matmul, 1.0f / std::sqrt(head_size), attention_mask);


    auto post_softmax_bmm_program_config = tt::operations::primary::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 2,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .per_core_M = 12,
        .per_core_N = 2,
    };
    auto post_softmax_bmm_output = tt::operations::primary::matmul(pre_softmax_bmm_matmul, value, post_softmax_bmm_program_config, l1_memory_config);
    pre_softmax_bmm_matmul.deallocate();
    value.deallocate();


    auto concat_heads_output = tt::operations::primary::transformers::concatenate_heads(post_softmax_bmm_output, CoreCoord{12, batch_size}, l1_memory_config);
    post_softmax_bmm_output.deallocate();


    auto selfout_bmm_program_config = tt::operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .per_core_M = 12,
        .per_core_N = 3,
        .transpose_mcast = false,
        .fused_activation = std::nullopt,
    };
    auto selfout_bmm_output = tt::operations::primary::matmul(
        concat_heads_output,
        parameters.at(fmt::format("selfout_weight_{}", encoder_index)),
        parameters.at(fmt::format("selfout_bias_{}", encoder_index)),
        selfout_bmm_program_config,
        l1_memory_config
    );
    concat_heads_output.deallocate();


    auto attention_layernorm_output = tt::operations::primary::add_layernorm(
        hidden_states,
        selfout_bmm_output,
        1e-12,
        parameters.at(fmt::format("attention_layernorm_weight_{}", encoder_index)),
        parameters.at(fmt::format("attention_layernorm_bias_{}", encoder_index)),
        l1_memory_config
    );
    hidden_states.deallocate();
    selfout_bmm_output.deallocate();


    auto ff1_matmul_program_config = tt::operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .per_core_M = 12,
        .per_core_N = 11,
        .transpose_mcast = false,
        .fused_activation = UnaryWithParam{.op_type=UnaryOpType::GELU, .param=1.0f},
    };
    auto ff1_matmul_output = tt::operations::primary::matmul(
        attention_layernorm_output,
        parameters.at(fmt::format("ff1_weight_{}", encoder_index)),
        parameters.at(fmt::format("ff1_bias_{}", encoder_index)),
        ff1_matmul_program_config,
        dram_memory_config
    );


    auto ff2_matmul_program_config = tt::operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .per_core_M = 12,
        .per_core_N = 3,
        .transpose_mcast = false,
        .fused_activation = std::nullopt,
    };
    auto ff2_matmul_output = tt::operations::primary::matmul(
        ff1_matmul_output,
        parameters.at(fmt::format("ff2_weight_{}", encoder_index)),
        parameters.at(fmt::format("ff2_bias_{}", encoder_index)),
        ff2_matmul_program_config,
        l1_memory_config
    );
    ff1_matmul_output.deallocate();


    auto feedforward_layernorm_output = tt::operations::primary::add_layernorm(
        attention_layernorm_output,
        ff2_matmul_output,
        1e-12,
        parameters.at(fmt::format("feedforward_layernorm_weight_{}", encoder_index)),
        parameters.at(fmt::format("feedforward_layernorm_bias_{}", encoder_index)),
        l1_memory_config
    );
    attention_layernorm_output.deallocate();
    ff2_matmul_output.deallocate();


    return feedforward_layernorm_output;
}

Tensor qa_head(Tensor&& hidden_states, const Parameters& parameters) {

    auto output = matmul(hidden_states, parameters.at("qa_head_weight"));
    hidden_states.deallocate();


    return bcast(output, parameters.at("qa_head_bias"), tt::tt_metal::BcastOpMath::ADD, tt::tt_metal::BcastOpDim::H, l1_memory_config);
}


void test_bert() {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    using tt::tt_metal::DataType;
    using tt::tt_metal::Device;

    using tt::tt_metal::Layout;
    using tt::tt_metal::Tensor;

    tt::log_info(tt::LogTest, "Running {}", __func__);

    int device_id = 0;
    const auto &device = tt::tt_metal::CreateDevice(device_id);
    CoreCoord compute_grid_size = device.compute_with_storage_grid_size();

    if (compute_grid_size.x * compute_grid_size.y == 88) {
        tt::log_info(tt::LogTest, "Skipping test_bert for E75");
        return;
    }


    std::size_t num_iterations = 2;
    std::size_t num_encoders = 24;
    std::uint32_t batch_size = 9;
    std::uint32_t sequence_size = 384;
    std::uint32_t num_heads = 16;
    std::uint32_t head_size = 64;
    std::uint32_t hidden_size = num_heads * head_size;
    std::uint32_t intermediate_size = hidden_size * 4;

    auto attention_mask = tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {batch_size, 1, TILE_HEIGHT, sequence_size}, Layout::TILE).to(device, l1_memory_config);

    auto parameters = Parameters{};
    for (auto encoder_index = 0; encoder_index < num_encoders; encoder_index++) {
        parameters.emplace(fmt::format("fused_qkv_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, hidden_size, hidden_size * 3}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("fused_qkv_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, hidden_size * 3}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("selfout_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, hidden_size, hidden_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("selfout_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, hidden_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("attention_layernorm_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, TILE_WIDTH}, Layout::ROW_MAJOR).to(device, dram_memory_config));
        parameters.emplace(fmt::format("attention_layernorm_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, TILE_WIDTH}, Layout::ROW_MAJOR).to(device, dram_memory_config));
        parameters.emplace(fmt::format("ff1_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, hidden_size, intermediate_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("ff1_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, intermediate_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("ff2_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, intermediate_size, hidden_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("ff2_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, hidden_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("feedforward_layernorm_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, TILE_WIDTH}, Layout::ROW_MAJOR).to(device, dram_memory_config));
        parameters.emplace(fmt::format("feedforward_layernorm_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, TILE_WIDTH}, Layout::ROW_MAJOR).to(device, dram_memory_config));
    };
    parameters.emplace("qa_head_weight", tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, hidden_size, TILE_WIDTH}, Layout::TILE).to(device, dram_memory_config));
    parameters.emplace("qa_head_bias", tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, TILE_WIDTH}, Layout::TILE).to(device, dram_memory_config));

    auto run_bert = [&]() {
        tt::log_info(tt::LogTest, "run_bert started");
        auto begin = std::chrono::steady_clock::now();
        auto hidden_states = tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {batch_size, 1, sequence_size, hidden_size}, Layout::TILE).to(device, l1_memory_config);
        for (auto encoder_index = 0; encoder_index < num_encoders; encoder_index++) {
            hidden_states = encoder(std::move(hidden_states), attention_mask, parameters, encoder_index, head_size);
        }
        auto output = qa_head(std::move(hidden_states), parameters).cpu();
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        tt::log_info(tt::LogTest, "run_bert finished in {} microseconds", duration);
        return duration;
    };

    auto run_loop = [&]() {
        auto total_duration = 0;
        for (int iteration = 0; iteration < num_iterations; iteration++) {
            total_duration += run_bert();
        }
        auto average_duration = total_duration / num_iterations;
        auto num_samples_per_second = 1e6 / average_duration * batch_size;
        tt::log_info(tt::LogTest, "total duration: {} microseconds", total_duration);
        tt::log_info(tt::LogTest, "average duration: {} average_duration", total_duration);
        tt::log_info(tt::LogTest, "samples per second: {}", num_samples_per_second);
    };

    tt::tt_metal::program_cache::enable();
    run_bert();
    run_loop();
    tt::tt_metal::program_cache::disable_and_clear();

    TT_FATAL(tt::tt_metal::CloseDevice(device));
}

int main(int argc, char** argv) {
    test_bert();
    return 0;
}
