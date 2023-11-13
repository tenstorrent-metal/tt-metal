// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>


#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

namespace untilize_helpers {
uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks) {
    int32_t ncores_x = grid_size.x;
    int32_t ncores_y = grid_size.y;
    int32_t ncores = ncores_x * ncores_y;
    if (nblocks <= ncores) {
        ncores = nblocks;
    } else {
        uint32_t nblocks_per_core = ceil((float) nblocks / ncores);
        ncores = ceil((float) nblocks / nblocks_per_core);
    }
    return ncores;
}
}

inline std::tuple<int32_t, int32_t, int32_t, int32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t>
    split_blocks_across_cores(CoreCoord grid_size, uint32_t nblocks) {

    int32_t ncores_x = grid_size.x;
    int32_t ncores_y = grid_size.y;
    int32_t ncores = ncores_x * ncores_y;
    uint32_t nblocks_per_core = nblocks;
    uint32_t nblocks_per_core_cliff = 0;
    int32_t ncores_x_cliff = 0;
    std::set<CoreRange> all_cores;
    std::set<CoreRange> core_range, core_range_cliff;
    if (nblocks <= ncores) {
        nblocks_per_core = 1;
        ncores = nblocks;
        ncores_y = ceil((float) ncores / ncores_x);
        ncores_x_cliff = ncores - (ncores_x * (ncores_y - 1));
        if (ncores_x_cliff == ncores_x) {
            // no cliff, all is perfectly divisible
            ncores_x_cliff = 0;
            core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
            all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
        } else if (ncores_x_cliff == 1) {
            // just one cliff core in the last row
            nblocks_per_core_cliff = 1;
            if (ncores_y > 1) {
                core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
            }
            core_range_cliff.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
            all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
        } else if (ncores_x_cliff > 1) {
            // both normal and cliff cores in the last row
            nblocks_per_core_cliff = 1;
            if (ncores_y > 1) {
                core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
            }
            core_range.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 2, ncores_y - 1)});
            core_range_cliff.insert(CoreRange{.start = CoreCoord(ncores_x_cliff - 1, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
            all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
        } else {
            TT_ASSERT(false, "Something went really wrong in splitting blocks across cores {} {}!!", ncores_x, ncores_x_cliff);
        }
    } else {
        nblocks_per_core = ceil((float) nblocks / ncores);
        ncores = ceil((float) nblocks / nblocks_per_core);
        nblocks_per_core_cliff = nblocks - nblocks_per_core * (ncores - 1);
        ncores_y = ceil((float) ncores / ncores_x);
        ncores_x_cliff = ncores - ncores_x * (ncores_y - 1);
        if (nblocks_per_core_cliff == nblocks_per_core) {
            // no special cliff at block level for per core
            if (ncores_x_cliff == ncores_x) {
                // no x_cliff row => all cores are equal
                ncores_x_cliff = 0;
                core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
            } else if (ncores_x_cliff == 1) {
                // just 1 core as cliff in the last core row
                if (ncores_y > 1) {
                    core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                    all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                }
                core_range_cliff.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
            } else if (ncores_x_cliff < ncores_x) {
                // last core row has last core as cliff, rest are normal
                if (ncores_y > 1) {
                    core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                    all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                }
                core_range.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 2, ncores_y - 1)});
                core_range_cliff.insert(CoreRange{.start = CoreCoord(ncores_x_cliff - 1, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
            } else {
                TT_ASSERT("Something went really wrong in calculating the core ranges {} {}", ncores_x, ncores_x_cliff);
            }
        } else if (nblocks_per_core_cliff < nblocks_per_core) {
            // last core has unequal blocks
            if (ncores_x_cliff == ncores_x) {
                // ncores x is same throughout
                ncores_x_cliff = 0;
                if (ncores_y > 1) {
                    core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                }
                core_range.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 2, ncores_y - 1)});
                core_range_cliff.insert(CoreRange{.start = CoreCoord(ncores_x_cliff - 1, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
            } else if (ncores_x_cliff == 1) {
                // last core row only has 1 core, as cliff
                if (ncores_y > 1) {
                    core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                    all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                }
                core_range_cliff.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
            } else if (ncores_x_cliff < ncores_x) {
                if (ncores_y > 1) {
                    core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                    all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                }
                core_range.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 2, ncores_y - 1)});
                core_range_cliff.insert(CoreRange{.start = CoreCoord(ncores_x_cliff - 1, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
            } else {
                TT_ASSERT(false, "Something went very wrong in calculating core ranges (case 2)");
            }
        } else {
            TT_ASSERT(false, "Somehting went really wrong in splitting blocks across cores (case else)");
        }
    }
    return std::make_tuple(ncores, ncores_x, ncores_x_cliff, ncores_y, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff);
}

operation::ProgramWithCallbacks untilize_multi_core(const Tensor& a, Tensor& output) {
    tt_metal::Program program = tt_metal::Program();

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    Device *device = a.device();

    uint32_t ntiles = a.volume() / TILE_HW;
    uint32_t ntiles_per_block = a.shape()[3] / TILE_WIDTH;
    uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = a.shape()[3] * output.element_size();

    {
        log_debug(LogOp, "ntiles: {}", ntiles);
        log_debug(LogOp, "ntiles_per_block: {}", ntiles_per_block);
        log_debug(LogOp, "nblocks: {}", nblocks);
    }

    auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, ncores_x, ncores_x_cliff, ncores_y, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] = split_blocks_across_cores(grid_size, nblocks);
    bool row_major = true;
    bool src_block_sharded = false;
    uint32_t num_rows_block = 0, block_row_size = 0, output_row_size = 0, last_block_row_size_unpadded = 0, num_output_rows_unpadded = 0;
    CoreCoord end_core;
    if (src_sharded) {
        auto shard_spec = a.shard_spec().value();
        src_block_sharded = a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
        row_major = shard_spec.shard_orientation == ShardOrientation::ROW_MAJOR;
        ncores_x = device->compute_with_storage_grid_size().x;
        ncores_y = device->compute_with_storage_grid_size().y;
        all_cores = shard_spec.shard_grid;
        uint32_t num_cores = all_cores.num_cores();
        ncores = num_cores;
        core_range = all_cores;
        core_range_cliff = CoreRangeSet({});
        ntiles_per_block = shard_spec.shard_shape[1] / TILE_WIDTH;
        nblocks_per_core = shard_spec.shard_shape[0] / TILE_HEIGHT;
        nblocks_per_core_cliff = 0;

        num_rows_block = shard_spec.shard_shape[0];
        block_row_size = shard_spec.shard_shape[1] * output.element_size();     // in0_block_w * TILE_WIDTH * dtype_nbytes
        output_row_size = output.shape()[-1] * output.element_size();    // output row size bytes
        last_block_row_size_unpadded = block_row_size - (round_up(output.shape()[-1], shard_spec.shard_shape[1]) - output.shape()[-1]) * output.element_size();
        uint32_t num_output_rows = output.volume() / output.shape()[-1];
        num_output_rows_unpadded = num_rows_block - (round_up(num_output_rows, shard_spec.shard_shape[0]) - num_output_rows);
        end_core = (*shard_spec.shard_grid.ranges().begin()).end;

    }
    {
        log_debug(LogOp, "ncores: {}", ncores);
        log_debug(LogOp, "ncores_x: {}", ncores_x);
        log_debug(LogOp, "ncores_x_cliff: {}", ncores_x_cliff);
        log_debug(LogOp, "ncores_y: {}", ncores_y);
        log_debug(LogOp, "nblocks_per_core: {}", nblocks_per_core);
        log_debug(LogOp, "nblocks_per_core_cliff: {}", nblocks_per_core_cliff);
    }

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_tiles = src_sharded ? ntiles_per_block * nblocks_per_core : ntiles_per_block * 2;
    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * input_single_tile_size, {{src0_cb_index, input_cb_data_format}})
        .set_page_size(src0_cb_index, input_single_tile_size);
    if (src_sharded) {
        src0_cb_config = src0_cb_config.set_globally_allocated_address(a.buffer()->address());
    }
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t output_cb_index = CB::c_out0;
    uint32_t num_output_tiles = out_sharded ? ntiles_per_block * nblocks_per_core : ntiles_per_block * 2;
    tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
        .set_page_size(output_cb_index, output_single_tile_size);
    if (out_sharded) {
        output_cb_config = output_cb_config.set_globally_allocated_address(output.buffer()->address());
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    Buffer *src0_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    KernelID unary_reader_kernel_id;

    if (src_sharded) {
        std::vector<uint32_t> reader_ct_args = {
            (std::uint32_t) src0_cb_index
        };

        unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary_sharded.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_ct_args});
    } else {
        bool src0_is_dram = src0_buffer->buffer_storage() == BufferStorage::DRAM ? 1 : 0;
        vector<uint32_t> reader_ct_args = {
            (uint32_t) src0_is_dram
        };

        unary_reader_kernel_id = CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_ct_args});
    }

    /** writer
     */
    KernelID unary_writer_kernel_id;
    if (out_sharded) {
        std::vector<uint32_t> writer_ct_args = {
            (std::uint32_t) output_cb_index
        };
        unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_sharded.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_ct_args});
    } else {
        bool out_is_dram = dst_buffer->buffer_storage() == BufferStorage::DRAM ? 1 : 0;
        if (src_block_sharded) {
            vector<uint32_t> writer_ct_args = {
                (uint32_t) out_is_dram
            };
            unary_writer_kernel_id = CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank_blocks.cpp",
                all_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = writer_ct_args});
        } else {
            bool stick_size_is_power_of_two = is_power_of_two_at_least_32(block_size_nbytes);
            uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t) std::log2(block_size_nbytes) : 0;
            vector<uint32_t> writer_ct_args = {
                (uint32_t) out_is_dram,
                (uint32_t) stick_size_is_power_of_two,
                (uint32_t) log2_stick_size,
            };

            unary_writer_kernel_id = CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/writer_unary_stick_layout_split_rows_interleaved.cpp",
                all_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = writer_ct_args});
        }
    }

    /** compute
     */
    vector<uint32_t> compute_args = {
        (uint32_t) nblocks_per_core,    // per_core_block_cnt
        (uint32_t) ntiles_per_block,    // per_block_ntiles
    };
    vector<uint32_t> compute_args_cliff = {
        (uint32_t) nblocks_per_core_cliff,
        (uint32_t) ntiles_per_block,    // per_block_ntiles
    };

    if (core_range.ranges().size() > 0) {
        auto untilize_kernel_id = CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/untilize.cpp",
            core_range,
            ComputeConfig{
                .compile_args = compute_args});
    }
    if (core_range_cliff.ranges().size() > 0) {
        auto untilize_cliff_kernel_id = CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/untilize.cpp",
            core_range_cliff,
            ComputeConfig{
                .compile_args = compute_args_cliff});
    }

    // 1D distribution of blocks across all cores
    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        // unequal case with cliff
        ncores_full -= 1;
        full_cores = core_range;
    }
    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);
    for (uint32_t i = 0; i < cores.size(); i++){
        CoreCoord core = cores[i];
        if (!full_cores.core_coord_in_core_ranges(core)) {
            continue;
        }
        // reader runtime args
        vector<uint32_t> reader_rt_args;

        if (src_sharded) {
            reader_rt_args = {
                ntiles_per_block * nblocks_per_core // ntiles
            };
        } else {
            reader_rt_args = {
                src0_buffer->address(),     // src_addr
                ntiles_per_block * nblocks_per_core, // ntiles
                tile_start_id                           // start_id
            };
        }
        // log_debug("reader[{}]: {},{} = {} ({})", src0_buffer->address(), core.x, core.y, tile_start_id, ntiles_per_block * nblocks_per_core);

        // writer runtime args
        vector<uint32_t> writer_rt_args;
         if (out_sharded) {
            writer_rt_args = {
                ntiles_per_block * nblocks_per_core // ntiles
            };
        } else {
            if (src_block_sharded) {
                uint32_t block_start_row_offset;
                uint32_t block_start_row_id_offset;
                uint32_t row_size_unpadded = block_row_size;
                uint32_t num_rows_unpadded = num_rows_block;
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }

                writer_rt_args = {
                    dst_buffer->address(),      // dst_addr
                    num_rows_block,
                    block_row_size,
                    1,
                    1,
                    1,
                    output_row_size,
                    row_size_unpadded,
                    num_rows_unpadded,
                    block_start_row_id_offset,
                    block_start_row_offset
                };
            } else {
                writer_rt_args = {
                    dst_buffer->address(),      // dst_addr
                    nblocks_per_core * TILE_HEIGHT,           // nblocks per core
                    block_size_nbytes,          // block_size_nbytes
                    ntiles_per_block,           // ntiles_per_block
                    block_size_nbytes,          // block_size_nbytes
                    1,                          // full blocks in a row
                    0,
                    0,
                    row_start_id
                };
            }
        }
        // log_debug("writer[{}]: {},{} = {} {}", dst_buffer->address(), core.x, core.y, block_size_nbytes, row_start_id);

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            reader_rt_args
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            writer_rt_args
        );

        tile_start_id += ntiles_per_block * nblocks_per_core;
        row_start_id += TILE_HEIGHT * nblocks_per_core;
    }
    if (ncores_full < ncores) {
        // last core is the cliff core with nblocks_per_core_cliff blocks
        CoreCoord core = row_major ? CoreCoord{ ncores_full % ncores_x, ncores_full / ncores_x} : CoreCoord{ ncores_full / ncores_y, ncores_full % ncores_y};
        // reader runtime args
        vector<uint32_t> reader_rt_args;

        if (src_sharded) {
            reader_rt_args = {
                ntiles_per_block * nblocks_per_core_cliff // ntiles
            };
        } else {
            reader_rt_args = {
                src0_buffer->address(),     // src_addr
                (uint32_t) ntiles_per_block * nblocks_per_core_cliff,       // ntiles
                tile_start_id                           // start_id
            };
        }
        // log_debug("reader: {},{} = {} ({})", core.x, core.y, tile_start_id, ntiles_per_block * nblocks_per_core_cliff);

        // writer runtime args
        vector<uint32_t> writer_rt_args;
        if (out_sharded) {
            writer_rt_args = {
                ntiles_per_block * nblocks_per_core_cliff // ntiles
            };
        } else {
            if (src_block_sharded) {
                uint32_t block_start_row_offset;
                uint32_t block_start_row_id_offset;
                uint32_t row_size_unpadded = block_row_size;
                uint32_t num_rows_unpadded = num_rows_block;
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }
                writer_rt_args = {
                    dst_buffer->address(),      // dst_addr
                    num_rows_block,
                    block_row_size,
                    1,
                    1,
                    1,
                    output_row_size,
                    row_size_unpadded,
                    num_rows_unpadded,
                    block_start_row_id_offset,
                    block_start_row_offset
                };
            } else {
                writer_rt_args = {
                    dst_buffer->address(),      // dst_addr
                    nblocks_per_core_cliff * TILE_HEIGHT,                 // nsticks
                    block_size_nbytes,          // stick_size_nbytes
                    ntiles_per_block,        // ntiles_per_block
                    block_size_nbytes,         // block_width_nbytes
                    1,     // full blocks in a row
                    0,         // UNUSED
                    0,      // UNUSED
                    row_start_id
                };
            }
        }
        // log_debug("writer: {},{} = {} {}", core.x, core.y, block_size_nbytes, row_start_id);

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            reader_rt_args
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            writer_rt_args
        );
    }

    auto override_runtime_arguments_callback = [
            reader_kernel_id=unary_reader_kernel_id,
            writer_kernel_id=unary_writer_kernel_id,
            cb_src0=cb_src0,
            cb_output=cb_output,
            all_cores=all_cores,
            ncores=ncores,
            ncores_x=ncores_x,
            ncores_y=ncores_y,
            row_major=row_major
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        bool src_sharded = input_tensors.at(0).memory_config().is_sharded();
        bool out_sharded = output_tensors.at(0).memory_config().is_sharded();

        if (src_sharded) {
            auto& src0_cb_config = GetCircularBufferConfig(program, cb_src0);
            src0_cb_config.set_globally_allocated_address(src_buffer->address());
        } else {
            auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);
            for (uint32_t i = 0; i < cores.size(); i++){
                CoreCoord core = cores[i];
                if (!all_cores.core_coord_in_core_ranges(core)) {
                    continue;
                }
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
                SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
            }
        }

        if (out_sharded) {
            auto& output_cb_config = GetCircularBufferConfig(program, cb_output);
            output_cb_config.set_globally_allocated_address(dst_buffer->address());
        } else {
            auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);
            for (uint32_t i = 0; i < cores.size(); i++){
                CoreCoord core = cores[i];
                if (!all_cores.core_coord_in_core_ranges(core)) {
                    continue;
                }
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

// This purely supports input block shard -> output interleaved for now
operation::ProgramWithCallbacks untilize_with_unpadding_multi_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end) {

    tt_metal::Program program = tt_metal::Program();

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    Device *device = a.device();

    uint32_t ntiles = a.volume() / TILE_HW;
    uint32_t ntiles_per_block = a.shape()[3] / TILE_WIDTH;
    uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = a.shape()[3] * output.element_size();

    auto grid_size = device->compute_with_storage_grid_size();
    bool row_major = true;
    uint32_t num_rows_block = 0, block_row_size = 0, output_row_size = 0, last_block_row_size_unpadded = 0, num_output_rows_unpadded = 0;
    CoreCoord end_core;
    auto shard_spec = a.shard_spec().value();
    row_major = shard_spec.shard_orientation == ShardOrientation::ROW_MAJOR;
    auto grid = *shard_spec.shard_grid.ranges().begin();
    uint32_t ncores_x = grid.end.x + 1;
    uint32_t ncores_y = grid.end.y + 1;
    auto all_cores = shard_spec.shard_grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t ncores = num_cores;
    ntiles_per_block = shard_spec.shard_shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = shard_spec.shard_shape[0] / TILE_HEIGHT;
    uint32_t batch = a.volume() / (a.shape()[-2] * a.shape()[-1]);
    uint32_t ntiles_per_batch = ntiles_per_block * nblocks_per_core / batch;

    num_rows_block = shard_spec.shard_shape[0];
    block_row_size = shard_spec.shard_shape[1] * output.element_size();     // in0_block_w * TILE_WIDTH * dtype_nbytes
    output_row_size = output.shape()[-1] * output.element_size();    // output row size bytes
    last_block_row_size_unpadded = block_row_size - (round_up(output.shape()[-1], shard_spec.shard_shape[1]) - output.shape()[-1]) * output.element_size();
    uint32_t num_output_rows = output.volume() / output.shape()[-1];
    num_output_rows_unpadded = num_rows_block - (round_up(num_output_rows, shard_spec.shard_shape[0]) - num_output_rows);
    end_core = {round_up(output.shape()[-1], shard_spec.shard_shape[1]) /shard_spec.shard_shape[1] - 1, round_up(num_output_rows, shard_spec.shard_shape[0]) / shard_spec.shard_shape[0] - 1};
    if (!row_major) {
        std::swap(end_core.x, end_core.y);
    }
    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * input_single_tile_size, {{src0_cb_index, input_cb_data_format}})
        .set_page_size(src0_cb_index, input_single_tile_size);
    if (src_sharded) {
        src0_cb_config = src0_cb_config.set_globally_allocated_address(a.buffer()->address());
    }
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t output_cb_index = CB::c_out0;
    uint32_t num_output_tiles = out_sharded ? ntiles_per_batch * 2 : ntiles_per_block * 2;
    tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
        .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    CircularBufferID cb_sharded_output = 0;
    uint32_t sharded_output_cb_index = CB::c_out1;
    if (out_sharded) {
        tt_metal::CircularBufferConfig sharded_output_cb_config = tt_metal::CircularBufferConfig(num_output_rows_unpadded * block_row_size, {{sharded_output_cb_index, output_cb_data_format}})
            .set_page_size(sharded_output_cb_index, block_row_size).set_globally_allocated_address(output.buffer()->address());
        cb_sharded_output = tt_metal::CreateCircularBuffer(program, all_cores, sharded_output_cb_config);
    }

    Buffer *src0_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    KernelID unary_reader_kernel_id;
    std::vector<uint32_t> reader_ct_args = {
            (std::uint32_t) src0_cb_index
        };

    unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_ct_args});

    /** writer
     */
    KernelID unary_writer_kernel_id;
    if (out_sharded) {
        vector<uint32_t> writer_ct_args = {
            (uint32_t) output_cb_index,
            (uint32_t) sharded_output_cb_index
        };
        unary_writer_kernel_id = CreateDataMovementKernel(
            program,
            "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/writer_unary_unpad_batch_rows_sharded.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_ct_args});
    } else {
        bool out_is_dram = dst_buffer->buffer_storage() == BufferStorage::DRAM ? 1 : 0;
        vector<uint32_t> writer_ct_args = {
            (uint32_t) out_is_dram
        };
        unary_writer_kernel_id = CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank_blocks.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_ct_args});
    }

    /** compute
     */
    vector<uint32_t> compute_args = {
        (uint32_t) nblocks_per_core,    // per_core_block_cnt
        (uint32_t) ntiles_per_block,    // per_block_ntiles
    };

    auto untilize_kernel_id = CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/untilize.cpp",
        all_cores,
        ComputeConfig{
            .compile_args = compute_args});

    // reader runtime args
    vector<uint32_t> reader_rt_args = {
        ntiles_per_block * nblocks_per_core // ntiles
    };
    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        all_cores,
        reader_rt_args
    );

    if (out_sharded) {
        vector<uint32_t> writer_rt_args = {
            num_output_rows_unpadded,
            ntiles_per_batch,
            num_output_rows_unpadded / batch * block_row_size,
            batch
        };
        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            all_cores,
            writer_rt_args
        );
    } else {
        uint32_t tile_start_id = 0;
        uint32_t row_start_id = 0;
        auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);
        for (uint32_t i = 0; i < cores.size(); i++){
            CoreCoord core = cores[i];
            if (!all_cores.core_coord_in_core_ranges(core)) {
                continue;
            }

            // writer runtime args
            vector<uint32_t> writer_rt_args;
            uint32_t block_start_row_offset;
            uint32_t block_start_row_id_offset;
            uint32_t row_size_unpadded = block_row_size;
            uint32_t num_rows_unpadded = num_rows_block;
            if (a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                block_start_row_offset = i * block_row_size;
                block_start_row_id_offset = 0;
            } else if (a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                block_start_row_offset = 0;
                block_start_row_id_offset = i * num_rows_block;
            } else {
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }
                if (core.x > end_core.x || core.y > end_core.y) {
                    row_size_unpadded =  0;
                    num_rows_unpadded =  0;
                }
            }

            writer_rt_args = {
                dst_buffer->address(),      // dst_addr
                num_rows_block,
                block_row_size,
                1,
                1,
                1,
                output_row_size,
                row_size_unpadded,
                num_rows_unpadded,
                block_start_row_id_offset,
                block_start_row_offset
            };

            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                writer_rt_args
            );
        }
    }


    auto override_runtime_arguments_callback = [
            reader_kernel_id=unary_reader_kernel_id,
            writer_kernel_id=unary_writer_kernel_id,
            cb_src0=cb_src0,
            cb_sharded_output=cb_sharded_output,
            all_cores=all_cores,
            ncores=ncores,
            ncores_x=ncores_x,
            ncores_y=ncores_y,
            row_major=row_major
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        bool src_sharded = input_tensors.at(0).memory_config().is_sharded();
        bool out_sharded = output_tensors.at(0).memory_config().is_sharded();

        auto& src0_cb_config = GetCircularBufferConfig(program, cb_src0);
        src0_cb_config.set_globally_allocated_address(src_buffer->address());

        if (out_sharded) {
            auto& sharded_output_cb_config = GetCircularBufferConfig(program, cb_sharded_output);
            sharded_output_cb_config.set_globally_allocated_address(dst_buffer->address());
        } else {
            auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);
            for (uint32_t i = 0; i < cores.size(); i++){
                CoreCoord core = cores[i];
                if (!all_cores.core_coord_in_core_ranges(core)) {
                    continue;
                }
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
