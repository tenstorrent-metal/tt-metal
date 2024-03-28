// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/groupnorm/groupnorm_op.hpp"

#include <optional>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_eager/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

Tensor groupnorm(
    const Tensor& a,
    uint32_t group_size,
    float eps,
    std::optional<const Tensor> gamma,
    std::optional<const Tensor> beta,
    const MemoryConfig& output_mem_config) {
    TT_ASSERT(a.get_legacy_shape()[3] % TILE_WIDTH == 0, "Normalizing on last dim cannot be padded");
    if (gamma.has_value()) {
        TT_ASSERT(gamma.value().get_legacy_shape()[3] == a.get_legacy_shape()[3], "Gamma width must be equal to input width");
    }
    if (beta.has_value()) {
        TT_ASSERT(beta.value().get_legacy_shape()[3] == a.get_legacy_shape()[3], "Beta width must be equal to input width");
    }

    TT_ASSERT(group_size == 1 && "group norm size is only supported for size = 1");
    /**
     * shortcut when group size = 1 we use layernorm with transpose and non-transpose
     */

    Shape shape = a.get_legacy_shape();
    Tensor ar = reshape(const_cast<Tensor&>(a),shape[0],1,shape[1]*shape[2],shape[3],output_mem_config);
    Tensor group_norm_1 = normalize_hw(ar,output_mem_config);
    Tensor output = reshape (group_norm_1,shape[0],shape[1],shape[2],shape[3],output_mem_config);
    if (gamma.has_value() && beta.has_value()) {
        output = mac(output,gamma.value(),beta.value(),output_mem_config);
    } else {
        if (gamma.has_value()) {
            output = mul(output,gamma.value()); //gamma_t);
        } else if (beta.has_value()) {
            output = add(output,beta.value());
        }
    }
    return output;
}

}  // namespace tt_metal

namespace operations {
namespace primary {

inline bool is_dram(const Tensor& input_tensor) { return input_tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> input_tensor) {
     return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

int get_max_subblock(uint32_t n, uint32_t max_subblock_w) {
    if (n <= max_subblock_w) {
        return n;
    }

    for (int quotient = max_subblock_w; quotient > 1; --quotient) {
        if (n % quotient == 0) {
            return quotient;
        }
    }
    return 1;
}
bool is_rectangle_grid(const std::vector<CoreCoord>& core_coords) {
    if (core_coords.empty()) {
        return true;
    }

    int min_x = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max();
    int max_y = std::numeric_limits<int>::min();

    for (const auto& coord : core_coords) {
        min_x = std::min(min_x, static_cast<int>(coord.x));
        max_x = std::max(max_x, static_cast<int>(coord.x));
        min_y = std::min(min_y, static_cast<int>(coord.y));
        max_y = std::max(max_y, static_cast<int>(coord.y));
    }

    return ((max_x - min_x + 1) * (max_y - min_y + 1)) == core_coords.size();
}
void split_and_form_rectangle_grids(std::vector<CoreCoord>& group, std::vector<CoreCoord>& mcast_group_first, std::vector<CoreCoord>& mcast_group_mid, std::vector<CoreCoord>& mcast_group_last) {

    int remove_front = 0;
    int remove_back = 0;
    int min_total_removal = group.size();

    for (int front = 0; front <= group.size(); ++front) {
        for (int back = 0; front + back <= group.size(); ++back) {
            if (is_rectangle_grid(std::vector<CoreCoord>(group.begin() + front, group.end() - back))) {
                int total_removal = front + back;
                if (total_removal < min_total_removal) {
                    min_total_removal = total_removal;
                    remove_front = front;
                    remove_back = back;
                }
            }
        }
    }

    // Pop and push the front outliers
    for (int i = 0; i < remove_front; ++i) {
        mcast_group_first.push_back(mcast_group_mid.front());
        mcast_group_mid.erase(mcast_group_mid.begin());
    }

    // Pop and push the back outliers
    for (int i = 0; i < remove_back; ++i) {
        mcast_group_last.push_back(mcast_group_mid.back());
        mcast_group_mid.pop_back();
    }
}
operation::ProgramWithCallbacks groupnorm_sharded_(
    const Tensor &a,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    float eps,
    const uint32_t num_groups,
    const uint32_t num_batches,
    MathFidelity fidelity,
    DataType im_data_format,
    CoreCoord grid_size
) {
    bool is_row_major_layout = a.get_layout() == Layout::ROW_MAJOR;
    bool is_height_sharding = a.get_legacy_shape()[3] == a.shard_spec().value().shape[1];
    // convert data format
    tt::DataFormat in_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat out_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(im_data_format);
    tt::DataFormat gamma_beta_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t datum_size_bytes = 2; // bfloat16
    // tile sizes
    uint32_t in_single_tile_size = tt_metal::detail::TileSize(in_data_format);
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    uint32_t out_single_tile_size = tt_metal::detail::TileSize(out_data_format);
    uint32_t gamma_beta_single_tile_size = tt_metal::detail::TileSize(gamma_beta_cb_data_format);
    // shard shape per core
    uint32_t per_core_M = a.shard_spec().value().shape[0];
    uint32_t per_core_N = a.shard_spec().value().shape[1];
    uint32_t per_core_Mt = per_core_M / TILE_HEIGHT;
    uint32_t per_core_Nt = (per_core_N + TILE_WIDTH - 1) / TILE_WIDTH;
    // uint32_t per_core_N = per_core_N % TILE_WIDTH != 0 ? int(ceil(double(per_core_N) / double(TILE_WIDTH)) * TILE_WIDTH) : per_core_N;
    // tensor shape
    const auto shape = a.get_legacy_shape();
    uint32_t H = shape[2] * num_batches;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t W = shape[3];
    uint32_t Wt = W / TILE_WIDTH;
    uint32_t num_datum_row_per_group = W / num_groups;
    // grid
    uint32_t num_cores_c = grid_size.x;
    uint32_t num_cores_r = grid_size.y;
    // uint32_t num_cores = num_cores_c * num_cores_r;
    auto all_cores = a.shard_spec().value().grid;
    uint32_t num_cores = all_cores.num_cores();
    auto shard_orientation = a.shard_spec().value().orientation;
    // split each batch into multiple cores
    uint32_t num_shards_r = H / per_core_M;
    uint32_t num_cores_per_batch = num_batches > num_shards_r ? 1 : num_shards_r / num_batches;
    uint32_t num_shards_c = W / per_core_N;
    uint32_t num_cores_per_group = num_groups > num_shards_c ? 1 : num_shards_c / num_groups;
    // each core contains multiple batches
    uint32_t num_batches_per_core = num_batches > num_shards_r ? num_batches / num_shards_r : 1;
    uint32_t num_groups_per_core = num_groups > num_shards_c ? num_groups / num_shards_c : 1;

    TT_ASSERT(per_core_N % num_datum_row_per_group == 0);
    TT_ASSERT(per_core_M % TILE_HEIGHT == 0);
    if (per_core_N != W) {
        if (shard_orientation == ShardOrientation::COL_MAJOR) {
            TT_ASSERT(per_core_N * num_cores_r == W);
            TT_ASSERT(per_core_M * num_cores_c == H);
        } else {
            TT_ASSERT(per_core_N * num_cores_c == W);
            TT_ASSERT(per_core_M * num_cores_r == H);
        }
    }

    TT_ASSERT(per_core_M % TILE_HEIGHT == 0 && "per_core_M must be divisble by TILE_HEIGHT");

    TT_ASSERT(W % num_groups == 0 && "Tensor W must be divisble by num_groups");
    TT_ASSERT(H % per_core_M == 0 && "H dim must be divisible by per_core_M");
    TT_ASSERT(W % per_core_N == 0 && "W dim must be divisible by per_core_N");
    if (num_batches >= num_shards_r) {
        TT_ASSERT(num_batches % num_shards_r == 0 && "num_batches must be divisible by number of cores in a full column");
    } else {
        TT_ASSERT(num_shards_r % num_batches == 0 && "number of cores in a full column must be divisible by num_batches");
    }
    if (num_groups >= num_shards_c) {
        TT_ASSERT(num_groups % num_shards_c == 0 && "num_groups must be divisible by number of cores in a full row");
    } else {
        TT_ASSERT(num_shards_c % num_groups == 0 && "number of cores in a full row must be divisible by num_groups");
    }

    // subblock
    bool is_channel_divisible_by_tile = (num_datum_row_per_group % TILE_WIDTH == 0);
    uint32_t block_wt = per_core_Nt / num_groups_per_core;
    if (per_core_Nt % num_groups_per_core != 0) {
        block_wt = per_core_Nt / num_groups_per_core + 1;
    }

    uint32_t num_rows_per_batch_per_core = per_core_M / num_batches_per_core;
    uint32_t num_nz_rows_per_tile = TILE_HEIGHT;
    bool is_row_per_batch_divisible_by_tile = true;
    uint32_t block_ht = per_core_Mt / num_batches_per_core;
    if (num_rows_per_batch_per_core % TILE_HEIGHT != 0) {
        block_ht = per_core_Mt / num_batches_per_core + 1;
        num_nz_rows_per_tile = num_rows_per_batch_per_core < TILE_HEIGHT ? num_rows_per_batch_per_core : TILE_HEIGHT;
        is_row_per_batch_divisible_by_tile = false;
    }
    uint32_t block_w = block_wt * TILE_WIDTH;
    uint32_t block_h = block_ht * TILE_HEIGHT;
    uint32_t subblock_wt = get_max_subblock(block_wt, 8);
    uint32_t num_subblocks_w = block_wt / subblock_wt;

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "num_rows_per_batch_per_core: {}", per_core_M / num_batches_per_core);
    log_debug(tt::LogOp, "num_nz_rows_per_tile: {}", num_nz_rows_per_tile);
    log_debug(tt::LogOp, "per_core_M: {}", per_core_M);
    log_debug(tt::LogOp, "per_core_N: {}", per_core_N);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "H: {}", H);
    log_debug(tt::LogOp, "num_datum_row_per_group: {}", num_datum_row_per_group);
    log_debug(tt::LogOp, "num_batches: {}", num_batches);
    log_debug(tt::LogOp, "num_groups: {}", num_groups);
    log_debug(tt::LogOp, "num_cores_r: {}", num_cores_r);
    log_debug(tt::LogOp, "num_cores_c: {}", num_cores_c);
    log_debug(tt::LogOp, "num_cores_per_batch: {}", num_cores_per_batch);
    log_debug(tt::LogOp, "num_cores_per_group: {}", num_cores_per_group);
    log_debug(tt::LogOp, "num_batches_per_core: {}", num_batches_per_core);
    log_debug(tt::LogOp, "num_groups_per_core: {}", num_groups_per_core);
    log_debug(tt::LogOp, "block_wt: {}", block_wt);
    log_debug(tt::LogOp, "block_ht: {}", block_ht);
    log_debug(tt::LogOp, "block_w: {}", block_w);
    log_debug(tt::LogOp, "block_h: {}", block_h);
    log_debug(tt::LogOp, "subblock_wt: {}", subblock_wt);
    log_debug(tt::LogOp, "num_subblocks_w: {}", num_subblocks_w);

    TT_ASSERT(per_core_M % num_batches_per_core == 0 && "shard height must be div by per_core_batch");
    TT_ASSERT(W % num_groups == 0 && "tensor width must be divisible by num_groups!");
    if (shard_orientation == ShardOrientation::ROW_MAJOR and num_groups_per_core == 1) {
        TT_ASSERT(num_cores_c % num_groups == 0 && "for RM shard, when each group is split across cores, num_cores_c must be divisible by num_groups!");
    } else if (shard_orientation == ShardOrientation::COL_MAJOR and num_groups_per_core == 1) {
        TT_ASSERT(num_cores_r % num_groups == 0 && "for CM shard, when each group is split across cores, num_cores_r must be divisible by num_groups!");
    }

    if (per_core_N != W) { // block sharded
        if (shard_orientation == ShardOrientation::ROW_MAJOR and num_batches_per_core == 1) {
            TT_ASSERT(num_cores_r % num_batches == 0 && "for RM shard, when each batch is split across cores, num_cores_r must be divisible by num_batches!");
        } else if (shard_orientation == ShardOrientation::COL_MAJOR and num_groups_per_core == 1) {
            TT_ASSERT(num_cores_c % num_batches == 0 && "for CM shard, when each batch is split across cores, num_cores_c must be divisible by num_batches!");
        }
    } else { // height sharded
        if (num_batches_per_core == 1)
            TT_ASSERT((num_cores_c*num_cores_r) % num_batches == 0 && "for height shard, number of cores must be divisible by num_batches!");
    }

    // get sharded addr
    auto in0_addr = a.buffer()->address();
    auto out_addr = output.buffer()->address();
    // gamma, beta addr
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    // num tiles for a, gamma, beta
    uint32_t num_tiles = a.volume()/TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_HW : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    Device *device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_block_tiles = per_core_Nt * per_core_Mt;
    uint32_t interm_block_tiles = block_ht * block_wt;
    uint32_t in0_CB_tiles = in0_block_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * in_single_tile_size;
    // in2 - scaler
    uint32_t in2_CB_size = single_tile_size;
    // in3 - eps
    uint32_t in3_CB_size = single_tile_size;
    uint32_t inz_CB_size = single_tile_size;
    // gamma
    uint32_t gamma_beta_num_cols_tile_per_core = block_wt * num_groups_per_core;
    uint32_t in5_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    // beta
    uint32_t in6_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    // itermediate buffers change later
    uint32_t in_CB_size = in_single_tile_size * block_ht * block_wt;
    uint32_t im_out_CB_size = out_single_tile_size * block_ht * block_wt;
    uint32_t x_CB_size = interm_block_tiles * single_tile_size;
    uint32_t xmm_CB_size = interm_block_tiles * single_tile_size;
    uint32_t ex_partial_CB_size = num_groups_per_core * num_batches_per_core * single_tile_size;
    uint32_t ex_CB_size = ex_partial_CB_size;
    uint32_t ex_global_CB_size = ex_partial_CB_size;
    uint32_t ex_external_CB_size = ex_partial_CB_size;
    uint32_t xmm2_CB_size = interm_block_tiles * single_tile_size;
    uint32_t ex2pe_CB_size = ex_partial_CB_size;
    // output buffer size
    uint32_t out_CB_size = in0_block_tiles * out_single_tile_size;

    log_debug(tt::LogOp, "per_core_Nt: {}", per_core_Nt);
    log_debug(tt::LogOp, "per_core_Mt: {}", per_core_Mt);
    log_debug(tt::LogOp, "in0_CB_tiles: {}", in0_CB_tiles);
    log_debug(tt::LogOp, "in0_CB_size: {}", in0_CB_size);
    log_debug(tt::LogOp, "in_CB_size: {}", in_CB_size);
    log_debug(tt::LogOp, "gamma_beta_num_cols_tile_per_core: {}", gamma_beta_num_cols_tile_per_core);
    log_debug(tt::LogOp, "in5_CB_size: {}", in5_CB_size);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = Program();
    // define core ranges
    bool use_mcast = num_cores_per_batch > 1 or num_cores_per_group > 1;
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    // CoreRange all_cores(
    //     {(std::size_t) start_core_x, (std::size_t) start_core_y},
    //     {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1});
    // create a vector of cores, in either RM or CM
    std::vector<CoreCoord> core_coords = grid_to_cores(num_cores, num_cores_c, num_cores_r, shard_orientation == ShardOrientation::ROW_MAJOR);
    // std::vector<CoreCoord> core_coords;
    // for (int i=0; i < num_cores_r * num_cores_c; ++i) {
    //     if (shard_orientation == ShardOrientation::ROW_MAJOR) {
    //         core_coords.push_back(CoreCoord{i % num_cores_c, i / num_cores_c});
    //     } else {
    //         core_coords.push_back(CoreCoord{i / num_cores_r, i % num_cores_r});
    //     }
    // }
    for (int i=0; i < core_coords.size(); ++i) {
        log_debug(tt::LogOp, "worker coord: {} {}", core_coords[i].x, core_coords[i].y);
    }
    std::vector<std::vector<CoreCoord> > core_coords2D;
    if (shard_orientation == ShardOrientation::ROW_MAJOR) {
        for (int i=0; i < num_cores_c / num_cores_per_group; ++i) {
            for (int j=0; j < num_cores_r; ++j) {
                std::vector<CoreCoord> temp;
                for (int k=0; k < num_cores_per_group; ++k) {
                    temp.push_back(CoreCoord{(std::size_t)(k + i * num_cores_per_group), (std::size_t)j});
                }
                core_coords2D.push_back(temp);
            }
        }
    } else {
        for (int i=0; i < num_cores_r / num_cores_per_group; ++i) {
            for (int j=0; j < num_cores_c; ++j) {
                std::vector<CoreCoord> temp;
                for (int k=0; k < num_cores_per_group; ++k) {
                    temp.push_back(CoreCoord{(std::size_t)j, (std::size_t)(k + i * num_cores_per_group)});
                }
                core_coords2D.push_back(temp);
            }
        }
    }

    // one mcast core per batch per group
    std::set<CoreRange> mcast_sender_core_ranges;
    std::set<CoreRange> mcast_receiver_core_ranges;
    uint32_t core_index = 0;
    uint32_t core_index_offset = 0;
    for (int i=0; i < num_batches / num_batches_per_core; ++i) {
        uint32_t core_index = core_index_offset;
        for (int j=0; j < num_groups / num_groups_per_core; ++j) {
            mcast_sender_core_ranges.insert(CoreRange(core_coords[core_index]));
            core_index += num_cores_per_group;
            core_index_offset += num_cores_per_batch * num_cores_per_group;
        }
    }
    for (auto& coord : mcast_sender_core_ranges) {
        log_debug(tt::LogOp, "mcast sender coord: {} {}", coord.start.x, coord.start.y);
    }
    for (int i=0; i < num_cores; ++i) {
        // not found in mcast sender
        if (mcast_sender_core_ranges.find(CoreRange(core_coords[i])) == mcast_sender_core_ranges.end()) {
            mcast_receiver_core_ranges.insert(CoreRange(core_coords[i]));
        }
    }
    for (auto& coord : mcast_receiver_core_ranges) {
        log_debug(tt::LogOp, "mcast receiver coord: {} {}", coord.start.x, coord.start.y);
    }
    CoreRangeSet mcast_sender_cores = CoreRangeSet(mcast_sender_core_ranges);
    CoreRangeSet mcast_receiver_cores = CoreRangeSet(mcast_receiver_core_ranges);
    // mcast groups
    std::vector<std::vector<CoreCoord> > mcast_groups;
    int group_index = -1;
    if (is_height_sharding) {
        for (int i=0; i < num_cores; ++i) {
            if (mcast_sender_core_ranges.find(CoreRange(core_coords[i])) != mcast_sender_core_ranges.end()) {
                group_index += 1;
            }
            if (group_index >= mcast_groups.size()) {
                mcast_groups.push_back(std::vector<CoreCoord>()); // Add a new group
            }
            mcast_groups[group_index].push_back(core_coords[i]);
        }
    } else {
        for (int i=0; i < core_coords2D.size(); ++i) {
            for (int j=0; j < core_coords2D[i].size(); ++j) {
                if (mcast_sender_core_ranges.find(CoreRange(core_coords2D[i][j])) != mcast_sender_core_ranges.end()) {
                    group_index += 1;
                }
                if (group_index >= mcast_groups.size()) {
                    mcast_groups.push_back(std::vector<CoreCoord>()); // Add a new group
                }
                mcast_groups[group_index].push_back(core_coords2D[i][j]);
            }
        }
    }
    for (int i=0; i < mcast_groups.size(); ++i) {
        for (int j=0; j < mcast_groups[i].size(); ++j) {
            log_debug(tt::LogOp, "mcast group: {} coord: {} {}", i, mcast_groups[i][j].x, mcast_groups[i][j].y);
        }
    }
    // how many cores in a mcast group
    uint32_t num_cores_per_mcast_group = mcast_groups[0].size();
    // Mcast args
    auto reduce_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto reduce_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // reader defines
    std::map<string, string> reader_mcast_sender_defines;
    std::map<string, string> reader_mcast_receiver_defines;
    if (gamma.has_value()) {
        reader_mcast_sender_defines["FUSE_GAMMA"] = "1";
        reader_mcast_receiver_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_mcast_sender_defines["FUSE_BETA"] = "1";
        reader_mcast_receiver_defines["FUSE_BETA"] = "1";
    }
    // reader compile time args
    std::vector<uint32_t> reader_mcast_sender_compile_time_args = {
        (std::uint32_t) reduce_receiver_semaphore,
        (std::uint32_t) reduce_sender_semaphore,
        (std::uint32_t) num_cores_per_mcast_group,
        (std::uint32_t) num_groups_per_core,
        (std::uint32_t) num_batches_per_core,
        (std::uint32_t) per_core_N,
        (std::uint32_t) is_channel_divisible_by_tile,
        (std::uint32_t) num_datum_row_per_group % TILE_WIDTH,    // num_cols_last_group
        (std::uint32_t) num_datum_row_per_group,    // group_offset
        (std::uint32_t) num_nz_rows_per_tile, // num_rows_per_batch
        (std::uint32_t) (per_core_M / num_batches_per_core) * per_core_N,
        (std::uint32_t) block_ht,
        (std::uint32_t) block_wt,
        (std::uint32_t) per_core_N * num_nz_rows_per_tile,
        (std::uint32_t) TILE_WIDTH,
        (std::uint32_t) datum_size_bytes
    };
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        (std::uint32_t) reduce_receiver_semaphore,
        (std::uint32_t) reduce_sender_semaphore,
        (std::uint32_t) num_groups_per_core,
        (std::uint32_t) num_batches_per_core,
        (std::uint32_t) per_core_N,
        (std::uint32_t) is_channel_divisible_by_tile,
        (std::uint32_t) num_datum_row_per_group % TILE_WIDTH,    // num_cols_per_group
        (std::uint32_t) num_datum_row_per_group,    // group_offset
        (std::uint32_t) num_nz_rows_per_tile, // num_rows_per_batch
        (std::uint32_t) (per_core_M / num_batches_per_core) * per_core_N,
        (std::uint32_t) block_ht,
        (std::uint32_t) block_wt,
        (std::uint32_t) per_core_N * num_nz_rows_per_tile,
        (std::uint32_t) TILE_WIDTH
    };
    tt_metal::NOC reader_noc = detail::GetPreferredNOCForDRAMRead(device->arch());
    tt_metal::NOC writer_noc = detail::GetPreferredNOCForDRAMWrite(device->arch());
    // reader kernel
    auto reader_mcast_sender_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/groupnorm/kernels/dataflow/reader_mcast_sender_unary_sharded_gn.cpp",
        mcast_sender_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = reader_noc, .compile_args = reader_mcast_sender_compile_time_args, .defines = reader_mcast_sender_defines}
    );
    KernelHandle reader_mcast_receiver_kernels_id = -1;
    if (use_mcast) {
        reader_mcast_receiver_kernels_id = CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/groupnorm/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn.cpp",
            mcast_receiver_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = reader_noc, .compile_args = reader_mcast_receiver_compile_time_args, .defines = reader_mcast_receiver_defines}
        );
    }

    // writer defines
    std::map<string, string> writer_defines;
    // writer compile time args
    std::vector<uint32_t> writer_mcast_sender_compile_time_args = {
        1,
        (std::uint32_t) gamma.has_value(),
        (std::uint32_t) beta.has_value(),
        (std::uint32_t) is_dram(gamma),
        (std::uint32_t) is_dram(beta),
        (std::uint32_t) gamma_beta_num_cols_tile_per_core,
        (std::uint32_t) per_core_N,
        (std::uint32_t) is_channel_divisible_by_tile,
        (std::uint32_t) num_datum_row_per_group % TILE_WIDTH,    // num_cols_per_group
        (std::uint32_t) num_datum_row_per_group,    // group_offset
        (std::uint32_t) num_nz_rows_per_tile, // num_rows_per_batch
        (std::uint32_t) (per_core_M / num_batches_per_core) * per_core_N,
        (std::uint32_t) num_groups_per_core,
        (std::uint32_t) num_batches_per_core,
        (std::uint32_t) block_ht,
        (std::uint32_t) block_wt,
        (std::uint32_t) per_core_N * num_nz_rows_per_tile,
        (std::uint32_t) TILE_WIDTH
    };

    if (gamma.has_value() and gamma.value().get_layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().get_legacy_shape()[3] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = is_power_of_two_at_least_32(gamma_stick_size);
        writer_mcast_sender_compile_time_args.push_back((std::uint32_t) gamma_stick_size_is_power_of_two);
        if (gamma_stick_size_is_power_of_two) {
            uint32_t gamma_log2_stick_size = gamma_stick_size_is_power_of_two ? (std::uint32_t)std::log2(gamma_stick_size) : 0;
            writer_mcast_sender_compile_time_args.push_back((std::uint32_t) gamma_log2_stick_size);
        } else {
            writer_mcast_sender_compile_time_args.push_back(gamma_stick_size);
        }
    } else if (beta.has_value() and beta.value().get_layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().get_legacy_shape()[3] * beta.value().element_size();
        bool beta_stick_size_is_power_of_two = is_power_of_two_at_least_32(beta_stick_size);
        writer_mcast_sender_compile_time_args.push_back((std::uint32_t) beta_stick_size_is_power_of_two);
        if (beta_stick_size_is_power_of_two) {
            uint32_t beta_log2_stick_size = beta_stick_size_is_power_of_two ? (std::uint32_t)std::log2(beta_stick_size) : 0;
            writer_mcast_sender_compile_time_args.push_back((std::uint32_t) beta_log2_stick_size);
        } else {
            writer_mcast_sender_compile_time_args.push_back(beta_stick_size);
        }
    } else {
        writer_mcast_sender_compile_time_args.push_back(0);
        writer_mcast_sender_compile_time_args.push_back(0);
    }

    // writer kernel
    bool use_row_major_kernel = (gamma.has_value() and gamma.value().get_layout() == Layout::ROW_MAJOR) or (beta.has_value() and beta.value().get_layout() == Layout::ROW_MAJOR);
    std::string writer_kernel = use_row_major_kernel ? "tt_eager/tt_dnn/op_library/groupnorm/kernels/dataflow/writer_unary_sharded_gn_rm_gb.cpp" : "tt_eager/tt_dnn/op_library/groupnorm/kernels/dataflow/writer_unary_sharded_gn.cpp";
    auto writer_kernels_id = CreateKernel(
        program,
        writer_kernel,
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = writer_noc, .compile_args = writer_mcast_sender_compile_time_args, .defines = writer_defines}
    );
    // defines
    std::map<string, string> eltwise_binary_defines;
    // compute kernel compile time args
    std::vector<uint32_t> mcast_sender_compute_compile_time_args = {
        1,
        gamma.has_value(),
        beta.has_value(),
        num_cores_per_mcast_group,
        num_batches_per_core,
        num_groups_per_core,
        num_batches_per_core * num_groups_per_core,
        block_ht,
        block_wt,
        block_ht * block_wt,
        subblock_wt,
        num_subblocks_w,
        is_row_major_layout,
        per_core_Mt,
        per_core_Nt,
        per_core_Mt * per_core_Nt,
        num_batches_per_core * block_ht * block_wt,
        (std::uint32_t) is_channel_divisible_by_tile,
        (std::uint32_t) is_row_per_batch_divisible_by_tile
    };
    std::vector<uint32_t> mcast_receiver_compute_compile_time_args = {
        0,
        gamma.has_value(),
        beta.has_value(),
        num_cores_per_mcast_group,
        num_batches_per_core,
        num_groups_per_core,
        num_batches_per_core * num_groups_per_core,
        block_ht,
        block_wt,
        block_ht * block_wt,
        subblock_wt,
        num_subblocks_w,
        is_row_major_layout,
        per_core_Mt,
        per_core_Nt,
        per_core_Mt * per_core_Nt,
        num_batches_per_core * block_ht * block_wt,
        (std::uint32_t) is_channel_divisible_by_tile,
        (std::uint32_t) is_row_per_batch_divisible_by_tile
    };
    // compute kernel
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto mcast_sender_compute_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/groupnorm/kernels/compute/groupnorm_sharded.cpp",
        mcast_sender_cores,
        tt_metal::ComputeConfig{.math_fidelity = fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = mcast_sender_compute_compile_time_args, .defines = eltwise_binary_defines}
    );
    auto mcast_receiver_compute_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/groupnorm/kernels/compute/groupnorm_sharded.cpp",
        mcast_receiver_cores,
        tt_metal::ComputeConfig{.math_fidelity = fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = mcast_receiver_compute_compile_time_args, .defines = eltwise_binary_defines}
    );
    // Create circular buffers
    // in0 sharded
    uint32_t in0_cb_index = CB::c_in0;
    tt_metal::CircularBufferConfig in0_cb_config = tt_metal::CircularBufferConfig(in0_CB_size, {{in0_cb_index, in_data_format}})
		.set_page_size(in0_cb_index, in_single_tile_size).set_globally_allocated_address(*a.buffer());
    auto cb_in0 = tt_metal::CreateCircularBuffer(program, all_cores, in0_cb_config);
    // in2 scaler
    uint32_t in2_cb_index = CB::c_in2;
    tt_metal::CircularBufferConfig in2_cb_config = tt_metal::CircularBufferConfig(in2_CB_size, {{in2_cb_index, cb_data_format}})
		.set_page_size(in2_cb_index, single_tile_size);
    auto cb_in2 = tt_metal::CreateCircularBuffer(program, all_cores, in2_cb_config);
    // in4 scaler-c
    uint32_t in4_cb_index = CB::c_in4;
    tt_metal::CircularBufferConfig in4_cb_config = tt_metal::CircularBufferConfig(in2_CB_size, {{in4_cb_index, cb_data_format}})
		.set_page_size(in4_cb_index, single_tile_size);
    auto cb_in4 = tt_metal::CreateCircularBuffer(program, all_cores, in4_cb_config);
    // in3 eps
    uint32_t in3_cb_index = CB::c_in3;
    tt_metal::CircularBufferConfig in3_cb_config = tt_metal::CircularBufferConfig(in3_CB_size, {{in3_cb_index, cb_data_format}})
		.set_page_size(in3_cb_index, single_tile_size);
    auto cb_in3 = tt_metal::CreateCircularBuffer(program, all_cores, in3_cb_config);
    // zero mask
    uint32_t inz_cb_index = CB::c_intermed4;
    tt_metal::CircularBufferConfig inz_cb_config = tt_metal::CircularBufferConfig(inz_CB_size, {{inz_cb_index, cb_data_format}})
		.set_page_size(inz_cb_index, single_tile_size);
    auto cb_inz = tt_metal::CreateCircularBuffer(program, all_cores, inz_cb_config);
    // zero mask full row
    uint32_t inzf_cb_index = CB::c_intermed6;
    tt_metal::CircularBufferConfig inzf_cb_config = tt_metal::CircularBufferConfig(inz_CB_size, {{inzf_cb_index, cb_data_format}})
		.set_page_size(inzf_cb_index, single_tile_size);
    auto cb_inzf = tt_metal::CreateCircularBuffer(program, all_cores, inzf_cb_config);
    // gamma
    if (gamma.has_value()) {
        uint32_t in5_cb_index = CB::c_in5;
        tt_metal::CircularBufferConfig in5_cb_config = tt_metal::CircularBufferConfig(in5_CB_size, {{in5_cb_index, gamma_beta_cb_data_format}})
            .set_page_size(in5_cb_index, gamma_beta_single_tile_size);
        auto cb_in5 = tt_metal::CreateCircularBuffer(program, all_cores, in5_cb_config);
    }
    // beta
    if (beta.has_value()) {
        uint32_t in6_cb_index = CB::c_in6;
        tt_metal::CircularBufferConfig in6_cb_config = tt_metal::CircularBufferConfig(in6_CB_size, {{in6_cb_index, gamma_beta_cb_data_format}})
            .set_page_size(in6_cb_index, gamma_beta_single_tile_size);
        auto cb_in6 = tt_metal::CreateCircularBuffer(program, all_cores, in6_cb_config);
    }
    // in, for pick values in sharded cb
    uint32_t in_cb_index;
    in_cb_index = CB::c_in7;
    tt_metal::CircularBufferConfig in_cb_config = tt_metal::CircularBufferConfig(in_CB_size, {{in_cb_index, in_data_format}})
        .set_page_size(in_cb_index, in_single_tile_size);
    auto cb_in = tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config);
    // im out
    uint32_t im_out_cb_index;
    im_out_cb_index = CB::c_intermed2;
    tt_metal::CircularBufferConfig im_out_cb_config = tt_metal::CircularBufferConfig(im_out_CB_size, {{im_out_cb_index, out_data_format}})
        .set_page_size(im_out_cb_index, out_single_tile_size);
    auto cb_im_out = tt_metal::CreateCircularBuffer(program, all_cores, im_out_cb_config);
    // xmm_temp
    uint32_t xmm_temp_cb_index;
    xmm_temp_cb_index = CB::c_intermed5;
    tt_metal::CircularBufferConfig xmm_temp_cb_config = tt_metal::CircularBufferConfig(xmm_CB_size, {{xmm_temp_cb_index, cb_data_format}})
        .set_page_size(xmm_temp_cb_index, single_tile_size);
    auto cb_xmm_temp = tt_metal::CreateCircularBuffer(program, all_cores, xmm_temp_cb_config);
    // x
    uint32_t x_cb_index;
    x_cb_index = CB::c_intermed0;
    tt_metal::CircularBufferConfig x_cb_config = tt_metal::CircularBufferConfig(x_CB_size, {{x_cb_index, cb_data_format}})
        .set_page_size(x_cb_index, single_tile_size);
    auto cb_x = tt_metal::CreateCircularBuffer(program, all_cores, x_cb_config);
    // xmm
    uint32_t xmm_cb_index;
    xmm_cb_index = CB::c_intermed1;
    tt_metal::CircularBufferConfig xmm_cb_config = tt_metal::CircularBufferConfig(xmm_CB_size, {{xmm_cb_index, cb_data_format}})
        .set_page_size(xmm_cb_index, single_tile_size);
    auto cb_xmm = tt_metal::CreateCircularBuffer(program, all_cores, xmm_cb_config);
    // ex_partial
    uint32_t ex_cb_partial_index = CB::dataflow0;
    tt_metal::CircularBufferConfig ex_cb_partial_config = tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial_index, cb_data_format}})
		.set_page_size(ex_cb_partial_index, single_tile_size);
    auto cb_ex_partial = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial_config);
    // ex
    uint32_t ex_cb_index = CB::dataflow1;
    // ex_external
    uint32_t ex_cb_external_index = CB::dataflow2;
    tt_metal::CircularBufferConfig ex_cb_external_config = tt_metal::CircularBufferConfig(ex_external_CB_size, {{ex_cb_external_index, cb_data_format}})
		.set_page_size(ex_cb_external_index, single_tile_size);
    auto cb_ex_external = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_external_config);
    // ex_global
    uint32_t ex_global_cb_index = CB::dataflow7;
    std::map<uint8_t, tt::DataFormat> ex_global_cb_data_format_spec {
        {ex_global_cb_index, cb_data_format},
        {ex_cb_index, cb_data_format}
    };
    auto ex_global_cb_config = tt_metal::CircularBufferConfig(ex_global_CB_size, ex_global_cb_data_format_spec)
        .set_page_size(ex_global_cb_index, single_tile_size)
        .set_page_size(ex_cb_index, single_tile_size);
    auto cb_ex_global = tt_metal::CreateCircularBuffer(program, all_cores, ex_global_cb_config);
    // ex2pe
    uint32_t cb_ex2pe_index;
    cb_ex2pe_index = CB::c_intermed3;
    tt_metal::CircularBufferConfig ex2pe_cb_config = tt_metal::CircularBufferConfig(ex2pe_CB_size, {{cb_ex2pe_index, cb_data_format}})
        .set_page_size(cb_ex2pe_index, single_tile_size);
    auto cb_ex2pe = tt_metal::CreateCircularBuffer(program, all_cores, ex2pe_cb_config);
    // out
    uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, {{output_cb_index, out_data_format}})
		.set_page_size(output_cb_index, out_single_tile_size).set_globally_allocated_address(*output.buffer());
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    // Runtime Args
    std::vector<KernelHandle> writer_kernel_ids;
    float winv = 1.0f / std::sqrt(num_rows_per_batch_per_core * num_datum_row_per_group); // bcast-w scaler
    float cinv = 1.0f / std::sqrt(num_cores_per_batch * num_cores_per_group); // bcast-cores scaler
    bfloat16 bfloat_cinv_value = bfloat16(cinv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv_value, bfloat_cinv_value});
    bfloat16 bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union { float f; uint32_t u; } e; e.f = eps;

    log_debug(tt::LogOp, "num_rows_per_batch_per_core: {}", num_rows_per_batch_per_core);
    log_debug(tt::LogOp, "num_datum_row_per_group: {}", num_datum_row_per_group);
    log_debug(tt::LogOp, "num_cores_per_batch: {}", num_cores_per_batch);
    log_debug(tt::LogOp, "num_cores_per_group: {}", num_cores_per_group);

    uint32_t gamma_tile_start_id = 0;
    uint32_t beta_tile_start_id = 0;
    for (int i=0; i < mcast_groups.size(); ++i) {
        auto group = mcast_groups[i];
        bool rectangle_grid = is_rectangle_grid(group);

        for (int j=0; j < group.size(); ++j) {
            CoreCoord core = group[j];
            CoreCoord core_physical = device->worker_core_from_logical_core(core);

            if (j == 0) { // mcast sender
                // get the bounding box for the mcast
                std::vector<CoreCoord> mcast_group_first;
                std::vector<CoreCoord> mcast_group_mid(group);
                std::vector<CoreCoord> mcast_group_last;
                if (not rectangle_grid) {
                    split_and_form_rectangle_grids(group, mcast_group_first, mcast_group_mid, mcast_group_last);
                }

                CoreCoord mcast_start = device->worker_core_from_logical_core(mcast_group_mid.front());
                CoreCoord mcast_end = device->worker_core_from_logical_core(mcast_group_mid.back());

                // if ((mcast_start.x < mcast_end.x) or (mcast_start.y < mcast_end.y)) {
                //     std::swap(mcast_start, mcast_end);
                // }

                if (reader_noc == NOC::NOC_1) {
                    std::swap(mcast_start, mcast_end);
                }
                std::vector<uint32_t> mcast_sender_args;
                mcast_sender_args.push_back(not mcast_group_first.empty());
                mcast_sender_args.push_back(not mcast_group_last.empty());
                mcast_sender_args.push_back(mcast_start.x);
                mcast_sender_args.push_back(mcast_start.y);
                mcast_sender_args.push_back(mcast_end.x);
                mcast_sender_args.push_back(mcast_end.y);
                if (not mcast_group_first.empty()) {
                    mcast_sender_args.push_back(mcast_group_mid.size());
                    log_debug(tt::LogOp, "mcast mid group size: {}", mcast_group_mid.size());
                } else {
                    mcast_sender_args.push_back(mcast_group_mid.size() - 1); // mcast w/o itself
                    log_debug(tt::LogOp, "mcast mid group size: {}", mcast_group_mid.size() - 1);
                }

                log_debug(tt::LogOp, "mcast mid group start coord: {} {} end coord: {} {}", mcast_start.x, mcast_start.y, mcast_end.x, mcast_end.y);


                if (not mcast_group_first.empty()) {
                    CoreCoord mcast_first_start = device->worker_core_from_logical_core(mcast_group_first.front());
                    CoreCoord mcast_first_end = device->worker_core_from_logical_core(mcast_group_first.back());

                    // if ((mcast_first_start.x < mcast_first_end.x) or (mcast_first_start.y < mcast_first_end.y)) {
                    //     std::swap(mcast_first_start, mcast_first_end);
                    // }
                    if (reader_noc == NOC::NOC_1) {
                        std::swap(mcast_start, mcast_end);
                    }
                    mcast_sender_args.push_back(mcast_first_start.x);
                    mcast_sender_args.push_back(mcast_first_start.y);
                    mcast_sender_args.push_back(mcast_first_end.x);
                    mcast_sender_args.push_back(mcast_first_end.y);
                    mcast_sender_args.push_back(mcast_group_first.size() - 1); // mcast w/0 itself

                    log_debug(tt::LogOp, "mcast first group start coord: {} {} end coord: {} {}", mcast_first_start.x, mcast_first_start.y, mcast_first_end.x, mcast_first_end.y);
                    log_debug(tt::LogOp, "mcast first group size: {}", mcast_group_first.size() - 1);
                }
                if (not mcast_group_last.empty()) {
                    CoreCoord mcast_last_start = device->worker_core_from_logical_core(mcast_group_last.front());
                    CoreCoord mcast_last_end = device->worker_core_from_logical_core(mcast_group_last.back());

                    // if ((mcast_last_start.x < mcast_last_end.x) or (mcast_last_start.y < mcast_last_end.y)) {
                    //     std::swap(mcast_last_start, mcast_last_end);
                    // }
                    if (reader_noc == NOC::NOC_1) {
                        std::swap(mcast_start, mcast_end);
                    }
                    mcast_sender_args.push_back(mcast_last_start.x);
                    mcast_sender_args.push_back(mcast_last_start.y);
                    mcast_sender_args.push_back(mcast_last_end.x);
                    mcast_sender_args.push_back(mcast_last_end.y);
                    mcast_sender_args.push_back(mcast_group_last.size());

                    log_debug(tt::LogOp, "mcast last group start coord: {} {} end coord: {} {}", mcast_last_start.x, mcast_last_start.y, mcast_last_end.x, mcast_last_end.y);
                    log_debug(tt::LogOp, "mcast last group size: {}", mcast_group_last.size());
                }

                // add all coords within a group
                std::vector<uint32_t> mcast_noc_xy;
                for (int c=0; c < group.size(); ++c) {
                    CoreCoord coord = device->worker_core_from_logical_core(group[c]);
                    mcast_noc_xy.push_back(coord.x);
                }
                for (int c=0; c < group.size(); ++c) {
                    CoreCoord coord = device->worker_core_from_logical_core(group[c]);
                    mcast_noc_xy.push_back(coord.y);
                }
                mcast_sender_args.insert(mcast_sender_args.end(), mcast_noc_xy.begin(), mcast_noc_xy.end());
                tt_metal::SetRuntimeArgs(program, reader_mcast_sender_kernels_id, core, mcast_sender_args);

            } else { // mcast receiver
                log_debug(tt::LogOp, "mcast receiver receive from coord: {} {}", group.front().x, group.front().y);
                std::vector<uint32_t> mcast_receiver_args;
                mcast_receiver_args.push_back(device->worker_core_from_logical_core(group.front()).x);
                mcast_receiver_args.push_back(device->worker_core_from_logical_core(group.front()).y);
                tt_metal::SetRuntimeArgs(program, reader_mcast_receiver_kernels_id, core, mcast_receiver_args);
            }

        }
    }

    // writer
    for (int i=0; i < core_coords.size(); ++i) {

        auto core = core_coords[i];

        std::vector<uint32_t> writer_mcast_sender_args;
        writer_mcast_sender_args.push_back(packed_cinv_value);
        writer_mcast_sender_args.push_back(packed_winv_value);
        writer_mcast_sender_args.push_back(e.u);
        writer_mcast_sender_args.push_back(gamma_dram_addr);
        writer_mcast_sender_args.push_back(beta_dram_addr);
        writer_mcast_sender_args.push_back(gamma_tile_start_id);
        writer_mcast_sender_args.push_back(beta_tile_start_id);
        tt_metal::SetRuntimeArgs(program, writer_kernels_id, core, writer_mcast_sender_args);
        writer_kernel_ids.push_back(writer_kernels_id);

        if (gamma.has_value()) {
            gamma_tile_start_id = (gamma_tile_start_id + gamma_beta_num_cols_tile_per_core) % (gamma.value().volume() / TILE_WIDTH);
        }
        if (beta.has_value()) {
            beta_tile_start_id = (beta_tile_start_id + gamma_beta_num_cols_tile_per_core) % (beta.value().volume() / TILE_WIDTH);
        }
    }

    auto override_runtime_args_callback = [
            writer_kernel_ids,
            cb_in0,
            cb_output,
            num_cores,
            grid_size
        ]
    (
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer_a = input_tensors.at(0).buffer();
        auto gamma_tensor = optional_input_tensors.at(0);
        auto beta_tensor = optional_input_tensors.at(1);
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_in0, *src_buffer_a);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);

        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};

            auto writer_kernel_id = writer_kernel_ids.at(i);

            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);

            if (gamma_tensor.has_value()) {
                runtime_args[3] = gamma_tensor.value().buffer()->address();
            }
            if (beta_tensor.has_value()) {
                runtime_args[4] = beta_tensor.value().buffer()->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

std::pair<uint32_t, uint32_t> find_max_tile_span(uint32_t W, uint32_t group_size) {
    uint32_t current_position = 0;
    uint32_t max_tile_span = 0;
    uint32_t num_groups_before_start_again_at_tile_beginning = -1;
    bool calc_num_groups_before_start_again_at_tile_beginning = true;

    while (current_position < W) {
        uint32_t group_end = current_position + group_size;
        uint32_t start_tile = current_position / TILE_WIDTH;
        uint32_t end_tile = (group_end - 1) / TILE_WIDTH;
        uint32_t current_tile_span = end_tile - start_tile + 1;

        max_tile_span = std::max(max_tile_span, current_tile_span);

        current_position = group_end;

        if (current_position % TILE_WIDTH == 0 and calc_num_groups_before_start_again_at_tile_beginning == true) {
            num_groups_before_start_again_at_tile_beginning = current_position / group_size;
            calc_num_groups_before_start_again_at_tile_beginning = false;
        }
    }

    return {max_tile_span, num_groups_before_start_again_at_tile_beginning};
}

operation::ProgramWithCallbacks groupnorm_sharded_v2_(
    const Tensor &a,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    const std::optional<const Tensor> input_mask,
    Tensor& output,
    float eps,
    const uint32_t num_groups,
    const uint32_t num_batches,
    MathFidelity fidelity,
    DataType im_data_format,
    CoreCoord grid_size,
    bool inplace
) {
    if (gamma.has_value()) {
        TT_ASSERT(gamma.value().get_layout() == Layout::ROW_MAJOR);
    }
    if (beta.has_value()) {
        TT_ASSERT(beta.value().get_layout() == Layout::ROW_MAJOR);
    }
    TT_ASSERT(a.get_layout() == Layout::ROW_MAJOR);

    bool is_height_sharding = a.get_legacy_shape()[3] == a.shard_spec().value().shape[1];
    // convert data format
    tt::DataFormat in_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat out_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(im_data_format);
    tt::DataFormat gamma_beta_cb_data_format = tt::DataFormat::Float16_b;
    if (gamma.has_value()) {
        gamma_beta_cb_data_format = tt_metal::datatype_to_dataformat_converter(gamma.value().get_dtype());
    }
    if (beta.has_value()) {
        gamma_beta_cb_data_format = tt_metal::datatype_to_dataformat_converter(beta.value().get_dtype());
    }
    tt::DataFormat in_mask_cb_data_format = input_mask.has_value() ? tt_metal::datatype_to_dataformat_converter(input_mask.value().get_dtype()) : tt::DataFormat::Float16_b;
    uint32_t datum_size_bytes = 2; // bfloat16

    TT_ASSERT(out_data_format == in_data_format && "input and output must be the same data format");

    // tile sizes
    uint32_t in_single_tile_size = tt_metal::detail::TileSize(in_data_format);
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    uint32_t out_single_tile_size = tt_metal::detail::TileSize(out_data_format);
    uint32_t gamma_beta_single_tile_size = tt_metal::detail::TileSize(gamma_beta_cb_data_format);
    uint32_t in_mask_single_tile_size = tt_metal::detail::TileSize(in_mask_cb_data_format);
    // shard shape per core
    uint32_t per_core_M = a.shard_spec().value().shape[0];
    uint32_t per_core_N = a.shard_spec().value().shape[1];
    uint32_t per_core_Mt = per_core_M / TILE_HEIGHT;
    uint32_t per_core_Nt = (per_core_N + TILE_WIDTH - 1) / TILE_WIDTH;
    uint32_t per_core_N_bytes_padded = round_up_to_mul32(per_core_N * datum_size_bytes);
    bool reader_repack_output = (per_core_N % TILE_WIDTH) != 0;
    // tensor shape
    const auto shape = a.get_legacy_shape();
    uint32_t H = shape[2] * num_batches;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t W = shape[3];
    uint32_t Wt = W / TILE_WIDTH;
    uint32_t num_datum_row_per_group = W / num_groups;
    uint32_t num_datum_row_per_group_mod_tile_w = num_datum_row_per_group % TILE_WIDTH == 0 ? TILE_WIDTH : num_datum_row_per_group % TILE_WIDTH;
    uint32_t group_size = W / num_groups;
    // grid
    uint32_t num_cores_c = grid_size.x;
    uint32_t num_cores_r = grid_size.y;
    // uint32_t num_cores = num_cores_c * num_cores_r;
    auto all_cores = a.shard_spec().value().grid;
    uint32_t num_cores = all_cores.num_cores();
    auto shard_orientation = a.shard_spec().value().orientation;
    // split each batch into multiple cores
    uint32_t num_shards_r = H / per_core_M;
    uint32_t num_cores_per_batch = num_batches > num_shards_r ? 1 : num_shards_r / num_batches;
    uint32_t num_shards_c = W / per_core_N;
    uint32_t num_cores_per_group = num_groups > num_shards_c ? 1 : num_shards_c / num_groups;
    // each core contains multiple batches
    uint32_t num_batches_per_core = num_batches > num_shards_r ? num_batches / num_shards_r : 1;
    uint32_t num_groups_per_core = num_groups > num_shards_c ? num_groups / num_shards_c : 1;

    TT_ASSERT(per_core_N % num_datum_row_per_group == 0);
    TT_ASSERT(per_core_M % TILE_HEIGHT == 0);
    if (per_core_N != W) {
        if (shard_orientation == ShardOrientation::COL_MAJOR) {
            TT_ASSERT(per_core_N * num_cores_r == W);
            TT_ASSERT(per_core_M * num_cores_c == H);
        } else {
            TT_ASSERT(per_core_N * num_cores_c == W);
            TT_ASSERT(per_core_M * num_cores_r == H);
        }
    }

    TT_ASSERT(per_core_M % TILE_HEIGHT == 0 && "per_core_M must be divisble by TILE_HEIGHT");

    TT_ASSERT(W % num_groups == 0 && "Tensor W must be divisble by num_groups");
    TT_ASSERT(H % per_core_M == 0 && "H dim must be divisible by per_core_M");
    TT_ASSERT(W % per_core_N == 0 && "W dim must be divisible by per_core_N");
    if (num_batches >= num_shards_r) {
        TT_ASSERT(num_batches % num_shards_r == 0 && "num_batches must be divisible by number of cores in a full column");
    } else {
        TT_ASSERT(num_shards_r % num_batches == 0 && "number of cores in a full column must be divisible by num_batches");
    }
    if (num_groups >= num_shards_c) {
        TT_ASSERT(num_groups % num_shards_c == 0 && "num_groups must be divisible by number of cores in a full row");
    } else {
        TT_ASSERT(num_shards_c % num_groups == 0 && "number of cores in a full row must be divisible by num_groups");
    }

    // subblock
    uint32_t num_rows_per_batch_per_core = per_core_M / num_batches_per_core;
    auto [block_wt, num_groups_per_reset] = find_max_tile_span(per_core_N, group_size);
    uint32_t block_ht = per_core_Mt / num_batches_per_core;
    uint32_t subblock_wt = get_max_subblock(block_wt, 8);
    uint32_t num_subblocks_w = block_wt / subblock_wt;
    bool block_wt_last = (per_core_Nt + num_groups_per_core - 1) / num_groups_per_core;

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "num_rows_per_batch_per_core: {}", per_core_M / num_batches_per_core);
    log_debug(tt::LogOp, "per_core_M: {}", per_core_M);
    log_debug(tt::LogOp, "per_core_N: {}", per_core_N);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "H: {}", H);
    log_debug(tt::LogOp, "num_datum_row_per_group: {}", num_datum_row_per_group);
    log_debug(tt::LogOp, "num_batches: {}", num_batches);
    log_debug(tt::LogOp, "num_groups: {}", num_groups);
    log_debug(tt::LogOp, "num_cores_r: {}", num_cores_r);
    log_debug(tt::LogOp, "num_cores_c: {}", num_cores_c);
    log_debug(tt::LogOp, "num_cores_per_batch: {}", num_cores_per_batch);
    log_debug(tt::LogOp, "num_cores_per_group: {}", num_cores_per_group);
    log_debug(tt::LogOp, "num_batches_per_core: {}", num_batches_per_core);
    log_debug(tt::LogOp, "num_groups_per_core: {}", num_groups_per_core);
    log_debug(tt::LogOp, "block_wt: {}", block_wt);
    log_debug(tt::LogOp, "block_wt_last: {}", block_wt_last);
    log_debug(tt::LogOp, "block_ht: {}", block_ht);
    log_debug(tt::LogOp, "subblock_wt: {}", subblock_wt);
    log_debug(tt::LogOp, "num_subblocks_w: {}", num_subblocks_w);
    log_debug(tt::LogOp, "reader_repack_output: {}", reader_repack_output);

    TT_ASSERT(per_core_M % num_batches_per_core == 0 && "shard height must be div by per_core_batch");
    TT_ASSERT(W % num_groups == 0 && "tensor width must be divisible by num_groups!");
    if (shard_orientation == ShardOrientation::ROW_MAJOR and num_groups_per_core == 1) {
        TT_ASSERT(num_cores_c % num_groups == 0 && "for RM shard, when each group is split across cores, num_cores_c must be divisible by num_groups!");
    } else if (shard_orientation == ShardOrientation::COL_MAJOR and num_groups_per_core == 1) {
        TT_ASSERT(num_cores_r % num_groups == 0 && "for CM shard, when each group is split across cores, num_cores_r must be divisible by num_groups!");
    }

    if (per_core_N != W) { // block sharded
        if (shard_orientation == ShardOrientation::ROW_MAJOR and num_batches_per_core == 1) {
            TT_ASSERT(num_cores_r % num_batches == 0 && "for RM shard, when each batch is split across cores, num_cores_r must be divisible by num_batches!");
        } else if (shard_orientation == ShardOrientation::COL_MAJOR and num_groups_per_core == 1) {
            TT_ASSERT(num_cores_c % num_batches == 0 && "for CM shard, when each batch is split across cores, num_cores_c must be divisible by num_batches!");
        }
    } else { // height sharded
        if (num_batches_per_core == 1)
            TT_ASSERT((num_cores_c*num_cores_r) % num_batches == 0 && "for height shard, number of cores must be divisible by num_batches!");
    }

    if (input_mask.has_value()) {
        TT_ASSERT(input_mask.value().get_legacy_shape()[3] == block_wt * TILE_WIDTH && "input mask must have the same width as block_wt");
    }

    // get sharded addr
    auto in0_addr = a.buffer()->address();
    auto out_addr = output.buffer()->address();
    // gamma, beta addr
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    auto input_mask_dram_addr = input_mask.has_value() ? input_mask.value().buffer()->address() : 0;
    // num tiles for a, gamma, beta
    uint32_t num_tiles = a.volume()/TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_HW : 0;
    uint32_t num_input_mask_tiles = input_mask.has_value() ? input_mask.value().volume()/TILE_HW : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    Device *device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_block_tiles = per_core_Nt * per_core_Mt;
    uint32_t in0_CB_size = in0_block_tiles * in_single_tile_size;
    // in2 - scaler
    uint32_t in2_CB_size = single_tile_size;
    // in3 - eps
    uint32_t in3_CB_size = single_tile_size;
    // gamma
    // uint32_t gamma_beta_num_cols_tile_per_core = block_wt * num_groups_per_core;
    uint32_t gamma_beta_num_cols_tile_per_core = per_core_Nt;
    uint32_t in5_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    // beta
    uint32_t in6_CB_size = gamma_beta_num_cols_tile_per_core * gamma_beta_single_tile_size;
    // input mask
    uint32_t input_mask_num_tiles_per_core = block_wt * num_groups_per_core;
    uint32_t in_mask_CB_size = block_wt * in_mask_single_tile_size * 2; // double buffer
    // repack cb
    uint32_t repack_CB_size = per_core_Nt * in_single_tile_size * 2; // double buffer
    // itermediate buffers
    uint32_t interm_block_tiles = block_ht * block_wt;
    uint32_t in_CB_size = in0_CB_size;
    uint32_t im_out_CB_size = out_single_tile_size * interm_block_tiles;
    uint32_t x_CB_size = interm_block_tiles * single_tile_size;
    uint32_t xmm_CB_size = interm_block_tiles * single_tile_size;
    uint32_t ex_partial_CB_size = single_tile_size; // partial Ex
    uint32_t ex_global_CB_size = ex_partial_CB_size; // the final result Ex
    uint32_t xmm2_CB_size = interm_block_tiles * single_tile_size;
    uint32_t ex2pe_CB_size = ex_partial_CB_size;
    // output buffer size
    uint32_t out_CB_size = in0_block_tiles * out_single_tile_size;

    log_debug(tt::LogOp, "per_core_Nt: {}", per_core_Nt);
    log_debug(tt::LogOp, "per_core_Mt: {}", per_core_Mt);
    log_debug(tt::LogOp, "in0_CB_size: {}", in0_CB_size);
    log_debug(tt::LogOp, "in_CB_size: {}", in_CB_size);
    log_debug(tt::LogOp, "gamma_beta_num_cols_tile_per_core: {}", gamma_beta_num_cols_tile_per_core);
    log_debug(tt::LogOp, "in5_CB_size: {}", in5_CB_size);
    log_debug(tt::LogOp, "repack_CB_size: {}", repack_CB_size);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = Program();
    // define core ranges
    bool use_mcast = num_cores_per_batch > 1 or num_cores_per_group > 1;
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    // create a vector of cores, in either RM or CM
    std::vector<CoreCoord> core_coords = grid_to_cores(num_cores, num_cores_c, num_cores_r, shard_orientation == ShardOrientation::ROW_MAJOR);
    for (int i=0; i < core_coords.size(); ++i) {
        log_debug(tt::LogOp, "worker coord: {} {}", core_coords[i].x, core_coords[i].y);
    }
    std::vector<std::vector<CoreCoord> > core_coords2D;
    if (shard_orientation == ShardOrientation::ROW_MAJOR) {
        for (int i=0; i < num_cores_c / num_cores_per_group; ++i) {
            for (int j=0; j < num_cores_r; ++j) {
                std::vector<CoreCoord> temp;
                for (int k=0; k < num_cores_per_group; ++k) {
                    temp.push_back(CoreCoord{(std::size_t)(k + i * num_cores_per_group), (std::size_t)j});
                }
                core_coords2D.push_back(temp);
            }
        }
    } else {
        for (int i=0; i < num_cores_r / num_cores_per_group; ++i) {
            for (int j=0; j < num_cores_c; ++j) {
                std::vector<CoreCoord> temp;
                for (int k=0; k < num_cores_per_group; ++k) {
                    temp.push_back(CoreCoord{(std::size_t)j, (std::size_t)(k + i * num_cores_per_group)});
                }
                core_coords2D.push_back(temp);
            }
        }
    }

    // one mcast core per batch per group
    std::set<CoreRange> mcast_sender_core_ranges;
    std::set<CoreRange> mcast_receiver_core_ranges;
    uint32_t core_index = 0;
    uint32_t core_index_offset = 0;
    for (int i=0; i < num_batches / num_batches_per_core; ++i) {
        uint32_t core_index = core_index_offset;
        for (int j=0; j < num_groups / num_groups_per_core; ++j) {
            mcast_sender_core_ranges.insert(CoreRange(core_coords[core_index]));
            core_index += num_cores_per_group;
            core_index_offset += num_cores_per_batch * num_cores_per_group;
        }
    }
    for (auto& coord : mcast_sender_core_ranges) {
        log_debug(tt::LogOp, "mcast sender coord: {} {}", coord.start.x, coord.start.y);
    }
    for (int i=0; i < num_cores; ++i) {
        // not found in mcast sender
        if (mcast_sender_core_ranges.find(CoreRange(core_coords[i])) == mcast_sender_core_ranges.end()) {
            mcast_receiver_core_ranges.insert(CoreRange(core_coords[i]));
        }
    }
    for (auto& coord : mcast_receiver_core_ranges) {
        log_debug(tt::LogOp, "mcast receiver coord: {} {}", coord.start.x, coord.start.y);
    }
    CoreRangeSet mcast_sender_cores = CoreRangeSet(mcast_sender_core_ranges);
    CoreRangeSet mcast_receiver_cores = CoreRangeSet(mcast_receiver_core_ranges);
    // mcast groups
    std::vector<std::vector<CoreCoord> > mcast_groups;
    int group_index = -1;
    if (is_height_sharding) {
        for (int i=0; i < num_cores; ++i) {
            if (mcast_sender_core_ranges.find(CoreRange(core_coords[i])) != mcast_sender_core_ranges.end()) {
                group_index += 1;
            }
            if (group_index >= mcast_groups.size()) {
                mcast_groups.push_back(std::vector<CoreCoord>()); // Add a new group
            }
            mcast_groups[group_index].push_back(core_coords[i]);
        }
    } else {
        for (int i=0; i < core_coords2D.size(); ++i) {
            for (int j=0; j < core_coords2D[i].size(); ++j) {
                if (mcast_sender_core_ranges.find(CoreRange(core_coords2D[i][j])) != mcast_sender_core_ranges.end()) {
                    group_index += 1;
                }
                if (group_index >= mcast_groups.size()) {
                    mcast_groups.push_back(std::vector<CoreCoord>()); // Add a new group
                }
                mcast_groups[group_index].push_back(core_coords2D[i][j]);
            }
        }
    }
    for (int i=0; i < mcast_groups.size(); ++i) {
        for (int j=0; j < mcast_groups[i].size(); ++j) {
            log_debug(tt::LogOp, "mcast group: {} coord: {} {}", i, mcast_groups[i][j].x, mcast_groups[i][j].y);
        }
    }
    // how many cores in a mcast group
    uint32_t num_cores_per_mcast_group = mcast_groups[0].size();
    // Mcast args
    auto reduce_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto reduce_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // reader defines
    std::map<string, string> reader_mcast_sender_defines;
    std::map<string, string> reader_mcast_receiver_defines;
    if (gamma.has_value()) {
        reader_mcast_sender_defines["FUSE_GAMMA"] = "1";
        reader_mcast_receiver_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_mcast_sender_defines["FUSE_BETA"] = "1";
        reader_mcast_receiver_defines["FUSE_BETA"] = "1";
    }
    if (reader_repack_output) {
        reader_mcast_sender_defines["READER_REPACK"] = "1";
        reader_mcast_receiver_defines["READER_REPACK"] = "1";
    }
    // reader compile time args
    std::vector<uint32_t> reader_mcast_sender_compile_time_args = {
        (std::uint32_t) reduce_receiver_semaphore,
        (std::uint32_t) reduce_sender_semaphore,
        (std::uint32_t) num_cores_per_mcast_group,
        (std::uint32_t) num_groups_per_core * num_batches_per_core,
        (std::uint32_t) per_core_Nt,
        (std::uint32_t) per_core_N_bytes_padded,
        (std::uint32_t) per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t) datum_size_bytes,
        (std::uint32_t) per_core_Mt,
        (std::uint32_t) TILE_HEIGHT
    };
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        (std::uint32_t) reduce_receiver_semaphore,
        (std::uint32_t) reduce_sender_semaphore,
        (std::uint32_t) num_groups_per_core * num_batches_per_core,
        (std::uint32_t) per_core_Nt,
        (std::uint32_t) per_core_N_bytes_padded,
        (std::uint32_t) per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t) per_core_Mt,
        (std::uint32_t) TILE_HEIGHT
    };
    tt_metal::NOC reader_noc = detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC writer_noc = detail::GetPreferredNOCForDRAMRead(device->arch());
    // reader kernel
    auto reader_mcast_sender_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/groupnorm/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp",
        mcast_sender_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = reader_noc, .compile_args = reader_mcast_sender_compile_time_args, .defines = reader_mcast_sender_defines}
    );
    KernelHandle reader_mcast_receiver_kernels_id = -1;
    if (use_mcast) {
        reader_mcast_receiver_kernels_id = CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/groupnorm/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp",
            mcast_receiver_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = reader_noc, .compile_args = reader_mcast_receiver_compile_time_args, .defines = reader_mcast_receiver_defines}
        );
    }

    // writer defines
    std::map<string, string> writer_defines;
    // writer compile time args
    std::vector<uint32_t> writer_mcast_sender_compile_time_args = {
        1,
        (std::uint32_t) gamma.has_value(),
        (std::uint32_t) beta.has_value(),
        (std::uint32_t) is_dram(gamma),
        (std::uint32_t) is_dram(beta),
        (std::uint32_t) is_dram(input_mask),
        (std::uint32_t) gamma_beta_num_cols_tile_per_core,
        (std::uint32_t) per_core_N,
        (std::uint32_t) per_core_N * datum_size_bytes,
        (std::uint32_t) per_core_Nt * TILE_WIDTH * datum_size_bytes,
        (std::uint32_t) num_groups_per_core,
        (std::uint32_t) num_batches_per_core,
        (std::uint32_t) block_wt
    };

    if (gamma.has_value() and gamma.value().get_layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().get_legacy_shape()[3] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = is_power_of_two_at_least_32(gamma_stick_size);
        writer_mcast_sender_compile_time_args.push_back((std::uint32_t) gamma_stick_size_is_power_of_two);
        if (gamma_stick_size_is_power_of_two) {
            uint32_t gamma_log2_stick_size = gamma_stick_size_is_power_of_two ? (std::uint32_t)std::log2(gamma_stick_size) : 0;
            writer_mcast_sender_compile_time_args.push_back((std::uint32_t) gamma_log2_stick_size);
        } else {
            writer_mcast_sender_compile_time_args.push_back(gamma_stick_size);
        }
    } else if (beta.has_value() and beta.value().get_layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().get_legacy_shape()[3] * beta.value().element_size();
        bool beta_stick_size_is_power_of_two = is_power_of_two_at_least_32(beta_stick_size);
        writer_mcast_sender_compile_time_args.push_back((std::uint32_t) beta_stick_size_is_power_of_two);
        if (beta_stick_size_is_power_of_two) {
            uint32_t beta_log2_stick_size = beta_stick_size_is_power_of_two ? (std::uint32_t)std::log2(beta_stick_size) : 0;
            writer_mcast_sender_compile_time_args.push_back((std::uint32_t) beta_log2_stick_size);
        } else {
            writer_mcast_sender_compile_time_args.push_back(beta_stick_size);
        }
    } else {
        writer_mcast_sender_compile_time_args.push_back(0);
        writer_mcast_sender_compile_time_args.push_back(0);
    }

    // writer kernel
    bool use_row_major_kernel = true;
    std::string writer_kernel = "tt_eager/tt_dnn/op_library/groupnorm/kernels/dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp";
    auto writer_kernels_id = CreateKernel(
        program,
        writer_kernel,
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = writer_noc, .compile_args = writer_mcast_sender_compile_time_args, .defines = writer_defines}
    );
    // defines
    std::map<string, string> eltwise_binary_defines;
    if (reader_repack_output) {
        eltwise_binary_defines["READER_REPACK"] = "1";
    }
    // compute kernel compile time args
    std::vector<uint32_t> mcast_sender_compute_compile_time_args = {
        (std::uint32_t) 1,
        (std::uint32_t) gamma.has_value(),
        (std::uint32_t) beta.has_value(),
        (std::uint32_t) num_cores_per_mcast_group,
        (std::uint32_t) num_batches_per_core,
        (std::uint32_t) num_groups_per_core,

        (std::uint32_t) num_datum_row_per_group_mod_tile_w,

        (std::uint32_t) block_ht,
        (std::uint32_t) block_wt,
        (std::uint32_t) block_ht * block_wt,

        (std::uint32_t) subblock_wt,
        (std::uint32_t) num_subblocks_w,

        (std::uint32_t) per_core_Mt,
        (std::uint32_t) per_core_Nt,
        (std::uint32_t) per_core_Mt * per_core_Nt,

        (std::uint32_t) per_core_Nt * TILE_HW * datum_size_bytes,  // per_core_N_tile_bytes
        (std::uint32_t) num_groups_per_reset,
        (std::uint32_t) single_tile_size,
        (std::uint32_t) per_core_Mt * per_core_Nt / num_batches_per_core,
        (std::uint32_t) num_groups_per_core * block_wt,
        (std::uint32_t) block_wt_last,
        (std::uint32_t) (num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t) num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t) num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH

    };
    std::vector<uint32_t> mcast_receiver_compute_compile_time_args = {
        (std::uint32_t) 0,
        (std::uint32_t) gamma.has_value(),
        (std::uint32_t) beta.has_value(),
        (std::uint32_t) num_cores_per_mcast_group,
        (std::uint32_t) num_batches_per_core,
        (std::uint32_t) num_groups_per_core,

        (std::uint32_t) num_datum_row_per_group_mod_tile_w,

        (std::uint32_t) block_ht,
        (std::uint32_t) block_wt,
        (std::uint32_t) block_ht * block_wt,

        (std::uint32_t) subblock_wt,
        (std::uint32_t) num_subblocks_w,

        (std::uint32_t) per_core_Mt,
        (std::uint32_t) per_core_Nt,
        (std::uint32_t) per_core_Mt * per_core_Nt,

        (std::uint32_t) per_core_Nt * TILE_HW * datum_size_bytes,  // per_core_N_tile_bytes
        (std::uint32_t) num_groups_per_reset,
        (std::uint32_t) single_tile_size,
        (std::uint32_t) per_core_Mt * per_core_Nt / num_batches_per_core,
        (std::uint32_t) num_groups_per_core * block_wt,
        (std::uint32_t) block_wt_last,
        (std::uint32_t) (num_datum_row_per_group_mod_tile_w & (num_datum_row_per_group_mod_tile_w - 1)) == 0,
        (std::uint32_t) num_datum_row_per_group < TILE_WIDTH,
        (std::uint32_t) num_datum_row_per_group - (block_wt - 1) * TILE_WIDTH
    };
    // compute kernel
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto mcast_sender_compute_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/groupnorm/kernels/compute/groupnorm_sharded_v2.cpp",
        mcast_sender_cores,
        tt_metal::ComputeConfig{.math_fidelity = fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = mcast_sender_compute_compile_time_args, .defines = eltwise_binary_defines}
    );
    auto mcast_receiver_compute_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/groupnorm/kernels/compute/groupnorm_sharded_v2.cpp",
        mcast_receiver_cores,
        tt_metal::ComputeConfig{.math_fidelity = fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = mcast_receiver_compute_compile_time_args, .defines = eltwise_binary_defines}
    );
    // Create circular buffers
    uint32_t in0_cb_index = CB::c_in0;
    uint32_t output_cb_index = CB::c_out0;
    CBHandle cb_in0;
    CBHandle cb_output;
    if (inplace) {
        std::map<uint8_t, tt::DataFormat> in0_out0_cb_data_format_spec {
            {in0_cb_index, in_data_format},
            {output_cb_index, in_data_format}
        };
        CircularBufferConfig in0_out0_cb_config = tt_metal::CircularBufferConfig(in0_CB_size, in0_out0_cb_data_format_spec)
            .set_page_size(in0_cb_index, in_single_tile_size)
            .set_page_size(output_cb_index, in_single_tile_size).set_globally_allocated_address(*a.buffer());

        cb_in0 = tt_metal::CreateCircularBuffer(program, all_cores, in0_out0_cb_config);
        cb_output = cb_in0;
    } else {
        tt_metal::CircularBufferConfig in0_cb_config = tt_metal::CircularBufferConfig(in0_CB_size, {{in0_cb_index, in_data_format}})
		    .set_page_size(in0_cb_index, in_single_tile_size).set_globally_allocated_address(*a.buffer());

        tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, {{output_cb_index, out_data_format}})
		    .set_page_size(output_cb_index, out_single_tile_size).set_globally_allocated_address(*output.buffer());

        cb_in0 = tt_metal::CreateCircularBuffer(program, all_cores, in0_cb_config);
        cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    }

    // in - stores tilized input
    uint32_t in_cb_index = CB::c_intermed5;
    tt_metal::CircularBufferConfig in_cb_config = tt_metal::CircularBufferConfig(in_CB_size, {{in_cb_index, in_data_format}})
        .set_page_size(in_cb_index, in_single_tile_size);
    auto cb_in = tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config);
    // out - stores tilized output
    uint32_t out_cb_index = CB::c_intermed6;
    tt_metal::CircularBufferConfig out_cb_config = tt_metal::CircularBufferConfig(in_CB_size, {{out_cb_index, in_data_format}})
        .set_page_size(out_cb_index, in_single_tile_size);
    auto cb_out = tt_metal::CreateCircularBuffer(program, all_cores, out_cb_config);
    // in2 scaler - for partial Ex
    uint32_t in2_cb_index = CB::c_in2;
    tt_metal::CircularBufferConfig in2_cb_config = tt_metal::CircularBufferConfig(in2_CB_size, {{in2_cb_index, cb_data_format}})
		.set_page_size(in2_cb_index, single_tile_size);
    auto cb_in2 = tt_metal::CreateCircularBuffer(program, all_cores, in2_cb_config);
    // in3 eps
    uint32_t in3_cb_index = CB::c_in3;
    tt_metal::CircularBufferConfig in3_cb_config = tt_metal::CircularBufferConfig(in3_CB_size, {{in3_cb_index, cb_data_format}})
		.set_page_size(in3_cb_index, single_tile_size);
    auto cb_in3 = tt_metal::CreateCircularBuffer(program, all_cores, in3_cb_config);
    // in4 scaler-c
    uint32_t in4_cb_index = CB::c_in4;
    tt_metal::CircularBufferConfig in4_cb_config = tt_metal::CircularBufferConfig(in2_CB_size, {{in4_cb_index, cb_data_format}})
		.set_page_size(in4_cb_index, single_tile_size);
    auto cb_in4 = tt_metal::CreateCircularBuffer(program, all_cores, in4_cb_config);
    // gamma
    if (gamma.has_value()) {
        uint32_t in5_cb_index = CB::c_in5;
        tt_metal::CircularBufferConfig in5_cb_config = tt_metal::CircularBufferConfig(in5_CB_size, {{in5_cb_index, gamma_beta_cb_data_format}})
            .set_page_size(in5_cb_index, gamma_beta_single_tile_size);
        auto cb_in5 = tt_metal::CreateCircularBuffer(program, all_cores, in5_cb_config);
    }
    // beta
    if (beta.has_value()) {
        uint32_t in6_cb_index = CB::c_in6;
        tt_metal::CircularBufferConfig in6_cb_config = tt_metal::CircularBufferConfig(in6_CB_size, {{in6_cb_index, gamma_beta_cb_data_format}})
            .set_page_size(in6_cb_index, gamma_beta_single_tile_size);
        auto cb_in6 = tt_metal::CreateCircularBuffer(program, all_cores, in6_cb_config);
    }
    // input mask
    if (input_mask.has_value()) {
        uint32_t in_mask_cb_index = CB::c_intermed4;
        tt_metal::CircularBufferConfig in_mask_cb_config = tt_metal::CircularBufferConfig(in_mask_CB_size, {{in_mask_cb_index, in_mask_cb_data_format}})
            .set_page_size(in_mask_cb_index, in_mask_single_tile_size);
        auto cb_inz = tt_metal::CreateCircularBuffer(program, all_cores, in_mask_cb_config);
    }
    if (reader_repack_output) {
        uint32_t repack_cb_index = CB::c_intermed2;
        uint32_t repack_out_cb_index = CB::c_intermed7;
        std::map<uint8_t, tt::DataFormat> in0_out0_cb_data_format_spec {
            {repack_cb_index, in_data_format},
            {repack_out_cb_index, in_data_format}
        };
        tt_metal::CircularBufferConfig repack_cb_config = tt_metal::CircularBufferConfig(repack_CB_size, in0_out0_cb_data_format_spec)
            .set_page_size(repack_cb_index, in_single_tile_size)
            .set_page_size(repack_out_cb_index, in_single_tile_size);
        auto cb_inz = tt_metal::CreateCircularBuffer(program, all_cores, repack_cb_config);
    }
    // x
    uint32_t x_cb_index = CB::c_intermed0;
    tt_metal::CircularBufferConfig x_cb_config = tt_metal::CircularBufferConfig(x_CB_size, {{x_cb_index, cb_data_format}})
        .set_page_size(x_cb_index, single_tile_size);
    auto cb_x = tt_metal::CreateCircularBuffer(program, all_cores, x_cb_config);
    // xmm
    uint32_t xmm_cb_index = CB::c_intermed1;
    tt_metal::CircularBufferConfig xmm_cb_config = tt_metal::CircularBufferConfig(xmm_CB_size, {{xmm_cb_index, cb_data_format}})
        .set_page_size(xmm_cb_index, single_tile_size);
    auto cb_xmm = tt_metal::CreateCircularBuffer(program, all_cores, xmm_cb_config);
    // ex_partial
    uint32_t ex_cb_partial_index = CB::dataflow0;
    tt_metal::CircularBufferConfig ex_cb_partial_config = tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial_index, cb_data_format}})
		.set_page_size(ex_cb_partial_index, single_tile_size);
    auto cb_ex_partial = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial_config);
    // ex_external
    uint32_t ex_cb_external_index = CB::dataflow2;
    tt_metal::CircularBufferConfig ex_cb_external_config = tt_metal::CircularBufferConfig(single_tile_size * num_cores_per_mcast_group, {{ex_cb_external_index, cb_data_format}})
		.set_page_size(ex_cb_external_index, single_tile_size);
    auto cb_ex_external = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_external_config);
    // ex_global
    uint32_t ex_cb_index = CB::dataflow1;
    uint32_t ex_global_cb_index = CB::dataflow7;
    std::map<uint8_t, tt::DataFormat> ex_global_cb_data_format_spec {
        {ex_global_cb_index, cb_data_format},
        {ex_cb_index, cb_data_format}
    };
    auto ex_global_cb_config = tt_metal::CircularBufferConfig(ex_global_CB_size, ex_global_cb_data_format_spec)
        .set_page_size(ex_global_cb_index, single_tile_size)
        .set_page_size(ex_cb_index, single_tile_size);
    auto cb_ex_global = tt_metal::CreateCircularBuffer(program, all_cores, ex_global_cb_config);
    // ex2pe
    uint32_t cb_ex2pe_index;
    cb_ex2pe_index = CB::c_intermed3;
    tt_metal::CircularBufferConfig ex2pe_cb_config = tt_metal::CircularBufferConfig(ex2pe_CB_size, {{cb_ex2pe_index, cb_data_format}})
        .set_page_size(cb_ex2pe_index, single_tile_size);
    auto cb_ex2pe = tt_metal::CreateCircularBuffer(program, all_cores, ex2pe_cb_config);

    // Runtime Args
    std::vector<KernelHandle> writer_kernel_ids;
    float winv = 1.0f / std::sqrt(num_rows_per_batch_per_core * num_datum_row_per_group); // bcast-w scaler
    bfloat16 bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    float cinv = 1.0f / std::sqrt(num_cores_per_batch * num_cores_per_group); // bcast-cores scaler
    bfloat16 bfloat_cinv_value = bfloat16(cinv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv_value, bfloat_cinv_value});
    union { float f; uint32_t u; } e; e.f = eps;

    log_debug(tt::LogOp, "num_rows_per_batch_per_core: {}", num_rows_per_batch_per_core);
    log_debug(tt::LogOp, "num_datum_row_per_group: {}", num_datum_row_per_group);
    log_debug(tt::LogOp, "num_cores_per_batch: {}", num_cores_per_batch);
    log_debug(tt::LogOp, "num_cores_per_group: {}", num_cores_per_group);

    for (int i=0; i < mcast_groups.size(); ++i) {
        auto group = mcast_groups[i];
        bool rectangle_grid = is_rectangle_grid(group);

        for (int j=0; j < group.size(); ++j) {
            CoreCoord core = group[j];
            CoreCoord core_physical = device->worker_core_from_logical_core(core);

            if (j == 0) { // mcast sender
                // get the bounding box for the mcast
                std::vector<CoreCoord> mcast_group_first;
                std::vector<CoreCoord> mcast_group_mid(group);
                std::vector<CoreCoord> mcast_group_last;
                if (not rectangle_grid) {
                    split_and_form_rectangle_grids(group, mcast_group_first, mcast_group_mid, mcast_group_last);
                }

                CoreCoord mcast_start = device->worker_core_from_logical_core(mcast_group_mid.front());
                CoreCoord mcast_end = device->worker_core_from_logical_core(mcast_group_mid.back());

                if (reader_noc == NOC::NOC_1) {
                    std::swap(mcast_start, mcast_end);
                }
                std::vector<uint32_t> mcast_sender_args;
                mcast_sender_args.push_back(not mcast_group_first.empty());
                mcast_sender_args.push_back(not mcast_group_last.empty());
                mcast_sender_args.push_back(mcast_start.x);
                mcast_sender_args.push_back(mcast_start.y);
                mcast_sender_args.push_back(mcast_end.x);
                mcast_sender_args.push_back(mcast_end.y);
                if (not mcast_group_first.empty()) {
                    mcast_sender_args.push_back(mcast_group_mid.size());
                    log_debug(tt::LogOp, "mcast mid group size: {}", mcast_group_mid.size());
                } else {
                    mcast_sender_args.push_back(mcast_group_mid.size() - 1); // mcast w/o itself
                    log_debug(tt::LogOp, "mcast mid group size: {}", mcast_group_mid.size() - 1);
                }

                log_debug(tt::LogOp, "mcast mid group start coord: {} {} end coord: {} {}", mcast_start.x, mcast_start.y, mcast_end.x, mcast_end.y);

                if (not mcast_group_first.empty()) {
                    CoreCoord mcast_first_start = device->worker_core_from_logical_core(mcast_group_first.front());
                    CoreCoord mcast_first_end = device->worker_core_from_logical_core(mcast_group_first.back());

                    if (reader_noc == NOC::NOC_1) {
                        std::swap(mcast_start, mcast_end);
                    }
                    mcast_sender_args.push_back(mcast_first_start.x);
                    mcast_sender_args.push_back(mcast_first_start.y);
                    mcast_sender_args.push_back(mcast_first_end.x);
                    mcast_sender_args.push_back(mcast_first_end.y);
                    mcast_sender_args.push_back(mcast_group_first.size() - 1); // mcast w/0 itself

                    log_debug(tt::LogOp, "mcast first group start coord: {} {} end coord: {} {}", mcast_first_start.x, mcast_first_start.y, mcast_first_end.x, mcast_first_end.y);
                    log_debug(tt::LogOp, "mcast first group size: {}", mcast_group_first.size() - 1);
                }
                if (not mcast_group_last.empty()) {
                    CoreCoord mcast_last_start = device->worker_core_from_logical_core(mcast_group_last.front());
                    CoreCoord mcast_last_end = device->worker_core_from_logical_core(mcast_group_last.back());

                    if (reader_noc == NOC::NOC_1) {
                        std::swap(mcast_start, mcast_end);
                    }
                    mcast_sender_args.push_back(mcast_last_start.x);
                    mcast_sender_args.push_back(mcast_last_start.y);
                    mcast_sender_args.push_back(mcast_last_end.x);
                    mcast_sender_args.push_back(mcast_last_end.y);
                    mcast_sender_args.push_back(mcast_group_last.size());

                    log_debug(tt::LogOp, "mcast last group start coord: {} {} end coord: {} {}", mcast_last_start.x, mcast_last_start.y, mcast_last_end.x, mcast_last_end.y);
                    log_debug(tt::LogOp, "mcast last group size: {}", mcast_group_last.size());
                }

                // add all coords within a group
                std::vector<uint32_t> mcast_noc_xy;
                for (int c=0; c < group.size(); ++c) {
                    CoreCoord coord = device->worker_core_from_logical_core(group[c]);
                    mcast_noc_xy.push_back(coord.x);
                }
                for (int c=0; c < group.size(); ++c) {
                    CoreCoord coord = device->worker_core_from_logical_core(group[c]);
                    mcast_noc_xy.push_back(coord.y);
                }
                mcast_sender_args.insert(mcast_sender_args.end(), mcast_noc_xy.begin(), mcast_noc_xy.end());
                tt_metal::SetRuntimeArgs(program, reader_mcast_sender_kernels_id, core, mcast_sender_args);

            } else { // mcast receiver
                log_debug(tt::LogOp, "mcast receiver receive from coord: {} {}", group.front().x, group.front().y);
                std::vector<uint32_t> mcast_receiver_args;
                mcast_receiver_args.push_back(device->worker_core_from_logical_core(group.front()).x);
                mcast_receiver_args.push_back(device->worker_core_from_logical_core(group.front()).y);
                tt_metal::SetRuntimeArgs(program, reader_mcast_receiver_kernels_id, core, mcast_receiver_args);
            }

        }
    }

    // writer
    uint32_t gamma_tile_start_id = 0;
    uint32_t beta_tile_start_id = 0;
    uint32_t input_mask_tile_start_id = 0;
    for (int i=0; i < core_coords.size(); ++i) {

        auto core = core_coords[i];

        std::vector<uint32_t> writer_mcast_sender_args;
        writer_mcast_sender_args.push_back(packed_cinv_value);
        writer_mcast_sender_args.push_back(packed_winv_value);
        writer_mcast_sender_args.push_back(e.u);
        writer_mcast_sender_args.push_back(gamma_dram_addr);
        writer_mcast_sender_args.push_back(beta_dram_addr);
        writer_mcast_sender_args.push_back(input_mask_dram_addr);
        writer_mcast_sender_args.push_back(gamma_tile_start_id);
        writer_mcast_sender_args.push_back(beta_tile_start_id);
        writer_mcast_sender_args.push_back(input_mask_tile_start_id);
        tt_metal::SetRuntimeArgs(program, writer_kernels_id, core, writer_mcast_sender_args);
        writer_kernel_ids.push_back(writer_kernels_id);

        if (gamma.has_value()) {
            gamma_tile_start_id = (gamma_tile_start_id + gamma_beta_num_cols_tile_per_core) % (gamma.value().volume() / TILE_WIDTH);
        }
        if (beta.has_value()) {
            beta_tile_start_id = (beta_tile_start_id + gamma_beta_num_cols_tile_per_core) % (beta.value().volume() / TILE_WIDTH);
        }
        if (input_mask.has_value()) {
            input_mask_tile_start_id = (input_mask_tile_start_id + input_mask_num_tiles_per_core) % (input_mask.value().volume() / TILE_HW);
        }
    }

    auto override_runtime_args_callback = [
            writer_kernel_ids,
            cb_in0,
            cb_output,
            num_cores,
            grid_size
        ]
    (
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer_a = input_tensors.at(0).buffer();
        auto gamma_tensor = optional_input_tensors.at(0);
        auto beta_tensor = optional_input_tensors.at(1);
        auto mask_tensor = optional_input_tensors.at(2);
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_in0, *src_buffer_a);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);

        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};

            auto writer_kernel_id = writer_kernel_ids.at(i);

            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);

            if (gamma_tensor.has_value()) {
                runtime_args[3] = gamma_tensor.value().buffer()->address();
            }
            if (beta_tensor.has_value()) {
                runtime_args[4] = beta_tensor.value().buffer()->address();
            }
            if (mask_tensor.has_value()) {
                runtime_args[5] = mask_tensor.value().buffer()->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

void GroupNorm::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 1 and optional_input_tensors.size() <= 3, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);
    const auto& input_mask = optional_input_tensors.at(2);
    TT_FATAL(a.get_layout() == Layout::ROW_MAJOR);
    TT_FATAL(a.get_dtype() == DataType::BFLOAT16);
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    TT_FATAL(a.get_legacy_shape()[3] % this->num_groups == 0,  "channel must be divisible by num_groups!");
    TT_FATAL(a.get_legacy_shape()[1] == 1,  "input tensor shape[1] must be 1!");

    if (gamma.has_value()) {
        if (gamma.value().get_layout() == Layout::TILE) {
            TT_FATAL(a.get_legacy_shape()[3] == gamma.value().get_legacy_shape()[3], fmt::format("{} != {}", a.get_legacy_shape()[3], gamma.value().get_legacy_shape()[3]));
            TT_FATAL(a.device() == gamma.value().device());
            TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(gamma.value().get_legacy_shape()[2] == TILE_HEIGHT);
        } else {
            TT_FATAL(gamma.value().get_layout() == Layout::ROW_MAJOR);
            TT_FATAL((gamma.value().get_legacy_shape()[3] == TILE_WIDTH));
            TT_FATAL(a.device() == gamma.value().device());
            TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(gamma.value().get_dtype() == DataType::BFLOAT16);
        }
        if (beta.has_value()) {
            TT_FATAL(gamma.value().get_layout() == beta.value().get_layout());
        }
    }

    if (beta.has_value()) {
        if (beta.value().get_layout() == Layout::TILE) {
            TT_FATAL(a.get_legacy_shape()[3] == beta.value().get_legacy_shape()[3]);
            TT_FATAL(a.device() == beta.value().device());
            TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(beta.value().get_legacy_shape()[2] == TILE_HEIGHT);
        } else {
            TT_FATAL(beta.value().get_layout() == Layout::ROW_MAJOR);
            TT_FATAL(beta.value().get_legacy_shape()[3] == TILE_WIDTH);
            TT_FATAL(a.device() == beta.value().device());
            TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(beta.value().get_dtype() == DataType::BFLOAT16);
        }
    }

    if (input_mask.has_value()) {
        TT_FATAL(input_mask.value().get_layout() == Layout::TILE);
        TT_FATAL(input_mask.value().get_legacy_shape()[1] == this->num_groups);
        TT_FATAL(input_mask.value().get_legacy_shape()[2] == TILE_HEIGHT);
        TT_FATAL(input_mask.value().get_legacy_shape()[3] % TILE_WIDTH == 0);
    }
}
std::vector<Shape> GroupNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}
std::vector<Tensor> GroupNorm::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->program_config.inplace) {
        return {input_tensors.at(0)};
    } else {
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = input_tensor.shard_spec();
        return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), program_config.out_data_format, Layout::ROW_MAJOR, input_tensor.device(), mem_config)};
    }
}
operation::ProgramWithCallbacks GroupNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);
    const auto& input_mask = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);

    MathFidelity fidelity = this->program_config.math_fidelity;
    uint32_t num_cores_x = this->program_config.compute_with_storage_grid_size.x;
    uint32_t num_cores_y = this->program_config.compute_with_storage_grid_size.y;
    bool inplace = this->program_config.inplace;
    CoreCoord grid_size = CoreCoord(num_cores_x, num_cores_y);
    uint32_t batch = a.get_legacy_shape()[0];

    if (input_mask.has_value()) {
        return groupnorm_sharded_v2_(
                                    a, gamma, beta, input_mask, output_tensor, this->eps,
                                    this->num_groups, batch,
                                    fidelity,
                                    program_config.im_data_format,
                                    program_config.compute_with_storage_grid_size,
                                    inplace
                                    );
    } else {
        return groupnorm_sharded_(
                                    a, gamma, beta, output_tensor, this->eps,
                                    this->num_groups, batch,
                                    fidelity,
                                    program_config.im_data_format,
                                    program_config.compute_with_storage_grid_size
                                    );
    }
}
tt::stl::reflection::Attributes GroupNorm::attributes() const {
    return {
        {"eps", this->eps},
        {"num_groups", this->num_groups},
        {"output_mem_config", this->output_mem_config}
    };
}

}   // namespace primary
}   // namespace operations

}  // namespace tt
