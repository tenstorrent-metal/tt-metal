// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

inline bool is_dram(const Tensor& input_tensor) { return input_tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> input_tensor) {
     return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

// computes layernorm(a+*b)*gamma + beta
// if b is nullptr it's treated as zero (no addition)
operation::ProgramWithCallbacks layernorm_multi_core(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    LayerNormType norm_type,
    float eps,
    DeviceComputeKernelConfig compute_kernel_config
) {
    bool rms_norm = norm_type == LayerNormType::RMSNORM;
    const auto shape = a.get_legacy_shape();
    uint32_t W = shape[3], H = shape[2];
    uint32_t HW = H*W;
    uint32_t NC = a.volume() / HW;

    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute
    const auto& a_dtype = a.get_dtype();

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = a.volume() / TILE_HW;

    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    //////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device *device = a.device();
    auto dst_addr = output.buffer()->address();


    ////////////////////////////////////////////////////////////////////////////
    //                Circular Buffer Data Format Setup
    //////////////////////////////////////////////////////////////////////////
    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::WORMHOLE_B0, "kernel config is not for wormhole_b0");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = tt_metal::datatype_to_dataformat_converter(a.get_dtype()) == tt::DataFormat::Float32 ? true : compute_kernel_config.fp32_dest_acc_en;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);

    uint32_t block_size = fp32_dest_acc_en ? find_max_divisor(Wt, 4) : find_max_divisor(Wt, 8);

    tt::DataFormat in_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat out_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format = gamma.has_value() ? tt_metal::datatype_to_dataformat_converter(gamma.value().get_dtype()) : tt::DataFormat::Float16_b;
    tt::DataFormat beta_cb_data_format = beta.has_value() ? tt_metal::datatype_to_dataformat_converter(beta.value().get_dtype()) : tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt_metal::detail::TileSize(in_data_format);
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    uint32_t out_single_tile_size = tt_metal::detail::TileSize(out_data_format);
    uint32_t bfloat16_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    uint32_t gamma_single_tile_size = tt_metal::detail::TileSize(gamma_cb_data_format);
    uint32_t beta_single_tile_size = tt_metal::detail::TileSize(beta_cb_data_format);

    log_debug("in_data_format: {}", in_data_format);
    log_debug("out_data_format: {}", out_data_format);
    log_debug("cb_data_format: {}", cb_data_format);
    log_debug("gamma_cb_data_format: {}", gamma_cb_data_format);
    log_debug("beta_cb_data_format: {}", beta_cb_data_format);
    log_debug("math_fidelity: {}", math_fidelity);
    log_debug("math_approx_mode: {}", math_approx_mode);
    log_debug("fp32_dest_acc_en: {}", fp32_dest_acc_en);

    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;
    if (b) {
        inb_data_format = tt_metal::datatype_to_dataformat_converter(b.value().get_dtype());
        inb_single_tile_size = tt_metal::detail::TileSize(inb_data_format);
    }

    auto a_addr = a.buffer()->address();
    auto b_dram_addr = b ? b.value().buffer()->address() : 0;
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;

    uint32_t num_tiles = a.volume()/TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_HW : 0;

    // For bert, tensor is packed as RM with width 32
    if (gamma.has_value() and gamma.value().get_layout() == Layout::ROW_MAJOR) {
        num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_WIDTH : 0;
    }
    if (beta.has_value() and beta.value().get_layout() == Layout::ROW_MAJOR) {
        num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_WIDTH : 0;
    }


    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    // TODO(AP): this will not work for all Wts possibly, but should work for Wt=8, 12, 16, 32
    // TODO(AP): can also add support for block_size=7 -> 63, 28
    uint32_t WtB    =  div_up(Wt, block_size)*block_size; // Wt padded to be divisible by block size
    uint32_t in0_t  =  WtB; // cb_x for no pre-add variant, x=a+b for fused pre-add, extra space for some buffering
    uint32_t in1_t  =  block_size*2; // buffer for fused pre-add b tensor
    uint32_t out0_t =  block_size*2;
    uint32_t im0_t  =  WtB; // buffer for saving xmm
    uint32_t im3_t  =  WtB; // buffer for xmm^2
    uint32_t in5_t  =  WtB; // buffer for gamma
    uint32_t in6_t  =  WtB; // buffer for beta
    uint32_t im6_t  =  block_size*2; // x=a+b reuse for x-E[x] computation plus a bit extra for buffering
    if (b) {
        im6_t = WtB;
        //cout << "im6_t=WtB=" << WtB << endl;
        in0_t = 2*block_size;
    }
    uint32_t im5_t  =  2*block_size; // for buffering to/from *gamma/+beta
    uint32_t im4_t  =  8; // 8 just in case, 4 would prob suffice
    uint32_t im1_t  =  2;
    uint32_t in2_t  =  2; // scaler for reduce coming from reader
    uint32_t in3_t  =  2; // epsilon coming from reader
    uint32_t im2_t  =  2; //

    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    TT_ASSERT(in0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in1_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(out0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im3_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in5_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in6_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im6_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(Wt % block_size == 0);
    TT_ASSERT(num_gamma_tiles % block_size == 0);
    TT_ASSERT(num_beta_tiles % block_size == 0);

    uint32_t num_tile_rows = NC * Ht;
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] = split_work_to_cores(grid_size, num_tile_rows, true);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) is_dram(a),
        (std::uint32_t) is_dram(b),
        (std::uint32_t) is_dram(gamma),
        (std::uint32_t) is_dram(beta),
        (std::uint32_t) block_size
    };

    if (gamma.has_value() and gamma.value().get_layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().get_legacy_shape()[3] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = is_power_of_two_at_least_32(gamma_stick_size);
        reader_compile_time_args.push_back((std::uint32_t) gamma_stick_size_is_power_of_two);
        if (gamma_stick_size_is_power_of_two) {
            uint32_t gamma_log2_stick_size = gamma_stick_size_is_power_of_two ? (std::uint32_t)log2(gamma_stick_size) : 0;
            reader_compile_time_args.push_back((std::uint32_t) gamma_log2_stick_size);
        } else {
            reader_compile_time_args.push_back(gamma_stick_size);
        }
    } else if (beta.has_value() and beta.value().get_layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().get_legacy_shape()[3] * beta.value().element_size();
        bool beta_stick_size_is_power_of_two = is_power_of_two_at_least_32(beta_stick_size);
        reader_compile_time_args.push_back((std::uint32_t) beta_stick_size_is_power_of_two);
        if (beta_stick_size_is_power_of_two) {
            uint32_t beta_log2_stick_size = beta_stick_size_is_power_of_two ? (std::uint32_t)log2(beta_stick_size) : 0;
            reader_compile_time_args.push_back((std::uint32_t) beta_log2_stick_size);
        } else {
            reader_compile_time_args.push_back(beta_stick_size);
        }
    } else {
        reader_compile_time_args.push_back(0);
        reader_compile_time_args.push_back(0);
    }

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) is_dram(output),
        (std::uint32_t) block_size
    };


    bool tile_dtype_is_bfloat16 = a.get_dtype() == tt::tt_metal::DataType::BFLOAT16;
    std::map<string, string> reader_defines;
    std::map<string, string> compute_defines;
    if (b) {
        reader_defines["FUSE_PRE_ADD"] = "1";
        compute_defines["FUSE_PRE_ADD"] = "1";
    }
    if (gamma.has_value()) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_defines["FUSE_BETA"] = "1";
    }

    if (rms_norm) {
        compute_defines["RMSNORM"] = "1";
    }

    auto use_row_major_kernel = (gamma.has_value() and gamma.value().get_layout() == Layout::ROW_MAJOR) or (beta.has_value() and beta.value().get_layout() == Layout::ROW_MAJOR);
    auto reader_kernels_id = CreateKernel(
        program,
        use_row_major_kernel ? "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/reader_unary_interleaved_ln_rm_gb.cpp" : "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/reader_unary_interleaved_ln.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines)
    );

    auto writer_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args)
    );

    vector<uint32_t> compute_args = { Wt, block_size, gamma.has_value(), beta.has_value(), fp32_dest_acc_en };

    auto compute_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm/kernels/compute/layernorm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compute_args, .defines = compute_defines}
    );

    // Create circular buffers
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_t*in_single_tile_size, {{CB::c_in0, in_data_format}}).set_page_size(CB::c_in0, in_single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_src0_config );
    CircularBufferConfig cb_out0_config = CircularBufferConfig(out0_t*out_single_tile_size, {{CB::c_out0, out_data_format}}).set_page_size(CB::c_out0, out_single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_out0_config );
    if (!rms_norm) {
        CircularBufferConfig cb_intermed1_config = CircularBufferConfig(im1_t*single_tile_size, {{CB::c_intermed1, cb_data_format}}).set_page_size(CB::c_intermed1, single_tile_size);
        CreateCircularBuffer( program, all_cores,  cb_intermed1_config );
    }
    CircularBufferConfig cb_in2_config = CircularBufferConfig(in2_t*bfloat16_tile_size, {{CB::c_in2, DataFormat::Float16_b}}).set_page_size(CB::c_in2, bfloat16_tile_size);
    CreateCircularBuffer( program, all_cores, cb_in2_config );
    CircularBufferConfig cb_in3_config = CircularBufferConfig(in3_t*bfloat16_tile_size, {{CB::c_in3, DataFormat::Float16_b}}).set_page_size(CB::c_in3, bfloat16_tile_size);
    CreateCircularBuffer( program, all_cores, cb_in3_config );
    CircularBufferConfig cb_intermed2_config = CircularBufferConfig(im2_t*single_tile_size, {{CB::c_intermed2, cb_data_format}}).set_page_size(CB::c_intermed2, single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_intermed2_config );
    if (!(rms_norm && !b.has_value())) {
        CircularBufferConfig cb_intermed0_config = CircularBufferConfig(im0_t*single_tile_size, {{CB::c_intermed0, cb_data_format}}).set_page_size(CB::c_intermed0, single_tile_size);
        CreateCircularBuffer( program, all_cores, cb_intermed0_config );
    }
    CircularBufferConfig c_intermed3_config = CircularBufferConfig(im3_t*single_tile_size, {{CB::c_intermed3, cb_data_format}}).set_page_size(CB::c_intermed3, single_tile_size);
    CreateCircularBuffer( program, all_cores, c_intermed3_config );
    CircularBufferConfig c_intermed4_config = CircularBufferConfig(im4_t*single_tile_size, {{CB::c_intermed4, cb_data_format}}).set_page_size(CB::c_intermed4, single_tile_size);
    CreateCircularBuffer( program, all_cores, c_intermed4_config );
    if (gamma.has_value() || beta.has_value()) {
        CircularBufferConfig c_intermed5_config = CircularBufferConfig(im5_t*single_tile_size, {{CB::c_intermed5, cb_data_format}}).set_page_size(CB::c_intermed5, single_tile_size);
        CreateCircularBuffer( program, all_cores, c_intermed5_config );
    }
    if (gamma.has_value()) {
        CircularBufferConfig c_in5_config = CircularBufferConfig(in5_t * gamma_single_tile_size, {{CB::c_in5, gamma_cb_data_format}})
            .set_page_size(CB::c_in5, gamma_single_tile_size);
        CreateCircularBuffer( program, all_cores, c_in5_config );
    }
    if (beta.has_value()) {
        CircularBufferConfig c_in6_config = CircularBufferConfig(in6_t * beta_single_tile_size, {{CB::c_in6, beta_cb_data_format}})
            .set_page_size(CB::c_in6, beta_single_tile_size);
        CreateCircularBuffer( program, all_cores, c_in6_config );
    }
    if (b) {
        // x = a+b in this notation
        // result = ln(x)*gamma + beta
        // if there's no pre-add we use cb_in0 for x, otherwise a is pre-buffered into in0, added into im6, then im6 is used as x
        // b is buffered into c_in1
        if (!rms_norm) {
            CircularBufferConfig c_intermed6_config = CircularBufferConfig(im6_t*single_tile_size, {{CB::c_intermed6, cb_data_format}}).set_page_size(CB::c_intermed6, single_tile_size);
            CreateCircularBuffer( program, all_cores, c_intermed6_config );
        }
        // c_in1 is input buffer for b
        CircularBufferConfig c_in1_config = CircularBufferConfig(in1_t*inb_single_tile_size, {{CB::c_in1, inb_data_format}}).set_page_size(CB::c_in1, inb_single_tile_size);
        CreateCircularBuffer( program, all_cores, c_in1_config);
    }

    uint32_t curr_row = 0;
    float winv = 1.0f / W; // bcast-w scaler
    bfloat16 bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union { float f; uint32_t u; } e; e.f = eps; // epsilon
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t tile_offset = curr_row * Wt;

        SetRuntimeArgs(program, reader_kernels_id, core,
            { a_addr, num_tile_rows_per_core, Wt, tile_offset, packed_winv_value, e.u, // 0-5
            gamma_dram_addr, beta_dram_addr, b_dram_addr } // 6-8
        );
        SetRuntimeArgs(program, compute_kernels_id, core, { num_tile_rows_per_core });
        SetRuntimeArgs(program, writer_kernels_id, core, { dst_addr, num_tile_rows_per_core * Wt, tile_offset } );
        curr_row += num_tile_rows_per_core;
    }

    auto override_runtime_args_callback = [
            reader_kernel_id=reader_kernels_id,
            writer_kernel_id=writer_kernels_id,
            num_cores,
            grid_size
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_a_dram_buffer = input_buffers.at(0);
        auto src_b_dram_buffer = input_buffers.at(1);
        auto gamma_dram_buffer = input_buffers.at(2);
        auto beta_dram_buffer = input_buffers.at(3);

        auto dst_dram_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};

            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_a_dram_buffer->address();
                if (src_b_dram_buffer != nullptr) {
                    runtime_args[8] = src_b_dram_buffer->address();
                }
                if (gamma_dram_buffer != nullptr) {
                    runtime_args[6] = gamma_dram_buffer->address();
                }
                if (beta_dram_buffer != nullptr) {
                    runtime_args[7] = beta_dram_buffer->address();
                }
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks layernorm_multi_core_sharded(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    LayerNormType norm_type,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_ht,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config
) {
    bool rms_norm = norm_type == LayerNormType::RMSNORM;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    Device *device = a.device();

    // convert data format
    tt::DataFormat in_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::WORMHOLE_B0, "kernel config is not for wormhole_b0");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = in_data_format == tt::DataFormat::Float32 ? true : compute_kernel_config.fp32_dest_acc_en;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);

    if (fp32_dest_acc_en) {
        TT_ASSERT(subblock_wt <= 4, "subblock width must less than 4 in fp32 mode");
    }

    tt::DataFormat out_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format = gamma.has_value() ? tt_metal::datatype_to_dataformat_converter(gamma.value().get_dtype()) : tt::DataFormat::Float16_b;
    tt::DataFormat beta_cb_data_format = beta.has_value() ? tt_metal::datatype_to_dataformat_converter(beta.value().get_dtype()) : tt::DataFormat::Float16_b;
    // tile sizes
    uint32_t in_single_tile_size = tt_metal::detail::TileSize(in_data_format);
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    uint32_t out_single_tile_size = tt_metal::detail::TileSize(out_data_format);
    uint32_t gamma_single_tile_size = tt_metal::detail::TileSize(gamma_cb_data_format);
    uint32_t beta_single_tile_size = tt_metal::detail::TileSize(beta_cb_data_format);
    uint32_t bfloat16_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    log_debug("in_data_format: {}", in_data_format);
    log_debug("out_data_format: {}", out_data_format);
    log_debug("cb_data_format: {}", cb_data_format);
    log_debug("gamma_cb_data_format: {}", gamma_cb_data_format);
    log_debug("beta_cb_data_format: {}", beta_cb_data_format);
    log_debug("math_fidelity: {}", math_fidelity);
    log_debug("math_approx_mode: {}", math_approx_mode);
    log_debug("fp32_dest_acc_en: {}", fp32_dest_acc_en);

    // tensor shape
    const auto shape = a.get_legacy_shape();
    uint32_t M = a.volume() / shape[-1];
    uint32_t K = shape[-1];
    uint32_t Mt = M / TILE_WIDTH;
    uint32_t Kt = K / TILE_WIDTH;
    // block
    uint32_t block_w = block_wt * TILE_WIDTH;
    uint32_t block_h = block_ht * TILE_HEIGHT;
    uint32_t num_blocks = 0;
    ShardSpec shard_spec = a.shard_spec().value();

    bool mcast_1d = M == block_h;
    bool row_wise = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end.x + 1, bbox.end.y+1};
    if (mcast_1d) {
        num_blocks = shard_spec.num_cores();
    } else if (row_wise) {
        num_blocks = grid_size.x;
    } else {
        num_blocks = grid_size.y;
    }

    uint32_t num_subblocks_w = block_wt / subblock_wt;

    // get sharded addr
    auto in0_addr = a.buffer()->address();
    uint32_t in1_addr;
    bool b_sharded;
    if (b) {
        in1_addr = b.value().buffer()->address();
    } else {
        in1_addr = 0;
    }
    auto out_addr = output.buffer()->address();
    // b, gamma, beta addr
    auto in1_dram_addr = b ? b.value().buffer()->address() : 0;
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    // num tiles for a, gamma, beta
    uint32_t num_tiles = a.volume()/TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_HW : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t num_rows_per_all_to_all_worker = div_up(block_ht, num_blocks);
    uint32_t num_rows_per_all_to_all_worker_last = block_ht - (block_ht / num_rows_per_all_to_all_worker) * num_rows_per_all_to_all_worker;
    uint32_t in0_block_tiles = block_wt * block_ht;
    uint32_t in0_CB_tiles = in0_block_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * in_single_tile_size;
    // block size for in1 (tensor b)
    uint32_t in1_CB_size = in0_CB_size;
    // in2 - scaler
    uint32_t in2_CB_size = bfloat16_tile_size;
    // in3 - eps
    uint32_t in3_CB_size = bfloat16_tile_size;
    // gamma
    uint32_t in5_CB_size = in0_block_tiles * gamma_single_tile_size / block_ht;
    // beta
    uint32_t in6_CB_size = in0_block_tiles * beta_single_tile_size / block_ht;
    // itermediate buffers change later
    uint32_t x_CB_size = in0_block_tiles * single_tile_size;
    uint32_t xmm_CB_size = in0_block_tiles * single_tile_size;
    uint32_t ex_partial_CB_size = in0_block_tiles * single_tile_size / block_wt;
    uint32_t ex_CB_size = ex_partial_CB_size;
    uint32_t ex_global_CB_size = ex_partial_CB_size;
    uint32_t ex_external_CB_size = Kt / block_wt * single_tile_size;
    uint32_t xmm2_CB_size = in0_block_tiles * single_tile_size / block_ht;
    uint32_t ex2pe_CB_size = num_rows_per_all_to_all_worker * single_tile_size;
    // output buffer size
    uint32_t out_CB_size = in0_block_tiles * out_single_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = Program();
    // define core ranges
    bool use_mcast = num_blocks > 1;

    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;
    uint32_t num_cores = num_cores_x * num_cores_y;
    uint32_t num_cores_all_to_all = div_up(block_ht, num_rows_per_all_to_all_worker);
    uint32_t num_none_all_to_all_workers = num_blocks - num_cores_all_to_all;
    if (num_rows_per_all_to_all_worker_last == 0)
        num_rows_per_all_to_all_worker_last = num_rows_per_all_to_all_worker;

    CoreCoord start_core = {0, 0};
    CoreRangeSet all_cores = shard_spec.grid;
    CoreRange sender_cores(start_core, start_core);
    CoreRangeSet all_to_all_cores({});
    CoreRangeSet all_to_all_workers_except_sender({});
    CoreRangeSet not_all_to_all_workers({});
    uint32_t num_cores_x_mcast, num_cores_y_mcast;
    if (mcast_1d) {
        sender_cores = {start_core, start_core};
        all_to_all_cores = num_cores_to_corerange_set(start_core, num_cores_all_to_all, grid_size, row_wise);
        if (row_wise) {
            if (use_mcast) {
                CoreCoord all_start_core;
                CoreCoord end_core = sender_cores.end;
                if (end_core.x == bbox.end.x) {
                    all_start_core = {0, end_core.y + 1};
                } else {
                    all_start_core = {end_core.x + 1, end_core.y};
                }
                all_to_all_workers_except_sender = num_cores_to_corerange_set(all_start_core, num_cores_all_to_all - 1, grid_size, row_wise);
            }
            if (num_none_all_to_all_workers > 0) {
                CoreCoord none_start_core;
                CoreCoord end_core = (*all_to_all_cores.ranges().rbegin()).end;
                if (end_core.x == bbox.end.x) {
                    none_start_core = {0, end_core.y + 1};
                } else {
                    none_start_core = {end_core.x + 1, end_core.y};
                }
                not_all_to_all_workers = num_cores_to_corerange_set(none_start_core, num_none_all_to_all_workers, grid_size, row_wise);
            }
        } else {
            if (use_mcast) {
                CoreCoord all_start_core;
                CoreCoord end_core = sender_cores.end;
                if (end_core.y == bbox.end.y) {
                    all_start_core = {end_core.x + 1, 0};
                } else {
                    all_start_core = {end_core.x, end_core.y + 1};
                }
                all_to_all_workers_except_sender = num_cores_to_corerange_set(CoreCoord{start_core.x, start_core.y + 1}, num_cores_all_to_all - 1, grid_size, row_wise);
            }
            if (num_none_all_to_all_workers > 0) {
                CoreCoord none_start_core;
                CoreCoord end_core = (*all_to_all_cores.ranges().rbegin()).end;
                if (end_core.y == bbox.end.y) {
                    none_start_core = {end_core.x + 1, 0};
                } else {
                    none_start_core = {end_core.x, end_core.y + 1};
                }
                not_all_to_all_workers = num_cores_to_corerange_set(none_start_core, num_none_all_to_all_workers, grid_size, row_wise);
            }
        }
        num_cores_x_mcast = num_cores_x;
        num_cores_y_mcast = num_cores_y;
    } else {
        if (row_wise) {
            sender_cores = {
                {(std::size_t) start_core.x, (std::size_t) start_core.y},
                {(std::size_t) start_core.x, (std::size_t) start_core.y + num_cores_y - 1}};
            all_to_all_cores = CoreRangeSet({CoreRange(
                {(std::size_t) start_core.x, (std::size_t) start_core.y},
                {(std::size_t) start_core.x + num_cores_all_to_all - 1, (std::size_t) start_core.y + num_cores_y - 1})});
            if (use_mcast && num_cores_all_to_all > 1) {
                all_to_all_workers_except_sender = CoreRangeSet({CoreRange(
                    {(std::size_t) start_core.x + 1, (std::size_t) start_core.y},
                    {(std::size_t) start_core.x + num_cores_all_to_all - 1, (std::size_t) start_core.y + num_cores_y - 1})});
            }
            if (num_none_all_to_all_workers > 0) {
                not_all_to_all_workers = CoreRangeSet({CoreRange(
                    {(std::size_t) start_core.x + num_cores_all_to_all, (std::size_t) start_core.y},
                    {(std::size_t) start_core.x + num_cores_x - 1, (std::size_t) start_core.y + num_cores_y - 1})});
            }
            num_cores_x_mcast = num_cores_x;
            num_cores_y_mcast = 1;
        } else {
            sender_cores = {
                {(std::size_t) start_core.x, (std::size_t) start_core.y},
                {(std::size_t) start_core.x + num_cores_x - 1, (std::size_t) start_core.y}};
            all_to_all_cores = CoreRangeSet({CoreRange(
                {(std::size_t) start_core.x, (std::size_t) start_core.y},
                {(std::size_t) start_core.x + num_cores_x - 1, (std::size_t) start_core.y + num_cores_all_to_all - 1})});
            if (use_mcast && num_cores_all_to_all > 1) {
                all_to_all_workers_except_sender = CoreRangeSet({CoreRange(
                    {(std::size_t) start_core.x, (std::size_t) start_core.y + 1},
                    {(std::size_t) start_core.x + num_cores_x - 1, (std::size_t) start_core.y + num_cores_all_to_all - 1})});
            }
            if (num_none_all_to_all_workers > 0) {
                not_all_to_all_workers = CoreRangeSet({CoreRange(
                    {(std::size_t) start_core.x, (std::size_t) start_core.y + num_cores_all_to_all},
                    {(std::size_t) start_core.x + num_cores_x - 1, (std::size_t) start_core.y + num_cores_y - 1})});
            }
            num_cores_x_mcast = 1;
            num_cores_y_mcast = num_cores_y;
        }
    }
    // Mcast args
    auto reduce_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto reduce_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // reader defines
    std::map<string, string> reader_mcast_sender_defines;
    std::map<string, string> reader_mcast_receiver_defines;
    if (b) {
        reader_mcast_sender_defines["FUSE_PRE_ADD"] = "1";
        reader_mcast_receiver_defines["FUSE_PRE_ADD"] = "1";
    }
    if (gamma.has_value()) {
        reader_mcast_sender_defines["FUSE_GAMMA"] = "1";
        reader_mcast_receiver_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_mcast_sender_defines["FUSE_BETA"] = "1";
        reader_mcast_receiver_defines["FUSE_BETA"] = "1";
    }
    if (rms_norm) {
        reader_mcast_sender_defines["RMSNORM"] = "1";
        reader_mcast_receiver_defines["RMSNORM"] = "1";
    }
    // reader compile time args
    std::vector<uint32_t> reader_mcast_sender_compile_time_args = {
        (std::uint32_t) reduce_receiver_semaphore,
        (std::uint32_t) reduce_sender_semaphore,
        (std::uint32_t) num_blocks,
        (std::uint32_t) block_ht,
        (std::uint32_t) block_ht * single_tile_size,
        (std::uint32_t) num_cores_all_to_all,
        (std::uint32_t) num_rows_per_all_to_all_worker,
        (std::uint32_t) num_rows_per_all_to_all_worker * single_tile_size,
        (std::uint32_t) num_rows_per_all_to_all_worker_last,
        (std::uint32_t) num_rows_per_all_to_all_worker_last * single_tile_size,
        (std::uint32_t) row_wise,
        (std::uint32_t) num_cores_x_mcast,
        (std::uint32_t) num_cores_y_mcast,
    };
    std::vector<uint32_t> reader_mcast_receiver_all_to_all_compile_time_args = {
        (std::uint32_t) reduce_receiver_semaphore,
        (std::uint32_t) reduce_sender_semaphore,
        (std::uint32_t) num_blocks,
        (std::uint32_t) block_ht,
        (std::uint32_t) 1,
        (std::uint32_t) num_cores_all_to_all,
        (std::uint32_t) num_rows_per_all_to_all_worker,
        (std::uint32_t) num_rows_per_all_to_all_worker_last,
        (std::uint32_t) row_wise,
        (std::uint32_t) num_cores_x_mcast,
        (std::uint32_t) num_cores_y_mcast,
    };
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        (std::uint32_t) reduce_receiver_semaphore,
        (std::uint32_t) reduce_sender_semaphore,
        (std::uint32_t) num_blocks,
        (std::uint32_t) block_ht,
        (std::uint32_t) 0,
        (std::uint32_t) num_cores_all_to_all,
        (std::uint32_t) num_rows_per_all_to_all_worker,
        (std::uint32_t) num_rows_per_all_to_all_worker_last,
        (std::uint32_t) row_wise,
        (std::uint32_t) 1,
        (std::uint32_t) 1,
    };

    tt_metal::NOC reader_noc = detail::GetPreferredNOCForDRAMRead(device->arch());
    tt_metal::NOC writer_noc = detail::GetPreferredNOCForDRAMWrite(device->arch());

    // reader kernel
    auto reader_mcast_sender_kernels_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp",
        sender_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = reader_noc, .compile_args = reader_mcast_sender_compile_time_args, .defines = reader_mcast_sender_defines}
    );
    KernelHandle reader_mcast_receiver_kernels_id_all_to_all = -1;
    KernelHandle reader_mcast_receiver_kernels_id = -1;
    if (use_mcast) {
        reader_mcast_receiver_kernels_id_all_to_all = CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp",
            all_to_all_workers_except_sender,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = reader_noc, .compile_args = reader_mcast_receiver_all_to_all_compile_time_args, .defines = reader_mcast_receiver_defines}
        );
    }
    if (num_none_all_to_all_workers > 0) {
        reader_mcast_receiver_kernels_id = CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp",
            not_all_to_all_workers,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = reader_noc, .compile_args = reader_mcast_receiver_compile_time_args, .defines = reader_mcast_receiver_defines}
        );
    }

    // writer defines
    std::map<string, string> writer_defines;
    if (rms_norm) {
        writer_defines["RMSNORM"] = 1;
    }
    // writer compile time args
    std::vector<uint32_t> writer_mcast_sender_compile_time_args = {
        1,
        (std::uint32_t) gamma.has_value(),
        (std::uint32_t) beta.has_value(),
        (std::uint32_t) is_dram(gamma),
        (std::uint32_t) is_dram(beta),
        (std::uint32_t) block_wt
    };
    std::vector<uint32_t> writer_mcast_receiver_compile_time_args = {
        0,
        (std::uint32_t) gamma.has_value(),
        (std::uint32_t) beta.has_value(),
        (std::uint32_t) is_dram(gamma),
        (std::uint32_t) is_dram(beta),
        (std::uint32_t) block_wt
    };

    if (gamma.has_value() and gamma.value().get_layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().get_legacy_shape()[3] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = is_power_of_two_at_least_32(gamma_stick_size);
        writer_mcast_sender_compile_time_args.push_back((std::uint32_t) gamma_stick_size_is_power_of_two);
        writer_mcast_receiver_compile_time_args.push_back((std::uint32_t) gamma_stick_size_is_power_of_two);
        if (gamma_stick_size_is_power_of_two) {
            uint32_t gamma_log2_stick_size = gamma_stick_size_is_power_of_two ? (std::uint32_t)log2(gamma_stick_size) : 0;
            writer_mcast_sender_compile_time_args.push_back((std::uint32_t) gamma_log2_stick_size);
            writer_mcast_receiver_compile_time_args.push_back((std::uint32_t) gamma_log2_stick_size);
        } else {
            writer_mcast_sender_compile_time_args.push_back(gamma_stick_size);
            writer_mcast_receiver_compile_time_args.push_back(gamma_stick_size);
        }
    } else if (beta.has_value() and beta.value().get_layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().get_legacy_shape()[3] * beta.value().element_size();
        bool beta_stick_size_is_power_of_two = is_power_of_two_at_least_32(beta_stick_size);
        writer_mcast_sender_compile_time_args.push_back((std::uint32_t) beta_stick_size_is_power_of_two);
        writer_mcast_receiver_compile_time_args.push_back((std::uint32_t) beta_stick_size_is_power_of_two);
        if (beta_stick_size_is_power_of_two) {
            uint32_t beta_log2_stick_size = beta_stick_size_is_power_of_two ? (std::uint32_t)log2(beta_stick_size) : 0;
            writer_mcast_sender_compile_time_args.push_back((std::uint32_t) beta_log2_stick_size);
            writer_mcast_receiver_compile_time_args.push_back((std::uint32_t) beta_log2_stick_size);
        } else {
            writer_mcast_sender_compile_time_args.push_back(beta_stick_size);
            writer_mcast_receiver_compile_time_args.push_back(beta_stick_size);

        }
    } else {
        writer_mcast_sender_compile_time_args.push_back(0);
        writer_mcast_sender_compile_time_args.push_back(0);
        writer_mcast_receiver_compile_time_args.push_back(0);
        writer_mcast_receiver_compile_time_args.push_back(0);
    }

    writer_mcast_sender_compile_time_args.push_back(gamma_cb_data_format == tt::DataFormat::Float32);
    writer_mcast_sender_compile_time_args.push_back(beta_cb_data_format == tt::DataFormat::Float32);
    writer_mcast_receiver_compile_time_args.push_back(gamma_cb_data_format == tt::DataFormat::Float32);
    writer_mcast_receiver_compile_time_args.push_back(beta_cb_data_format == tt::DataFormat::Float32);

    // writer kernel
    bool use_row_major_kernel = (gamma.has_value() and gamma.value().get_layout() == Layout::ROW_MAJOR) or (beta.has_value() and beta.value().get_layout() == Layout::ROW_MAJOR);
    std::string writer_kernel = use_row_major_kernel ? "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/writer_unary_sharded_ln_rm_gb.cpp" : "tt_eager/tt_dnn/op_library/layernorm/kernels/dataflow/writer_unary_sharded_ln.cpp";
    auto writer_mcast_sender_kernels_id = CreateKernel(
        program,
        writer_kernel,
        all_to_all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = writer_noc, .compile_args = writer_mcast_sender_compile_time_args, .defines = writer_defines}
    );
    KernelHandle writer_mcast_receiver_kernels_id = -1;
    if (num_none_all_to_all_workers > 0) {
        writer_mcast_receiver_kernels_id = CreateKernel(
            program,
            writer_kernel,
            not_all_to_all_workers,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = writer_noc, .compile_args = writer_mcast_receiver_compile_time_args, .defines = writer_defines}
        );
    }
    // defines
    std::map<string, string> compute_defines;
    if (b) {
        compute_defines["FUSE_PRE_ADD"] = "1";
    }
    if (rms_norm) {
        compute_defines["RMSNORM"] = "1";
    }
    // compute kernel compile time args
    std::vector<uint32_t> top_row_compute_compile_time_args = {
        1,
        gamma.has_value(),
        beta.has_value(),
        num_blocks,
        block_ht,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        1,
        block_ht * block_wt,
        fp32_dest_acc_en
    };
    std::vector<uint32_t> all_to_all_except_top_compute_compile_time_args = {
        0,
        gamma.has_value(),
        beta.has_value(),
        num_blocks,
        block_ht,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        1,
        block_ht * block_wt,
        fp32_dest_acc_en
    };
    std::vector<uint32_t> not_all_to_all_compute_compile_time_args = {
        0,
        gamma.has_value(),
        beta.has_value(),
        num_blocks,
        block_ht,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        0,
        block_ht * block_wt,
        fp32_dest_acc_en
    };
    // compute kernel
    KernelHandle compute_kernels_id = -1;
    auto compute_kernels_id_all_to_all = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/layernorm/kernels/compute/layernorm_sharded.cpp",
        all_to_all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = all_to_all_except_top_compute_compile_time_args, .defines = compute_defines}
    );
    if (num_none_all_to_all_workers > 0) {
        compute_kernels_id = CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/layernorm/kernels/compute/layernorm_sharded.cpp",
            not_all_to_all_workers,
            tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = not_all_to_all_compute_compile_time_args, .defines = compute_defines}
        );
    }
    // Create circular buffers
    // in0 sharded
    uint32_t in0_cb_index = CB::c_in0;
    tt_metal::CircularBufferConfig in0_cb_config = tt_metal::CircularBufferConfig(in0_CB_size, {{in0_cb_index, in_data_format}})
		.set_page_size(in0_cb_index, in_single_tile_size).set_globally_allocated_address(*a.buffer());
    auto cb_in0 = tt_metal::CreateCircularBuffer(program, all_cores, in0_cb_config);
    // in1 sharded
    uint32_t in1_cb_index = CB::c_in1;
    CBHandle cb_in1 = 0;
    if (b) {
        tt_metal::CircularBufferConfig in1_cb_config = tt_metal::CircularBufferConfig(in1_CB_size, {{in1_cb_index, in_data_format}})
            .set_page_size(in1_cb_index, in_single_tile_size).set_globally_allocated_address(*b.value().buffer());
        cb_in1 = tt_metal::CreateCircularBuffer(program, all_cores, in1_cb_config);
    }
    // in2 scaler
    uint32_t in2_cb_index = CB::c_in2;
    tt_metal::CircularBufferConfig in2_cb_config = tt_metal::CircularBufferConfig(in2_CB_size, {{in2_cb_index, tt::DataFormat::Float16_b}})
		.set_page_size(in2_cb_index, bfloat16_tile_size);
    auto cb_in2 = tt_metal::CreateCircularBuffer(program, all_cores, in2_cb_config);
    // in4 scaler-c
    uint32_t in4_cb_index = CB::c_in4;
    tt_metal::CircularBufferConfig in4_cb_config = tt_metal::CircularBufferConfig(in2_CB_size, {{in4_cb_index, tt::DataFormat::Float16_b}})
		.set_page_size(in4_cb_index, bfloat16_tile_size);
    auto cb_in4 = tt_metal::CreateCircularBuffer(program, all_cores, in4_cb_config);
    // in3 eps
    uint32_t in3_cb_index = CB::c_in3;
    tt_metal::CircularBufferConfig in3_cb_config = tt_metal::CircularBufferConfig(in3_CB_size, {{in3_cb_index, tt::DataFormat::Float16_b}})
		.set_page_size(in3_cb_index, bfloat16_tile_size);
    auto cb_in3 = tt_metal::CreateCircularBuffer(program, all_cores, in3_cb_config);
    // gamma
    if (gamma.has_value()) {
        uint32_t in5_cb_index = CB::c_in5;
        tt_metal::CircularBufferConfig in5_cb_config = tt_metal::CircularBufferConfig(in5_CB_size, {{in5_cb_index, gamma_cb_data_format}})
            .set_page_size(in5_cb_index, gamma_single_tile_size);
        auto cb_in5 = tt_metal::CreateCircularBuffer(program, all_cores, in5_cb_config);
    }
    // beta
    if (beta.has_value()) {
        uint32_t in6_cb_index = CB::c_in6;
        tt_metal::CircularBufferConfig in6_cb_config = tt_metal::CircularBufferConfig(in6_CB_size, {{in6_cb_index, beta_cb_data_format}})
            .set_page_size(in6_cb_index, beta_single_tile_size);
        auto cb_in6 = tt_metal::CreateCircularBuffer(program, all_cores, in6_cb_config);
    }
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
    if(!rms_norm) {
        uint32_t ex_cb_partial_index = CB::dataflow0;
        tt_metal::CircularBufferConfig ex_cb_partial_config = tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial_index, cb_data_format}})
            .set_page_size(ex_cb_partial_index, single_tile_size);
        auto cb_ex_partial = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial_config);
        // ex
        uint32_t ex_cb_index = CB::dataflow1;
        tt_metal::CircularBufferConfig ex_cb_config = tt_metal::CircularBufferConfig(ex_CB_size, {{ex_cb_index, cb_data_format}})
            .set_page_size(ex_cb_index, single_tile_size);
        auto cb_ex = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_config);
        // ex_external
        uint32_t ex_cb_external_index = CB::dataflow2;
        tt_metal::CircularBufferConfig ex_cb_external_config = tt_metal::CircularBufferConfig(ex_external_CB_size, {{ex_cb_external_index, cb_data_format}})
            .set_page_size(ex_cb_external_index, single_tile_size);
        auto cb_ex_external = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_external_config);
    }
    // ex_partial2
    uint32_t ex_cb_partial2_index = CB::dataflow3;
    tt_metal::CircularBufferConfig ex_cb_partial2_config = tt_metal::CircularBufferConfig(ex_partial_CB_size, {{ex_cb_partial2_index, cb_data_format}})
		.set_page_size(ex_cb_partial2_index, single_tile_size);
    auto cb_ex_partial2 = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_partial2_config);
    // ex2
    uint32_t ex2_cb_index = CB::dataflow4;
    tt_metal::CircularBufferConfig ex2_cb_config = tt_metal::CircularBufferConfig(ex_CB_size, {{ex2_cb_index, cb_data_format}})
		.set_page_size(ex2_cb_index, single_tile_size);
    auto cb_ex2 = tt_metal::CreateCircularBuffer(program, all_cores, ex2_cb_config);
    // ex_external2
    uint32_t ex_cb_external2_index = CB::dataflow5;
    tt_metal::CircularBufferConfig ex_cb_external2_config = tt_metal::CircularBufferConfig(ex_external_CB_size, {{ex_cb_external2_index, cb_data_format}})
		.set_page_size(ex_cb_external2_index, single_tile_size);
    auto cb_ex_external2 = tt_metal::CreateCircularBuffer(program, all_cores, ex_cb_external2_config);
    // ex_global
    uint32_t ex_global_cb_index = CB::dataflow7;
    tt_metal::CircularBufferConfig ex_global_cb_config = tt_metal::CircularBufferConfig(ex_global_CB_size, {{ex_global_cb_index, cb_data_format}})
		.set_page_size(ex_global_cb_index, single_tile_size);
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

    const auto& cores = grid_to_cores(all_cores.num_cores(), num_cores_x, num_cores_y, row_wise);

    // Runtime Args
    std::vector<KernelHandle> writer_kernel_ids;
    writer_kernel_ids.reserve(cores.size());
    float winv = 1.0f / block_w; // bcast-w scaler
    float cinv = 1.0f / num_blocks; // bcast-cores scaler
    bfloat16 bfloat_cinv_value = bfloat16(cinv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv_value, bfloat_cinv_value});
    bfloat16 bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union { float f; uint32_t u; } e; e.f = eps;

    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    in0_mcast_noc_x.reserve(num_cores_x);
    in0_mcast_noc_y.reserve(num_cores_y);
    for(uint32_t core_idx_x = 0; core_idx_x < num_cores_x; ++core_idx_x) {
        in0_mcast_noc_x.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
    }
    for(uint32_t core_idx_y = 0; core_idx_y < num_cores_y; ++core_idx_y) {
        in0_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
    }

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        uint32_t height_index = 0, width_index = 0;
        if (mcast_1d) {
            height_index = 0;
            width_index = i;
        } else {
            if (row_wise) {
                height_index = core.y;
                width_index = core.x;
            } else {
                height_index = core.x;
                width_index = core.y;
            }
        }
        uint32_t all_to_all_worker_tile_offset_size_bytes = (width_index * num_rows_per_all_to_all_worker) * single_tile_size;
        uint32_t in1_tile_start_id = (height_index * block_ht * Kt) + (width_index * block_wt);
        uint32_t gamma_tile_start_id = width_index * block_wt;
        uint32_t beta_tile_start_id = width_index * block_wt;

        if (width_index < num_cores_all_to_all) {
            std::vector<uint32_t> compute_args;
            uint32_t num_rows = width_index == num_cores_all_to_all - 1 ? num_rows_per_all_to_all_worker_last : num_rows_per_all_to_all_worker;
            compute_args.push_back(num_rows);
            tt_metal::SetRuntimeArgs(program, compute_kernels_id_all_to_all, core, compute_args);
        }

        if (width_index == 0) {
            CoreCoord mcast_start, mcast_end;
            if (mcast_1d) {
                CoreCoord top_left_core = {(std::size_t) start_core.x, (std::size_t) start_core.y};
                CoreCoord bottom_right_core = {(std::size_t) start_core.x + num_cores_x - 1, (std::size_t) start_core.y + num_cores_y - 1};
                auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
                auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);
                mcast_start = top_left_core_physical;
                mcast_end = bottom_right_core_physical;
            } else {
                if (row_wise) {
                    CoreCoord left_core_plus_one    = {(std::size_t) start_core.x + 1, (std::size_t) core.y};
                    CoreCoord right_core   = {(std::size_t) start_core.x + num_cores_x - 1, (std::size_t) core.y};
                    auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
                    auto right_core_physical = device->worker_core_from_logical_core(right_core);
                    mcast_start = left_core_plus_one_physical;
                    mcast_end = right_core_physical;
                } else {
                    CoreCoord top_core_plus_one     = {(std::size_t) core.x, (std::size_t) start_core.y + 1};
                    CoreCoord bottom_core  = {(std::size_t) core.x, (std::size_t) start_core.y + num_cores_y - 1};
                    auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
                    auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);
                    mcast_start = top_core_plus_one_physical;
                    mcast_end = bottom_core_physical;
                }
            }
            if (reader_noc == NOC::NOC_1) {
                std::swap(mcast_start, mcast_end);
            }
            std::vector<uint32_t> mcast_sender_args;
            mcast_sender_args.push_back(mcast_start.x);
            mcast_sender_args.push_back(mcast_start.y);
            mcast_sender_args.push_back(mcast_end.x);
            mcast_sender_args.push_back(mcast_end.y);
            if (mcast_1d) {
                mcast_sender_args.push_back(core.x - start_core.x);
                mcast_sender_args.push_back(core.y - start_core.y);
                mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
                mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
            } else {
                if (row_wise) {
                    mcast_sender_args.push_back(core.x - start_core.x);
                    mcast_sender_args.push_back(0);
                    mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
                    mcast_sender_args.push_back(in0_mcast_noc_y[height_index]);
                } else {
                    mcast_sender_args.push_back(0);
                    mcast_sender_args.push_back(core.y - start_core.y);
                    mcast_sender_args.push_back(in0_mcast_noc_x[height_index]);
                    mcast_sender_args.insert(mcast_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
                }
            }
            tt_metal::SetRuntimeArgs(program, reader_mcast_sender_kernels_id, core, mcast_sender_args);
        } else if (width_index < num_cores_all_to_all) {
            std::vector<uint32_t> mcast_receiver_args;
            bool is_last_all_to_all_worker = width_index == num_cores_all_to_all - 1 ? true : false;
            mcast_receiver_args.push_back(is_last_all_to_all_worker);
            mcast_receiver_args.push_back(all_to_all_worker_tile_offset_size_bytes);
            if (mcast_1d) {
                mcast_receiver_args.push_back(core.x - start_core.x);
                mcast_receiver_args.push_back(core.y - start_core.y);
                mcast_receiver_args.insert(mcast_receiver_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
                mcast_receiver_args.insert(mcast_receiver_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
            } else {
                if (row_wise) {
                    mcast_receiver_args.push_back(core.x - start_core.x);
                    mcast_receiver_args.push_back(0);
                    mcast_receiver_args.insert(mcast_receiver_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
                    mcast_receiver_args.push_back(in0_mcast_noc_y[height_index]);
                } else {
                    mcast_receiver_args.push_back(0);
                    mcast_receiver_args.push_back(core.y - start_core.y);
                    mcast_receiver_args.push_back(in0_mcast_noc_x[height_index]);
                    mcast_receiver_args.insert(mcast_receiver_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
                }
            }
            tt_metal::SetRuntimeArgs(program, reader_mcast_receiver_kernels_id_all_to_all, core, mcast_receiver_args);
        } else {
            std::vector<uint32_t> mcast_receiver_args;
            bool is_last_all_to_all_worker = false;
            mcast_receiver_args.push_back(is_last_all_to_all_worker);
            mcast_receiver_args.push_back(all_to_all_worker_tile_offset_size_bytes);
            mcast_receiver_args.push_back(0);
            mcast_receiver_args.push_back(0);
            if (mcast_1d) {
                mcast_receiver_args.push_back(in0_mcast_noc_x[0]);
                mcast_receiver_args.push_back(in0_mcast_noc_y[0]);
            } else {
                if (row_wise) {
                    mcast_receiver_args.push_back(in0_mcast_noc_x[0]);
                    mcast_receiver_args.push_back(in0_mcast_noc_y[height_index]);
                } else {
                    mcast_receiver_args.push_back(in0_mcast_noc_x[height_index]);
                    mcast_receiver_args.push_back(in0_mcast_noc_y[0]);
                }
            }
            tt_metal::SetRuntimeArgs(program, reader_mcast_receiver_kernels_id, core, mcast_receiver_args);
        }

        if (width_index < num_cores_all_to_all) { // all to all workers
            std::vector<uint32_t> writer_mcast_sender_args;
            writer_mcast_sender_args.push_back(packed_cinv_value);
            writer_mcast_sender_args.push_back(packed_winv_value);
            writer_mcast_sender_args.push_back(e.u);
            writer_mcast_sender_args.push_back(gamma_dram_addr);
            writer_mcast_sender_args.push_back(beta_dram_addr);
            writer_mcast_sender_args.push_back(gamma_tile_start_id);
            writer_mcast_sender_args.push_back(beta_tile_start_id);
            tt_metal::SetRuntimeArgs(program, writer_mcast_sender_kernels_id, core, writer_mcast_sender_args);
            writer_kernel_ids.push_back(writer_mcast_sender_kernels_id);
        } else {
            std::vector<uint32_t> writer_mcast_receiver_args;
            writer_mcast_receiver_args.push_back(packed_cinv_value);
            writer_mcast_receiver_args.push_back(packed_winv_value);
            writer_mcast_receiver_args.push_back(e.u);
            writer_mcast_receiver_args.push_back(gamma_dram_addr);
            writer_mcast_receiver_args.push_back(beta_dram_addr);
            writer_mcast_receiver_args.push_back(gamma_tile_start_id);
            writer_mcast_receiver_args.push_back(beta_tile_start_id);
            tt_metal::SetRuntimeArgs(program, writer_mcast_receiver_kernels_id, core, writer_mcast_receiver_args);
            writer_kernel_ids.push_back(writer_mcast_receiver_kernels_id);
        }

    }

    auto override_runtime_args_callback = [
            writer_kernel_ids,
            cb_in0,
            cb_in1,
            cb_output,
            cores
        ]
    (
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer_a = input_tensors.at(0).buffer();
        auto b_tensor = optional_input_tensors.at(0);
        auto gamma_tensor = optional_input_tensors.at(1);
        auto beta_tensor = optional_input_tensors.at(2);
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_in0, *src_buffer_a);

        if (b_tensor.has_value()) {
            UpdateDynamicCircularBufferAddress(program, cb_in1, *b_tensor.value().buffer());
        }

        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);

        for (uint32_t i = 0; i < cores.size(); ++i) {
            const CoreCoord& core = cores[i];

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

}  // namespace tt_metal

}  // namespace tt
