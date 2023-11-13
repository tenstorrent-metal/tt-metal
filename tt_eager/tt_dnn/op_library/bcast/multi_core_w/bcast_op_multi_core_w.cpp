// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"


using namespace tt::tt_metal;
using namespace tt::constants;


namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks bcast_multi_core_w(const Tensor &a, const Tensor &b, Tensor& output, BcastOpMath bcast_math, BcastOpDim bcast_dim) {
    TT_ASSERT(bcast_dim == BcastOpDim::W);

    const auto ashape = a.shape();
    const auto bshape = b.shape();
    uint32_t N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    uint32_t bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    uint32_t NC = N*C;
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*Ht*Wt;
    uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

	uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;

    tt_metal::Program program = tt_metal::Program();

    tt_metal::Device *device = a.device();

	tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

	auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, Wt_per_core_group_1, Wt_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, Wt);

	auto src0_buffer = a.buffer();
	auto src1_buffer = b.buffer();
	auto dst_buffer = output.buffer();
	TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const char* reader_name = bcast_op_utils::get_reader_name(bcast_dim, BcastOpParallelizationStrategy::MULTI_CORE_W);
    const char* compute_name = bcast_op_utils::get_compute_name(bcast_dim);

	uint32_t src0_cb_index = 0;
	uint32_t num_input_tiles = 2;

	tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_device_cores, src0_cb_config);

	uint32_t src1_cb_index = 1;
	tt_metal::CircularBufferConfig src1_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
	auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_device_cores, src1_cb_config);

	uint32_t output_cb_index = 16; // output operands start at index 16
	uint32_t num_output_tiles = 2;
	tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
	auto cb_output = tt_metal::CreateCircularBuffer(program, all_device_cores, output_cb_config);


	bool src0_is_dram = src0_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

	KernelID binary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
		program,
		reader_name,
		all_device_cores,
		tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

	KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
		program,
		"tt_metal/kernels/dataflow/writer_unary_8bank_input_cols_batched.cpp",
		all_device_cores,
		tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

	std::map<std::string, std::string> bcast_defines = bcast_op_utils::get_defines(bcast_dim, bcast_math);
	auto bcast_kernel_id = tt_metal::CreateComputeKernel(
		program,
		compute_name,
		all_device_cores,
		tt_metal::ComputeConfig{.compile_args = {}, .defines = bcast_defines}
	);

	for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_y * num_cores_x; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
		uint32_t Wt_per_core;
		if (core_group_1.core_coord_in_core_ranges(core)) {
			Wt_per_core = Wt_per_core_group_1;
		} else if (core_group_2.core_coord_in_core_ranges(core)) {
			Wt_per_core = Wt_per_core_group_2;
		} else {
			tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, std::vector<uint32_t>(16, 0));
			tt_metal::SetRuntimeArgs(program, bcast_kernel_id, core, std::vector<uint32_t>(3, 0));
			tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::vector<uint32_t>(9, 0));
			continue;
		}
        uint32_t num_tensor_tiles_per_core = NC*Ht*Wt_per_core;
        uint32_t Wt_skip = Wt - Wt_per_core;

        tt_metal::SetRuntimeArgs(
			program,
            binary_reader_kernel_id,
            core,
            {
				a.buffer()->address(), // 0
				0, // 1
				0, // 2
				num_tensor_tiles_per_core, // 3
				b.buffer()->address(), // 4
				0, // 5
				0, // 6
				num_btensor_tiles, // 7
				num_tensor_tiles_per_core, // 8
				NC, // 9
				Ht, // 10
				Wt_per_core, // 11
				bnc1, // 12
				num_Wtiles_read, // 13
				Ht*Wt, // 14
				Wt_skip, // 15
			}
		);

        tt_metal::SetRuntimeArgs(
			program,
            bcast_kernel_id,
            core,
            {
				NC, // B
				Ht, // Ht
				Wt_per_core  // Wt
			}
		);

        tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core,
            {
                output.buffer()->address(),
                0,
                0,
                Ht,
                Wt_per_core,
                num_Wtiles_read,
                Wt_skip,
                NC,
                Ht*Wt,
            }
		);
		num_Wtiles_read+=Wt_per_core;
	}

    auto override_runtime_arguments_callback = [
            binary_reader_kernel_id,
            unary_writer_kernel_id,
			bcast_kernel_id,
            compute_with_storage_grid_size
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
		uint32_t num_cores_x = compute_with_storage_grid_size.x;
		uint32_t num_cores_y = compute_with_storage_grid_size.y;

        auto src_dram_buffer_a = input_tensors.at(0).buffer();
        auto src_dram_buffer_b = input_tensors.at(1).buffer();

        auto dst_dram_buffer = output_tensors.at(0).buffer();

		const auto ashape = input_tensors.at(0).shape();
		const auto bshape = input_tensors.at(1).shape();
		uint32_t N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
		uint32_t bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
		uint32_t NC = N*C;
		uint32_t HW = H*W;

		uint32_t Wt = W/TILE_WIDTH;
		uint32_t Ht = H/TILE_HEIGHT;

		uint32_t num_tensor_tiles = NC*Ht*Wt;
		uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

		uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;

    	auto [num_cores, all_cores, core_group_1, core_group_2, Wt_per_core_group_1, Wt_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, Wt);

		for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_y * num_cores_x; i++) {
			CoreCoord core = {i / num_cores_y, i % num_cores_y};
			uint32_t Wt_per_core;
			if (core_group_1.core_coord_in_core_ranges(core)) {
				Wt_per_core = Wt_per_core_group_1;
			} else if (core_group_2.core_coord_in_core_ranges(core)) {
				Wt_per_core = Wt_per_core_group_2;
			} else {
				tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, std::vector<uint32_t>(16, 0));
				tt_metal::SetRuntimeArgs(program, bcast_kernel_id, core, std::vector<uint32_t>(3, 0));
				tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::vector<uint32_t>(9, 0));
				continue;
			}
			uint32_t num_tensor_tiles_per_core = NC*Ht*Wt_per_core;
			uint32_t Wt_skip = Wt - Wt_per_core;

			tt_metal::SetRuntimeArgs(
				program,
				binary_reader_kernel_id,
				core,
				{
					src_dram_buffer_a->address(), // 0
					0, // 1
					0, // 2
					num_tensor_tiles_per_core, // 3
					src_dram_buffer_b->address(), // 4
					0, // 5
					0, // 6
					num_btensor_tiles, // 7
					num_tensor_tiles_per_core, // 8
					NC, // 9
					Ht, // 10
					Wt_per_core, // 11
					bnc1, // 12
					num_Wtiles_read, // 13
					Ht*Wt, // 14
					Wt_skip, // 15
				}
			);

			tt_metal::SetRuntimeArgs(
				program,
				bcast_kernel_id,
				core,
				{
					NC, // B
					Ht, // Ht
					Wt_per_core  // Wt
				}
			);

			tt_metal::SetRuntimeArgs(
				program, unary_writer_kernel_id, core,
				{
					dst_dram_buffer->address(),
					0,
					0,
					Ht,
					Wt_per_core,
					num_Wtiles_read,
					Wt_skip,
					NC,
					Ht*Wt,
				}
			);
			num_Wtiles_read+=Wt_per_core;
		}
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
