// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include "impl/buffers/buffer.hpp"
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

operation::ProgramWithCallbacks bcast_multi_core_hw(const Tensor &a, const Tensor &b, const Tensor& output, BcastOpMath bcast_math, BcastOpDim bcast_dim) {
    TT_ASSERT(bcast_dim == BcastOpDim::HW);
	// Todo.pp : fix
	TT_ASSERT(a.memory_config().memory_layout == output.memory_config().memory_layout);

    const auto ashape = a.get_legacy_shape();
    const auto bshape = b.get_legacy_shape();
    uint32_t N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    uint32_t bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    uint32_t NC = N*C;
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;
	uint32_t HtWt = Ht * Wt;

    uint32_t num_tensor_tiles = NC*Ht*Wt;

	uint32_t bnc1 = (bN*bC == 1);

    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::Device *device = a.device();

	std::optional<ShardSpec> shard_spec = std::nullopt;
	bool src0_sharded = a.memory_config().is_sharded();
	bool output_sharded = output.memory_config().is_sharded();
	if (src0_sharded) {
		shard_spec = a.shard_spec().value();
	} else if (output_sharded) {
		shard_spec = output.shard_spec().value();
	}

	tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

	auto src0_buffer = a.buffer();
	auto src1_buffer = b.buffer();
	auto dst_buffer = output.buffer();
	TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const char* reader_name = bcast_op_utils::get_reader_name(bcast_dim, BcastOpParallelizationStrategy::MULTI_CORE_HW);
    const char* compute_name = bcast_op_utils::get_compute_name(bcast_dim);

	uint32_t src0_cb_index = 0;
	uint32_t num_input_tiles = 2;
	uint32_t num_tiles_per_shard = 0;
	if (shard_spec.has_value()) {
		num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
		num_tiles_per_core_group_1 = num_tiles_per_shard;
		num_tiles_per_core_group_2 = 0;
		all_cores = shard_spec.value().grid;
		core_group_1 = all_cores;
		core_group_2 = CoreRangeSet({});
	}

	uint32_t num_input_tiles_cb0 = src0_sharded ? num_tiles_per_shard : num_input_tiles;

	tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles_cb0 * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
	if (src0_sharded) {
		src0_cb_config = src0_cb_config.set_globally_allocated_address(*a.buffer());
	}
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_device_cores, src0_cb_config);

	uint32_t src1_cb_index = 1;
	tt_metal::CircularBufferConfig src1_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
	auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_device_cores, src1_cb_config);

	uint32_t output_cb_index = 16; // output operands start at index 16
	uint32_t num_output_tiles = output_sharded ? num_tiles_per_shard : 2;
	tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
	if (output_sharded) {
		output_cb_config = output_cb_config.set_globally_allocated_address(*output.buffer());
	}
	auto cb_output = tt_metal::CreateCircularBuffer(program, all_device_cores, output_cb_config);

	bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

	std::map<string, string> reader_defines;
	std::map<string, string> bcast_compute_defines = bcast_op_utils::get_defines(bcast_dim, bcast_math);
	if(bnc1) {
		reader_defines["BCAST_SCALAR"] = "1";
		bcast_compute_defines["BCAST_SCALAR"] = "1";
	}
	if (src0_sharded) {
		reader_defines["IN0_SHARDED"] = "1";
	}
	KernelHandle binary_reader_kernel_id = tt_metal::CreateKernel(
		program,
		reader_name,
		all_device_cores,
		tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

	std::map<string, string> writer_defines;
	if (output_sharded) {
		writer_defines["OUT_SHARDED"] = "1";
	}
	KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
		program,
		"tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
		all_device_cores,
		tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

	auto bcast_kernel_id = tt_metal::CreateKernel(
		program,
		compute_name,
		all_device_cores,
		tt_metal::ComputeConfig{.compile_args = {}, .defines = bcast_compute_defines}
	);

	for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_y * num_cores_x; i++){
		CoreCoord core = {i / num_cores_y, i % num_cores_y};
		uint32_t num_tensor_tiles_per_core;
		if (core_group_1.core_coord_in_core_ranges(core)) {
			num_tensor_tiles_per_core = num_tiles_per_core_group_1;
		} else if (core_group_2.core_coord_in_core_ranges(core)) {
			num_tensor_tiles_per_core = num_tiles_per_core_group_2;
		} else {
			tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, std::vector<uint32_t>(7, 0));
			tt_metal::SetRuntimeArgs(program, bcast_kernel_id, core, std::vector<uint32_t>(3, 0));
			tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::vector<uint32_t>(3, 0));
			continue;
		}

		tt_metal::SetRuntimeArgs(
			program,
			binary_reader_kernel_id,
			core,
			{
				a.buffer()->address(), // 0
				b.buffer()->address(),
				num_tensor_tiles_per_core,
				HtWt,
				num_tiles_read / HtWt * HtWt,
				num_tiles_read % HtWt,
				bnc1 ? 0 : num_tiles_read / HtWt
			}
		);

		tt_metal::SetRuntimeArgs(
			program,
			bcast_kernel_id,
			core,
			{
				1, // B
				1, // Ht
				num_tensor_tiles_per_core  // Wt
			}
		);

		tt_metal::SetRuntimeArgs(
			program, unary_writer_kernel_id, core,
			{
				output.buffer()->address(),
				num_tensor_tiles_per_core,
				num_tiles_read,
			}
		);
		num_tiles_read += num_tensor_tiles_per_core;
	}

    auto override_runtime_arguments_callback = [
            binary_reader_kernel_id,
            unary_writer_kernel_id,
			bcast_kernel_id,
            compute_with_storage_grid_size,
			cb_src0,
			single_tile_size,
			cb_output
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
		uint32_t num_cores_x = compute_with_storage_grid_size.x;
		uint32_t num_cores_y = compute_with_storage_grid_size.y;

        auto src_buffer_a = input_tensors.at(0).buffer();
        auto src_dram_buffer_b = input_tensors.at(1).buffer();
		std::optional<ShardSpec> shard_spec = std::nullopt;
		bool src0_sharded = input_tensors.at(0).memory_config().is_sharded();
		bool out_sharded = output_tensors.at(0).memory_config().is_sharded();

		if (src0_sharded) {
			shard_spec = input_tensors.at(0).shard_spec().value();
		} else if (out_sharded) {
			shard_spec = output_tensors.at(0).shard_spec().value();
		}

        auto dst_buffer= output_tensors.at(0).buffer();

		const auto ashape = input_tensors.at(0).get_legacy_shape();
		const auto bshape = input_tensors.at(1).get_legacy_shape();
		uint32_t N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
		uint32_t bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
		uint32_t NC = N*C;
		uint32_t HW = H*W;

		uint32_t Wt = W/TILE_WIDTH;
		uint32_t Ht = H/TILE_HEIGHT;
		uint32_t HtWt = Ht * Wt;

		uint32_t num_tensor_tiles = NC*Ht*Wt;

		uint32_t bnc1 = (bN*bC == 1);

   	 	auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

		if (shard_spec.has_value()) {
			uint32_t num_tiles_per_shard = 0;
			num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
			num_tiles_per_core_group_1 = num_tiles_per_shard;
			num_tiles_per_core_group_2 = 0;
			all_cores = shard_spec.value().grid;
			core_group_1 = all_cores;
			core_group_2 = CoreRangeSet({});
		}

        for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_y * num_cores_x; i++){
			CoreCoord core = {i / num_cores_y, i % num_cores_y};
			uint32_t num_tensor_tiles_per_core;
			if (core_group_1.core_coord_in_core_ranges(core)) {
				num_tensor_tiles_per_core = num_tiles_per_core_group_1;
			} else if (core_group_2.core_coord_in_core_ranges(core)) {
				num_tensor_tiles_per_core = num_tiles_per_core_group_2;
			} else {
				tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, std::vector<uint32_t>(7, 0));
				tt_metal::SetRuntimeArgs(program, bcast_kernel_id, core, std::vector<uint32_t>(3, 0));
				tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::vector<uint32_t>(3, 0));
				continue;
			}

			tt_metal::SetRuntimeArgs(
				program,
				binary_reader_kernel_id,
				core,
				{
					src_buffer_a->address(), // 0
					src_dram_buffer_b->address(),
					num_tensor_tiles_per_core,
					HtWt,
					num_tiles_read / HtWt * HtWt,
					num_tiles_read % HtWt,
					bnc1 ? 0 : num_tiles_read / HtWt
				}
			);

			tt_metal::SetRuntimeArgs(
				program,
				bcast_kernel_id,
				core,
				{
					1, // B
					1, // Ht
					num_tensor_tiles_per_core  // Wt
				}
			);

			tt_metal::SetRuntimeArgs(
				program, unary_writer_kernel_id, core,
				{
					dst_buffer->address(),
					num_tensor_tiles_per_core,
					num_tiles_read,
				}
			);
			num_tiles_read += num_tensor_tiles_per_core;
		}

		if (src0_sharded) {
			UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer_a);
            UpdateCircularBufferTotalSize(program, cb_src0, num_tiles_per_core_group_1 * single_tile_size);
		}

		if (out_sharded) {
			UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
			UpdateCircularBufferTotalSize(program, cb_output, num_tiles_per_core_group_1 * single_tile_size);
		}
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
