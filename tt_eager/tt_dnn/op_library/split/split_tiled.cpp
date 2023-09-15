// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/split/split_tiled.hpp"

#include <iostream>

#include "common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

void SplitTiled::boiler_plate_asserts(const Tensor &a) const {
    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_ASSERT(
        a.dtype() == tt::tt_metal::DataType::BFLOAT16 || a.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
}

void SplitTiled::shape_asserts(const Tensor &a) const {
    int chunk_size = a.shape()[dim] / num_chunks;
    TT_ASSERT(a.shape()[0] == 1, "Only batch 1 implemented");
    TT_ASSERT(a.shape()[dim] % num_chunks == 0, "Incorrect shape on last dim");
    TT_ASSERT(dim <= a.shape().rank() && dim >= 0, "Improper dims");
    TT_ASSERT(a.shape().rank() == 4, "W,Z,Y,X tensor");
    TT_ASSERT(a.layout() == Layout::TILE, "Currently only tile layout support");
    TT_ASSERT((a.shape()[2] % TILE_HEIGHT == 0), "Shape not divisible by tile");
    TT_ASSERT((a.shape()[3] % TILE_WIDTH == 0), "Shape not divisible by tile");
    if (dim == 3)
        TT_ASSERT((chunk_size % TILE_WIDTH == 0), "Chunk not divisible by tile");
    else if (dim == 2)
        TT_ASSERT((chunk_size % TILE_HEIGHT == 0), "Chunk not divisible by tile");

    //For now until we have other dim support
    TT_ASSERT(dim == 3, "Currently only last dim support");
}

inline bool is_dram(const Tensor &a) { return a.memory_config().buffer_type == BufferType::DRAM; }

Shape SplitTiled::get_single_output_shape(const Shape &input_shape) const {
    auto output_shape = input_shape;
    output_shape[dim] /= num_chunks;
    return output_shape;
}

tt::DataFormat get_data_format(const Tensor &a) {
    tt::DataFormat cb_data_format = tt::DataFormat::Bfp8_b;
    if (a.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        cb_data_format = tt::DataFormat::Float16_b;
    }
    return cb_data_format;
}

void SplitTiled::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    tt_metal::Buffer *in0_buffer = input_tensor.buffer();
    auto cb_data_format = get_data_format(input_tensor);
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);
    boiler_plate_asserts((const Tensor &)input_tensor);
    shape_asserts((const Tensor &)input_tensor);
}

std::vector<Shape> SplitTiled::compute_output_shapes(
    const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    auto input_shape = input_tensor.shape();
    auto output_shape = get_single_output_shape(input_tensor.shape());
    
    std::vector<Shape> ret_vec(this->num_chunks, output_shape);
    return ret_vec;
}

std::vector<Tensor> SplitTiled::create_output_tensors(
    const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks get_program(const Tensor &input_tensor, std::vector<Tensor> &output_tensors, 
                                            const uint32_t num_chunks, 
                                            const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG){
    
    
    Program program{};
    tt_metal::Device *device = input_tensor.device();
    auto input_shape = input_tensor.shape();


    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    tt_metal::Buffer *in0_buffer = input_tensor.buffer();

    // Output buffers
    TT_ASSERT(output_tensors.size() == num_chunks);

    uint32_t z = input_shape[1];
    uint32_t num_tiles_dim_2 = input_shape[2] / TILE_HEIGHT;
    uint32_t num_tiles_dim_3 = input_shape[3] / TILE_WIDTH;
    uint32_t num_cores_x_limit = device->compute_with_storage_grid_size().x;
    uint32_t num_cores_y_limit = device->compute_with_storage_grid_size().y;



    // parallelize z
    auto num_cores_z = z;

    // parallelize y
    auto [num_cores_y, per_core_tiles_y] = get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_dim_3, num_cores_y_limit);

    // parallelize x
    auto [num_cores_x, per_core_tiles_x] = get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_dim_2,
                                                                                        num_cores_x_limit/ num_cores_z);


    //Adjust num_cores_y to be divisible by num chunks
    if(num_cores_y % num_chunks != 0){
        num_cores_y = (num_cores_y/num_chunks) * num_chunks;
        // parallelize y
        auto [num_cores_y_temp, per_core_tiles_y_temp] = get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_dim_3, num_cores_y);
        num_cores_y = num_cores_y_temp;
        per_core_tiles_y = per_core_tiles_y_temp;
    }



    uint32_t per_core_tiles = per_core_tiles_x * per_core_tiles_y * (z / num_cores_z);

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t num_cores_c = num_cores_y;
    uint32_t num_cores_r = num_cores_x * num_cores_z;

    CoreRange all_cores{
        .start = {(std::size_t)start_core_x, (std::size_t)start_core_y},
        .end = {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1},
    };

    bool tile_dtype_is_bfloat16 = input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t num_tiles_per_z = (per_core_tiles_x * num_cores_x) * (per_core_tiles_y * num_cores_y);
    uint32_t z_stride_read = num_tiles_per_z;
    uint32_t y_stride_read = per_core_tiles_y * num_cores_y;

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)tile_dtype_is_bfloat16,
        // by default in dram
        (std::uint32_t)in0_is_dram,

        // READER COMPILE TIME ARGS
        (std::uint32_t)(z / num_cores_z),
        (std::uint32_t) per_core_tiles_x,  // out_num_tiles_per_tensor
        (std::uint32_t) per_core_tiles_y,  // out_num_tiles_per_tensor
        (std::uint32_t)z_stride_read,
        (std::uint32_t)y_stride_read};

    uint32_t z_stride_write = num_tiles_per_z / num_chunks;
    uint32_t y_stride_write = per_core_tiles_y * (num_cores_c/num_chunks);
    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) per_core_tiles_x,  // out_num_tiles_per_tensor
        (std::uint32_t) per_core_tiles_y,  // out_num_tiles_per_tensor

        (std::uint32_t)(z / num_cores_z),
        (std::uint32_t)z_stride_write,
        (std::uint32_t)y_stride_write

    };

    auto reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_tm_tile_layout_split.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    auto writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_tm_tile_layout_split.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    uint32_t cb_index = 0;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program, cb_index, all_cores, 2, 2 * single_tile_size, cb_data_format);


    if(num_cores_c > 1){
        TT_ASSERT(num_cores_c %2 == 0, "Must be even number of cores");
    }
    uint32_t idc_outer_limit = 1;
    uint32_t idc_inner_limit = num_cores_c;


    for (int id_r_outer = 0; id_r_outer < z; id_r_outer++) {
        for(int id_r_inner = 0; id_r_inner < num_cores_x; id_r_inner++){
            uint32_t id_r = id_r_outer*num_cores_x + id_r_inner;

            uint32_t id_r_reader = id_r_outer*num_tiles_per_z + id_r_inner*per_core_tiles_y*num_cores_c*per_core_tiles_x;
            uint32_t id_r_writer = id_r_reader/2;
            if(num_cores_c > 1){
                idc_outer_limit = num_chunks;
                idc_inner_limit = num_cores_c/num_chunks;
            }
            for(int id_c_outer = 0; id_c_outer < idc_outer_limit; id_c_outer++){
                for (int id_c_inner = 0; id_c_inner < idc_inner_limit; id_c_inner++) {
                    uint32_t id_c = id_c_outer*idc_inner_limit + id_c_inner;
                    CoreCoord core = {(std::size_t)start_core_x + id_c, (std::size_t)start_core_y + id_r};

                    uint32_t reader_core_id = id_c*per_core_tiles_y;
                    reader_core_id += id_r_reader;


                    std::vector<uint32_t> reader_runtime_args = {
                        (std::uint32_t)reader_core_id,
                        (std::uint32_t)(in0_buffer->address()),  // in0_tensor_addr
                        (std::uint32_t) 0 //split on last dim
                    };
                    

                    bool parallelize_last_dim = num_cores_c > 1;
                    uint32_t writer_core_id = id_c_inner*per_core_tiles_y + (id_r_writer);

                    std::vector<uint32_t> writer_runtime_args = {
                        writer_core_id,
                        num_chunks,
                        parallelize_last_dim,
                        (uint32_t)id_c_outer
                    };
                    for(int tensor_idx = 0; tensor_idx <num_chunks; tensor_idx++){
                        writer_runtime_args.push_back(
                            (std::uint32_t)output_tensors[tensor_idx].buffer()->address());
                    }
                    for(int tensor_idx = 0; tensor_idx <num_chunks; tensor_idx++){
                        writer_runtime_args.push_back(
                            (std::uint32_t)(output_tensors[tensor_idx].buffer()->buffer_type() 
                                                        == tt_metal::BufferType::DRAM ? 1 : 0));
                    }
                    tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
                    tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
                }
            }
        }
    }


}

operation::ProgramWithCallbacks SplitTiled::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    return get_program(input_tensor, output_tensors, this->num_chunks, this->output_mem_config);
}

std::vector<Tensor> split_last_dim(const Tensor &input_tensor, uint32_t num_chunks, const MemoryConfig &mem_config) {
    uint32_t dim = 3;
    SplitTiled op = {.dim=dim, .num_chunks=num_chunks, .output_mem_config=mem_config};

    tt_metal::Device *device;
    // Get the device
    if (input_tensor.storage_type() == StorageType::OWNED) {
        device = AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    auto input_shape = input_tensor.shape();
    auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_shape);
    if (AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape)) {
        return operation::run(op, {input_tensor});
    } else {
        auto device = input_tensor.device();
        auto output_shape = op.compute_output_shapes({input_tensor}).at(0);
        const auto padded_tensor = AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, 0.0, Layout::TILE);
        auto output_tensors = operation::run(op, {padded_tensor});
        for (auto &output_tensor : output_tensors) {
            output_tensor = AutoFormat::format_output_tensor(output_tensor, output_shape, device, Layout::TILE);
        }
        return output_tensors;
    }
}


tt::stl::reflection::Attributes SplitTiled::attributes() const {
    return {
        {"dim", this->dim},
        {"num_chunks", this->num_chunks},
        {"output_mem_config", this->output_mem_config},
    };
}

}  // namespace tt_metal

}  // namespace tt
