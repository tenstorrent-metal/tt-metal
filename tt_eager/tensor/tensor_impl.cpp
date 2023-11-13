// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_impl_wrapper.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: os << "bfloat16"; break;
        case DataType::FLOAT32: os << "float32"; break;
        case DataType::UINT32: os << "uint32"; break;
        case DataType::BFLOAT8_B: os << "bfloat8_b"; break;
        default: throw std::invalid_argument("Unknown data type");
    }
    return os;
}

uint32_t get_page_size(DataType dtype, Layout layout, uint32_t total_size_bytes, const Shape& shape) {
    uint32_t W = shape[3];
    uint32_t page_size = 0;
    switch (layout) {
        case Layout::ROW_MAJOR: {
            uint32_t size_of_element = element_size_bytes_wrapper(dtype);
            page_size = W * size_of_element;
        }
        break;
        case Layout::TILE: {
            // TODO: Update to be generic for data type (issue 462)
            switch (dtype) {
                case DataType::BFLOAT16:
                case DataType::FLOAT32: {
                    // Float is converted to bfloat16 before being written to device
                    uint32_t size_of_element = element_size_bytes_wrapper(DataType::BFLOAT16);
                    page_size = constants::TILE_HW * size_of_element;
                }
                break;
                case DataType::UINT32: {
                    uint32_t size_of_element = element_size_bytes_wrapper(dtype);
                    page_size = constants::TILE_HW * size_of_element;
                }
                break;
                case DataType::BFLOAT8_B:  {
                    page_size = constants::BFLOAT8_B_TILE_HW;
                }
                break;
                default:
                    TT_ASSERT(false && "Unsupported data type!");
            }
            TT_ASSERT(total_size_bytes % page_size == 0);
        }
        break;
        default:
            TT_ASSERT(false && "Unsupported layout to write to device");
    }
    TT_ASSERT(page_size != 0);
    return page_size;
}

namespace detail {

DeviceBuffer allocate_interleaved_buffer_on_device(uint32_t buffer_size_bytes, Device *device,
            const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    uint32_t page_size = get_page_size(data_type, layout, buffer_size_bytes, shape);
    return std::make_shared<Buffer>(device, buffer_size_bytes, page_size, memory_config.buffer_storage);
}

DeviceBuffer allocate_contiguous_buffer_on_device(uint32_t buffer_size_bytes, Device *device,
                                                    const MemoryConfig& memory_config) {
    return std::make_shared<Buffer>(device, buffer_size_bytes, buffer_size_bytes, memory_config.buffer_storage);
}


DeviceBuffer allocate_sharded_buffer_on_device(uint32_t buffer_size_bytes, Device *device,
                                            const Shape& shape, DataType data_type, Layout layout,
                                            std::optional<ShardSpec> shard_params,
                                            const MemoryConfig& memory_config) {
    uint32_t page_size = get_page_size(data_type, layout, buffer_size_bytes, shape);
    std::cout << "About to make buffer in allocate_sharded_buffer" << std::endl;
    return std::make_shared<Buffer>(device, buffer_size_bytes, page_size,
                                 memory_config.buffer_storage,
                                 memory_config.memory_layout,
                                 shard_params);
}


}



DeviceBuffer allocate_buffer_on_device(uint32_t buffer_size_bytes, Device *device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config, std::optional<ShardSpec> shard_params) {
    if (memory_config.memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return detail::allocate_interleaved_buffer_on_device(buffer_size_bytes, device, shape, data_type, layout, memory_config);
    }
    else if(memory_config.memory_layout == tt::tt_metal::TensorMemoryLayout::SINGLE_BANK){
        return detail::allocate_contiguous_buffer_on_device(buffer_size_bytes, device, memory_config);
    }
    else {
        std::cout << "About to allocate sharded buffer " << std::endl;
        TT_ASSERT( memory_config.is_sharded() && "Incorrect Memory Layout");
        return detail::allocate_sharded_buffer_on_device(buffer_size_bytes, device, shape, data_type, layout, shard_params, memory_config);
    }
}

void validate_on_device_dtype_and_layout(Device *device, DataType dtype, Layout layout) {
    // TODO: Get supported layout and dtypes from device
    auto supported_dtype = [&dtype]() {
        TT_ASSERT(
            (dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B || dtype == DataType::UINT32) &&
            "Only BFLOAT16 , BFLOAT8_B or UINT32 is supported on device!"
        );
    };
    auto supported_layout = [&dtype, &layout]() {
        switch (dtype) {
            case DataType::UINT32:
                break;
            case DataType::BFLOAT16:
                break;
            case DataType::BFLOAT8_B:
                TT_ASSERT(layout == Layout::TILE && "Only TILE layout is supported for BFLOAT8_B dtype!");
                break;
            default:
                TT_ASSERT(false && "Only BFLOAT16 or BFLOAT8_B is supported on device!");
                break;
            }
    };
    supported_dtype();
    supported_layout();
}

Tensor to_layout_bfloat8_b(const Tensor &tensor, Layout target_layout) {
    // TODO(arakhmati): do not convert to FLOAT32

    if(tensor.layout() == target_layout) {
        return tensor;
    }

    // Convert to FLOAT32 tensor and change layout
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data = unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor = Tensor(OwnedStorage{input_float_buffer}, tensor.shape(), DataType::FLOAT32, tensor.layout()).to(target_layout);

    // Convert back to BFLOAT8_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data = pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), tensor.shape(), DataType::BFLOAT8_B, target_layout);
}


Tensor pad_bfloat8_b(const Tensor &tensor, const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value) {
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and pad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data = unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor = Tensor(OwnedStorage{input_float_buffer}, tensor.shape(), DataType::FLOAT32, tensor.layout()).pad(output_tensor_shape, input_tensor_start, pad_value);

    // Convert back to BFLOAT8_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data = pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), float_tensor.shape(), DataType::BFLOAT8_B, tensor.layout());
}

Tensor unpad_bfloat8_b(const Tensor &tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and unpad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data = unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor = Tensor(OwnedStorage{input_float_buffer}, tensor.shape(), DataType::FLOAT32, tensor.layout()).unpad(output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT8_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data = pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), float_tensor.shape(), DataType::BFLOAT8_B, tensor.layout());
}


}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
