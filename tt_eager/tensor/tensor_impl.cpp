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
        case DataType::BFLOAT8_B: os << "bfloat8_b"; break;
        case DataType::BFLOAT4_B: os << "bfloat4_b"; break;
        case DataType::BFLOAT16: os << "bfloat16"; break;
        case DataType::FLOAT32: os << "float32"; break;
        case DataType::UINT16: os << "uint16"; break;
        case DataType::UINT32: os << "uint32"; break;
        default: throw std::invalid_argument("Unknown data type");
    }
    return os;
}



uint32_t get_page_size(DataType dtype, Layout layout, uint32_t total_size_bytes, const Shape& shape) {
    uint32_t W = shape[-1];
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
                case DataType::BFLOAT16: {
                    // Float is converted to bfloat16 before being written to device
                    uint32_t size_of_element = element_size_bytes_wrapper(DataType::BFLOAT16);
                    page_size = constants::TILE_HW * size_of_element;
                }
                break;
                case DataType::FLOAT32: {
                    uint32_t size_of_element = element_size_bytes_wrapper(DataType::FLOAT32);
                    page_size = constants::TILE_HW * size_of_element;
                }
                break;
                case DataType::UINT32:
                case DataType::UINT16: {
                    uint32_t size_of_element = element_size_bytes_wrapper(dtype);
                    page_size = constants::TILE_HW * size_of_element;
                }
                break;
                case DataType::BFLOAT4_B: {
                    page_size = constants::BFLOAT4_B_TILE_HW;
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



std::array<uint32_t, 2> get_sharded_page_shape(Layout layout,  DataType dtype, std::array<uint32_t, 2> shard_shape) {
    uint32_t page_size = 0;

    std::array<uint32_t, 2> page_shape = {constants::TILE_HEIGHT, constants::TILE_WIDTH};

    //Physical limitation in FD for now
    switch (layout) {
        case Layout::ROW_MAJOR: {
            //TODO: Explore valid page shapes other than 1,W
            page_shape = {1, shard_shape[1]};
        }
        break;
        case Layout::TILE: {;}
        break;
        default:
            TT_ASSERT(false && "Unsupported layout to write to device");
    }

    return page_shape;
}

namespace detail {

DeviceBuffer allocate_interleaved_buffer_on_device(uint32_t buffer_size_bytes, Device *device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    uint32_t page_size = get_page_size(data_type, layout, buffer_size_bytes, shape);
    return std::make_shared<Buffer>(device, buffer_size_bytes, page_size, memory_config.buffer_type);
}

DeviceBuffer allocate_contiguous_buffer_on_device(uint32_t buffer_size_bytes, Device *device, const MemoryConfig& memory_config) {
    return std::make_shared<Buffer>(device, buffer_size_bytes, buffer_size_bytes, memory_config.buffer_type);
}


DeviceBuffer allocate_sharded_buffer_on_device(uint32_t buffer_size_bytes, Device *device,
                                            const Shape& shape, DataType data_type, Layout layout,
                                            std::optional<ShardSpecBuffer> shard_params,
                                            const MemoryConfig& memory_config) {
    auto page_shape = shard_params.value().page_shape;
    uint32_t size_of_element = element_size_bytes_wrapper(data_type);
    uint32_t page_size = page_shape[0] * page_shape[1] * size_of_element;
    if(layout == Layout::TILE){
        page_size = get_page_size(data_type, layout, buffer_size_bytes, shape);
    }

    return std::make_shared<Buffer>(device, buffer_size_bytes, page_size,
                                 memory_config.buffer_type,
                                 memory_config.memory_layout,
                                 shard_params);
}


}




DeviceBuffer allocate_buffer_on_device(uint32_t buffer_size_bytes, Device *device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config, std::optional<ShardSpecBuffer> shard_spec) {
    if (memory_config.memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return detail::allocate_interleaved_buffer_on_device(buffer_size_bytes, device, shape, data_type, layout, memory_config);
    }
    else if(memory_config.memory_layout == tt::tt_metal::TensorMemoryLayout::SINGLE_BANK){
        return detail::allocate_contiguous_buffer_on_device(buffer_size_bytes, device, memory_config);
    }
    else {
        TT_ASSERT( memory_config.is_sharded() && "Incorrect Memory Layout");
        return detail::allocate_sharded_buffer_on_device(buffer_size_bytes, device, shape, data_type, layout, shard_spec, memory_config);
    }
}

void validate_on_device_dtype_and_layout(Device *device, DataType dtype, Layout layout) {
    // TODO: Get supported layout and dtypes from device
    auto supported_dtype = [&dtype]() {
        TT_ASSERT(
            (dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT4_B || dtype == DataType::UINT32 || dtype == DataType::UINT16) &&
            "Only BFLOAT16, BFLOAT8_B, BFLOAT4_B, UINT32, or UINT16 is supported on device!"
        );
    };
    auto supported_layout = [&dtype, &layout]() {
        switch (dtype) {
            case DataType::UINT32:
            case DataType::UINT16:
            case DataType::BFLOAT16:
            case DataType::FLOAT32:
                break;
            case DataType::BFLOAT4_B:
            case DataType::BFLOAT8_B:
                TT_ASSERT(layout == Layout::TILE && "Only TILE layout is supported for BFLOAT8_B dtype!");
                break;
            default:
                TT_ASSERT(false && "Only BFLOAT16, BFLOAT8_B, BFLOAT4_B, UINT32, or UINT16 is supported on device!");
                break;
            }
    };
    supported_dtype();
    supported_layout();
}

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
