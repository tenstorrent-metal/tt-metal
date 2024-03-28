// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <common/math.hpp>
#include <optional>
#include <random>
#include <tensor/owned_buffer.hpp>
#include <tensor/owned_buffer_functions.hpp>
#include <tensor/tensor.hpp>
#include <tensor/tensor_utils.hpp>
#include <tensor/types.hpp>
#include <tt_eager/tensor/tensor_impl.hpp>

namespace tt {

namespace numpy {

using tt_metal::DataType;
using tt_metal::Device;
using tt_metal::Layout;
using tt_metal::MemoryConfig;
using tt_metal::OwnedStorage;
using tt_metal::Shape;
using tt_metal::StorageType;
using tt_metal::Tensor;
namespace detail {

template <typename T>
constexpr static DataType get_data_type() {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return DataType::UINT32;
    } else if constexpr (std::is_same_v<T, float>) {
        return DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        return DataType::BFLOAT16;
    } else {
        TT_THROW("Unsupported DataType!");
    }
}

template <typename T>
static Tensor full(
    const Shape& shape,
    T value,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    constexpr DataType data_type = detail::get_data_type<T>();
    auto owned_buffer = tt_metal::owned_buffer::create<T>(tt_metal::compute_volume(shape));
    std::fill(std::begin(owned_buffer), std::end(owned_buffer), value);
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

}  // namespace detail

template <typename T>
static Tensor full(
    const Shape& shape,
    const T value,
    const DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    switch (data_type) {
        case DataType::UINT32: {
            return detail::full<uint32_t>(shape, uint32_t(value), layout, device, output_mem_config);
        }
        case DataType::FLOAT32: {
            return detail::full<float>(shape, float(value), layout, device, output_mem_config);
        }
        case DataType::BFLOAT16: {
            return detail::full<bfloat16>(shape, bfloat16(value), layout, device, output_mem_config);
        }
        default: TT_THROW("Unsupported DataType!");
    }
}

static Tensor zeros(
    const Shape& shape,
    const DataType data_type = DataType::BFLOAT16,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return full(shape, 0.0f, data_type, layout, device, output_mem_config);
}

static Tensor ones(
    const Shape& shape,
    const DataType data_type = DataType::BFLOAT16,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return full(shape, 1.0f, data_type, layout, device, output_mem_config);
}

template <typename T>
static Tensor full_like(
    const Tensor& input_tensor,
    const T value,
    std::optional<DataType> data_type = std::nullopt,
    std::optional<Layout> layout = std::nullopt,
    std::optional<MemoryConfig> output_mem_config = std::nullopt) {
    DataType data_type_to_use = input_tensor.get_dtype();
    if (data_type.has_value()) {
        data_type_to_use = data_type.value();
    }
    Layout layout_to_use = input_tensor.get_layout();
    if (layout.has_value()) {
        layout_to_use = layout.value();
    }
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        MemoryConfig output_mem_config_to_use = input_tensor.memory_config();
        if (output_mem_config.has_value()) {
            output_mem_config_to_use = output_mem_config.value();
        }
        return full(
            input_tensor.get_legacy_shape(),
            value,
            data_type_to_use,
            layout_to_use,
            input_tensor.device(),
            output_mem_config_to_use);
    } else {
        return full(input_tensor.get_legacy_shape(), value, data_type_to_use, layout_to_use);
    }
}

static Tensor zeros_like(
    const Tensor& input_tensor,
    std::optional<DataType> data_type = std::nullopt,
    std::optional<Layout> layout = std::nullopt,
    std::optional<MemoryConfig> output_mem_config = std::nullopt) {
    return full_like(input_tensor, 0.0f, data_type, layout, output_mem_config);
}

static Tensor ones_like(
    const Tensor& input_tensor,
    std::optional<DataType> data_type = std::nullopt,
    std::optional<Layout> layout = std::nullopt,
    std::optional<MemoryConfig> output_mem_config = std::nullopt) {
    return full_like(input_tensor, 1.0f, data_type, layout, output_mem_config);
}

template <typename T>
static Tensor arange(
    int64_t start,
    int64_t stop,
    int64_t step,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    constexpr DataType data_type = detail::get_data_type<T>();
    // Current implementation restrictions
    TT_ASSERT(step > 0, "Step must be greater than 0");
    TT_ASSERT(start < stop, "Start must be less than step");
    auto size = div_up((stop - start), step);
    auto owned_buffer = tt_metal::owned_buffer::create<T>(size);

    auto index = 0;
    for (auto value = start; value < stop; value += step) {
        if constexpr (std::is_same_v<T, bfloat16>) {
            owned_buffer[index++] = T(static_cast<float>(value));
        } else {
            owned_buffer[index++] = static_cast<T>(value);
        }
    }
    auto output = Tensor(OwnedStorage{owned_buffer}, {1, 1, 1, static_cast<uint32_t>(size)}, data_type, layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor arange(
    const int64_t& start,
    const int64_t& step,
    const Shape& shape,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    constexpr DataType data_type = detail::get_data_type<T>();
    // Current implementation restrictions
    TT_ASSERT(step > 0, "Step must be greater than 0");
    auto owned_buffer = tt_metal::owned_buffer::create<T>(tt_metal::compute_volume(shape));

    auto value = start;
    for (auto index = 0; index < owned_buffer.size(); index++) {
        if constexpr (std::is_same_v<T, bfloat16>) {
            owned_buffer[index++] = T(static_cast<float>(value));
        } else {
            owned_buffer[index++] = static_cast<T>(value);
        }
        value += step;
    }
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T, bool IS_UPPER>
static Tensor index_trilu(
    const Shape& shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    // Current implementation restrictions
    auto owned_buffer = tt_metal::owned_buffer::create<T>(tt_metal::compute_volume(shape));

    auto index = 0;
    auto rank = shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    auto offset = shape[penultimate] * shape[ultimate];
    auto iterations = 1;
    for (int itr = 0; itr < rank - 2; itr++) iterations *= shape[itr];
    for (uint32_t itr = 0; itr < iterations; itr++) {
        for (int32_t y = 0; y < shape[penultimate]; y++) {
            for (int32_t x = 0; x < shape[ultimate]; x++) {
                int32_t value = (IS_UPPER) ? (x >= (y + diag)) : (y >= (x - diag));
                if constexpr (std::is_same_v<T, bfloat16>) {
                    owned_buffer[index + y * shape[ultimate] + x] = T(static_cast<float>(value));
                } else {
                    owned_buffer[index + y * shape[ultimate] + x] = static_cast<T>(value);
                }
            }  // dim X
        }      // dim Y
        index += offset;
    }
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_width(
    const Shape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt_metal::owned_buffer::create<T>(tt_metal::compute_volume(shape));

    auto index = 0;
    auto value = 0;
    auto rank = shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    auto offset = shape[penultimate] * shape[ultimate];
    auto iterations = 1;
    for (int itr = 0; itr < rank - 2; itr++) iterations *= shape[itr];
    for (uint32_t itr = 0; itr < iterations; itr++) {
        for (int32_t y = 0; y < shape[penultimate]; y++) {
            for (int32_t x = 0; x < shape[ultimate]; x++) {
                owned_buffer[index + y * shape[ultimate] + x] = T(static_cast<float>(value));
                value = value + 1;
            }  // dim X
            value = 0;
        }  // dim Y
        index += offset;
    }  // dim W
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_height(
    const Shape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt_metal::owned_buffer::create<T>(tt_metal::compute_volume(shape));

    auto index = 0;
    auto value = 0;
    auto rank = shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    auto offset = shape[penultimate] * shape[ultimate];
    auto iterations = 1;
    for (int itr = 0; itr < rank - 2; itr++) iterations *= shape[itr];
    for (uint32_t itr = 0; itr < iterations; itr++) {
        for (int32_t y = 0; y < shape[penultimate]; y++) {
            for (int32_t x = 0; x < shape[ultimate]; x++) {
                owned_buffer[index + y * shape[ultimate] + x] = T(static_cast<float>(value));
            }  // dim X
            value = value + 1;
        }  // dim Y
        value = 0;
        index += offset;
    }  // dim W
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_all(
    const Shape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt_metal::owned_buffer::create<T>(tt_metal::compute_volume(shape));

    auto index = 0;
    auto value = 0;
    auto rank = shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < shape[penultimate]; y++) {
                for (uint32_t x = 0; x < shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                    value = value + 1;
                }  // dim W
            }      // dim H
        }          // dim C
    }              // dim N
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template<typename T>
static Tensor prod_result_computation_GS(const Tensor& input_tensor, DataType data_type,
			  const Layout layout , Device * device = nullptr,
			  const MemoryConfig& output_mem_config = MemoryConfig{.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    const Shape& s_a = input_tensor.shape();
    auto owned_buffer = tt_metal::owned_buffer::create<T>(tt_metal::compute_volume(s_a)); //ouput
    auto device_buffer = input_tensor.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    vector<T> data_vec;
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(input_tensor.device()->command_queue(), device_buffer, data_vec.data(), true);
    } else {
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto input_buffer = owned_buffer::create<T>(std::move(data_vec));
    const Shape input_tensor_strides = input_tensor.strides();
    auto result = static_cast<T>(1.0f);
    for(uint32_t i = s_a[0]-1; i < s_a[0]; i++) {
        for(int32_t j = s_a[1]-1; j < s_a[1]; j++) {
            for(int32_t k = s_a[2]-32; k < s_a[2]; k++) { //access last tile
                for(int32_t l = s_a[3]-32; l < s_a[3]; l++) {
                    auto input_index = l + input_tensor_strides[2] * k + input_tensor_strides[1] * j + input_tensor_strides[0] * i;
                    if(k>=s_a[2]-2 && l>=s_a[3]-32){ //to access 2*32 in TILE layout
                        result = result * static_cast<T>(input_buffer[input_index]);
                        owned_buffer[input_index] = static_cast<T>(0.0f);
                    }else{
                        owned_buffer[input_index] = static_cast<T>(0.0f);
                    }
                }
            }
        }
    }
    owned_buffer[0] = result; //store the result at the first position of the tensor,and the rest of the values as 0.0f
    auto output = Tensor(OwnedStorage{owned_buffer}, s_a, data_type, layout).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template<typename T>
static Tensor prod_result_computation_WH_B0(const Tensor& input_tensor, DataType data_type,
			  const Layout layout , Device * device = nullptr,
			  const MemoryConfig& output_mem_config = MemoryConfig{.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    const Shape& s_a = input_tensor.shape();
    auto owned_buffer = tt_metal::owned_buffer::create<T>(tt_metal::compute_volume(s_a)); //ouput
    auto device_buffer = input_tensor.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    vector<T> data_vec;
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(input_tensor.device()->command_queue(), device_buffer, data_vec.data(), true);
    } else {
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto input_buffer = owned_buffer::create<T>(std::move(data_vec));
    const Shape input_tensor_strides = input_tensor.strides();
    auto result = static_cast<T>(1.0f);
    // need to access the last 4 rows and alternating columns of index 17 ,19, 21, 23, 25, 27, 29, 31
    for(uint32_t i = s_a[0]-1; i < s_a[0]; i++) {
        for(int32_t j = s_a[1]-1; j < s_a[1]; j++) {
            for(int32_t k = s_a[2]-32; k < s_a[2]; k++) { //access last tile
                for(int32_t l = s_a[3]-32; l < s_a[3]; l++) {
                    auto input_index = l + input_tensor_strides[2] * k + input_tensor_strides[1] * j + input_tensor_strides[0] * i;
                    if(k>=s_a[2]-4 && (l==s_a[3]-15 || l==s_a[3]-13 || l==s_a[3]-11 || l==s_a[3]-9 || l==s_a[3]-7 || l==s_a[3]-5 || l==s_a[3]-3 || l==s_a[3]-1)){ //to access 4*16 elements placed alternatively starting from index 17W in TILE layout
                        result = result * static_cast<T>(input_buffer[input_index]);
                        owned_buffer[input_index] = static_cast<T>(0.0f);
                    }else{
                        owned_buffer[input_index] = static_cast<T>(0.0f);
                    }
                }
            }
        }
    }
    owned_buffer[0] = result; //store the result at the first position of the tensor,and the rest of the values as 0.0f
    auto output = Tensor(OwnedStorage{owned_buffer}, s_a, data_type, layout).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}


template <typename T>
static Tensor index_channel(
    const Shape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt_metal::owned_buffer::create<T>(tt_metal::compute_volume(shape));

    auto index = 0;
    auto value = 0;
    auto rank = shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < shape[penultimate]; y++) {
                for (uint32_t x = 0; x < shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                }  // dim W
            }      // dim H
            value = value + 1;
        }  // dim C
        value = 0;
    }  // dim N
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_batch(
    const Shape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt_metal::owned_buffer::create<T>(tt_metal::compute_volume(shape));

    auto index = 0;
    auto value = 0;
    auto rank = shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < shape[penultimate]; y++) {
                for (uint32_t x = 0; x < shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                }  // dim W
            }      // dim H
        }          // dim C
        value = value + 1;
    }  // dim N

    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor manual_insertion(
    const Tensor& input_tensor,
    const Shape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    TT_ASSERT(input_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_ASSERT(
        shape[0] * shape[1] * shape[2] * shape[3] == input_tensor.volume(),
        "Required shape volume must match old shape volume");
    auto device_buffer = input_tensor.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(
            input_tensor.device()->command_queue(), device_buffer, data_vec.data(), true);
    } else {
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto owned_buffer = owned_buffer::create<T>(std::move(data_vec));
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_tril(
    const Shape& shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return index_trilu<T, false>(shape, diag, data_type, layout, device, output_mem_config);
}

template <typename T>
static Tensor index_triu(
    const Shape& shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return index_trilu<T, true>(shape, diag, data_type, layout, device, output_mem_config);
}

namespace random {

inline auto RANDOM_GENERATOR = std::mt19937(0);

static void seed(std::size_t seed) { RANDOM_GENERATOR = std::mt19937(seed); }

template <typename T>
static Tensor uniform(T low, T high, const Shape& shape, const Layout layout = Layout::ROW_MAJOR) {
    constexpr DataType data_type = detail::get_data_type<T>();

    auto owned_buffer = tt_metal::owned_buffer::create<T>(tt_metal::compute_volume(shape));

    if constexpr (std::is_same_v<T, uint32_t>) {
        auto rand_value = std::bind(std::uniform_int_distribution<T>(low, high), RANDOM_GENERATOR);
        for (auto index = 0; index < owned_buffer.size(); index++) {
            owned_buffer[index] = rand_value();
        }
    } else if constexpr (std::is_same_v<T, float>) {
        auto rand_value = std::bind(std::uniform_real_distribution<T>(low, high), RANDOM_GENERATOR);
        for (auto index = 0; index < owned_buffer.size(); index++) {
            owned_buffer[index] = rand_value();
        }
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        auto rand_value =
            std::bind(std::uniform_real_distribution<float>(low.to_float(), high.to_float()), RANDOM_GENERATOR);
        for (auto index = 0; index < owned_buffer.size(); index++) {
            owned_buffer[index] = bfloat16(rand_value());
        }
    }

    return Tensor(OwnedStorage{owned_buffer}, shape, data_type, layout);
}

static Tensor random(
    const Shape& shape, const DataType data_type = DataType::BFLOAT16, const Layout layout = Layout::ROW_MAJOR) {
    switch (data_type) {
        case DataType::UINT32: return uniform(0u, 1u, shape, layout);
        case DataType::FLOAT32: return uniform(0.0f, 1.0f, shape, layout);
        case DataType::BFLOAT16: return uniform(bfloat16(0.0f), bfloat16(1.0f), shape, layout);
        default: TT_THROW("Unsupported DataType!");
    };
}

}  // namespace random

namespace detail {
static bool nearly_equal(float a, float b, float epsilon = 1e-5f, float abs_threshold = 1e-5f) {
    auto diff = std::abs(a - b);
    auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
    auto result = diff < std::max(abs_threshold, epsilon * norm);
    if (not result) {
        tt::log_error(tt::LogTest, "{} != {}", a, b);
    }
    return result;
}

template <typename... Args>
static bool nearly_equal(bfloat16 a, bfloat16 b, Args... args) {
    return nearly_equal(a.to_float(), b.to_float(), args...);
}
}  // namespace detail

template <typename DataType, typename... Args>
static bool allclose(const Tensor& tensor_a, const Tensor& tensor_b, Args... args) {
    if (tensor_a.get_legacy_shape() != tensor_b.get_legacy_shape()) {
        return false;
    }

    if (tensor_a.get_dtype() != tensor_b.get_dtype()) {
        return false;
    }

    auto tensor_a_buffer = tt_metal::owned_buffer::get_as<DataType>(tensor_a);
    auto tensor_b_buffer = tt_metal::owned_buffer::get_as<DataType>(tensor_b);

    for (int index = 0; index < tensor_a_buffer.size(); index++) {
        if (not detail::nearly_equal(tensor_a_buffer[index], tensor_b_buffer[index], args...)) {
            return false;
        }
    }
    return true;
}

}  // namespace numpy
}  // namespace tt
