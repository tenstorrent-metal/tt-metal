// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/run_operation.hpp"

#include <chrono>
#include <tt_eager/tensor/tensor.hpp>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_stl/reflection.hpp"

namespace tt::tt_metal::operation {

namespace detail {

static const Device& get_device(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}) {
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            return input_tensor.device();
        }
    }
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() == StorageType::DEVICE) {
            return optional_input_tensor.value().device();
        }
    }

    return AutoFormat::GetDefaultDevice();
}

}  // namespace detail

namespace detail {

void override_addresses(
    const OverrideAddressesCallback& override_addresses_callback,
    const Program &program,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<Tensor>& output_tensors
) {
    std::vector<Buffer*> input_buffers;
    for (auto& tensor : input_tensors) {
        input_buffers.push_back(tensor.buffer());
    }
    for (auto& tensor : optional_input_tensors) {
        auto buffer = tensor.has_value() ? tensor.value().buffer() : nullptr;
        input_buffers.push_back(buffer);
    }

    std::vector<Buffer*> output_buffers;
    for (auto& tensor : output_tensors) {
        output_buffers.push_back(tensor.buffer());
    }

    override_addresses_callback(program, input_buffers, output_buffers);
}

void setup_profiler(const HostOperation& operation, const std::vector<Tensor>& input_tensors) {
    auto profiler_info = operation.create_profiler_info(input_tensors);
    if (profiler_info.preferred_name.has_value()) {
        op_profiler::set_preferred_name(profiler_info.preferred_name.value());
    }
    op_profiler::append_meta_data(fmt::format("{}", operation.attributes()));
}

void setup_profiler(const DeviceOperation& operation, const std::vector<Tensor>& input_tensors, const Program& program) {
    auto profiler_info = operation.create_profiler_info(input_tensors);
    if (profiler_info.preferred_name.has_value()) {
        op_profiler::set_preferred_name(profiler_info.preferred_name.value());
    }
    if (profiler_info.parallelization_strategy.has_value()) {
        op_profiler::set_parallelization_strategy(profiler_info.parallelization_strategy.value());
    }

    op_profiler::append_math_fidelities(program);
    op_profiler::append_meta_data(fmt::format("{}", operation.attributes()));
}

std::vector<Tensor> run_without_program_cache(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {
    ZoneScoped;
    ZoneText( operation.get_type_name().c_str(), operation.get_type_name().length() );

    const auto &device = detail::get_device(input_tensors, optional_input_tensors);
    auto output_tensors = operation.create_output_tensors(input_tensors);

    auto program_with_callbacks = operation.create_program(input_tensors, optional_input_tensors, output_tensors);
    auto& program = program_with_callbacks.program;

    auto do_profile = op_profiler::get_profiler_flag();
    if (do_profile) { detail::setup_profiler(operation, input_tensors, program); }

    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        EnqueueProgram(*tt::tt_metal::detail::GLOBAL_CQ, program, false);
        // Only need to dump device data when in dispatch mode
        // LaunchKernel automatically dumps device data
        op_profiler::dump_device_profiler_results(device, program);
    } else {
        ::detail::LaunchProgram(device, program);
    }

    return output_tensors;
}

std::vector<Tensor> run_with_program_cache(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {
    ZoneScoped;
    ZoneText( operation.get_type_name().c_str(), operation.get_type_name().length() );

    const auto &device = detail::get_device(input_tensors, optional_input_tensors);
    auto output_tensors = operation.create_output_tensors(input_tensors);

    auto&& [program_with_callbacks, cache_hit] = program_cache::get_or_create(
        operation, input_tensors, optional_input_tensors, output_tensors, device
    );
    TT_ASSERT(program_with_callbacks.supports_program_cache());

    auto& program = program_with_callbacks.program;
    if (cache_hit) {
        ZoneScopedN ("Cache_hit_set_runtime_args");
        if (program_with_callbacks.override_addresses_callback.has_value()) {
            auto override_addresses_callback = program_with_callbacks.override_addresses_callback.value();
            override_addresses(
                override_addresses_callback,
                program, input_tensors, optional_input_tensors, output_tensors
            );
        }

        if (program_with_callbacks.override_runtime_arguments_callback.has_value()) {
            auto override_runtime_arguments_callback = program_with_callbacks.override_runtime_arguments_callback.value();
            operation.override_runtime_arguments(override_runtime_arguments_callback, program, input_tensors, optional_input_tensors, output_tensors);
        }
    }

    {
        ZoneScopedN ("Profiler Check");
        auto do_profile = op_profiler::get_profiler_flag();
        if (do_profile) { detail::setup_profiler(operation, input_tensors, program); }
    }

    const char *TT_METAL_SLOW_DISPATCH_MODE = nullptr;
    {
        ZoneScopedN ("Get Env Var");
        TT_METAL_SLOW_DISPATCH_MODE =  std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    }
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        EnqueueProgram(*tt::tt_metal::detail::GLOBAL_CQ, program, false);
        // Only need to dump device data when in dispatch mode
        // LaunchKernel automatically dumps device data
        op_profiler::dump_device_profiler_results(device, program);
    } else {
        ::detail::LaunchProgram(device, program);
    }

    return output_tensors;
}

template <typename OperationType>
constexpr op_profiler::OpType get_profiler_operation_type() {
    if constexpr (std::is_same_v<OperationType, HostOperation>) {
        return op_profiler::OpType::tt_dnn_cpu;
    } else if constexpr (std::is_same_v<OperationType, DeviceOperation>) {
        return op_profiler::OpType::tt_dnn_device;
    } else if constexpr (std::is_same_v<OperationType, ExternalOperation>) {
        return op_profiler::OpType::python_fallback;
    } else {
        static_assert(tt::stl::concepts::always_false_v<OperationType>, "OperationType is not supported!");
    }
}

template <typename Function>
constexpr auto decorate_operation(const Function& function) {
#ifndef DEBUG
    return function;
#else
    return [function]<typename Operation, typename... Args>(const Operation& operation, Args&&... args) {
        const auto start{std::chrono::steady_clock::now()};

        log_operation(operation, args...);

        auto output_tensors = function(operation, args...);

        const auto end{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{end - start};
        tt::log_debug(
            tt::LogOp, "Operation {:50} finished in {:15} seconds", operation.get_type_name(), elapsed_seconds.count());

        return output_tensors;
    };
#endif
}
std::vector<Tensor> run_host_operation(const HostOperation& operation, const std::vector<Tensor>& input_tensors) {
    ZoneScoped;
    ZoneName(operation.get_type_name().c_str(), operation.get_type_name().size());

    auto profile_scope = op_profiler::OpProfileScope(operation.get_type_name(), op_profiler::OpType::tt_dnn_cpu);
    auto do_profile = op_profiler::get_profiler_flag();
    if (do_profile) {
        detail::setup_profiler(operation, input_tensors);
    }

    operation.validate(input_tensors);
    auto output_tensors = operation.compute_output_tensors(input_tensors);

    op_profiler::append_all_tensor_io_data(input_tensors, {}, output_tensors);

    return output_tensors;
}

std::vector<Tensor> run_device_operation(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {
    auto profile_scope = op_profiler::OpProfileScope(operation.get_type_name(), op_profiler::OpType::tt_dnn_device);

    operation.validate(input_tensors, optional_input_tensors);

    std::vector<Tensor> output_tensors;
    if (program_cache::is_enabled()) {
        output_tensors = detail::run_with_program_cache(operation, input_tensors, optional_input_tensors);
    } else {
        output_tensors = detail::run_without_program_cache(operation, input_tensors, optional_input_tensors);
    }

    op_profiler::append_all_tensor_io_data(input_tensors, optional_input_tensors, output_tensors);

    return output_tensors;
}
}  // namespace detail

std::vector<Tensor> run(const HostOperation& operation, const std::vector<Tensor>& input_tensors) {
    return detail::decorate_operation(detail::run_host_operation)(operation, input_tensors);
}

std::vector<Tensor> run(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {
    return detail::decorate_operation(detail::run_device_operation)(operation, input_tensors, optional_input_tensors);
}

std::vector<Tensor> run_without_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors
) {
    const Device& device = detail::get_device(input_tensors, optional_input_tensors);

    std::vector<Tensor> input_tensors_on_dev;
    input_tensors_on_dev.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() == StorageType::OWNED) {
            input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(input_tensor, device));
        } else {
            input_tensors_on_dev.push_back(input_tensor);
        }
    }
    std::vector<std::optional<const Tensor>> optional_input_tensors_on_dev;
    optional_input_tensors_on_dev.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() == StorageType::OWNED) {
            optional_input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(optional_input_tensor.value(), device));
        } else {
            optional_input_tensors_on_dev.push_back(optional_input_tensor);
        }
    }
    return run(operation, input_tensors_on_dev, optional_input_tensors_on_dev);
}

// To be deprecated/removed in favor of new implementation where ops specifically request how to format inputs/outputss
std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const float pad_value,
    const bool pad_c
) {
    const Device& device = detail::get_device(input_tensors, optional_input_tensors);

    auto output_shapes = operation.compute_output_shapes(input_tensors);

    std::vector<Tensor> formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape(), pad_c);
        auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
        if (pad_input) {
            formatted_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value, Layout::TILE));
        } else {
            formatted_input_tensors.push_back(input_tensor);
        }
    }

    std::vector<std::optional<const Tensor>> formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value()) {
            auto& input_tensor = optional_input_tensor.value();
            auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape(), pad_c);
            auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
            if (pad_input) {
                formatted_optional_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value, Layout::TILE));
            } else {
                formatted_optional_input_tensors.push_back(input_tensor);
            }
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensor);
        }
    }

    auto output_tensors = run(operation, formatted_input_tensors, formatted_optional_input_tensors);

    std::vector<Tensor> formatted_output_tensors;
    formatted_output_tensors.reserve(output_tensors.size());
    for (auto i = 0; i < output_tensors.size(); i++) {
        formatted_output_tensors.push_back(AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device, Layout::TILE));
    }
    return formatted_output_tensors;
}

std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<FormatParams>>& optional_input_formatting
) {
    const Device& device = detail::get_device(input_tensors, optional_input_tensors);

    auto output_shapes = operation.compute_output_shapes(input_tensors);

    std::vector<Tensor> formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); i++) {
        formatted_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensors[i], device, input_formatting[i].pad_shape, input_formatting[i].pad_value, input_formatting[i].target_layout));
    }

    std::vector<std::optional<const Tensor>> formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (uint32_t i = 0; i < optional_input_tensors.size(); i++) {
        if (optional_input_tensors[i].has_value()) {
            auto& input_tensor = optional_input_tensors[i].value();
            TT_ASSERT(optional_input_formatting.at(i).has_value());
            auto& input_formatting = optional_input_formatting.at(i).value();
            formatted_optional_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensor, device, input_formatting.pad_shape, input_formatting.pad_value, input_formatting.target_layout));
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensors[i]);
        }
    }

    auto output_tensors = run(operation, formatted_input_tensors, formatted_optional_input_tensors);

    std::vector<Tensor> formatted_output_tensors;
    formatted_output_tensors.reserve(output_tensors.size());
    for (auto i = 0; i < output_tensors.size(); i++) {
        formatted_output_tensors.push_back(AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device, output_layouts[i]));
    }

    return formatted_output_tensors;
}

}
