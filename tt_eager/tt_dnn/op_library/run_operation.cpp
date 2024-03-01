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
#include "tt_metal/tt_stl/reflection.hpp"

namespace tt::tt_metal::operation {

namespace detail {

static Device* get_device(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}) {
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            TT_FATAL(input_tensor.buffer() != nullptr, "Operands need to be allocated in buffers on device");
            return input_tensor.device();
        }
    }
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() == StorageType::DEVICE) {
            return optional_input_tensor.value().device();
        }
    }
    auto device = AutoFormat::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to operation are on device");
    return device;
}

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

template <typename Function>
constexpr auto decorate_host_operation(const Function& function) {
    return [function]<typename Operation, typename... Args>(const Operation& operation, Args&&... args) {
        log_operation(operation, args...);
        auto output_tensors = function(operation, args...);
        return output_tensors;
    };
}

template <typename Function>
constexpr auto decorate_device_operation(const Function& function) {
    return [function]<typename Operation, typename... Tensors>(
               std::optional<std::reference_wrapper<CommandQueue>> queue,
               const Operation& operation,
               Tensors&&... tensors) {
        log_operation(operation, tensors...);
        auto output_tensors = function(queue, operation, tensors...);
        return output_tensors;
    };
}

std::vector<Tensor> run_host_operation(const HostOperation& operation, const std::vector<Tensor>& input_tensors) {
    ZoneScoped;

    operation.validate(input_tensors);
    auto output_tensors = operation.compute_output_tensors(input_tensors);

    //std::string op_message = op_profiler::op_meta_data_serialized_json(operation, input_tensors, output_tensors);
    //std::string op_text = fmt::format("id:{}", operation.get_uniqe_id());
    //ZoneText(op_text.c_str(), op_text.size());
    //TracyMessage(op_message.c_str(), op_message.size());

    return output_tensors;
}

inline const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;

std::vector<Tensor> run_device_operation(
    std::optional<std::reference_wrapper<CommandQueue>> queue,
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    ZoneScopedN("TT_DNN_DEVICE_OP");
    uint32_t op_id = assign_id();

    std::function<std::variant<std::shared_ptr<Program>, std::reference_wrapper<Program>>(
        const DeviceOperation&,
        const std::vector<Tensor>&,
        const std::vector<std::optional<const Tensor>>&,
        std::vector<Tensor>&)>
        get_or_create_program;
    if (program_cache::is_enabled()) {
        get_or_create_program = [](const DeviceOperation& operation,
                                   const std::vector<Tensor>& input_tensors,
                                   const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                   std::vector<Tensor>& output_tensors) -> std::reference_wrapper<Program> {
            auto&& [program_with_callbacks, cache_hit] =
                program_cache::get_or_create(operation, input_tensors, optional_input_tensors, output_tensors);
            TT_ASSERT(program_with_callbacks.supports_program_cache());

            auto& program = program_with_callbacks.program;
            if (cache_hit) {
                ZoneScopedN("Cache_hit_set_runtime_args");
                if (program_with_callbacks.override_addresses_callback.has_value()) {
                    auto override_addresses_callback = program_with_callbacks.override_addresses_callback.value();
                    override_addresses(
                        override_addresses_callback, program, input_tensors, optional_input_tensors, output_tensors);
                }

                if (program_with_callbacks.override_runtime_arguments_callback.has_value()) {
                    auto override_runtime_arguments_callback =
                        program_with_callbacks.override_runtime_arguments_callback.value();
                    operation.override_runtime_arguments(
                        override_runtime_arguments_callback,
                        program,
                        input_tensors,
                        optional_input_tensors,
                        output_tensors);
                }
            }
            return program;
        };
    } else {
        get_or_create_program = [](const DeviceOperation& operation,
                                   const std::vector<Tensor>& input_tensors,
                                   const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                   std::vector<Tensor>& output_tensors) -> std::shared_ptr<Program> {
            auto program_with_callbacks =
                operation.create_program(input_tensors, optional_input_tensors, output_tensors);
            return std::make_shared<Program>(std::move(program_with_callbacks.program));
        };
    }
    operation.validate(input_tensors, optional_input_tensors, optional_output_tensors);
    auto output_tensors = operation.create_output_tensors(input_tensors, optional_output_tensors);
    auto program = get_or_create_program(operation, input_tensors, optional_input_tensors, output_tensors);

    // Enqueue or Launch Program
    std::visit(
        [&operation, &input_tensors, &optional_input_tensors, queue](auto&& program) {
            auto device = detail::get_device(input_tensors, optional_input_tensors);
            using T = std::decay_t<decltype(program)>;
            if constexpr (std::is_same_v<T, std::reference_wrapper<Program>> || std::is_same_v<T, std::shared_ptr<Program>> ) {
                if (USE_FAST_DISPATCH) {
                    TT_ASSERT(queue.has_value(), "CommandQueue is required for fast dispatch mode");
                    CommandQueue& cq = queue.value().get();
                    EnqueueProgram(cq, program, false);
                } else {
                    ::detail::LaunchProgram(device, program);
                }
            }
        },
        program);


    // Will turn the following four lines to a macro that need to compile to nothing in a non tracy build
    // Needs to be a macro because it needs to be part of the same parent run_device_operation scope
    std::string op_message = op_profiler::op_meta_data_serialized_json(op_id, operation, program, input_tensors, optional_input_tensors, output_tensors);
    std::string op_text = fmt::format("id:{}", op_id);
    ZoneText(op_text.c_str(), op_text.size());
    TracyMessage(op_message.c_str(), op_message.size());

    return output_tensors;
}

}  // namespace detail

std::vector<Tensor> run(const HostOperation& operation, const std::vector<Tensor>& input_tensors) {
    return detail::decorate_host_operation(detail::run_host_operation)(operation, input_tensors);
}

std::vector<Tensor> run(
    CommandQueue& queue,
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    return detail::decorate_device_operation(detail::run_device_operation)(
        queue, operation, input_tensors, optional_input_tensors, optional_output_tensors);
}

std::vector<Tensor> run(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    auto device = detail::get_device(input_tensors, optional_input_tensors);
    return detail::decorate_device_operation(detail::run_device_operation)(
        detail::USE_FAST_DISPATCH ? std::make_optional(std::ref(device->command_queue())) : std::nullopt, operation, input_tensors, optional_input_tensors, optional_output_tensors);
}

std::vector<Tensor> run_without_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors
) {
    Device* device = detail::get_device(input_tensors, optional_input_tensors);

    std::vector<Tensor> input_tensors_on_dev;
    input_tensors_on_dev.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() != StorageType::DEVICE) {
            input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(input_tensor, device));
        } else {
            input_tensors_on_dev.push_back(input_tensor);
        }
    }
    std::vector<std::optional<const Tensor>> optional_input_tensors_on_dev;
    optional_input_tensors_on_dev.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() != StorageType::DEVICE) {
            optional_input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(optional_input_tensor.value(), device));
        } else {
            optional_input_tensors_on_dev.push_back(optional_input_tensor);
        }
    }
    return run(operation, input_tensors_on_dev, optional_input_tensors_on_dev, {});
}

std::vector<Tensor> run_without_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors
) {
    Device* device = detail::get_device(input_tensors, optional_input_tensors);

    std::vector<Tensor> input_tensors_on_dev;
    input_tensors_on_dev.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() != StorageType::DEVICE) {
            input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(input_tensor, device));
        } else {
            input_tensors_on_dev.push_back(input_tensor);
        }
    }
    std::vector<std::optional<const Tensor>> optional_input_tensors_on_dev;
    optional_input_tensors_on_dev.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() != StorageType::DEVICE) {
            optional_input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(optional_input_tensor.value(), device));
        } else {
            optional_input_tensors_on_dev.push_back(optional_input_tensor);
        }
    }
    return run(operation, input_tensors_on_dev, optional_input_tensors_on_dev, optional_output_tensors);
}

// To be deprecated/removed in favor of new implementation where ops specifically request how to format inputs/outputss
std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const float pad_value,
    const bool pad_c
) {
    Device* device = detail::get_device(input_tensors, optional_input_tensors);

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

    TT_ASSERT(output_tensors.size() == output_shapes.size());

    formatted_input_tensors.clear();
    formatted_optional_input_tensors.clear();

    for (auto i = 0; i < output_tensors.size(); ++i) {
        output_tensors[i] = AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device, Layout::TILE);
    }
    return output_tensors;
}

std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<FormatParams>>& optional_input_formatting
) {
    Device* device = detail::get_device(input_tensors, optional_input_tensors);

    auto output_shapes = operation.compute_output_shapes(input_tensors);

    TT_ASSERT(input_tensors.size() == input_formatting.size());
    TT_ASSERT(optional_input_tensors.size() == optional_input_formatting.size());

    std::vector<Tensor> formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        formatted_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensors[i], device, input_formatting[i].pad_shape, input_formatting[i].pad_value, input_formatting[i].target_layout));
    }

    std::vector<std::optional<const Tensor>> formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (uint32_t i = 0; i < optional_input_tensors.size(); ++i) {
        if (optional_input_tensors[i].has_value()) {
            auto& input_tensor = optional_input_tensors[i].value();
            TT_ASSERT(optional_input_formatting[i].has_value());
            auto& input_formatting = optional_input_formatting[i].value();
            formatted_optional_input_tensors.push_back(AutoFormat::format_input_tensor(input_tensor, device, input_formatting.pad_shape, input_formatting.pad_value, input_formatting.target_layout));
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensors[i]);
        }
    }

    auto output_tensors = run(operation, formatted_input_tensors, formatted_optional_input_tensors);

    TT_ASSERT(output_tensors.size() == output_shapes.size());
    TT_ASSERT(output_tensors.size() == output_layouts.size());

    formatted_input_tensors.clear();
    formatted_optional_input_tensors.clear();

    for (auto i = 0; i < output_tensors.size(); ++i) {
        output_tensors[i] = AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device, output_layouts[i]);
    }

    return output_tensors;
}

}
