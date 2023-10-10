/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/operation_history.hpp"
#include "tt_dnn/op_library/auto_format.hpp"

#include <tt_eager/tensor/tensor.hpp>

#include <optional>

namespace tt::tt_metal {

namespace operation {


std::vector<Tensor> generic_create_output_tensors(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const DataType output_dtype,
    const Layout output_layout,
    const MemoryConfig& output_mem_config
);
template<typename ConcreteOperation>
std::vector<Tensor> generic_create_output_tensors(
    const ConcreteOperation& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const DataType output_dtype,
    const Layout output_layout,
    const MemoryConfig& output_mem_config
) {
    if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        const auto operation = DeviceOperation(concrete_op);
        return generic_create_output_tensors(operation, input_tensors, output_dtype, output_layout, output_mem_config);
    } else {
        static_assert(detail::always_false<ConcreteOperation>, "Unsupported Operation");
    }
}



namespace run_operation_state {
namespace detail {
struct RunOperationState {

    RunOperationState() {}

    void push_composite_parent_name(const char* parent_name) {
        this->composite_parent_names.push_back(parent_name);
    }

    void pop_composite_parent_name() {
        this->composite_parent_names.pop_back();
    }

    bool is_composite_operation() const {
        return not composite_parent_names.empty();
    }

    const auto& get_composite_parent_names() const {
        return this->composite_parent_names;
    }

  private:
    std::vector<const char*> composite_parent_names{};
};

inline RunOperationState OPERATION_STATE{};

}  // namespace detail

inline void push_composite_parent_name(const char* parent_name) {
    detail::OPERATION_STATE.push_composite_parent_name(parent_name);
}

inline void pop_composite_parent_name() {
    detail::OPERATION_STATE.pop_composite_parent_name();
}

inline bool is_composite_operation() {
    return detail::OPERATION_STATE.is_composite_operation();
}

inline const auto& get_composite_parent_names() {
    return detail::OPERATION_STATE.get_composite_parent_names();
}

}  // namespace run_operation_state


namespace detail {
template<typename ReturnType, typename... Args>
struct CompositeOperation {

    const char* name;
    std::function<ReturnType(Args...)> function;

    constexpr ReturnType operator()(Args... args) const {
        run_operation_state::push_composite_parent_name(this->name);
        ReturnType output = this->function(args...);
        run_operation_state::pop_composite_parent_name();
        return output;
    }
};

}  // namespace detail

template<typename ReturnType, typename... Args>
constexpr auto decorate_as_composite(const char* name, std::function<ReturnType(Args...)>&& function) {
  return detail::CompositeOperation<ReturnType, Args...>{.name=name, .function=function};
}

template<typename FunctionType>
constexpr auto decorate_as_composite(const char* name, FunctionType function) {
  return decorate_as_composite(name, std::function(function));
}

#ifdef DEBUG
namespace detail {

template<typename OperationType>
static void print_operation(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {

    tt::log_debug(tt::LogOp, "Operation Type: {}", operation.get_type_name());

    if (run_operation_state::is_composite_operation()) {
        tt::log_debug(tt::LogOp, "Composite Parents: {}", run_operation_state::get_composite_parent_names());
    }

    tt::log_debug(tt::LogOp, "Operation Attributes:");
    for (auto&& [name, value] : operation.attributes()) {
        tt::log_debug(tt::LogOp, "\t{} = {}", name, value);
    }

    tt::log_debug(tt::LogOp, "Input Tensors:");
    for (auto index = 0; index < input_tensors.size(); index++) {
        const auto& tensor = input_tensors[index];
        tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
    }

    if (not optional_input_tensors.empty()) {
        tt::log_debug(tt::LogOp, "Optional Input Tensors:");
        for (auto index = 0; index < optional_input_tensors.size(); index++) {
            const auto& tensor = optional_input_tensors[index];
            tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
        }
    }

    tt::log_debug(tt::LogOp, "");
}

static operation_history::TensorRecord create_tensor_record(const Tensor& tensor) {
    return std::visit(
        [&] (const auto& storage) -> operation_history::TensorRecord {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.shape(), tensor.dtype(), tensor.layout(), std::nullopt
                };
            }
            else if constexpr (std::is_same_v<T, DeviceStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.shape(), tensor.dtype(), tensor.layout(), storage.memory_config
                };
            }
            else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.shape(), tensor.dtype(), tensor.layout()
                };
            }
            else {
                raise_unsupported_storage<T>();
            }
        },
        tensor.storage()
    );
}

template<typename OperationType>
static void append_operation_to_operation_history(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {

    std::vector<operation_history::TensorRecord> input_tensor_records;
    input_tensor_records.reserve(input_tensors.size() + optional_input_tensors.size());

    for (const auto& tensor : input_tensors) {
        input_tensor_records.emplace_back(create_tensor_record(tensor));
    }
    for (const auto& tensor : optional_input_tensors) {
        if (tensor.has_value()) {
            input_tensor_records.emplace_back(create_tensor_record(tensor.value()));
        }
    }
    operation_history::append(
        operation_history::OperationRecord{
            operation.get_type_name(),
            operation.attributes(),
            input_tensor_records,
            run_operation_state::get_composite_parent_names()
        }
    );
}

}  // namespace detail

template<typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}) {
    detail::print_operation(operation, input_tensors, optional_input_tensors);
    if (operation_history::enabled()) {
        detail::append_operation_to_operation_history(operation, input_tensors, optional_input_tensors);
    }
}
#else
template<typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}) {}
#endif

std::vector<Tensor> run(
    const HostOperation& operation,
    const std::vector<Tensor>& input_tensors
);
std::vector<Tensor> run(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
);
template<typename ConcreteOperation>
inline std::vector<Tensor> run(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
) {
    if constexpr (detail::is_host_operation<ConcreteOperation>()) {
        TT_ASSERT(optional_input_tensors.empty());
        const auto operation = HostOperation(concrete_op);
        return run(operation, input_tensors);
    } else if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        const auto operation = DeviceOperation(concrete_op);
        return run(operation, input_tensors, optional_input_tensors);
    } else {
        static_assert(detail::always_false<ConcreteOperation>, "Unsupported Operation");
    }
}

std::vector<Tensor> run_without_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
);
template<typename ConcreteOperation>
inline std::vector<Tensor> run_without_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {}
) {
    const auto operation = DeviceOperation(concrete_op);
    return run_without_autoformat(operation, input_tensors, optional_input_tensors);
}

std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false
);
template<typename ConcreteOperation>
inline std::vector<Tensor> run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false
) {
    const auto operation = DeviceOperation(concrete_op);
    return run_with_autoformat(operation, input_tensors, optional_input_tensors, pad_value, pad_c);
}

std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {}
);
template<typename ConcreteOperation>
inline std::vector<Tensor> run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {}
) {
    const auto operation = DeviceOperation(concrete_op);
    return run_with_autoformat(operation, input_tensors, input_formatting, output_layouts, optional_input_tensors, optional_input_formatting);
}

} //namespace operation

} //namespace tt::tt_metal
