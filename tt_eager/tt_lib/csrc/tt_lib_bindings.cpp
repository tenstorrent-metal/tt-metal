// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings.hpp"

#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"
#include "operations/module.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "tt_lib_bindings_tensor.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/module.hpp"
#include "type_caster.hpp"

namespace py = pybind11;

namespace tt {

namespace tt_metal {

void DeviceModule(py::module &m_device) {
    py::enum_<tt::ARCH>(m_device, "Arch", "Enum of types of Tenstorrent accelerator devices.")
        .value("GRAYSKULL", tt::ARCH::GRAYSKULL)
        .value("WORMHOLE_B0", tt::ARCH::WORMHOLE_B0);

    auto pyDevice = py::class_<Device>(m_device, "Device", "Class describing a Tenstorrent accelerator device.");
    pyDevice
        .def(
            py::init<>(
                [](int device_id) {
                    return Device(device_id, 1);
                }
            ), "Create device."
        )
        .def("id", &Device::id, "Device's ID")
        .def("arch", &Device::arch, "Device's arch")
        .def("compute_with_storage_grid_size", &Device::compute_with_storage_grid_size, "Grid size (x, y) denoting region that can be targeted by ops");

    // *** eps constant ***
    m_device.attr("EPS_GS") = EPS_GS;
    m_device.attr("EPS_WHB0") = EPS_WHB0;

    pyDevice.def("sfpu_eps", &Device::sfpu_eps, R"doc(
        Machine epsilon value for current device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | return machine epsilon | tt_lib.device.Device  |     NA      | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
        )doc");
    m_device.def("CreateDevice", [](int device_id) { return CreateDevice(device_id, 1); }, R"doc(
        Creates an instance of TT device.

        +------------------+------------------------+---------------------+------------------------------+----------+
        | Argument         | Description            | Data type           | Valid range                  | Required |
        +==================+========================+=====================+==============================+==========+
        | device_id        | Device index           | int                 |                              | Yes      |
        +------------------+------------------------+---------------------+------------------------------+----------+
    )doc");
    m_device.def("CloseDevice", &CloseDevice, R"doc(
        Reset an instance of TT accelerator device to default state and relinquish connection to device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to close     | tt_lib.device.Device  |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("SetDefaultDevice", &AutoFormat::SetDefaultDevice, R"doc(
        Sets the default device to use for ops when inputs aren't on device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to use       | tt_lib.device.Device  |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("GetDefaultDevice", &AutoFormat::GetDefaultDevice, R"doc(
        Gets the default device to use for ops when inputs aren't on device.
    )doc");

    m_device.def("EnablePersistentKernelCache", &detail::EnablePersistentKernelCache, R"doc(
        Enable kernel compilation cache to be persistent across runs. When this is called, kernels will not be compiled if the output binary path exists.
    )doc");
    m_device.def("DisablePersistentKernelCache", &detail::DisablePersistentKernelCache, R"doc(
        Disables kernel compilation cache from being persistent across runs
    )doc");
    m_device.def("EnableCompilationReports", &detail::EnableCompilationReports, R"doc(
        Enables tt-metal to generate reports of compilation statistics
    )doc");
    m_device.def("DisableCompilationReports", &detail::DisableCompilationReports, R"doc(
        Disables generation of compilation statistics reports in tt-metal
    )doc");

    m_device.def("EnableMemoryReports", &detail::EnableMemoryReports, R"doc(
        Enables tt-metal to generate reports of memory allocation statistics
    )doc");
    m_device.def("DisableMemoryReports", &detail::DisableMemoryReports, R"doc(
        Disables generation of memory allocation statistics reports in tt-metal
    )doc");

    m_device.def("DumpDeviceMemoryState", &detail::DumpDeviceMemoryState, R"doc(
        Generates reports to dump device memory state. Three reports are generated:
        - `l1_usage_summary.csv` has a table with an entry for each program indicating the minimum largest free L1 block and size of largest L1 buffer that can be interleaved across available free L1 blocks
        - `memory_usage_summary.csv` for each program there is an entry indicating total allocatable, allocated, free, and largest free block sizes for each DRAM and L1 bank
        - `detailed_memory_usage.csv` expands on the memory usage summary report by including each memory block address, size, and allocation status

        +------------------+----------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                      | Data type             | Valid range | Required |
        +==================+==================================+=======================+=============+==========+
        | device           | Device to dump memory state for  | tt_lib.device.Device  |             | Yes      |
        +------------------+----------------------------------+-----------------------+-------------+----------+
    )doc");

    m_device.def("Synchronize", &detail::Synchronize, R"doc(
        Wait for all kernels on TT device to complete.
    )doc");
    m_device.def("SetLazyCommandQueueMode", &detail::SetLazyCommandQueueMode, R"doc(
        If set to true, the host does not notify the device that there are commands available other than
        the FinishCommand. Once set to false, all subsequent commands will immediately notify the device
        that the write pointer has been updated.
    )doc");
    m_device.def("DumpDeviceProfiler", &detail::DumpDeviceProfiler, R"doc(
        Dump device side profiling data.
    )doc");
    m_device.def("DeallocateBuffers", &detail::DeallocateBuffers, R"doc(
        Deallocate all buffers associated with Device handle
    )doc");
}

void ProfilerModule(py::module &m_profiler) {
    py::enum_<op_profiler::OpType>(m_profiler, "OpType")
        .value("python_fallback", op_profiler::OpType::python_fallback)
        .value("custom_zone", op_profiler::OpType::custom_zone);

    m_profiler.def("get_profiler_flag", &op_profiler::get_profiler_flag, R"doc(
        Gets the profiling flag.
    )doc");

    m_profiler.def("set_profiler_location", &op_profiler::set_profiler_location, R"doc(
        Sets the profiling root folder.

        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | profilerLocation | Profiling output folder under the              | string                | Valid dir   | Yes      |
        |                  | parent folder generated/profiler/.logs         |                       |             |          |
        |                  | Default : generated/profiler/.logs/ops/        |                       |             |          |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def("append_meta_data", &op_profiler::append_meta_data, R"doc(
        Append extra information regardig the op.

        +------------------+------------------------+-----------------------+------------------+----------+
        | Argument         | Description            | Data type             | Valid range      | Required |
        +==================+========================+=======================+==================+==========+
        | metaData         | Meta Data              | string                | Non-empty string | Yes      |
        +------------------+------------------------+-----------------------+------------------+----------+
    )doc");

    m_profiler.def("append_input_data", &op_profiler::append_input_data, R"doc(
        Append op input information .

        +------------------+------------------------+-----------------------+------------------+----------+
        | Argument         | Description            | Data type             | Valid range      | Required |
        +==================+========================+=======================+==================+==========+
        | input            | Input tensor           | Tensor                | Valid Tensor     | Yes      |
        +------------------+------------------------+-----------------------+------------------+----------+
    )doc");

    m_profiler.def("append_output_data", &op_profiler::append_output_data, R"doc(
        Append op output information .

        +------------------+------------------------+-----------------------+------------------+----------+
        | Argument         | Description            | Data type             | Valid range      | Required |
        +==================+========================+=======================+==================+==========+
        | output           | output tensor          | Tensor                | Valid Tensor     | Yes      |
        +------------------+------------------------+-----------------------+------------------+----------+
    )doc");

    m_profiler.def("set_preferred_name", &op_profiler::set_preferred_name<string>, R"doc(
        Set a name to be appended to the name that profiler started with.

        +------------------+------------------------+-----------------------+------------------+----------+
        | Argument         | Description            | Data type             | Valid range      | Required |
        +==================+========================+=======================+==================+==========+
        | name             | Preferred Name         | String                | Valid String     | Yes      |
        +------------------+------------------------+-----------------------+------------------+----------+
    )doc");

    m_profiler.def("start_profiling",
		  &op_profiler::start_profiling,py::arg("opName"), py::arg("opType") = op_profiler::OpType::custom_zone, R"doc(
        Start profiling op.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | opName             | Name of the op or zone to be profiled          | string                |             | Yes      |
        | opType             | Fallback op or custom zone                     | string                |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def("stop_profiling", &op_profiler::stop_profiling, R"doc(
        Stop profiling op.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | name             | Name of the op or zone to stop profiling       | string                |             | Yes      |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def("start_tracy_zone",&op_profiler::start_tracy_zone,
            py::arg("source"), py::arg("functName"),py::arg("lineNum"), py::arg("color") = 0, R"doc(
        Stop profiling op with tracy.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | source           | Source file for the zone                       | string                |             | Yes      |
        | functName        | Function of the zone                           | string                |             | Yes      |
        | lineNum          | Line number of the zone marker                 | int                   |             | Yes      |
        | color            | Zone color                                     | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

    m_profiler.def("stop_tracy_zone",&op_profiler::stop_tracy_zone, py::arg("name") = "", py::arg("color") = 0, R"doc(
        Stop profiling op with tracy.
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
        | Argument         | Description                                    | Data type             | Valid range | Required |
        +==================+================================================+=======================+=============+==========+
        | name             | Replace name for the zone                          | string                |             | No       |
        | color            | Replace zone color                             | int                   |             | No       |
        +------------------+------------------------------------------------+-----------------------+-------------+----------+
    )doc");

}

void DTXModule(py::module &m_dtx) {
    auto pyDataTransformations = py::class_<DataTransformations>(m_dtx, "DataTransformations", "Class describing the data transformations.");
    m_dtx.def("evaluate", [](vector<float> data, vector<uint32_t> address_map, vector<vector<int>> output_shape){
        return evaluate(data, address_map, output_shape);
    }, R"doc(
        Evaluates data transformation on host cpu.
        +------------------+----------------------------+-----------------------+-------------+----------+
        | Argument         | Description                 | Data type            | Valid range | Required |
        +==================+=============================+======================+=============+==========+
        | data             | Input data to transform     | vector of floats     |             | Yes      |
        | address_map      | address mapping from src to dst  |  vector of uint32_t |      | Yes      |
        | output shape     | shape of the dst tensor |  vector of int |      | Yes      |
        +------------------+-----------------------------+----------------------+-------------+----------+
    )doc");
    m_dtx.def("conv_transform", [](vector<int> activation_shape,
                                        vector<int> weight_shape,
                                        vector<int> conv_params,
                                        uint32_t in0_block_h,
                                        uint32_t in0_block_w,
                                        uint32_t in1_block_w,
                                        uint32_t num_blocks_in0_h,
                                        uint32_t num_blocks_in1_w,
                                        uint32_t num_bytes_of_df,
                                        bool skip_activation_transform){
        return conv_transform(activation_shape, weight_shape, conv_params, in0_block_h, in0_block_w, in1_block_w, num_blocks_in0_h, num_blocks_in1_w, num_bytes_of_df, skip_activation_transform);
    });
}



void ProgramCacheModule(py::module &m_program_cache) {
   m_program_cache.def("enable", &tt::tt_metal::program_cache::enable);
   m_program_cache.def("disable_and_clear", &tt::tt_metal::program_cache::disable_and_clear);
   m_program_cache.def("num_entries", &tt::tt_metal::program_cache::num_entries);
}

} // end namespace tt_metal

} // end namespace tt


PYBIND11_MODULE(_C, m) {

    m.attr("__name__") = "tt_lib";
    m.doc() = "Python bindings for TT-Metal";

    py::module_ m_device = m.def_submodule("device", "Submodule defining a host or device");
    tt::tt_metal::DeviceModule(m_device);

    py::module_ m_profiler = m.def_submodule("profiler", "Submodule defining the profiler");
    tt::tt_metal::ProfilerModule(m_profiler);

    py::module_ m_tensor = m.def_submodule("tensor", "Submodule defining an tt_metal tensor");
    tt::tt_metal::TensorModule(m_tensor);

    py::module_ m_dtx = m.def_submodule("dtx", "Submodule defining data transformation engine");
    tt::tt_metal::DTXModule(m_dtx);

    py::module_ m_program_cache = m.def_submodule("program_cache", "Submodule for caching operations");
    tt::tt_metal::ProgramCacheModule(m_program_cache);

    py::module_ m_operations = m.def_submodule("operations", "Submodule for operations");
    tt::operations::py_module(m_operations);

    py::module_ m_ttnn = m.def_submodule("ttnn", "Submodule for ttnn");
    ttnn::py_module(m_ttnn);

#if defined(TRACY_ENABLE)
    py::function tracy_decorator = py::module::import("tt_eager.tt_lib_profiler_wrapper").attr("callable_decorator");

    tracy_decorator(m_device);
    tracy_decorator(m_tensor);
    tracy_decorator(m_dtx);
    tracy_decorator(m_program_cache);
    tracy_decorator(m_operations);
#endif
}
