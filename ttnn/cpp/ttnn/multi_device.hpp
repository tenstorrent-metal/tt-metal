// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_pool.hpp"
#include "ttnn/device.hpp"
#include "ttnn/types.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace ttnn {

namespace multi_device {

using DeviceGrid = std::pair<int, int>;
using DeviceIds = std::vector<int>;

class DeviceMesh
{
public:
    DeviceGrid device_grid;
    DeviceIds device_ids;
    std::vector<Device*> managed_devices;

    DeviceMesh(const DeviceGrid& device_grid, const DeviceIds &device_ids)
        : device_grid(device_grid), device_ids(device_ids)
    {
        auto num_requested_devices = device_ids.size();
        auto num_available_devices = tt::tt_metal::GetNumPCIeDevices();

        managed_devices.resize(num_requested_devices, nullptr);
        for (int i = 0; i < num_requested_devices; i++) { // assume linear ordering
            auto device_id = device_ids[i];
            TT_ASSERT(device_id < num_available_devices);
            if (managed_devices[i] == nullptr) {
                managed_devices[i] = &ttnn::device::open_device(device_id);
            }
        }
    }
    ~DeviceMesh() = default;

    DeviceMesh(const DeviceMesh &) = delete;
    DeviceMesh &operator=(const DeviceMesh &) = delete;

    DeviceMesh(DeviceMesh &&) = delete;
    DeviceMesh &operator=(DeviceMesh &&) = delete;

    Device &get_device(int index)
    {
        for (int i = 0; i < managed_devices.size(); i++) {
            if (device_ids[i] == index) {
                return *managed_devices[i];
            }
        }
        TT_THROW("User has provided an invalid device index");
    }

    const DeviceIds &get_device_ids() const
    {
        return device_ids;
    }

    int num_devices() const
    {
        return managed_devices.size();
    }
};

std::unordered_map<int, std::shared_ptr<DeviceMesh>> id_to_multi_device;

inline DeviceMesh &open_device_mesh(const DeviceGrid& device_grid, const DeviceIds& device_ids) {
    auto multi_device = std::make_shared<DeviceMesh>(device_grid, device_ids);
    for (auto device_id : device_ids) {
        id_to_multi_device[device_id] = multi_device;
    }
    return *multi_device;
}

inline void close_device_mesh(DeviceMesh &multi_device) {
    for (int i = 0; i < multi_device.managed_devices.size(); i++) {
        id_to_multi_device.erase(multi_device.managed_devices[i]->id());
        ttnn::device::close_device(*multi_device.managed_devices[i]);
        multi_device.managed_devices[i] = nullptr;
    }
}

ttnn::Tensor to_device(const ttnn::Tensor& tensor, ttnn::multi_device::DeviceMesh& device_mesh, const ttnn::MemoryConfig& memory_config) {
    return tensor;
    /*
    if (tensor.storage_type() == StorageType::DEVICE) {
        return tensor;
    }
    const auto& storage = tensor.storage();
    if (std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(storage)) {
        auto& host_storage = std::get<tt::tt_metal::MultiDeviceStorage>(storage);
    }
    return ::tt::metal::tensor_impl::to_device_wrapper(ttl_tensor, target_device, mem_config);
    */
}

}  // namespace multi_device

}  // namespace ttnn
