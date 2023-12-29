// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/llrt/tt_cluster.hpp"

using namespace tt::tt_metal;

// Core manager APIs act as an interface to access the correct dispatch and storage only cores for a given device and
// architecture
// TODO: Make this work for multi cq per device!!
// TODO: add APIs for storage only cores
namespace tt::tt_metal::detail {

/// @brief Gets collection of dispatch cores based for a given device. Each architecture can have different collection
/// of dispatch cores based on number of harvested rows
/// @param device_id to get dispatch cores for
/// @return vector of logical coordinates, ordered based on soc desc yaml specification
inline const std::vector<CoreCoord> &get_logical_dispatch_cores(chip_id_t device_id) {
    // Holds dispatch cores for a given arch based on number of harvested rows
    static std::unordered_map<ARCH, std::unordered_map<uint8_t, std::vector<CoreCoord>>> dispatch_cores_by_arch;

    ARCH arch = tt::Cluster::instance().arch();
    uint8_t num_harvested_rows = tt::Cluster::instance().get_harvested_rows(device_id);
    if (dispatch_cores_by_arch.find(arch) != dispatch_cores_by_arch.end() and
        dispatch_cores_by_arch.at(arch).find(num_harvested_rows) != dispatch_cores_by_arch.at(arch).end()) {
        return dispatch_cores_by_arch.at(arch).at(num_harvested_rows);
    }

    std::unordered_map<uint8_t, std::vector<CoreCoord>> &dispatch_cores_by_num_harvested_rows =
        dispatch_cores_by_arch[arch];
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(device_id);

    for (const RelativeCoreCoord &core : soc_desc.dispatch_cores) {
        const CoreCoord logical_coord = get_core_coord_from_relative(core, soc_desc.worker_grid_size);
        dispatch_cores_by_num_harvested_rows[num_harvested_rows].push_back(logical_coord);
    }

    return dispatch_cores_by_arch.at(arch).at(num_harvested_rows);
}

/// @brief The command fetcher core pops commands from the issue queue and relays them to the dispatch core
///     When running fast dispatch on remote devices, this core is on the associated MMIO device and pulls commands from
///     the remote device's issue queue
/// @param device_id device ID that carries out the fast dispatch commands
/// @return tt_cxy_pair chip and logical command fetcher core coordinate
inline tt_cxy_pair get_logical_command_fetcher_core(chip_id_t device_id) {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    const std::vector<CoreCoord> &all_dispatch_cores = get_logical_dispatch_cores(mmio_device_id);
    if (mmio_device_id != device_id) {  // running on remote device
        TT_ASSERT(all_dispatch_cores.size() > 2);
        return tt_cxy_pair(mmio_device_id, all_dispatch_cores.at(2));
    }
    return tt_cxy_pair(device_id, all_dispatch_cores.at(0));
}

/// @brief The command dispatcher core receives commands and relays them to worker cores. This core is on the same
/// device that carried out commands
/// @param device_id device ID that does the work specified by the fast dispatch commands
/// @return tt_cxy_pair chip and logical command dispatcher core coordinate
inline tt_cxy_pair get_logical_command_dispatcher_core(chip_id_t device_id) {
    const std::vector<CoreCoord> &all_dispatch_cores = get_logical_dispatch_cores(device_id);
    return tt_cxy_pair(device_id, all_dispatch_cores.at(1));
}

/// @brief The completion queue interface core writes to the completion queue.
///     This core is the same as the command dispatcher when running fast dispatch on a MMIO device, otherwise for
///     remote device dispatch this core is on the associated MMIO device and interfaces with the remote device's
///     completion queue
/// @param device_id device ID that does the work specified by the fast dispatch commands
/// @return tt_cxy_pair chip and logical command dispatcher core coordinate
inline tt_cxy_pair get_logical_completion_queue_interface_core(chip_id_t device_id) {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    if (mmio_device_id == device_id) {
        get_logical_command_dispatcher_core(device_id);
    }
    // running on remote device
    const std::vector<CoreCoord> &all_dispatch_cores = get_logical_dispatch_cores(mmio_device_id);
    TT_ASSERT(all_dispatch_cores.size() > 3);
    return tt_cxy_pair(mmio_device_id, all_dispatch_cores.at(3));
}

/// @brief Returns the physical core coordinate of given dispatch core
/// @param logical_core_coordinate tt_cxy_pair describing logical location of a dispatch core. tt_cxy_pair is needed
/// because logical to physical conversion needs to account for the device's harvesting config
/// @return tt_cxy_pair describing physical location of the specified dispatch core
inline tt_cxy_pair get_physical_dispatch_core(const tt_cxy_pair &logical_core_coordinate) {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(logical_core_coordinate.chip);
    return tt_cxy_pair(
        logical_core_coordinate.chip,
        static_cast<size_t>(soc_desc.worker_log_to_physical_routing_x.at(logical_core_coordinate.x)),
        static_cast<size_t>(soc_desc.worker_log_to_physical_routing_y.at(logical_core_coordinate.y)));
}

};  // namespace tt::tt_metal::detail
