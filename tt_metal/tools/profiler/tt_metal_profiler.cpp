// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "impl/debug/dprint_server.hpp"

#include "tools/profiler/profiler.hpp"
#include "hostdevcommon/profiler_common.h"

#include "tt_metal/detail/tt_metal.hpp"


namespace tt {

namespace tt_metal {

void DumpDeviceProfileResults(Device *device, const Program &program) {
    const auto &all_logical_cores = program.logical_cores();
    if (all_logical_cores.find(CoreType::WORKER) != all_logical_cores.end()) {
        detail::DumpDeviceProfileResults(device, program.logical_cores().at(CoreType::WORKER));
    }
    // TODO: add support for ethernet core device dumps
}


namespace detail {

Profiler tt_metal_profiler;

void InitDeviceProfiler(Device *device){
#if defined(PROFILER)
    ZoneScoped;

    CoreCoord compute_with_storage_size = device->logical_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};

    //std::vector<uint32_t> zero_buffer(PROFILER_RISC_COUNT * PROFILER_L1_VECTOR_SIZE + PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
    //{
        //ZoneScopedN("Clearing_profiler_L1");
        //for (size_t x=start_core.x; x <= end_core.x; x++)
        //{
            //for (size_t y=start_core.y; y <= end_core.y; y++)
            //{
                //CoreCoord curr_core = {x, y};
                //tt_metal::detail::WriteToDeviceL1(device, curr_core, PROFILER_L1_BUFFER_BR, zero_buffer);
            //}
        //}
    //}

    vector<uint32_t> huge_zero_buffer(PROFILER_HOST_BUFFER_SIZE / sizeof(uint32_t), 0);
    tt::Cluster::instance().write_sysmem_vec(huge_zero_buffer, PROFILER_HUGE_PAGE_ADDRESS, 0);

#endif
}

void DumpDeviceProfileResults(Device *device) {
#if defined(PROFILER)
    CoreCoord compute_with_storage_size = device->logical_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};

    std::vector<CoreCoord> workerCores;
    for (size_t x=start_core.x; x <= end_core.x; x++)
    {
        for (size_t y=start_core.y; y <= end_core.y; y++)
        {
            CoreCoord logical_core = {x, y};
            workerCores.push_back(device->worker_core_from_logical_core(logical_core));
        }
        TT_FATAL(tt_is_print_server_running() == false, "Debug print server is running, cannot dump device profiler data");
        auto worker_cores_used_in_program =\
            device->worker_cores_from_logical_cores(logical_cores);
    }
    DumpDeviceProfileResults(device, workerCores);
#endif
}

void DumpDeviceProfileResults(Device *device, const Program &program)
{
#if defined(PROFILER)
    auto worker_cores_used_in_program =\
                                       device->worker_cores_from_logical_cores(program.logical_cores());
    DumpDeviceProfileResults(device, worker_cores_used_in_program);
#endif
}


void DumpDeviceProfileResults(Device *device, const vector<CoreCoord>& worker_cores) {
#if defined(PROFILER)
    ZoneScoped;
    if (getDeviceProfilerState())
    {
        auto device_id = device->id();
        tt_metal_profiler.setDeviceArchitecture(device->arch());
        tt_metal_profiler.dumpDeviceResults(device_id, worker_cores);
        tt_metal_profiler.pushTracyDeviceResults(device_id);
        tt_metal_profiler.device_data.clear();
    }
#endif
}

void SetDeviceProfilerDir(std::string output_dir){
#if defined(PROFILER)
     tt_metal_profiler.setDeviceOutputDir(output_dir);
#endif
}

void SetHostProfilerDir(std::string output_dir){
#if defined(PROFILER)
     tt_metal_profiler.setHostOutputDir(output_dir);
#endif
}

void FreshProfilerHostLog(){
#if defined(PROFILER)
     tt_metal_profiler.setHostNewLogFlag(true);
#endif
}

void FreshProfilerDeviceLog(){
#if defined(PROFILER)
     tt_metal_profiler.setDeviceNewLogFlag(true);
#endif
}

ProfileTTMetalScope::ProfileTTMetalScope (const string& scopeNameArg) : scopeName(scopeNameArg){
#if defined(PROFILER)
    tt_metal_profiler.markStart(scopeName);
#endif
}

ProfileTTMetalScope::~ProfileTTMetalScope ()
{
#if defined(PROFILER)
    tt_metal_profiler.markStop(scopeName);
#endif
}

}  // namespace detail

}  // namespace tt_metal

}  // namespace tt
