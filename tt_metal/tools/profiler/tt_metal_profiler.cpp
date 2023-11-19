// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "llrt/tt_debug_print_server.hpp"

#include "tools/profiler/profiler.hpp"

#include "tt_metal/detail/tt_metal.hpp"

namespace tt {

namespace tt_metal {

void DumpDeviceProfileResults(const Device& device, const Program &program) {
    detail::DumpDeviceProfileResults(device, program.logical_cores());
}


namespace detail {

static Profiler tt_metal_profiler = Profiler();

void DumpDeviceProfileResults(const Device& device, const vector<CoreCoord> &logical_cores) {
#if defined(PROFILER)
    ZoneScoped;
    if (getDeviceProfilerState())
    {
        ProfileTTMetalScope profile_this = ProfileTTMetalScope("DumpDeviceProfileResults");
        //TODO: (MO) This global is temporary need to update once the new interface is in
        if (GLOBAL_CQ) {
            Finish(*GLOBAL_CQ);
        }
        TT_FATAL(tt_is_print_server_running() == false, "Debug print server is running, cannot dump device profiler data");
        auto worker_cores_used_in_program =\
            device.worker_cores_from_logical_cores(logical_cores);
        auto device_id = device.id();
        tt_metal_profiler.setDeviceArchitecture(device.arch());
        tt_metal_profiler.dumpDeviceResults(device_id, worker_cores_used_in_program);
    }
#endif
}


void SetProfilerDir(std::string output_dir){
#if defined(PROFILER)
     tt_metal_profiler.setOutputDir(output_dir);
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
