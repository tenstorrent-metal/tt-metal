// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>
#include <filesystem>

#include "tt_metal/impl/buffers/buffer.hpp"
#include "llrt/llrt.hpp"
#include "tools/profiler/profiler_state.hpp"
#include "tools/profiler/common.hpp"
#include "tt_metal/third_party/tracy/public/tracy/TracyOpenCL.hpp"

using std::chrono::steady_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

namespace tt {

namespace tt_metal {

// struct for holding start, stop and duration of a timer period in integer format
struct TimerPeriodInt {
    uint64_t start;
    uint64_t stop;
    uint64_t delta;
};

// struct for holding start, stop of a timer in steady_clock::time_point format
struct TimerPeriod {
    steady_clock::time_point start;
    steady_clock::time_point stop;
};

//Struct for holding marker data
struct markerInfo{
    uint32_t coreX;
    uint32_t coreY;
    uint32_t risc;
    uint32_t op;
};

class Profiler {
    private:

        // Holds name to timers
        std::unordered_map <std::string, TimerPeriod> name_to_timer_map;

        // Recreate host side log file with header
        bool host_new_log;

        // Recreate device side log file with header
        bool device_new_log;

        // Device architecture
        tt::ARCH device_architecture;

        // Device frequency
        int device_core_frequency;

        // Output Dir for Profile Logs
        std::filesystem::path host_output_dir;

        // Output Dir for device Profile Logs
        std::filesystem::path device_output_dir;

        // Turn steady clock start and stop into integer start, stop and duration
        TimerPeriodInt timerToTimerInt(TimerPeriod period);

        //Traverse all timers and dump the results, appending addtional fields
        void dumpHostResults(
                const std::string& timer_name,
                const std::vector<std::pair<std::string,std::string>>& additional_fields = {});

        // Dumping profile result to file
        void dumpDeviceResultToFile(
                uint16_t kernelID,
                int chip_id,
                int core_x,
                int core_y,
                int risc_num,
                uint64_t timestamp,
                uint32_t timer_id);

        // Helper function for reading risc profile results
        void readRiscProfilerResults(
                int device_id,
                vector<std::uint32_t> profile_buffer,
                const CoreCoord &worker_core);

    public:
        //Constructor
        Profiler();
        ~Profiler();

        // Map for storing dvice data
        std::map<uint64_t,std::list<uint64_t>> device_data;

        //TracyContext
        TracyCLCtx tracyTTCtx;

        //DRAM buffer for device side results
        Buffer output_dram_buffer;

        //Mark the steady_clock for the start of the asked name
        void markStart(const std::string& timer_name);

        //Mark the steady_clock time for the end of the asked name
        void markStop(const std::string& timer_name, const std::vector<std::pair<std::string,std::string>>& additional_fields = {});

        //Set the host side file flag
        void setHostNewLogFlag(bool new_log_flag);

        //Set the device side file flag
        void setDeviceNewLogFlag(bool new_log_flag);

        //Set the device architecture
        void setDeviceArchitecture(tt::ARCH device_arch);

        //Change the output dir of device profile logs
        void setDeviceOutputDir(const std::string& new_output_dir);

        //Change the output dir of host profile logs
        void setHostOutputDir(const std::string& new_output_dir);

        //Traverse all cores on the device and dump the device profile results
        void dumpDeviceResults(Device *device, const vector<CoreCoord> &worker_cores);

        //Push device results to tracy
        void pushTracyDeviceResults(int device_id);
};

}  // namespace tt_metal

}  // namespace tt
