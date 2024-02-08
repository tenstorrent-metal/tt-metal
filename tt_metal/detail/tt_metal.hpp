// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <mutex>
#include <variant>

#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_metal/jit_build/build.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/llrt/watcher.hpp"
#include "tt_metal/jit_build/genfiles.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

using std::unique_lock;
using std::mutex;

namespace tt::tt_metal{

    namespace detail {

        inline static bool DispatchStateCheck( bool isFastDispatch){
            static bool fd = isFastDispatch;
            TT_FATAL( fd == isFastDispatch, "Mixing fast and slow dispatch is prohibited!" );
            return fd;
        }

        std::map<chip_id_t, Device *> CreateDevices(
            std::vector<chip_id_t> device_ids,
            const uint8_t num_hw_cqs = 1,
            const std::vector<uint32_t> &l1_bank_remap = {});

        void CloseDevices(std::map<chip_id_t, Device *> devices);

        /**
        * Copies data from a host buffer into the specified buffer
        *
        * Return value: void
        *
        * | Argument    | Description                                     | Data type               | Valid range                                      | Required |
        * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
        * | buffer      | Buffer to send data to                          | const Buffer &          |                                                  | Yes      |
        * | host_buffer | Buffer on host to copy data from                | std::vector<uint32_t> & | Host buffer size must match buffer               | Yes      |
        */
        void WriteToBuffer(const Buffer &buffer, const std::vector<uint32_t> &host_buffer);
        void WriteToBuffer( std::shared_ptr<const Buffer> buffer, const std::vector<uint32_t> &host_buffer);
        /**
        * Copies data from a buffer into a host buffer
        *
        * Return value: void
        *
        * | Argument    | Description                                     | Data type               | Valid range                                      | Required |
        * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
        * | buffer      | Buffer to read data from                        | const Buffer &          |                                                  | Yes      |
        * | host_buffer | Buffer on host to copy data into                | std::vector<uint32_t> & |                                                  | Yes      |
        * | shard_order | For a sharded buffer we can read in shard order | bool                    |                                                  | No       |
        */
        void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order = false);
        void ReadFromBuffer(std::shared_ptr<const Buffer> buffer, std::vector<uint32_t> &host_buffer, bool shard_order = false);

        /**
        * Copies data from a buffer into a host buffer
        *
        * Return value: void
        *
        * | Argument    | Description                                     | Data type               | Valid range                                      | Required |
        * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
        * | buffer      | Buffer to read data from                        | const Buffer &          |                                                  | Yes      |
        * | host_buffer | Buffer on host to copy data into                | std::vector<uint32_t> & |                                                  | Yes      |
        * | core_id     | ID of core                                      | const uint32_t &        |                                                  | Yes      |
        */
        void ReadShard(const Buffer &buffer, std::vector<uint32_t> &host_buffer, const uint32_t & core_id);



        // Launches all kernels on cores specified with kernels in the program.
        // All kernels on a given Tensix core must be launched.
        void LaunchProgram(Device *device, Program &program);

        /**
         *  Compiles all kernels within the program, and generates binaries that are written to `$TT_METAL_HOME/built/<device>/kernels/<kernel name>/<kernel hash>`
         *
         *  To speed up compilation there is a kernel compilation cache that skips over generating binaries for the previously compiled kernels.
         *  Kernel uniqueness is determined by the kernel hash which is computed based on compile time args, defines, and kernel type specific attributes such as NOC for data movement kernels and math fidelity for compute kernels
         *  TODO: Kernel hash needs to account for device architecture as binaries are not the same across architectures.
         *  On cache hits the kernel is not recompiled if the output binary directory exists, otherwise the kernel is compiled.
         *  This cache is static is enabled for the duration of the running process.
         *  By default the cache does not persistent across runs, but can be enabled by calling EnablePersistentKernelCache(). Setting this will skip compilation when output binary directory exists.
         *
         *  Return value: void
         *
         * | Argument       | Description                                                      | Type      | Valid Range                                        | Required |
         * |----------------|------------------------------------------------------------------|-----------|----------------------------------------------------|----------|
         * | device         | Which device the program is compiled for                         | Device *  | Must be initialized via tt_metal::InitializeDevice | Yes      |
         * | program        | The program to compile                                           | Program & |                                                    | Yes      |
         */
        void CompileProgram(Device *device, Program &program);

        /**
         * Writes runtime args that are saved in the program to device
         *
         * Return value: void
         *
         * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
         * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
         * | device       | The device to whcih runtime args will be written                       | Device *                      |                                    | Yes      |
         * | program      | The program holding the runtime args                                   | const Program &               |                                    | Yes      |
         */
        void WriteRuntimeArgsToDevice(Device *device, const Program &program);

        // Configures a given device with a given program.
        // - Loads all kernel binaries into L1s of assigned Tensix cores
        // - Configures circular buffers (inits regs with buffer data)
        // - Takes the device out of reset
        bool ConfigureDeviceWithProgram(Device *device, Program &program);


        /**
         * Initialize device profiling data buffers
         *
         * Return value: void
         *
         * | Argument      | Description                                       | Type            | Valid Range               | Required |
         * |---------------|---------------------------------------------------|-----------------|---------------------------|----------|
         * | device        | The device holding the program being profiled.    | Device *        |                           | True     |
         * */
	void InitDeviceProfiler(Device *device);

        /**
         * Read device side profiler data and dump results into device side CSV log
         *
         * Return value: void
         *
         * | Argument      | Description                                       | Type                                                         | Valid Range               | Required |
         * |---------------|---------------------------------------------------|--------------------------------------------------------------|---------------------------|----------|
         * | device        | The device holding the program being profiled.    | Device *                                                     |                           | True     |
         * | core_coords   | The logical core coordinates being profiled.      | const std::unordered_map<CoreType, std::vector<CoreCoord>> & |                           | True     |
         * */
        void DumpDeviceProfileResults(Device *device, std::vector<CoreCoord> &worker_cores);

        /**
         * Traverse all cores and read device side profiler data and dump results into device side CSV log
         *
         * Return value: void
         *
         * | Argument      | Description                                       | Type                                                         | Valid Range               | Required |
         * |---------------|---------------------------------------------------|--------------------------------------------------------------|---------------------------|----------|
         * | device        | The device holding the program being profiled.    | Device *                                                     |                           | True     |
         * */
        void DumpDeviceProfileResults(Device *device);

        /**
         * Set the directory for device-side CSV logs produced by the profiler instance in the tt-metal module
         *
         * Return value: void
         *
         * | Argument     | Description                                             |  Data type  | Valid range              | required |
         * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
         * | output_dir   | The output directory that will hold the output CSV logs  | std::string | Any valid directory path | No       |
         * */
        void SetDeviceProfilerDir(std::string output_dir = "");

        /**
         * Set the directory for all host-side CSV logs produced by the profiler instance in the tt-metal module
         *
         * Return value: void
         *
         * | Argument     | Description                                             |  Data type  | Valid range              | required |
         * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
         * | output_dir   | The output directory that will hold the output CSV logs  | std::string | Any valid directory path | No       |
         * */
        void SetHostProfilerDir(std::string output_dir = "");

        /**
         * Start a fresh log for the host side profile results
         *
         * Return value: void
         *
         * | Argument     | Description                                             |  Data type  | Valid range              | required |
         * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
         * */
        void FreshProfilerHostLog();

        /**
         * Start a fresh log for the device side profile results
         *
         * Return value: void
         *
         * | Argument     | Description                                             |  Data type  | Valid range              | required |
         * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
         * */
        void FreshProfilerDeviceLog();

        /**
         * Profile scopes in tt_metal API
         *
         * */

        class ProfileTTMetalScope
        {
            private:
                string scopeName = "";
            public:
                ProfileTTMetalScope (const string& scopeNameArg);
                ~ProfileTTMetalScope ();
        };

        /**
         * Copies data from a host buffer into a buffer within the device DRAM channel
         *
         * Return value: bool
         *
         * | Argument     | Description                                            | Data type             | Valid range                               | required |
         * |--------------|--------------------------------------------------------|-----------------------|-------------------------------------------|----------|
         * | device       | The device whose DRAM to write data into               | Device *              |                                           | Yes      |
         * | dram_channel | Channel index of DRAM to write into                    | int                   | On Grayskull, [0, 7] inclusive            | Yes      |
         * | address      | Starting address on DRAM channel to begin writing data | uint32_t              | [DRAM_UNRESERVED_BASE, dram_size)         | Yes      |
         * | host_buffer  | Buffer on host to copy data from                       | std::vector<uint32_t> | Host buffer must be fully fit DRAM buffer | Yes      |
         */
        inline bool WriteToDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, std::vector<uint32_t> &host_buffer)
        {
            bool pass = true;
            TT_FATAL(address >= DRAM_UNRESERVED_BASE, "Cannot write to reserved DRAM region, addresses [0, {}) are reserved!", DRAM_UNRESERVED_BASE);
            tt::Cluster::instance().write_dram_vec(host_buffer, tt_target_dram{device->id(), dram_channel, 0}, address);
            return pass;
        }

        /**
         * Copy data from a device DRAM channel to a host buffer
         *
         * Return value: bool
         *
         * | Argument     | Description                                                  | Data type             | Valid range                    | required |
         * |--------------|--------------------------------------------------------------|-----------------------|--------------------------------|----------|
         * | device       | The device whose DRAM to read data from                      | Device *              |                                | Yes      |
         * | dram_channel | Channel index of DRAM to read from                           | int                   | On Grayskull, [0, 7] inclusive | Yes      |
         * | address      | Starting address on DRAM channel from which to begin reading | uint32_t              |                                | Yes      |
         * | size         | Size of buffer to read from device in bytes                  | uint32_t              |                                | Yes      |
         * | host_buffer  | Buffer on host to copy data into                             | std::vector<uint32_t> |                                | Yes      |
         */
        inline bool ReadFromDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer)
        {
            bool pass = true;
            tt::Cluster::instance().dram_barrier(device->id());
            tt::Cluster::instance().read_dram_vec(host_buffer, size, tt_target_dram{device->id(), dram_channel, 0}, address);
            return pass;
        }

        /**
         * Copy data from a host buffer into an L1 buffer. (Note: Current Can not be a CircularBuffer.)
         *
         * Return value: bool
         *
         * | Argument      | Description                                     | Data type             | Valid range                                         | required |
         * |---------------|-------------------------------------------------|-----------------------|-----------------------------------------------------|----------|
         * | device        | The device whose DRAM to write data into        | Device *              |                                                     | Yes      |
         * | logical_core  | Logical coordinate of core whose L1 to write to | CoreCoord            | On Grayskull, any valid logical worker coordinate   | Yes      |
         * | address       | Starting address in L1 to write into            | uint32_t              | Any non-reserved address in L1 that fits for buffer | Yes      |
         * | host_buffer   | Buffer on host whose data to copy from          | std::vector<uint32_t> | Buffer must fit into L1                             | Yes      |
         */
        inline bool WriteToDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, std::vector<uint32_t> &host_buffer)
        {
            ZoneScoped;
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            llrt::write_hex_vec_to_core(device->id(), worker_core, host_buffer, address);
            return true;
        }

        inline bool WriteRegToDevice(Device *device, const CoreCoord &logical_core, uint32_t address, const uint32_t &regval)
        {
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            tt::Cluster::instance().write_reg(&regval, tt_cxy_pair(device->id(), worker_core), address);
            return true;
        }


        /**
         * Copy data from an L1 buffer into a host buffer. Must be a buffer, and not a CB.
         *
         * Return value: bool
         *
         * | Argument             | Description                                 | Data type             | Valid range                                       | required |
         * |----------------------|---------------------------------------------|-----------------------|---------------------------------------------------|----------|
         * | device               | The device whose DRAM to read data from     | Device *              |                                                   | Yes      |
         * | logical_core         | Logical coordinate of core whose L1 to read | CoreCoord            | On Grayskull, any valid logical worker coordinate | Yes      |
         * | address              | Starting address in L1 to read from         | uint32_t              |                                                   | Yes      |
         * | size                 | Size of L1 buffer in bytes                  | uint32_t              |                                                   | Yes      |
         * | host_buffer          | Buffer on host to copy data into            | std::vector<uint32_t> | Buffer must fit L1 buffer                         | Yes      |
         */
        inline bool ReadFromDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer)
        {
            tt::Cluster::instance().l1_barrier(device->id());
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            host_buffer = llrt::read_hex_vec_from_core(device->id(), worker_core, address, size);
            return true;
        }

        inline bool ReadRegFromDevice(Device *device, const CoreCoord &logical_core, uint32_t address, uint32_t &regval)
        {
            tt::Cluster::instance().l1_barrier(device->id());
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            tt::Cluster::instance().read_reg(&regval, tt_cxy_pair(device->id(), worker_core), address);
            return true;
        }

        inline CommandQueue &GetCommandQueue(Device *device)
        {
            detail::DispatchStateCheck(true);
            // For now there is only one SW CommandQueue per device
            static std::vector<std::unique_ptr<CommandQueue>> command_queues( GetNumAvailableDevices() );
            chip_id_t id = device->id();
            TT_FATAL(id < command_queues.size(), "Invalid device {} detected", id);
            TT_FATAL(device->is_initialized(), "Cannot access command queue for closed device {}", id);
            static std::mutex cq_creation_mutex;
            {
                std::lock_guard<std::mutex> lock(cq_creation_mutex);
                command_queues[device->id()] = std::make_unique<CommandQueue>(device, 0);
            }
            return *(command_queues[id]);
        }

        inline void Synchronize(Device *device)
        {
            if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
                Finish(GetCommandQueue(device));
            }
        }

        inline void SetLazyCommandQueueMode(bool lazy)
        {
            DispatchStateCheck(true);
            LAZY_COMMAND_QUEUE_MODE = lazy;
        }
        inline void DumpDeviceProfiler(Device * device)
        {
            tt::tt_metal::detail::DumpDeviceProfileResults(device);
        }

        inline void DeallocateBuffers(Device * device)
        {
            device->deallocate_buffers();
        }

        inline void GenerateDeviceHeaders(Device *device,
                                          const std::string &path)
        {

            // Basic Allocator generates number of banks which may not be power of 2, so we could just pad and alias for now
            const size_t num_dram_banks = device->num_banks(BufferType::DRAM);
            const size_t num_dram_banks_pow2 = std::pow(2, std::ceil(std::log2(num_dram_banks)));
            std::vector<CoreCoord> dram_noc_coord_per_bank(num_dram_banks);
            std::vector<int32_t> dram_offsets_per_bank(num_dram_banks);
            for (unsigned bank_id = 0; bank_id < num_dram_banks; bank_id++) {
                dram_noc_coord_per_bank[bank_id] = device->core_from_dram_channel(device->dram_channel_from_bank_id(bank_id));
                dram_offsets_per_bank[bank_id] = device->dram_bank_offset_from_bank_id(bank_id);
            }
            const size_t num_l1_banks = device->num_banks(BufferType::L1); // 128
            const size_t num_l1_banks_pow2 = std::pow(2, std::ceil(std::log2(num_l1_banks)));
            std::vector<CoreCoord> l1_noc_coord_per_bank(num_l1_banks);
            std::vector<int32_t> l1_offset_per_bank(num_l1_banks);
            for (unsigned bank_id = 0; bank_id < num_l1_banks; bank_id++) {
                l1_noc_coord_per_bank[bank_id] = device->worker_core_from_logical_core(device->logical_core_from_bank_id(bank_id));
                l1_offset_per_bank[bank_id] = device->l1_bank_offset_from_bank_id(bank_id);
            }

            const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device->id());

            // Generate header file in proper location
            jit_build_genfiles_bank_to_noc_coord_descriptor (
                path,
                soc_d.grid_size,
                dram_noc_coord_per_bank,
                dram_offsets_per_bank,
                l1_noc_coord_per_bank,
                l1_offset_per_bank,
                soc_d.profiler_ceiled_core_count_perf_dram_bank,
                soc_d.physical_routing_to_profiler_flat_id
            );

            // Determine which noc-coords are harvested
            // TODO(PGK/Almeet): fix this w/ new UMD
            vector<uint32_t> harvested_rows;
            uint32_t harvested_noc_rows = tt::Cluster::instance().get_harvested_rows(device->id());
            for (uint32_t y = 0; y < soc_d.grid_size.y; y++) {
                bool row_harvested = (harvested_noc_rows >> y) & 0x1;
                if (row_harvested) {
                    harvested_rows.push_back(y);
                }
            }

            // Create valid PCIe address ranges
            // This implementation assumes contiguous ranges and aggregates the ranges into one bounds check
            // TODO: consider checking multiple ranges to detect straddling transactions
            uint64_t pcie_chan_base_addr = tt::Cluster::instance().get_pcie_base_addr_from_device(device->id());
            uint64_t pcie_chan_end_addr = pcie_chan_base_addr;
            for (int pcie_chan = 0; pcie_chan < tt::Cluster::instance().get_num_host_channels(device->id()); pcie_chan++) {
                pcie_chan_end_addr += tt::Cluster::instance().get_host_channel_size(device->id(), pcie_chan);
            }

            uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
            const uint8_t cq_id = 0; // Currently, only the first command queue is responsible for enqueuing programs
            tt_cxy_pair enqueue_program_dispatch_core = dispatch_core_manager::get(device->num_hw_cqs()).command_dispatcher_core(device->id(), channel, cq_id);
            CoreCoord physical_enqueue_program_dispatch_core = get_physical_core_coordinate(enqueue_program_dispatch_core, CoreType::WORKER);

            jit_build_genfiles_noc_addr_ranges_header(
                path,
                pcie_chan_base_addr,
                pcie_chan_end_addr - pcie_chan_base_addr,
                0,
                soc_d.dram_core_size,
                soc_d.get_pcie_cores(),
                soc_d.get_dram_cores(),
                soc_d.get_physical_ethernet_cores(),
                soc_d.grid_size,
                harvested_rows,
                physical_enqueue_program_dispatch_core);
        }

        inline void CheckDataMovementConfig(Program &program, const std::string &file_name, const CoreRangeSet &core_ranges) {
            bool riscv0_in_use = false; bool riscv1_in_use = false;
            bool noc0_in_use = false; bool noc1_in_use = false;

            auto set_global_and_local_noc_usage = [&](KernelHandle kernel_id, bool &local_noc0_usage, bool &local_noc1_usage) {
                const auto kernel = detail::GetKernel(program, kernel_id);
                auto kernel_config = std::get<DataMovementConfig>(kernel->config());
                auto noc_value = magic_enum::enum_integer(kernel_config.noc);
                noc0_in_use, local_noc0_usage = noc_value == 0;
                noc1_in_use, local_noc1_usage = noc_value == 1;
            };

            for (const auto &core_range : core_ranges.ranges()) {
                for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                    for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                        const KernelGroup * kernel_group = program.kernels_on_core(CoreCoord(x, y));
                        if (kernel_group != nullptr) {
                            bool local_noc0_in_use = false; bool local_noc1_in_use = false;
                            if (kernel_group->riscv0_id.has_value()) {
                                riscv0_in_use = true;
                                set_global_and_local_noc_usage(kernel_group->riscv0_id.value(), local_noc0_in_use, local_noc1_in_use);
                            }
                            if (kernel_group->riscv1_id.has_value()) {
                                riscv1_in_use = true;
                                set_global_and_local_noc_usage(kernel_group->riscv1_id.value(), local_noc0_in_use, local_noc1_in_use);
                            }
                            if (kernel_group->riscv0_id.has_value() and kernel_group->riscv1_id.has_value()) {
                                TT_FATAL(local_noc0_in_use and local_noc1_in_use, "Illegal NOC usage: data movement kernels on logical core {} cannot use the same NOC, doing so results in hangs!", CoreCoord(x, y).str());
                            }
                        }
                    }
                }
            }

            TT_FATAL(not (riscv0_in_use and riscv1_in_use), "DataMovementKernel creation failure: Cannot create data movement kernel for {} across specified cores because both data movement processors are in use!", file_name);
            TT_FATAL(not (noc0_in_use and noc1_in_use), "DataMovementKernel creation failure: Cannot create data movement kernels for {} across specified cores because both NOCs are in use!", file_name);
        }

        inline CoreRangeSet GetCoreRangeSet(const std::variant<CoreCoord, CoreRange, CoreRangeSet> &specified_core_spec) {
            ZoneScoped;
            return std::visit(
                [](auto&& core_spec) -> CoreRangeSet
                {
                    using T = std::decay_t<decltype(core_spec)>;
                    if constexpr (std::is_same_v<T, CoreCoord>) {
                        return CoreRangeSet({CoreRange{.start=core_spec, .end=core_spec}});
                    }
                    else if constexpr (std::is_same_v<T, CoreRange>) {
                        return CoreRangeSet({core_spec});
                    }
                    else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                        return core_spec;
                    }
                },
                specified_core_spec
            );
        }

        inline void CommandQueueInit(Device* device) {

            chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
            uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());

            std::vector<uint32_t> pointers(CQ_START / sizeof(uint32_t), 0);
            const uint32_t hugepage_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel);
            const uint32_t cq_size = hugepage_size / device->num_hw_cqs();

            TT_ASSERT(device->command_queue_programs.size() == 1);
            Program& command_queue_program = *device->command_queue_programs[0];

            for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); cq_id++) {
                // Reset the host manager's pointer for this command queue
                device->manager->reset(cq_id);

                pointers[HOST_CQ_ISSUE_READ_PTR / sizeof(uint32_t)] = (CQ_START + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[HOST_CQ_COMPLETION_WRITE_PTR / sizeof(uint32_t)] = (CQ_START + device->manager->get_issue_queue_size(cq_id) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;

                tt::Cluster::instance().write_sysmem(pointers.data(), pointers.size() * sizeof(uint32_t), cq_id * cq_size, mmio_device_id, channel);
            }

            if (device->is_mmio_capable()) {
                for (const chip_id_t &device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(device->id())) {
                    uint8_t curr_num_hw_cqs = device_id == device->id() ? device->num_hw_cqs() : 1;
                    uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
                    uint32_t curr_cq_size = tt::Cluster::instance().get_host_channel_size(device->id(), curr_channel) / curr_num_hw_cqs;

                    for (uint8_t cq_id = 0; cq_id < curr_num_hw_cqs; cq_id++) {
                        tt_cxy_pair issue_q_reader_location = dispatch_core_manager::get(curr_num_hw_cqs).issue_queue_reader_core(device_id, curr_channel, cq_id);
                        tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(curr_num_hw_cqs).completion_queue_writer_core(device_id, curr_channel, cq_id);

                        TT_ASSERT(issue_q_reader_location.chip == device->id() and completion_q_writer_location.chip == device->id(),
                            "Issue queue interface is on device {} and completion queue interface is on device {} but they are expected to be on device {}", issue_q_reader_location.chip, completion_q_writer_location.chip, device->id());

                        CoreCoord issue_q_logical_core(issue_q_reader_location.x, issue_q_reader_location.y);
                        CoreCoord completion_q_logical_core(completion_q_writer_location.x, completion_q_writer_location.y);

                        // Re-start the pointers
                        uint32_t issue_queue_start_addr = CQ_START + get_absolute_cq_offset(curr_channel, cq_id, curr_cq_size);
                        uint32_t issue_queue_size = tt::round_up((cq_size - CQ_START) * SystemMemoryCQInterface::default_issue_queue_split, 32);
                        uint32_t issue_queue_start_addr_16B = issue_queue_start_addr >> 4;
                        vector<uint32_t> issue_queue_read_ptr = {issue_queue_start_addr_16B};
                        WriteToDeviceL1(device, issue_q_logical_core, CQ_ISSUE_READ_PTR, issue_queue_read_ptr);
                        WriteToDeviceL1(device, issue_q_logical_core, CQ_ISSUE_WRITE_PTR, issue_queue_read_ptr);

                        // Currently remote device dispatch completion queue interface has not been brought up
                        // This will be updated with https://github.com/tenstorrent-metal/tt-metal/issues/3949
                        if (device_id == device->id()) {
                            uint32_t completion_queue_start_addr = CQ_START + issue_queue_size + get_absolute_cq_offset(curr_channel, cq_id, curr_cq_size);
                            uint32_t completion_queue_start_addr_16B = completion_queue_start_addr >> 4;
                            vector<uint32_t> completion_queue_wr_ptr = {completion_queue_start_addr_16B};
                            WriteToDeviceL1(device, completion_q_logical_core, CQ_COMPLETION_READ_PTR, completion_queue_wr_ptr);
                            WriteToDeviceL1(device, completion_q_logical_core, CQ_COMPLETION_WRITE_PTR, completion_queue_wr_ptr);
                        }
                    }
                }
            }

            detail::ConfigureDeviceWithProgram(device, command_queue_program);
            tt::Cluster::instance().l1_barrier(device->id());
        }

        inline void CompileCommandQueuePrograms(Device *device, vector<unique_ptr<Program, ProgramDeleter>>& command_queue_programs) {
            ZoneScoped;

            // TODO: Load dispatch kernels on dispatch cores of the remote chip
            //  https://github.com/tenstorrent-metal/tt-metal/issues/3953 and https://github.com/tenstorrent-metal/tt-metal/issues/3954
            TT_ASSERT(device->is_mmio_capable(), "Cannot compile a program for a non-mmio capable device TODO(abhullar)");

            unique_ptr<Program, ProgramDeleter> command_queue_program_ptr(new Program);

            for (const chip_id_t &device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(device->id())) {
                // TODO (abhullar): allow for multiple cqs on remote device, atm device initialization asserts one cq for the remote device
                uint8_t num_hw_cqs = device_id == device->id() ? device->num_hw_cqs() : 1;
                uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
                uint32_t cq_size = tt::Cluster::instance().get_host_channel_size(device->id(), channel) / num_hw_cqs;

                for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                    tt_cxy_pair issue_q_reader_location = dispatch_core_manager::get(num_hw_cqs).issue_queue_reader_core(device_id, channel, cq_id);
                    tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(num_hw_cqs).completion_queue_writer_core(device_id, channel, cq_id);

                    TT_ASSERT(issue_q_reader_location.chip == device->id() and completion_q_writer_location.chip == device->id(),
                        "Issue queue interface is on device {} and completion queue interface is on device {} but they are expected to be on device {}", issue_q_reader_location.chip, completion_q_writer_location.chip, device->id());

                    CoreCoord issue_q_logical_core(issue_q_reader_location.x, issue_q_reader_location.y);
                    CoreCoord completion_q_logical_core(completion_q_writer_location.x, completion_q_writer_location.y);
                    CoreCoord issue_q_physical_core = get_physical_core_coordinate(issue_q_reader_location, CoreType::WORKER);
                    CoreCoord completion_q_physical_core = get_physical_core_coordinate(completion_q_writer_location, CoreType::WORKER);

                    CoreCoord consumer_physical_core = completion_q_physical_core;
                    if (device_id != device->id()) {
                        // This means the issue queue and completion queue interfaces that service a remote device are being set up
                        // the issue queue interface needs to send fast dispatch packets to the "src" ethernet core
                        CoreCoord logical_eth_router_src = tt::Cluster::instance().get_eth_core_for_dispatch_core(
                            issue_q_reader_location, EthRouterMode::FD_SRC, device_id);
                        consumer_physical_core = device->ethernet_core_from_logical_core(logical_eth_router_src);

                        tt::Cluster::instance().configure_eth_core_for_dispatch_core(
                            issue_q_reader_location, EthRouterMode::FD_SRC, device_id);
                    }

                    std::map<string, string> producer_defines = {
                        {"DISPATCH_KERNEL", "1"},
                        {"CONSUMER_NOC_X", std::to_string(consumer_physical_core.x)},
                        {"CONSUMER_NOC_Y", std::to_string(consumer_physical_core.y)},
                    };
                    std::map<string, string> consumer_defines = {
                        {"DISPATCH_KERNEL", "1"},
                        {"PRODUCER_NOC_X", std::to_string(issue_q_physical_core.x)},
                        {"PRODUCER_NOC_Y", std::to_string(issue_q_physical_core.y)},
                    };

                    // Address in sysmem for CQ to write back its read ptr to
                    uint32_t host_issue_queue_read_ptr_addr = HOST_CQ_ISSUE_READ_PTR + get_absolute_cq_offset(channel, cq_id, cq_size);
                    uint32_t issue_queue_start_addr = CQ_START + get_absolute_cq_offset(channel, cq_id, cq_size);
                    uint32_t issue_queue_size = tt::round_up((cq_size - CQ_START) * SystemMemoryCQInterface::default_issue_queue_split, 32);
                    uint32_t command0_l1_addr = get_command_start_l1_address(/*use_eth_l1=*/false); // issue queue interface kernels are currently placed on tensix cores
                    uint32_t data_section_l1_addr = get_data_section_l1_address(/*use_eth_l1=*/false); // issue queue interface kernels are currently placed on tensix cores
                    uint32_t consumer_cmd_base_addr = get_command_start_l1_address(device_id != device->id()); // device is MMIO capable but current device_id being set up is remote
                    uint32_t consumer_data_buff_size = get_consumer_data_buffer_size(device_id != device->id()); // device is MMIO capable but current device_id being set up is remote
                    std::vector<uint32_t> producer_compile_args = {
                        host_issue_queue_read_ptr_addr, issue_queue_start_addr, issue_queue_size, command0_l1_addr, data_section_l1_addr, consumer_cmd_base_addr, consumer_data_buff_size};

                    uint32_t host_completion_queue_write_ptr_addr = HOST_CQ_COMPLETION_WRITE_PTR + get_absolute_cq_offset(channel, cq_id, cq_size);
                    uint32_t completion_queue_start_addr = CQ_START + issue_queue_size + get_absolute_cq_offset(channel, cq_id, cq_size);
                    uint32_t completion_queue_size = (cq_size - CQ_START) - issue_queue_size;
                    uint32_t host_finish_addr = HOST_CQ_FINISH_PTR + get_absolute_cq_offset(channel, cq_id, cq_size);
                    std::vector<uint32_t> consumer_compile_args = {host_completion_queue_write_ptr_addr, completion_queue_start_addr, completion_queue_size, host_finish_addr, consumer_cmd_base_addr, consumer_data_buff_size};

                    std::string issue_q_reader_kernel = (device_id == device->id()) ? "tt_metal/impl/dispatch/kernels/command_queue_producer.cpp" : "tt_metal/impl/dispatch/kernels/remote_issue_queue_reader.cpp";

                    tt::tt_metal::CreateKernel(
                        *command_queue_program_ptr,
                        issue_q_reader_kernel,
                        issue_q_logical_core,
                        tt::tt_metal::DataMovementConfig {
                            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                            .noc = tt::tt_metal::NOC::RISCV_0_default,
                            .compile_args = producer_compile_args,
                            .defines = producer_defines});


                    uint32_t num_command_slots = (device_id == device->id()) ? 2 : 1;
                    tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, issue_q_logical_core, num_command_slots);

                    // Currently remote device dispatch completion queue interface has not been brought up
                    // This will be updated with https://github.com/tenstorrent-metal/tt-metal/issues/3949
                    if (device_id == device->id()) {
                        tt::tt_metal::CreateKernel(
                            *command_queue_program_ptr,
                            "tt_metal/impl/dispatch/kernels/command_queue_consumer.cpp",
                            completion_q_logical_core,
                            tt::tt_metal::DataMovementConfig {
                                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                                .noc = tt::tt_metal::NOC::RISCV_0_default,
                                .compile_args = consumer_compile_args,
                                .defines = consumer_defines});

                        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, completion_q_logical_core, 0);
                    }
                }
            }

            CompileProgram(device, *command_queue_program_ptr);
            command_queue_programs.push_back(std::move(command_queue_program_ptr));
        }
    }
}
