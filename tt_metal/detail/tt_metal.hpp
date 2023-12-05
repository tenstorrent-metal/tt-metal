// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <mutex>
#include <variant>

#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_metal/build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/llrt/watcher.hpp"

using std::unique_lock;
using std::mutex;

namespace tt::tt_metal{

    namespace detail {
        // To be removed at a later time, but need a global
        // command queue for the time being.
        inline unique_ptr<CommandQueue> GLOBAL_CQ;

        inline static bool DispatchStateCheck( bool isFastDispatch){
            static bool fd = isFastDispatch;
            TT_FATAL( fd == isFastDispatch, "Mixing fast and slow dispatch is prohibited!" );
            return fd;
        }

        static Allocator &GetAllocator( const Device* device )
        {
            static std::vector<std::unique_ptr<Allocator>> allocators( Device::detect_num_available_devices() );
            static vector<std::once_flag> vflags( Device::detect_num_available_devices() );
            chip_id_t id = device->id();
            TT_FATAL(id < allocators.size(), "Invalid device {} detected", id);
            std::call_once(vflags[id], [&device](){
                allocators[device->id()] = std::make_unique<L1BankingAllocator>(*device); });

            return *(allocators[id]);
        }

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

        /**
        * Copies data from a buffer into a host buffer
        *
        * Return value: void
        *
        * | Argument    | Description                                     | Data type               | Valid range                                      | Required |
        * |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
        * | buffer      | Buffer to read data from                        | const Buffer &          |                                                  | Yes      |
        * | host_buffer | Buffer on host to copy data into                | std::vector<uint32_t> & |                                                  | Yes      |
        */
        void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer);


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
         * Read device side profiler data and dump results into device side CSV log
         *
         * Return value: void
         *
         * | Argument      | Description                                       | Type                      | Valid Range               | Required |
         * |---------------|---------------------------------------------------|---------------------------|---------------------------|----------|
         * | device        | The device holding the program being profiled.    | Device *                  |                           | True     |
         * | core_coords   | The logical core coordinates being profiled.      | const vector<CoreCoord> & |                           | True     |
         * */
        void DumpDeviceProfileResults(Device *device, const vector<CoreCoord>& logical_cores);

        /**
         * Set the directory for all CSV logs produced by the profiler instance in the tt-metal module
         *
         * Return value: void
         *
         * | Argument     | Description                                             |  Data type  | Valid range              | required |
         * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
         * | output_dir   | The output directory that will hold the outpu CSV logs  | std::string | Any valid directory path | No       |
         * */
        void SetProfilerDir(std::string output_dir = "");

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
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            llrt::write_hex_vec_to_core(device->id(), worker_core, host_buffer, address);
            return true;
        }

        inline bool WriteToDeviceL1(Device *device, const CoreCoord &core, op_info_t op_info, int op_idx)
        {
            auto worker_core = device->worker_core_from_logical_core(core);
            llrt::write_graph_interpreter_op_info_to_core(device->id(), worker_core, op_info, op_idx);
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

        inline void Synchronize()
        {
            if (detail::GLOBAL_CQ) {
                Finish(*detail::GLOBAL_CQ);
            }
        }

        inline void DeallocateBuffers(Device * device)
        {
            allocator::deallocate_buffers(GetAllocator(device));
        }

        inline void ClearCommandQueueProgramCache()
        {
            if (detail::GLOBAL_CQ) {
                ClearProgramCache(*detail::GLOBAL_CQ);
            }
        }

        inline void GenerateDeviceHeaders(Device *device,
                                          build_kernel_for_riscv_options_t *build_options,
                                          const std::string &op_path_suffix)
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
            generate_bank_to_noc_coord_descriptor (
                build_options,
                op_path_suffix,
                soc_d.grid_size,
                dram_noc_coord_per_bank,
                dram_offsets_per_bank,
                l1_noc_coord_per_bank,
                l1_offset_per_bank
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

            auto dispatch_cores = device->dispatch_cores().begin();
            CoreCoord producer_logical_core = *dispatch_cores++;
            CoreCoord consumer_logical_core = *dispatch_cores;

            // Create valid PCIe address ranges
            // This implementation assumes contiguous ranges and aggregates the ranges into one bounds check
            // TODO: consider checking multiple ranges to detect straddling transactions
            uint64_t pcie_chan_base_addr = tt::Cluster::instance().get_pcie_base_addr_from_device(device->id());
            uint64_t pcie_chan_end_addr = pcie_chan_base_addr;
            for (int pcie_chan = 0; pcie_chan < tt::Cluster::instance().get_num_host_channels(device->id()); pcie_chan++) {
                pcie_chan_end_addr += tt::Cluster::instance().get_host_channel_size(device->id(), pcie_chan);
            }

            generate_noc_addr_ranges_header(
                build_options,
                op_path_suffix,
                pcie_chan_base_addr,
                pcie_chan_end_addr - pcie_chan_base_addr,
                0,
                soc_d.dram_core_size,
                soc_d.get_pcie_cores(),
                soc_d.get_dram_cores(),
                soc_d.get_physical_ethernet_cores(),
                soc_d.grid_size,
                harvested_rows,
                {device->worker_core_from_logical_core(consumer_logical_core)});
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
    }
}
