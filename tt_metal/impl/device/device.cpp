// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/core_descriptor.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "impl/debug/dprint_server.hpp"
#include "impl/debug/watcher_server.hpp"
#include "tt_metal/third_party/umd/device/util.hpp"


#include "common/utils.hpp"
#include "llrt/llrt.hpp"
#include "dev_msgs.h"

namespace tt {

namespace tt_metal {

void ::detail::ProgramDeleter::operator()(Program *p) {
    delete p;
}

ActiveDevices Device::active_devices_;

ActiveDevices::ActiveDevices() {
}

ActiveDevices::~ActiveDevices() {
    for (size_t i = 0; i < active_devices_.size(); i++) {
        if (active_devices_[i] == ActiveState::ACTIVE) {
            TT_THROW("Process tear down with device {} still active", i);
        }
    }
}

bool ActiveDevices::activate_device(chip_id_t id) {
    bool already_initialized;
    const std::lock_guard<std::mutex> lock(lock_);
    if (this->active_devices_.size() < id + 1) {
        this->active_devices_.resize(id + 1);
        already_initialized = false;
    } else if (this->active_devices_[id] == ActiveState::ACTIVE) {
        TT_THROW("Cannot re-initialize device {}, must first call close()", id);
    } else {
        already_initialized = (this->active_devices_[id] == ActiveState::INACTIVE) ? true : false;
    }
    this->active_devices_[id] = ActiveState::ACTIVE;

    return already_initialized;
}

void ActiveDevices::deactivate_device(chip_id_t id) {
    const std::lock_guard<std::mutex> lock(lock_);
    this->active_devices_[id] = ActiveState::INACTIVE;
}

Device::Device(chip_id_t device_id, const uint8_t num_hw_cqs, const std::vector<uint32_t>& l1_bank_remap) : id_(device_id), num_hw_cqs_(num_hw_cqs)
{
    ZoneScoped;
    TT_ASSERT(num_hw_cqs > 0 and num_hw_cqs < 3, "num_hw_cqs can be between 1 and 2");
    this->initialize(l1_bank_remap);
}

void Device::initialize_cluster() {
    ZoneScoped;
    if (llrt::OptionsG.get_clear_l1()) {
        this->clear_l1_state();
    }
#ifdef TT_METAL_VERSIM_DISABLED
    int ai_clk = tt::Cluster::instance().get_device_aiclk(this->id_);
    log_info(tt::LogMetal, "AI CLK for device {} is:   {} MHz", this->id_, ai_clk);
#endif
}

void Device::initialize_allocator(const std::vector<uint32_t>& l1_bank_remap) {
    ZoneScoped;
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    // Construct allocator config from soc_desc
    AllocatorConfig config({
        .num_dram_channels = static_cast<size_t>(soc_desc.get_num_dram_channels()),
        .dram_bank_size = soc_desc.dram_bank_size,
        .dram_bank_offsets = {},
        .worker_grid_size = this->logical_grid_size(),
        .worker_l1_size = static_cast<size_t>(soc_desc.worker_l1_size),
        .l1_bank_size = static_cast<size_t>(get_storage_core_bank_size(this->id_, this->num_hw_cqs_)),
        .core_type_from_noc_coord_table = {}, // Populated later
        .worker_log_to_physical_routing_x=soc_desc.worker_log_to_physical_routing_x,
        .worker_log_to_physical_routing_y=soc_desc.worker_log_to_physical_routing_y,
        .l1_bank_remap = l1_bank_remap,
        .compute_grid_size = this->compute_with_storage_grid_size()
    });
    // Initialize dram_offsets from soc_descriptor
    for (auto channel = 0; channel < soc_desc.get_num_dram_channels(); channel++) {
        config.dram_bank_offsets.push_back(soc_desc.get_address_offset(channel));
    }
    // Initialize core_type_from_noc_coord_table table
    for (const auto& core: soc_desc.physical_cores) {
        config.core_type_from_noc_coord_table.insert({core.first, AllocCoreType::Invalid});
    }

    for (const CoreCoord& core : tt::get_logical_compute_cores(id_, num_hw_cqs_)) {
        this->compute_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::ComputeAndStore;
    }
    for (const CoreCoord& core : tt::get_logical_storage_cores(id_, num_hw_cqs_)) {
        this->storage_only_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::StorageOnly;
    }
    for (const CoreCoord& core : tt::get_logical_dispatch_cores(id_, num_hw_cqs_)) {
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::Dispatch;
    }
    for (const auto &core : soc_desc.get_logical_ethernet_cores()) {
        this->ethernet_cores_.insert(core);
    }

    // L1_BANKING scheme creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
    // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
    TT_ASSERT(this->allocator_scheme_ == MemoryAllocator::L1_BANKING);
    this->allocator_ = std::make_unique<L1BankingAllocator>(config);
}

void Device::initialize_build() {
    ZoneScoped;

    this->build_env_.init(this->id(), this->arch());

    auto init_helper = [this] (bool is_fw) -> JitBuildStateSet {
        std::vector<std::shared_ptr<JitBuildState>> build_states;

        build_states.resize(arch() == tt::ARCH::GRAYSKULL ? 5 : 6);

        build_states[build_processor_type_to_index(JitBuildProcessorType::DATA_MOVEMENT).first + 0] =
            std::make_shared<JitBuildDataMovement>(this->build_env_, 0, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::DATA_MOVEMENT).first + 1] =
            std::make_shared<JitBuildDataMovement>(this->build_env_, 1, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 0] =
            std::make_shared<JitBuildCompute>(this->build_env_, 0, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 1] =
            std::make_shared<JitBuildCompute>(this->build_env_, 1, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 2] =
            std::make_shared<JitBuildCompute>(this->build_env_, 2, is_fw);

        if (arch() != tt::ARCH::GRAYSKULL) {
            build_states[build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 0] =
                std::make_shared<JitBuildEthernet>(this->build_env_, 0, is_fw);
        }

       return build_states;
    };

    this->firmware_build_states_ = init_helper(true);
    this->kernel_build_states_ = init_helper(false);
}

void Device::build_firmware() {
    ZoneScoped;

    detail::GenerateDeviceHeaders(this, this->build_env_.get_out_firmware_root_path());
    jit_build_set(this->firmware_build_states_, nullptr, "");
}

void Device::initialize_firmware(CoreCoord phys_core, launch_msg_t *launch_msg) {
    ZoneScoped;

    if (llrt::is_ethernet_core(phys_core, this->id())) {
        int eriscv_id = build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 0;
        ll_api::memory binary_mem = llrt::get_risc_binary(firmware_build_states_[eriscv_id]->get_target_out_path(""));
        uint32_t kernel_size16 = llrt::get_binary_code_size16(binary_mem, eriscv_id);
        log_debug(LogDevice, "ERISC fw binary size: {} in bytes", kernel_size16 * 16);
        llrt::test_load_write_read_risc_binary(binary_mem, this->id(), phys_core, eriscv_id);
        llrt::launch_erisc_app_fw_on_core(this->id(), phys_core);
    } else {
        llrt::program_brisc_startup_addr(this->id(), phys_core);
        for (int riscv_id = 0; riscv_id < 5; riscv_id++) {
            ll_api::memory binary_mem =
                llrt::get_risc_binary(firmware_build_states_[riscv_id]->get_target_out_path(""));
            uint32_t kernel_size16 = llrt::get_binary_code_size16(binary_mem, riscv_id);
            if (riscv_id == 1) {
                launch_msg->ncrisc_kernel_size16 = kernel_size16;
            }
            log_debug(LogDevice, "RISC {} fw binary size: {} in bytes", riscv_id, kernel_size16 * 16);
            llrt::test_load_write_read_risc_binary(binary_mem, this->id(), phys_core, riscv_id);
        }
    }
    llrt::write_launch_msg_to_core(this->id(), phys_core, launch_msg);
}

void Device::initialize_and_launch_firmware() {
    ZoneScoped;

    launch_msg_t launch_msg = {
        .brisc_watcher_kernel_id = 0,
        .ncrisc_watcher_kernel_id = 0,
        .triscs_watcher_kernel_id = 0,
        .ncrisc_kernel_size16 = 0,
        .mode = DISPATCH_MODE_HOST,
        .brisc_noc_id = 0,
        .enable_brisc = 0,
        .enable_ncrisc = 0,
        .enable_triscs = 0,
        .enable_erisc = 0,
        .run = RUN_MSG_INIT,
    };

    // Download to worker cores
    log_debug("Initializing firmware");
    CoreCoord grid_size = this->logical_grid_size();
    std::unordered_set<CoreCoord> not_done_cores;

    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            if (!this->storage_only_cores_.count(logical_core)) {
                CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);
                this->initialize_firmware(worker_core, &launch_msg);
                not_done_cores.insert(worker_core);
            }
        }
    }

    // Load erisc app base FW to eth cores
    for (const auto &eth_core : this->get_active_ethernet_cores()) {
        CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
        this->initialize_firmware(phys_eth_core, &launch_msg);
    }

    // Barrier between L1 writes above and deassert below
    tt::Cluster::instance().l1_barrier(this->id());

    // Deassert worker cores
    for(const auto& worker_core : not_done_cores)
        tt::Cluster::instance().deassert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));

    // Wait until fw init is done, ensures the next launch msg doesn't get
    // written while fw is still in init
    log_debug("Waiting for firmware init complete");
    llrt::internal_::wait_until_cores_done(this->id(), RUN_MSG_INIT, not_done_cores);
    log_debug("Firmware init complete");
}

void Device::clear_l1_state() {
    CoreCoord logical_grid_size = this->logical_grid_size();
    TT_ASSERT(this->l1_size_per_core() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(this->l1_size_per_core() / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            detail::WriteToDeviceL1(this, logical_core, start_address, zero_vec);
        }
    }

    // Clear erisc sync info
    for (const auto &eth_core : this->get_active_ethernet_cores()) {
        CoreCoord physical_core = this->ethernet_core_from_logical_core(eth_core);
        std::vector<uint32_t> init_erisc_info_vec(
            (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE) / sizeof(uint32_t),
            0);

        llrt::write_hex_vec_to_core(
            this->id(), physical_core, init_erisc_info_vec, eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
    }
}

// TODO (abhullar): Refactor this with #2593 to allow each target fast dispatch (FD) device to program their associated FD cores regardless of whether they are on the target device or not.
// Currently we have to program FD cores for the remote device when initializing the MMIO device because completion queue cores are on MMIO device
//  and we don't have handle on MMIO device when initializing the remote device
void Device::compile_command_queue_programs() {
    ZoneScoped;
    unique_ptr<Program, detail::ProgramDeleter> command_queue_program_ptr(new Program);

    uint32_t dispatch_buffer_pages = DISPATCH_BUFFER_BLOCK_SIZE_PAGES * DISPATCH_BUFFER_SIZE_BLOCKS;
    constexpr uint32_t dispatch_cb_sem = 0;
    uint32_t dispatch_buffer_base = get_dispatch_buffer_base();
    uint32_t dev_hugepage_base = 0; // what is this????
    uint32_t prefetch_q_base = L1_UNRESERVED_BASE;
    uint32_t prefetch_q_rd_ptr_addr = L1_UNRESERVED_BASE - 4; // XXXXX hacks and hacks and hacks FIND A SPOT FOR THE FETCH Q
    uint32_t prefetch_q_size = PREFETCH_Q_ENTRIES * sizeof(uint16_t);
    constexpr uint32_t noc_read_alignment = 32;
    uint32_t cmddat_q_base = prefetch_q_base + ((prefetch_q_size + noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);
    uint32_t cmddat_q_size_g = CMDDAT_Q_SIZE;
    uint32_t scratch_db_base = cmddat_q_base + ((cmddat_q_size_g + noc_read_alignment - 1) / noc_read_alignment * noc_read_alignment);
    TT_ASSERT(scratch_db_base < MEM_L1_SIZE); // L1 size
    uint32_t scratch_db_size_g = SCRATCH_DB_SIZE;


    if (this->is_mmio_capable()) {
        for (const chip_id_t &device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(this->id())) {
            if (device_id != this->id()) {
                continue; // REMOVE WHEN R CHIP IS SUPPORTED
            }
            // TODO (abhullar): allow for multiple cqs on remote device, atm device initialization asserts one cq for the remote device
            uint8_t num_hw_cqs = device_id == this->id() ? this->num_hw_cqs() : 1;
            uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
            uint32_t cq_size = tt::Cluster::instance().get_host_channel_size(this->id(), channel) / num_hw_cqs;

            for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                tt_cxy_pair prefetcher_location = dispatch_core_manager::get(num_hw_cqs).prefetcher_core(device_id, channel, cq_id);
                tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(num_hw_cqs).completion_queue_writer_core(device_id, channel, cq_id);
                tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_core(device_id, channel, cq_id);

                TT_ASSERT(prefetcher_location.chip == this->id() and completion_q_writer_location.chip == this->id(),
                    "Issue queue interface is on device {} and completion queue interface is on device {} but they are expected to be on device {}", prefetcher_location.chip, completion_q_writer_location.chip, this->id());

                CoreCoord prefetcher_physical_core = get_physical_core_coordinate(prefetcher_location, CoreType::WORKER);
                CoreCoord completion_q_physical_core = get_physical_core_coordinate(completion_q_writer_location, CoreType::WORKER);
                CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, CoreType::WORKER);

                // CoreCoord consumer_physical_core = completion_q_physical_core;
                // CoreCoord producer_physical_core = prefetcher_physical_core;
                if (device_id != this->id()) {
                    // This means the issue queue and completion queue interfaces that service a remote device are being set up
                    // the issue queue interface needs to send fast dispatch packets to the "src" ethernet core and
                    // the completion queue writer receives packets from the "dst" ethernet core
                    // tt_cxy_pair logical_eth_router_src = tt::Cluster::instance().get_eth_core_for_dispatch_core(
                    //     issue_q_reader_location, EthRouterMode::BI_DIR_TUNNELING, device_id);
                    // consumer_physical_core = this->ethernet_core_from_logical_core(logical_eth_router_src);

                    // // remote_issue_q writing to eth SRC, semaphore 0
                    // tt::Cluster::instance().write_core(&accept_cmd_sem_value, sizeof(uint32_t), tt_cxy_pair(this->id(), consumer_physical_core), eth_l1_mem::address_map::SEMAPHORE_BASE);

                    // tt_cxy_pair logical_eth_router_dst = tt::Cluster::instance().get_eth_core_for_dispatch_core(
                    //     completion_q_writer_location, EthRouterMode::BI_DIR_TUNNELING, device_id);
                    // producer_physical_core = this->ethernet_core_from_logical_core(logical_eth_router_dst);

                    // // remote_command_processor receiving from eth DST, semaphore 1
                    // tt::Cluster::instance().write_core(&num_eth_command_slots, sizeof(uint32_t), tt_cxy_pair(this->id(), producer_physical_core), eth_l1_mem::address_map::SEMAPHORE_BASE + L1_ALIGNMENT);

                    // // Setup eth core for bidirectional tunneling
                    // std::map<string, string> eth_tunneller_defines = {
                    //     {"DISPATCH_KERNEL", "1"}, //TODO: do we need this?
                    //     {"CONSUMER_NOC_X", std::to_string(completion_q_physical_core.x)},
                    //     {"CONSUMER_NOC_Y", std::to_string(completion_q_physical_core.y)},
                    //     {"PRODUCER_NOC_X", std::to_string(issue_q_physical_core.x)},
                    //     {"PRODUCER_NOC_Y", std::to_string(issue_q_physical_core.y)},
                    // };
                    // std::vector<uint32_t> eth_tunneller_compile_args = {true, num_tensix_command_slots};
                    // std::string command_q_tunneller_kernel = "tt_metal/impl/dispatch/kernels/command_queue_bidirectional_tunneller.cpp";
                    // tt::tt_metal::CreateKernel(
                    //     *command_queue_program_ptr,
                    //     command_q_tunneller_kernel,
                    //     logical_eth_router_src,
                    //     tt::tt_metal::EthernetConfig {
                    //         .noc = tt::tt_metal::NOC::RISCV_0_default,
                    //         .compile_args = eth_tunneller_compile_args,
                    //         .defines = eth_tunneller_defines});
                }

                TT_ASSERT(tt::Cluster::instance().get_soc_desc(this->id()).pcie_cores.size() == 1);
                CoreCoord pcie_physical_core = tt::Cluster::instance().get_soc_desc(this->id()).pcie_cores.at(0);

                std::map<string, string> defines = {
                    {"PREFETCH_NOC_X", std::to_string(prefetcher_physical_core.x)},
                    {"PREFETCH_NOC_Y", std::to_string(prefetcher_physical_core.y)},
                    {"DISPATCH_NOC_X", std::to_string(dispatch_physical_core.x)},
                    {"DISPATCH_NOC_Y", std::to_string(dispatch_physical_core.y)},
                };

                std::vector<uint32_t> prefetch_compile_args = {
                    dispatch_buffer_base,
                    DISPATCH_BUFFER_LOG_PAGE_SIZE,
                    dispatch_buffer_pages,
                    dispatch_cb_sem,
                    dev_hugepage_base,
                    cq_size,
                    prefetch_q_base,
                    PREFETCH_Q_ENTRIES * (uint32_t)sizeof(uint16_t),
                    prefetch_q_rd_ptr_addr,
                    cmddat_q_base,
                    cmddat_q_size_g,
                    scratch_db_base,
                    scratch_db_size_g
                };

                tt::tt_metal::CreateKernel(
                    *command_queue_program_ptr,
                    "tt_metal/impl/dispatch/kernels/cq_prefetch_hd.cpp", // update this for remote device
                    prefetcher_location,
                    tt::tt_metal::DataMovementConfig {
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt::tt_metal::NOC::RISCV_0_default,
                        .compile_args = prefetch_compile_args,
                        .defines = defines});

                tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, prefetcher_location, dispatch_buffer_pages);

                if (device_id == this->id()) {
                    std::vector<uint32_t> dispatch_compile_args = {
                        dispatch_buffer_base,
                        DISPATCH_BUFFER_LOG_PAGE_SIZE,
                        DISPATCH_BUFFER_SIZE_BLOCKS * DISPATCH_BUFFER_BLOCK_SIZE_PAGES,
                        dispatch_cb_sem,
                        DISPATCH_BUFFER_SIZE_BLOCKS,
                    };

                    tt::tt_metal::CreateKernel(
                        *command_queue_program_ptr,
                        "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
                        dispatch_location,
                        tt::tt_metal::DataMovementConfig {
                            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                            .noc = tt::tt_metal::NOC::RISCV_0_default,
                            .compile_args = dispatch_compile_args,
                            .defines = defines});

                    tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_location, 0);
                } else {
                    // program the completion queue writer for the remote command queue

                    // std::map<string, string> completion_q_defines = {
                    //     {"DISPATCH_KERNEL", "1"},
                    //     {"PULL_NOC_X", std::to_string(producer_physical_core.x)},
                    //     {"PULL_NOC_Y", std::to_string(producer_physical_core.y)},
                    //     {"PUSH_NOC_X", std::to_string(pcie_physical_core.x)},
                    //     {"PUSH_NOC_Y", std::to_string(pcie_physical_core.y)},
                    //     {"DISPATCH_NOC_X", std::to_string(pcie_physical_core.x)},   // this is unused by completion queue writer
                    //     {"DISPATCH_NOC_Y", std::to_string(pcie_physical_core.y)},   // this is unused by completion queue writer
                    // };

                    // std::vector<uint32_t> completion_q_writer_args = {
                    //     host_issue_queue_read_ptr_addr,
                    //     issue_queue_start_addr,
                    //     issue_queue_size,
                    //     host_completion_queue_write_ptr_addr,
                    //     completion_queue_start_addr,
                    //     completion_queue_size,
                    //     host_finish_addr,
                    //     cmd_start_tensix,
                    //     data_section_addr_tensix,
                    //     producer_data_buffer_size_tensix,
                    //     consumer_cmd_base_addr,
                    //     consumer_data_buff_size,
                    //     (uint32_t)tt::PullAndPushConfig::PULL_FROM_REMOTE
                    // };

                    // tt::tt_metal::CreateKernel(
                    //     *command_queue_program_ptr,
                    //     pull_and_push_kernel,
                    //     completion_q_writer_location,
                    //     tt::tt_metal::DataMovementConfig {
                    //         .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    //         .noc = tt::tt_metal::NOC::RISCV_0_default,
                    //         .compile_args = completion_q_writer_args,
                    //         .defines = completion_q_defines});

                    // tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, completion_q_writer_location, num_eth_command_slots); // push semaphore
                    // tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, completion_q_writer_location, 0); // pull semaphore
                    // tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, completion_q_writer_location, num_tensix_command_slots); // semaphore between push&pull kernel and dispatch kernel
                }
            }
        }
    } else {
        // TT_THROW("FD2.0 does not support R chip yet");
        // TT_ASSERT(this->num_hw_cqs() == 1, "Currently can only support one command queue for remote device");
        // uint8_t num_hw_cqs = this->num_hw_cqs();
        // const uint8_t cq_id = 0;
        // chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id());
        // uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->id());
        // uint32_t cq_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / num_hw_cqs;

        // tt_cxy_pair remote_processor_location = dispatch_core_manager::get(num_hw_cqs).remote_push_and_pull_core(this->id(), channel, cq_id);
        // tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).command_dispatcher_core(this->id(), channel, cq_id);
        // CoreCoord remote_processor_physical_core = get_physical_core_coordinate(remote_processor_location, CoreType::WORKER);
        // CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, CoreType::WORKER);

        // // Set up the dst router to receive fast dispatch packets
        // tt_cxy_pair logical_eth_router_remote_dst = tt::Cluster::instance().get_eth_core_for_dispatch_core(remote_processor_location, EthRouterMode::BI_DIR_TUNNELING, mmio_device_id);
        // CoreCoord physical_eth_router_remote_dst = this->ethernet_core_from_logical_core(logical_eth_router_remote_dst);

        // // TODO (abhullar / aliu): there is no API to configure ethernet semaphores used for FD so manually write initial semaphore value
        // // remote_completion_writer receiving from eth DST, semaphore 1
        // tt::Cluster::instance().write_core(&num_eth_command_slots, sizeof(uint32_t), tt_cxy_pair(this->id(), physical_eth_router_remote_dst), eth_l1_mem::address_map::SEMAPHORE_BASE + L1_ALIGNMENT);

        // // Set up the src router on remote device to send fast dispatch packets on the return path to MMIO device
        // CoreCoord logical_eth_router_remote_src = tt::Cluster::instance().get_eth_core_for_dispatch_core(
        //     remote_processor_location, EthRouterMode::BI_DIR_TUNNELING, mmio_device_id);

        // // remote_signaller writing to eth SRC, semaphore 0
        // CoreCoord physical_eth_router_remote_src = this->ethernet_core_from_logical_core(logical_eth_router_remote_src);
        // tt::Cluster::instance().write_core(&accept_cmd_sem_value, sizeof(uint32_t), tt_cxy_pair(this->id(), physical_eth_router_remote_src), eth_l1_mem::address_map::SEMAPHORE_BASE);
        // // TODO: aliu add more bidirection tunneling kernels for multihop dispatch
        //   // Setup eth core for bidirectional tunneling
        //     std::map<string, string> eth_tunneller_defines = {
        //         {"DISPATCH_KERNEL", "1"}, //TODO: do we need this?
        //         {"CONSUMER_NOC_X", std::to_string(remote_processor_physical_core.x)},
        //         {"CONSUMER_NOC_Y", std::to_string(remote_processor_physical_core.y)},
        //         {"PRODUCER_NOC_X", std::to_string(remote_processor_physical_core.x)},
        //         {"PRODUCER_NOC_Y", std::to_string(remote_processor_physical_core.y)},
        //     };
        //     std::vector<uint32_t> eth_tunneller_compile_args = {false, num_tensix_command_slots}; // SENDER is ISSUE
        //     std::string command_q_tunneller_kernel = "tt_metal/impl/dispatch/kernels/command_queue_bidirectional_tunneller.cpp";
        //     tt::tt_metal::CreateKernel(
        //         *command_queue_program_ptr,
        //         command_q_tunneller_kernel,
        //         logical_eth_router_remote_src,
        //         tt::tt_metal::EthernetConfig {
        //             .noc = tt::tt_metal::NOC::RISCV_0_default,
        //             .compile_args = eth_tunneller_compile_args,
        //             .defines = eth_tunneller_defines});

        // std::vector<uint32_t> remote_pull_and_push_compile_args = {
        //     0, // host_issue_queue_read_ptr_addr,
        //     0, // issue_queue_start_addr,
        //     0, // issue_queue_size,
        //     0, // host_completion_queue_write_ptr_addr,
        //     0, // completion_queue_start_addr,
        //     0, // completion_queue_size,
        //     0, // host_finish_addr
        //     cmd_start_tensix,
        //     data_section_addr_tensix,
        //     producer_data_buffer_size_tensix,
        //     cmd_start_tensix,
        //     consumer_data_buffer_size_tensix,
        //     (uint32_t)tt::PullAndPushConfig::REMOTE_PULL_AND_PUSH
        // };

        // std::map<string, string> remote_pull_and_push_defines = {
        //     {"DISPATCH_KERNEL", "1"},
        //     {"PULL_NOC_X", std::to_string(physical_eth_router_remote_dst.x)},
        //     {"PULL_NOC_Y", std::to_string(physical_eth_router_remote_dst.y)},
        //     {"PUSH_NOC_X", std::to_string(physical_eth_router_remote_src.x)},
        //     {"PUSH_NOC_Y", std::to_string(physical_eth_router_remote_src.y)},
        //     {"DISPATCH_NOC_X", std::to_string(dispatch_physical_core.x)},
        //     {"DISPATCH_NOC_Y", std::to_string(dispatch_physical_core.y)},
        // };

        // tt::tt_metal::CreateKernel(
        //     *command_queue_program_ptr,
        //     "tt_metal/impl/dispatch/kernels/cq_prefetcher.cpp",
        //     remote_processor_location,
        //     tt::tt_metal::DataMovementConfig {
        //         .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        //         .noc = tt::tt_metal::NOC::RISCV_0_default,
        //         .compile_args = remote_pull_and_push_compile_args,
        //         .defines = remote_pull_and_push_defines});

        // // first semaphore is between pull_and_relay and pusher
        // tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, remote_processor_location, num_eth_command_slots);
        // // second semaphore is between processor and dispatcher to detect whether dispatcher can accept commands
        // tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, remote_processor_location, 0);
        // tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, remote_processor_location, num_tensix_command_slots); // semaphore between push&pull kernel and dispatch kernel

        // std::vector<uint32_t> dispatch_compile_args = {cmd_start_tensix, consumer_data_buffer_size_tensix};

        // std::map<string, string> remote_dispatch_defines = {
        //     {"DISPATCH_KERNEL", "1"},
        //     {"PRODUCER_NOC_X", std::to_string(remote_processor_physical_core.x)},
        //     {"PRODUCER_NOC_Y", std::to_string(remote_processor_physical_core.y)},
        // };

        // tt::tt_metal::CreateKernel(
        //     *command_queue_program_ptr,
        //     "tt_metal/impl/dispatch/kernels/cq_dispatcher.cpp",
        //     dispatch_location,
        //     tt::tt_metal::DataMovementConfig {
        //         .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        //         .noc = tt::tt_metal::NOC::RISCV_0_default,
        //         .compile_args = dispatch_compile_args,
        //         .defines = remote_dispatch_defines});

        // tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_location, 0);
    }
    detail::CompileProgram(this, *command_queue_program_ptr);
    this->command_queue_programs.push_back(std::move(command_queue_program_ptr));
}

// Writes issue and completion queue pointers to device and in sysmem and loads fast dispatch program onto dispatch cores
void Device::configure_command_queue_programs() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->id());

    std::vector<uint32_t> pointers(CQ_START / sizeof(uint32_t), 0);
    const uint32_t hugepage_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel);
    const uint32_t cq_size = hugepage_size / this->num_hw_cqs();

    TT_ASSERT(this->command_queue_programs.size() == 1);
    Program& command_queue_program = *this->command_queue_programs[0];

    uint32_t prefetch_q_base = L1_UNRESERVED_BASE;
    uint32_t prefetch_q_rd_ptr_addr = L1_UNRESERVED_BASE - 4; // XXXXX hacks and hacks and hacks

    for (uint8_t cq_id = 0; cq_id < this->num_hw_cqs(); cq_id++) {
        // Reset the host manager's pointer for this command queue
        this->sysmem_manager_->reset(cq_id);

        pointers[HOST_CQ_ISSUE_READ_PTR / sizeof(uint32_t)] = (CQ_START + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
        pointers[HOST_CQ_COMPLETION_WRITE_PTR / sizeof(uint32_t)] = (CQ_START + this->sysmem_manager_->get_issue_queue_size(cq_id) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;

        tt::Cluster::instance().write_sysmem(pointers.data(), pointers.size() * sizeof(uint32_t), cq_id * cq_size, mmio_device_id, channel);


    }

    if (this->is_mmio_capable()) {
        for (const chip_id_t &device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(this->id())) {
            if (device_id != this->id()) {
                continue; // UPDATE THIS FOR R CHIP SUPPORT!
            }

            uint8_t curr_num_hw_cqs = device_id == this->id() ? this->num_hw_cqs() : 1;
            uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
            uint32_t curr_cq_size = tt::Cluster::instance().get_host_channel_size(this->id(), curr_channel) / curr_num_hw_cqs;

            for (uint8_t cq_id = 0; cq_id < curr_num_hw_cqs; cq_id++) {
                tt_cxy_pair prefetcher_location = dispatch_core_manager::get(curr_num_hw_cqs).prefetcher_core(device_id, curr_channel, cq_id);
                tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(curr_num_hw_cqs).completion_queue_writer_core(device_id, curr_channel, cq_id);

                TT_ASSERT(prefetcher_location.chip == this->id() and completion_q_writer_location.chip == this->id(),
                    "Issue queue interface is on device {} and completion queue interface is on device {} but they are expected to be on device {}", prefetcher_location.chip, completion_q_writer_location.chip, this->id());

                // Initialize the FetchQ
                std::vector<uint32_t> prefetch_q(PREFETCH_Q_ENTRIES, 0);
                std::vector<uint32_t> prefetch_q_rd_ptr_addr_data = {
                    (uint32_t)(prefetch_q_base + PREFETCH_Q_ENTRIES * sizeof(uint16_t))
                };
                detail::WriteToDeviceL1(this, prefetcher_location, prefetch_q_rd_ptr_addr, prefetch_q_rd_ptr_addr_data);
                detail::WriteToDeviceL1(this, prefetcher_location, prefetch_q_base, prefetch_q);

                // Initialize completion queue write pointer and read pointer copy
                uint32_t issue_queue_size = tt::round_up((cq_size - CQ_START) * SystemMemoryCQInterface::default_issue_queue_split, 32);
                uint32_t completion_queue_start_addr = CQ_START + issue_queue_size + get_absolute_cq_offset(curr_channel, cq_id, curr_cq_size);
                uint32_t completion_queue_start_addr_16B = completion_queue_start_addr >> 4;
                vector<uint32_t> completion_queue_wr_ptr = {completion_queue_start_addr_16B};
                vector<uint32_t> completion_queue_last_event = {0x0}; // Reset state in case L1 Clear is disabled.
                detail::WriteToDeviceL1(this, completion_q_writer_location, CQ_COMPLETION_READ_PTR, completion_queue_wr_ptr);
                detail::WriteToDeviceL1(this, completion_q_writer_location, CQ_COMPLETION_WRITE_PTR, completion_queue_wr_ptr);
                // detail::WriteToDeviceL1(this, completion_q_writer_location, CQ_COMPLETION_LAST_EVENT, completion_queue_last_event);
            }
        }
    }
    detail::ConfigureDeviceWithProgram(this, command_queue_program, true);
    tt::Cluster::instance().l1_barrier(this->id());
}

void Device::initialize_command_queue() {
    TT_ASSERT(this->is_mmio_capable() or (not this->is_mmio_capable() and this->num_hw_cqs() == 1), "Only support one hardware command queue for fast dispatch on remote device");
    using_fast_dispatch = true;
    this->sysmem_manager_ = std::make_unique<SystemMemoryManager>(this->id_, this->num_hw_cqs());
    hw_command_queues_.resize(num_hw_cqs());
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        hw_command_queues_[cq_id] = std::make_unique<HWCommandQueue>(this, cq_id);
        // Need to do this since CommandQueue constructor is private
        sw_command_queues_.push_back(std::unique_ptr<CommandQueue>(new CommandQueue(this, cq_id)));
    }
    this->compile_command_queue_programs();
    TT_ASSERT(this->command_queue_programs.size() == 1);
    this->configure_command_queue_programs();
    Program& command_queue_program = *this->command_queue_programs[0];

    for (uint8_t cq_id = 0; cq_id < this->num_hw_cqs(); cq_id++) {
        for (const auto &[core_type, logical_dispatch_cores] : command_queue_program.logical_cores()) {
            for (const CoreCoord &logical_dispatch_core : logical_dispatch_cores) {
                launch_msg_t msg = command_queue_program.kernels_on_core(logical_dispatch_core, core_type)->launch_msg;
                tt::llrt::write_launch_msg_to_core(this->id(), this->physical_core_from_logical_core(logical_dispatch_core, core_type), &msg);
            }
        }
    }
    // Added this for safety while debugging hangs with FD v1.3 tunnel to R, should experiment with removing it
    tt::Cluster::instance().l1_barrier(this->id());
}

void Device::initialize_synchronous_sw_cmd_queue() {
    // Initialize a single Software Command Queue for SD, using passthrough mode.
    // This queue is used for all host bound functions using the Software CQ in SD mode.
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        // Need to do this since CommandQueue constructor is private
        sw_command_queues_.push_back(std::unique_ptr<CommandQueue>(new CommandQueue(this, cq_id)));
        sw_command_queues_[cq_id]->set_mode(CommandQueue::CommandQueueMode::PASSTHROUGH);
    }
}

bool Device::initialize(const std::vector<uint32_t>& l1_bank_remap) {
    ZoneScoped;
    log_info(tt::LogMetal, "Initializing device {}", this->id_);
    bool already_initialized = this->active_devices_.activate_device(this->id_);
    this->initialize_cluster();
    this->initialize_allocator(l1_bank_remap);
    this->initialize_build();
    if (!already_initialized) {
        this->build_firmware();
    }

    DprintServerAttach(this);
    watcher_init(this);

    this->initialize_and_launch_firmware();

    watcher_attach(this, build_env_.get_out_root_path());

    // Mark initialized before compiling and sending dispatch kernels to device because compilation expects device to be initialized
    this->initialized_ = true;

    // Create system memory writer for this device to have an associated interface to hardware command queue (i.e. hugepage)
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        detail::DispatchStateCheck(true);
        this->initialize_command_queue();
    } else {
        detail::DispatchStateCheck(false);
        this->initialize_synchronous_sw_cmd_queue();
        TT_ASSERT(this->num_hw_cqs() == 1, "num_hw_cqs must be 1 in slow dispatch");
    }

    return true;
}

bool Device::close() {
    log_info(tt::LogMetal, "Closing device {}", this->id_);
    if (not this->initialized_) {
        TT_THROW("Cannot close device {} that has not been initialized!", this->id_);
    }
    this->deallocate_buffers();
    watcher_detach(this);
    DprintServerDetach(this);

    // Assert worker cores
    CoreCoord grid_size = this->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

            if (this->storage_only_cores_.find(logical_core) == this->storage_only_cores_.end()) {
                tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));
            }
        }
    }

    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);

    if (llrt::OptionsG.get_clear_l1()) {
        this->clear_l1_state();
    }
    tt::Cluster::instance().l1_barrier(id_);
    allocator::clear(*this->allocator_);

    this->active_devices_.deactivate_device(this->id_);
    this->disable_and_clear_program_cache();
    this->sw_command_queues_.clear();
    this->hw_command_queues_.clear();

    this->initialized_ = false;

    return true;
}

Device::~Device() {
    if (this->initialized_) {
        this->close();
    }
}

tt::ARCH Device::arch() const {
    return tt::Cluster::instance().arch();
}

int Device::num_dram_channels() const {
    return tt::Cluster::instance().get_soc_desc(id_).get_num_dram_channels();
}

uint32_t Device::l1_size_per_core() const {
    return tt::Cluster::instance().get_soc_desc(id_).worker_l1_size;
}
uint32_t Device::dram_size_per_channel() const {
    return tt::Cluster::instance().get_soc_desc(id_).dram_bank_size;
}

CoreCoord Device::logical_grid_size() const {
    return tt::Cluster::instance().get_soc_desc(id_).worker_grid_size;
}

CoreCoord Device::compute_with_storage_grid_size() const {
    return tt::get_compute_grid_size(id_, num_hw_cqs_);
}

CoreCoord Device::physical_core_from_logical_core(const CoreCoord &logical_coord, const CoreType &core_type) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_physical_core_from_logical_core(logical_coord, core_type);
}

CoreCoord Device::worker_core_from_logical_core(const CoreCoord &logical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_physical_tensix_core_from_logical(logical_core);
}

std::vector<CoreCoord> Device::worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> worker_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        worker_cores[idx] = worker_core_from_logical_core(logical_cores[idx]);

    return worker_cores;
}

CoreCoord Device::ethernet_core_from_logical_core(const CoreCoord &logical_core) const {
    return tt::Cluster::instance().ethernet_core_from_logical_core(id_, logical_core);
}

std::vector<CoreCoord> Device::ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> ethernet_cores(logical_cores.size());

    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        ethernet_cores[idx] = ethernet_core_from_logical_core(logical_cores[idx]);
    return ethernet_cores;
}

void Device::check_allocator_is_initialized() const {
    if (this->allocator_ == nullptr) {
        TT_THROW("No memory allocator! Device has not been initialized, did you forget to call InitializeDevice?");
    }
}

uint32_t Device::num_banks(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::num_banks(*this->allocator_, buffer_type);
}

uint32_t Device::bank_size(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::bank_size(*this->allocator_, buffer_type);
}

uint32_t Device::dram_channel_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::dram_channel_from_bank_id(*this->allocator_, bank_id);
}

CoreCoord Device::core_from_dram_channel(uint32_t dram_channel) const {
    TT_ASSERT(
        dram_channel < this->num_dram_channels(),
        "Bounds-Error -- dram_channel={} is outside of num_dram_channels={}",
        dram_channel,
        this->num_dram_channels()
    );
    return tt::Cluster::instance().get_soc_desc(id_).get_preferred_worker_core_for_dram_channel(dram_channel);
}

int32_t Device::l1_bank_offset_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::l1_bank_offset_from_bank_id(*this->allocator_, bank_id);
}

int32_t Device::dram_bank_offset_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::dram_bank_offset_from_bank_id(*this->allocator_, bank_id);
}

CoreCoord Device::logical_core_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::logical_core_from_bank_id(*this->allocator_, bank_id);
}

const std::vector<uint32_t> &Device::bank_ids_from_dram_channel(uint32_t dram_channel) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_dram_channel(*this->allocator_, dram_channel);
}

const std::vector<uint32_t> &Device::bank_ids_from_logical_core(const CoreCoord &logical_core) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_logical_core(*this->allocator_, logical_core);
}

allocator::Statistics Device::get_memory_allocation_statistics(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::get_statistics(*this->allocator_, buffer_type);
}

void Device::dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out) const {
    this->check_allocator_is_initialized();
    return allocator::dump_memory_blocks(*this->allocator_, buffer_type, out);
}

void Device::deallocate_buffers(){
    allocator::deallocate_buffers(*allocator_);
}

float Device::sfpu_eps() const {

  float value = std::numeric_limits<float>::epsilon();
  if( arch() == tt::ARCH::GRAYSKULL  ) {
    value = tt::tt_metal::EPS_GS;
  } else if( arch() == tt::ARCH::WORMHOLE_B0 ) {
    value = tt::tt_metal::EPS_WHB0;
  }

  return value;
}

pair<int, int> Device::build_processor_type_to_index(JitBuildProcessorType t) const {
    constexpr int DataMovementBuildCount = 2;
    constexpr int ComputeBuildCount = 3;
    constexpr int EthernetBuildCount = 1;

    switch (t) {
    case JitBuildProcessorType::DATA_MOVEMENT: return pair<int, int>(0, DataMovementBuildCount);
    case JitBuildProcessorType::COMPUTE: return pair<int, int>(DataMovementBuildCount, ComputeBuildCount);
    case JitBuildProcessorType::ETHERNET: return pair<int, int>(DataMovementBuildCount + ComputeBuildCount, EthernetBuildCount);
    default: TT_ASSERT("Bad processor type: {}", static_cast<std::underlying_type<JitBuildProcessorType>::type>(t));
    }

    // shh the warnings
    return pair<int, int>(0, 0);
}

// Ideally the firmware getter would be private to the device, however, tests look for this
const JitBuildState& Device::build_firmware_state(JitBuildProcessorType t, int i) const {
    return *(this->firmware_build_states_[build_processor_type_to_index(t).first + i]);
}

const JitBuildState& Device::build_kernel_state(JitBuildProcessorType t, int i) const {
    return *(this->kernel_build_states_[build_processor_type_to_index(t).first + i]);
}

const JitBuildStateSubset Device::build_kernel_states(JitBuildProcessorType t) const {
    pair<int, int> bptti = build_processor_type_to_index(t);
    JitBuildStateSubset subset = {
        &this->kernel_build_states_[bptti.first],
        bptti.second
    };
    return subset;
}

const string Device::build_firmware_target_path(JitBuildProcessorType t, int i) const {
    const JitBuildState& bs = build_firmware_state(t, i);
    return bs.get_target_out_path("");
}

const string Device::build_kernel_target_path(JitBuildProcessorType t, int i, const string& kernel_name) const {
    const JitBuildState& bs = build_kernel_state(t, i);
    return bs.get_target_out_path(kernel_name);
}

HWCommandQueue& Device::hw_command_queue(size_t cq_id) {
    detail::DispatchStateCheck(true);
    TT_ASSERT( cq_id < hw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *hw_command_queues_[cq_id];
}

CommandQueue& Device::command_queue(size_t cq_id) {
    detail::DispatchStateCheck(using_fast_dispatch);
    TT_ASSERT( cq_id < sw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *sw_command_queues_[cq_id];
}

bool Device::using_slow_dispatch() const {
    return not (this->using_fast_dispatch);
}
}  // namespace tt_metal

}  // namespace tt
