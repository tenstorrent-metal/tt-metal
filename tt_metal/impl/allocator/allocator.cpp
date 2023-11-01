// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/allocator/algorithms/free_list.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "third_party/magic_enum/magic_enum.hpp"

namespace tt {
namespace tt_metal {

namespace allocator {

void BankManager::init_allocator(chip_id_t device_id, const BufferType &buffer_type, uint64_t size_bytes, uint64_t offset) {
    std::string free_list_name = buffer_type == BufferType::DRAM ? concurrent::dram_mem_blocks_name(device_id) : concurrent::l1_mem_blocks_name(device_id);
    uint64_t min_allocation_size_bytes = buffer_type == BufferType::DRAM ? MIN_ALLOCATABLE_DRAM_SIZE_BYTES : MIN_ALLOCATABLE_L1_SIZE_BYTES;
    this->allocator_ = std::make_unique<FreeList>(
        free_list_name,
        size_bytes,
        offset,
        min_allocation_size_bytes,
        ADDRESS_ALIGNMENT,
        FreeList::SearchPolicy::FIRST
    );
}

void validate_num_banks(uint32_t num_banks, const BufferType &buffer_type) {
    bool is_pow2_num_banks = num_banks && (!(num_banks & (num_banks - 1)));
    // Dataflow API does not have a working implementation of generic modulo to determine bank_id for interleaved address gen
    // For non pow2 num banks, special cases need to be added to avoid falling back to generic implementation.
    // See https://github.com/tenstorrent-metal/tt-metal/issues/3321
    bool custom_mod_bank_id_calculation_exists = (num_banks == 12 or num_banks == 94);
    bool valid_num_banks = (is_pow2_num_banks or custom_mod_bank_id_calculation_exists);
    if (not valid_num_banks) {
        log_fatal(LogMetal, "Invalid number of memory banks for {}. Num banks must be power of 2 or have a dedicated modulo implementation", magic_enum::enum_name(buffer_type));
    }
}

BankManager::BankManager(chip_id_t device_id, const BufferType &buffer_type, const std::vector<int64_t> &bank_offsets, uint64_t size_bytes, uint64_t alloc_offset) : device_id_(device_id), buffer_type_(buffer_type) {
    unsigned int bank_id = 0;
    for (const auto bank_offset : bank_offsets) {
        this->bank_id_to_bank_offset_.insert({bank_id, bank_offset});
        bank_id++;
    }
    validate_num_banks(this->bank_id_to_bank_offset_.size(), this->buffer_type_);
    this->init_allocator(device_id, buffer_type, size_bytes, alloc_offset);
}

BankManager::BankManager(chip_id_t device_id, const BufferType &buffer_type, const std::unordered_map<uint32_t, int64_t> &bank_id_to_bank_offset, uint64_t size_bytes, uint64_t alloc_offset) : device_id_(device_id), buffer_type_(buffer_type), bank_id_to_bank_offset_(bank_id_to_bank_offset) {
    validate_num_banks(this->bank_id_to_bank_offset_.size(), this->buffer_type_);
    this->init_allocator(device_id, buffer_type, size_bytes, alloc_offset);
}

uint32_t BankManager::num_banks() const {
    return this->bank_id_to_bank_offset_.size();
}

uint32_t BankManager::bank_size() const {
    uint64_t max_size_bytes_u64 = this->allocator_->max_size_bytes();
    if (max_size_bytes_u64 > std::numeric_limits<uint32_t>::max()) {
        tt::log_fatal(tt::LogMetal, "Bank size {} overflows uint32_t", max_size_bytes_u64);
    }
    uint32_t max_size_bytes = (uint32_t)max_size_bytes_u64;
    return max_size_bytes;
}

int64_t BankManager::bank_offset(uint32_t bank_id) const {
    this->validate_bank_id(bank_id);
    return this->bank_id_to_bank_offset_.at(bank_id);
}

void BankManager::validate_bank_id(uint32_t bank_id) const {
    log_assert(this->bank_id_to_bank_offset_.find(bank_id) != this->bank_id_to_bank_offset_.end(), "Expected bank {} to be tracked!", bank_id);
}

uint64_t BankManager::allocate_buffer(uint32_t size, uint32_t page_size, bool bottom_up) {
    uint32_t num_banks = this->num_banks();
    // Each page needs to be at a 32B aligned address
    uint32_t size_per_bank = tt::tt_metal::detail::SizeBytesPerBank(size, page_size, num_banks);

    auto address = this->allocator_->allocate(size_per_bank, bottom_up);
    if (not address.has_value()) {
        log_fatal(tt::LogMetal, "Out of Memory: Not enough space to allocate {} B {} buffer across {} banks on device {}, where each bank needs to store {} B", size, magic_enum::enum_name(this->buffer_type_), num_banks, this->device_id_, size_per_bank);
    }
    allocated_buffers_.insert(address.value());
    return address.value();
}

void BankManager::deallocate_buffer(uint64_t address) {
    this->allocator_->deallocate(address);
}

void BankManager::deallocate_all(){
    for (uint64_t addr : this->allocated_buffers_)
    {
        this->allocator_->deallocate(addr);
    }
}


void BankManager::clear() {
    this->allocator_->clear();
}

std::optional<uint64_t> BankManager::lowest_occupied_address(uint32_t bank_id) const {
    auto lowest_address = this->allocator_->lowest_occupied_address();
    if (not lowest_address.has_value()) {
        return lowest_address;
    }
    auto adjusted_abs_addr = lowest_address.value() + this->bank_offset(bank_id);
    return adjusted_abs_addr;
}

Statistics BankManager::get_statistics() const {
    return this->allocator_->get_statistics();
}

void BankManager::dump_blocks(std::ofstream &out) const {
    this->allocator_->dump_blocks(out);
}

void init_one_bank_per_channel(Allocator &allocator, const AllocatorConfig &alloc_config) {
    // Space up to DRAM_UNRESERVED_BASE is reserved for DRAM write barrier
    uint64_t offset_bytes = static_cast<uint64_t>(DRAM_UNRESERVED_BASE);
    uint32_t dram_bank_size = alloc_config.dram_bank_size - DRAM_UNRESERVED_BASE;
    std::vector<int64_t> bank_offsets (alloc_config.num_dram_channels);
    for (uint32_t channel_id = 0; channel_id < alloc_config.num_dram_channels; channel_id++) {
        bank_offsets.at(channel_id) = static_cast<int32_t>(alloc_config.dram_bank_offsets.at(channel_id));
    }
    allocator.dram_manager = BankManager(alloc_config.device_id, BufferType::DRAM, bank_offsets, dram_bank_size, offset_bytes);
    for (uint32_t bank_id = 0; bank_id < alloc_config.num_dram_channels; bank_id++) {
        allocator.bank_id_to_dram_channel.insert({bank_id, bank_id});
        allocator.dram_channel_to_bank_ids.insert({bank_id, {bank_id}});
    }
}

void init_one_bank_per_l1(Allocator &allocator, const AllocatorConfig &alloc_config) {
    uint32_t num_l1_banks = alloc_config.worker_grid_size.y * alloc_config.worker_grid_size.x;
    // Space up to L1_UNRESERVED_BASE is reserved for risc binaries, kernel args, debug and perf monitoring tools
    uint64_t offset_bytes = static_cast<uint64_t>(L1_UNRESERVED_BASE);
    uint32_t l1_bank_size = alloc_config.worker_l1_size - L1_UNRESERVED_BASE;
    std::vector<int64_t> bank_offsets (num_l1_banks, 0);
    allocator.l1_manager = BankManager(alloc_config.device_id, BufferType::L1, bank_offsets, l1_bank_size, offset_bytes);

    uint32_t bank_id = 0;
    for (uint32_t y = 0; y < alloc_config.worker_grid_size.y; y++) {
        for (uint32_t x = 0; x < alloc_config.worker_grid_size.x; x++) {
            CoreCoord logical_core = CoreCoord{x, y};
            allocator.bank_id_to_logical_core.insert({bank_id, logical_core});
            allocator.logical_core_to_bank_ids.insert({logical_core, {bank_id}});
            bank_id++;
        }
    }
}

uint32_t num_banks(const Allocator &allocator, const BufferType &buffer_type) {
    switch (buffer_type) {
        case BufferType::DRAM: return allocator.dram_manager.num_banks();
        case BufferType::L1: return allocator.l1_manager.num_banks();
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
    return 0;
}

uint32_t bank_size(const Allocator &allocator, const BufferType &buffer_type) {
    switch (buffer_type) {
        case BufferType::DRAM: return allocator.dram_manager.bank_size();
        case BufferType::L1: return allocator.l1_manager.bank_size();
        default: {
            log_fatal(tt::LogMetal, "Unsupported buffer type!");
        }
    }
    return 0;
}

uint32_t dram_channel_from_bank_id(const Allocator &allocator, uint32_t bank_id) {
    TT_ASSERT(allocator.bank_id_to_dram_channel.find(bank_id) != allocator.bank_id_to_dram_channel.end());
    return allocator.bank_id_to_dram_channel.at(bank_id);
}

CoreCoord logical_core_from_bank_id(const Allocator &allocator, uint32_t bank_id) {
    TT_ASSERT(allocator.bank_id_to_logical_core.find(bank_id) != allocator.bank_id_to_logical_core.end());
    return allocator.bank_id_to_logical_core.at(bank_id);
}

int32_t l1_bank_offset_from_bank_id(const Allocator &allocator, uint32_t bank_id) {
    return allocator.l1_manager.bank_offset(bank_id);
}

int32_t dram_bank_offset_from_bank_id(const Allocator &allocator, uint32_t bank_id) {
    return allocator.dram_manager.bank_offset(bank_id);
}

std::vector<uint32_t> bank_ids_from_dram_channel(const Allocator &allocator, uint32_t dram_channel) {
    if (allocator.dram_channel_to_bank_ids.find(dram_channel) == allocator.dram_channel_to_bank_ids.end()) {
        log_fatal(LogMetal, "No DRAM bank exists for DRAM channel {}", dram_channel);
    }
    return allocator.dram_channel_to_bank_ids.at(dram_channel);
}

std::vector<uint32_t> bank_ids_from_logical_core(const Allocator &allocator, const CoreCoord &logical_core) {
    if (allocator.logical_core_to_bank_ids.find(logical_core) == allocator.logical_core_to_bank_ids.end()) {
        log_fatal(LogMetal, "No L1 bank exists for core {}", logical_core.str());
    }
    return allocator.logical_core_to_bank_ids.at(logical_core);
}

Statistics get_statistics(const Allocator &allocator, const BufferType &buffer_type) {
    Statistics stats;
    switch (buffer_type) {
        case BufferType::DRAM: return allocator.dram_manager.get_statistics();
        case BufferType::L1: return allocator.l1_manager.get_statistics();
        default: {
            log_assert(false, "Unsupported buffer type!");
        }
    }
    return stats;
}

void dump_memory_blocks(const Allocator &allocator, const BufferType &buffer_type, std::ofstream &out) {
    switch (buffer_type) {
        case BufferType::DRAM: allocator.dram_manager.dump_blocks(out);
        break;
        case BufferType::L1: allocator.l1_manager.dump_blocks(out);
        break;
        default: {
            log_assert(false, "Unsupported buffer type!");
        }
    }
}

std::optional<uint64_t> lowest_occupied_l1_address(const Allocator &allocator, uint32_t bank_id) {
    return allocator.l1_manager.lowest_occupied_address(bank_id);
}

uint64_t base_alloc(const AllocatorConfig &config, BankManager &bank_manager, uint64_t size, uint64_t page_size, bool bottom_up) {
    return bank_manager.allocate_buffer(size, page_size, bottom_up);
}

uint64_t allocate_buffer(Allocator &allocator, uint32_t size, uint32_t page_size, const BufferType &buffer_type, bool bottom_up) {
    uint64_t address = 0;
    switch (buffer_type) {
        case BufferType::DRAM: return allocator.descriptor.dram.alloc(allocator.config, allocator.dram_manager, size, page_size, bottom_up);
        case BufferType::L1: return allocator.descriptor.l1.alloc(allocator.config, allocator.l1_manager, size, page_size, bottom_up);
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
    return address;
}

void deallocate_buffer(Allocator &allocator, uint64_t address, const BufferType &buffer_type) {
    switch (buffer_type) {
        case BufferType::DRAM:
            allocator.dram_manager.deallocate_buffer(address);
        break;
        case BufferType::L1:
            allocator.l1_manager.deallocate_buffer(address);
        break;
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
}

void deallocate_buffers(Allocator &allocator) {
    allocator.dram_manager.deallocate_all();
    allocator.l1_manager.deallocate_all();
}

void clear(Allocator &allocator) {
    allocator.dram_manager.clear();
    allocator.l1_manager.clear();
}

const AllocDescriptor generate_allocator_descriptor(const MemoryAllocator &memory_scheme) {
    AllocDescriptor descriptor;
    switch (memory_scheme) {
        case MemoryAllocator::BASIC: {
            descriptor.dram = InitAndAllocFuncs{.init = init_one_bank_per_channel, .alloc = base_alloc};
            descriptor.l1 = InitAndAllocFuncs{.init = init_one_bank_per_l1, .alloc = base_alloc};
        }
        break;
        case MemoryAllocator::L1_BANKING: {
            descriptor.dram = InitAndAllocFuncs{.init = init_one_bank_per_channel, .alloc = base_alloc};
            descriptor.l1 = InitAndAllocFuncs{.init = init_compute_and_storage_l1_bank_manager, .alloc = base_alloc};
        }
        break;
        default:
            log_fatal("Cannot generate allocator descriptor for unsupported memory allocator scheme");
    }
    return descriptor;
}

}  // namespace allocator

Allocator::Allocator(const AllocatorConfig &alloc_config, const allocator::AllocDescriptor &alloc_descriptor) : config(alloc_config), descriptor(alloc_descriptor) {
    // TODO: add validation for allocator_descriptor?
    this->descriptor.dram.init(*this, alloc_config);
    this->descriptor.l1.init(*this, alloc_config);
    // assert that bank managers have been initialized?
    TT_ASSERT(not bank_id_to_dram_channel.empty() and not dram_channel_to_bank_ids.empty());
    TT_ASSERT(not bank_id_to_logical_core.empty() and not bank_id_to_logical_core.empty());
}

}  // namespace tt_metal

}  // namespace tt
