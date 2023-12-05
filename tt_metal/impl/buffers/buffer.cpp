// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/common/math.hpp"
#include "common/assert.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

void validate_buffer_size_and_page_size(uint64_t size, uint64_t page_size, const BufferType &buffer_type) {
    TT_FATAL(size != 0 and page_size != 0, "Buffer size and page size should be larger than 0 bytes!");
    bool valid_page_size = (size % page_size == 0);
    TT_FATAL(valid_page_size, "For valid non-interleaved buffers page size {} must equal buffer size {}. For interleaved-buffers page size should be divisible by buffer size", page_size, size);
    TT_FATAL(page_size % sizeof(uint32_t) == 0, "Page size must be divisible by sizeof(uint32_t) because buffers hold uint32_t values");
}

Buffer::Buffer(Device *device, uint64_t size, uint64_t page_size, const BufferType buffer_type)
    : device_(device), size_(size), page_size_(page_size), buffer_type_(buffer_type) {
    TT_FATAL(this->device_ != nullptr, "Valid device required for Buffer construction!");
    validate_buffer_size_and_page_size(size, page_size, buffer_type);
    this->allocate();
}

Buffer::Buffer(const Buffer &other)
    : device_(other.device_), size_(other.size_), page_size_(other.page_size_), buffer_type_(other.buffer_type_) {
    this->allocate();
}

Buffer &Buffer::operator=(const Buffer &other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->size_ = other.size_;
        this->page_size_ = other.page_size_;
        this->buffer_type_ = other.buffer_type_;
        this->allocate();
    }
    return *this;
}

Buffer::Buffer(Buffer &&other) : device_(other.device_), size_(other.size_), address_(other.address_), page_size_(other.page_size_), buffer_type_(other.buffer_type_) {
    // Set `other.device_` to be nullptr so destroying other does not deallocate reserved address space that is transferred to `this`
    other.device_ = nullptr;
}

Buffer &Buffer::operator=(Buffer &&other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->size_ = other.size_;
        this->address_ = other.address_;
        this->page_size_ = other.page_size_;
        this->buffer_type_ = other.buffer_type_;
        // Set `other.device_` to be nullptr so destroying other does not deallocate reserved address space that is transferred to `this`
        other.device_ = nullptr;
    }
    return *this;
}

void Buffer::allocate() {
    TT_ASSERT(this->device_ != nullptr);
    // L1 buffers are allocated top down!
    bool bottom_up = this->buffer_type_ == BufferType::DRAM;
    this->address_ = allocator::allocate_buffer(detail::GetAllocator(this->device_), this->size_, this->page_size_, this->buffer_type_, bottom_up);
}

uint32_t Buffer::dram_channel_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(this->buffer_type_ == BufferType::DRAM, "Expected DRAM buffer!");
    return this->device_->dram_channel_from_bank_id(bank_id);
}

CoreCoord Buffer::logical_core_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(this->buffer_type_ == BufferType::L1, "Expected L1 buffer!");
    return this->device_->logical_core_from_bank_id(bank_id);
}

CoreCoord Buffer::noc_coordinates(uint32_t bank_id) const {
    switch (this->buffer_type_) {
        case BufferType::DRAM: {
            auto dram_channel = this->dram_channel_from_bank_id(bank_id);
            return llrt::get_core_for_dram_channel(dram_channel, this->device_->id());
        }
        case BufferType::L1: {
            auto logical_core = this->logical_core_from_bank_id(bank_id);
            return this->device_->worker_core_from_logical_core(logical_core);
        }
        break;
        case BufferType::SYSTEM_MEMORY: {
            TT_THROW("Host buffer is located in system memory! Cannot retrieve NoC coordinates for it");
        }
        break;
        default:
            TT_ASSERT(false && "Unsupported buffer type!");
    }
    return CoreCoord{.x=0, .y=0};
}

CoreCoord Buffer::noc_coordinates() const {
    return this->noc_coordinates(0);
}

uint64_t Buffer::page_address(uint32_t bank_id, uint32_t page_index) const {
    auto num_banks = this->device_->num_banks(this->buffer_type_);
    TT_ASSERT(bank_id < num_banks, "Invalid Bank ID: {} exceeds total numbers of banks ({})!", bank_id, num_banks);

    // DRAM readers and writers in Cluster add DRAM bank offset before doing a read but L1 readers and writers do not
    uint64_t base_page_address = this->buffer_type_ == BufferType::DRAM ?
        this->address_ :
        this->address_ + this->device_->l1_bank_offset_from_bank_id(bank_id);

    int pages_handled_in_bank = (int)page_index / num_banks;
    auto offset = (round_up(this->page_size_, ADDRESS_ALIGNMENT) * pages_handled_in_bank);
    return base_page_address + offset;
}

void Buffer::deallocate() {
    if (this->device_ == nullptr or not this->device_->initialized_) {
        return;
    }
    this->size_ = 0;
    allocator::deallocate_buffer(detail::GetAllocator(this->device_), this->address_, this->buffer_type_);
}

Buffer::~Buffer() {
    this->deallocate();
}

}  // namespace tt_metal

}  // namespace tt
