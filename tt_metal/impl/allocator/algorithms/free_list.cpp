// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/allocator/algorithms/free_list.hpp"
#include "common/assert.hpp"

#include <algorithm>
#include <cmath>

namespace tt {

namespace tt_metal {

namespace allocator {

FreeList::FreeList(uint64_t max_size_bytes, uint64_t offset_bytes, uint64_t min_allocation_size, uint64_t alignment, FreeList::SearchPolicy search_policy)
    : search_policy_(search_policy), Algorithm(max_size_bytes, offset_bytes, min_allocation_size, alignment) {
    this->init();
}

void FreeList::init() {
    this->block_head_.reset(  new FreeList::Block{.address = 0, .size = this->max_size_bytes_} );
    this->block_tail_ = block_head_;
    this->free_block_head_ = block_head_;
    this->free_block_tail_ = block_head_;
}

bool FreeList::is_allocated(const Block *block) const {
    return (block->prev_free == nullptr) and (block->next_free == nullptr and block != this->free_block_head_.get()) and (block != this->free_block_tail_.get());
}

std::vector<std::pair<uint64_t, uint64_t>> FreeList::available_addresses(uint64_t size_bytes) const {
    uint64_t alloc_size = size_bytes < this->min_allocation_size_ ? this->min_allocation_size_ : size_bytes;
    alloc_size = this->align(alloc_size);
    std::vector<std::pair<uint64_t, uint64_t>> addresses;
    std::shared_ptr<Block> curr_block = this->free_block_head_;
    while (curr_block != nullptr) {
        if (curr_block->size >= alloc_size) {
            uint64_t end_range = (curr_block->address + curr_block->size) - alloc_size;
            addresses.push_back({curr_block->address, end_range});
        }
        curr_block = curr_block->next_free;
    }
    return addresses;
}

/**
 std::shared_ptr<FreeList::Block> FreeList::search_best(uint64_t size_bytes, bool bottom_up) {
    std::shared_ptr<FreeList::Block> best_block;
    std::shared_ptr<FreeList::Block> curr_block = bottom_up ? this->free_block_head_ : this->free_block_tail_;
    while (curr_block != nullptr) {
        if (curr_block->size == size_bytes) {
            best_block = curr_block;
            return std::move(best_block);
        }
        curr_block = bottom_up ? curr_block->next_free : curr_block->prev_free;
    }
    //best search fail over to search first
    return search_first(size_bytes,bottom_up);
}
*/
FreeList::Block *FreeList::search_best(uint64_t size_bytes, bool bottom_up) {
    FreeList::Block *best_block = nullptr;
    FreeList::Block *curr_block = bottom_up ? this->free_block_head_ : this->free_block_tail_;
    while (curr_block != nullptr) {
        if (curr_block->size == size_bytes) {
            best_block = curr_block;
            break;
        } else if (curr_block->size >= size_bytes) {
            if (best_block == nullptr or curr_block->size < best_block->size) {
                best_block = curr_block;
            }
        }
        curr_block = bottom_up ? curr_block->next_free : curr_block->prev_free;
    }

    return best_block;
}

std::shared_ptr<FreeList::Block> FreeList::search_first(uint64_t size_bytes, bool bottom_up) {
    std::shared_ptr<FreeList::Block> curr_block = bottom_up ? this->free_block_head_ : this->free_block_tail_;
    while (curr_block != nullptr) {
        if (curr_block->size >= size_bytes) {
            return std::move(curr_block);
        }
        curr_block = bottom_up ? curr_block->next_free : curr_block->prev_free;
    }

    return std::shared_ptr<FreeList::Block>{};
}

std::shared_ptr<FreeList::Block> FreeList::search(uint64_t size_bytes, bool bottom_up) {
    switch (this->search_policy_) {
        case FreeList::SearchPolicy::BEST:
            return std::move(search_best(size_bytes, bottom_up));
        break;
        case FreeList::SearchPolicy::FIRST:
            return std::move(search_first(size_bytes, bottom_up));
        break;
        default:
            TT_ASSERT(false && "Unsupported search policy");
    }
    return std::shared_ptr<FreeList::Block>{};
}

void FreeList::allocate_entire_free_block(std::shared_ptr<FreeList::Block>& free_block_to_allocate) {
    TT_ASSERT(not is_allocated(free_block_to_allocate.get()));
    if (free_block_to_allocate->prev_free != nullptr) {
        free_block_to_allocate->prev_free->next_free = free_block_to_allocate->next_free;
    }
    if (free_block_to_allocate->next_free != nullptr) {
        free_block_to_allocate->next_free->prev_free = free_block_to_allocate->prev_free;
    }
    if (free_block_to_allocate == this->free_block_head_) {
        if (free_block_to_allocate->next_free == nullptr) {
            this->free_block_head_ = nullptr;
        } else {
            this->free_block_head_ = free_block_to_allocate->next_free;
        }
    }
    if (free_block_to_allocate == this->free_block_tail_) {
        if (free_block_to_allocate->prev_free == nullptr) {
            this->free_block_tail_ = nullptr;
        } else {
            this->free_block_tail_ = free_block_to_allocate->prev_free;
        }
    }
    free_block_to_allocate->prev_free = nullptr;
    free_block_to_allocate->next_free = nullptr;
}

// free_block range: [a, b)
// allocated_block range: [a, c), where c < b
void FreeList::update_left_aligned_allocated_block_connections(std::shared_ptr<FreeList::Block>& free_block,
                                                            std::shared_ptr<FreeList::Block>& allocated_block) {
    allocated_block->prev_block = free_block->prev_block;
    allocated_block->next_block = free_block;
    if (free_block->prev_block != nullptr) {
        free_block->prev_block->next_block = allocated_block;
    }
    if (free_block == this->block_head_) {
        this->block_head_ = allocated_block ;
    }
    // next_free and prev_free connections of free_block are still valid
    free_block->prev_block = allocated_block;
    free_block->address = allocated_block->address + allocated_block->size;
    free_block->size -= allocated_block->size;
}

// free_block range: [a, b)
// allocated_block range: [c, b), where c > a
void FreeList::update_right_aligned_allocated_block_connections(std::shared_ptr<FreeList::Block>& free_block, std::shared_ptr<FreeList::Block>& allocated_block) {
    allocated_block->prev_block = free_block;
    allocated_block->next_block = free_block->next_block;
    if (free_block->next_block != nullptr) {
        free_block->next_block->prev_block = allocated_block;
    }
    if (free_block == this->block_tail_) {
        this->block_tail_ = allocated_block;
    }
    // next_free and prev_free connections of free_block are still valid
    free_block->next_block = allocated_block;
    free_block->size -= allocated_block->size;
}

// Offset marks the start of the allocated block
std::shared_ptr<FreeList::Block> FreeList::allocate_slice_of_free_block(std::shared_ptr<FreeList::Block>& free_block, uint64_t offset, uint64_t size_bytes) {
    TT_ASSERT(free_block->address + offset + size_bytes <= free_block->address + free_block->size);

    // Allocated slice spans the entire space of free_block
    if (offset == 0 and size_bytes == free_block->size) {
        this->allocate_entire_free_block(free_block);
        return free_block;
    }

    std::shared_ptr<FreeList::Block> allocated_block(new FreeList::Block{
        .address = free_block->address + offset,
        .size = size_bytes,
    });

    // Allocated slice takes up a portion of free_block, three cases to consider:
    // 1. allocated_block is left aligned with free_block with free space remaining on the right
    // 2. allocated_block is right aligned with free_block with free space remaining on the left
    // 3. allocated_block is in the middle of free_block with free space on left and right sides
    bool case_one = offset == 0 and size_bytes < free_block->size;
    bool case_two = offset > 0 and ((free_block->address + offset + size_bytes) == (free_block->address + free_block->size));
    bool case_three = offset > 0 and ((free_block->address + offset + size_bytes) < (free_block->address + free_block->size));
    TT_ASSERT((int)(case_one + case_two + case_three) == 1);

    if (case_one) {
        this->update_left_aligned_allocated_block_connections(free_block, allocated_block);
    } else if (case_two) {
        this->update_right_aligned_allocated_block_connections(free_block, allocated_block);
    } else {
        TT_ASSERT(case_three);
        // Original: | .................... free_block ....................|
        // Result:   | free_block_mod | allocated_block | next_free_block  |
        uint64_t next_free_block_addr = free_block->address + offset + size_bytes;
        uint64_t next_free_block_size = (free_block->address + free_block->size) - next_free_block_addr;
        std::shared_ptr<FreeList::Block> next_free_block(new Block{
            .address = next_free_block_addr,
            .size = next_free_block_size,
            .prev_block = allocated_block,
            .next_block = free_block->next_block,
            .prev_free = free_block,
            .next_free = free_block->next_free});
        if (free_block->next_block != nullptr) {
            free_block->next_block->prev_block = next_free_block;
        }
        if (free_block->next_free != nullptr) {
            free_block->next_free->prev_free = next_free_block;
        }
        if (this->free_block_tail_ == free_block) {
            this->free_block_tail_ = next_free_block ;
        }
        if (this->block_tail_ == free_block) {
            this->block_tail_ = next_free_block ;
        }
        free_block->next_free = next_free_block;
        free_block->next_block = allocated_block;

        allocated_block->prev_block = free_block;
        allocated_block->next_block = next_free_block;

        free_block->size -= (allocated_block->size + next_free_block->size);
    }

    return std::move(allocated_block);
}

void FreeList::update_lowest_occupied_address(uint64_t address) {
    if (not this->lowest_occupied_address_.has_value()) {
        this->lowest_occupied_address_ = address;
    } else {
        this->lowest_occupied_address_ = std::min(this->lowest_occupied_address_.value(), address);
    }
}

std::optional<uint64_t> FreeList::allocate(uint64_t size_bytes, bool bottom_up, uint64_t address_limit) {
    uint64_t alloc_size = size_bytes < this->min_allocation_size_ ? this->min_allocation_size_ : size_bytes;
    alloc_size = this->align(alloc_size);
    auto free_block = search(alloc_size, bottom_up);

    if (free_block == nullptr) {
        return std::nullopt;
    }

    // offset denotes where allocation starts relative to free_block start
    uint64_t offset = bottom_up ? 0 : (((free_block->address + free_block->size) - alloc_size) - free_block->address);
    auto allocated_block = allocate_slice_of_free_block(free_block, offset, alloc_size);

    this->update_lowest_occupied_address(allocated_block->address);
    if (allocated_block->address + this->offset_bytes_ < address_limit) {
        TT_THROW("Out of Memory: Cannot allocate at an address below {}", address_limit);
    }
    return allocated_block->address + this->offset_bytes_;
}

std::optional<uint64_t> FreeList::allocate_at_address(uint64_t absolute_start_address, uint64_t size_bytes) {
    TT_ASSERT(absolute_start_address % this->alignment_ == 0, "Requested address " + std::to_string(absolute_start_address) + " should be " + std::to_string(this->alignment_) + "B aligned");
    auto start_address = absolute_start_address - this->offset_bytes_;
    auto& curr_block = this->free_block_head_;
    uint64_t alloc_size = size_bytes < this->min_allocation_size_ ? this->min_allocation_size_ : size_bytes;
    alloc_size = this->align(alloc_size);
    // Look for a free block of size at least size_bytes that encompasses start_address
    while (curr_block != nullptr) {
        if (curr_block->size >= alloc_size) {
            if (curr_block->address == start_address) {
                allocate_slice_of_free_block(curr_block, /*offset=*/0, alloc_size);
                break;
            } else if ((start_address > curr_block->address) and ((start_address + alloc_size) <= (curr_block->address + curr_block->size))) {
                uint64_t start_offset = start_address - curr_block->address;
                allocate_slice_of_free_block(curr_block, start_offset, alloc_size);
                break;
            }
        }
        curr_block = curr_block->next_free;
    }

    if (curr_block == nullptr) {
        return std::nullopt;
    }
    this->update_lowest_occupied_address(start_address);
    return absolute_start_address;
}

std::shared_ptr<FreeList::Block> FreeList::find_block(uint64_t address) {
    auto curr_block = this->block_head_;
    while (curr_block != nullptr) {
        if (curr_block->address == address) {
            return curr_block;
        }
        curr_block = curr_block->next_block;
    }
    return std::shared_ptr<FreeList::Block>(nullptr);
}

void FreeList::update_lowest_occupied_address() {
    auto& block = this->block_head_;
    while (block != nullptr) {
        if (this->is_allocated(block.get())) {
            break;
        }
        block = block->next_block;
    }
    if (block == nullptr) {
        this->lowest_occupied_address_ = std::nullopt;
    } else {
        this->lowest_occupied_address_ = block->address;
    }
}

void FreeList::deallocate(uint64_t absolute_address) {
    uint64_t address = absolute_address - this->offset_bytes_;
    auto block_to_free = find_block(address);
    if (block_to_free == nullptr or not this->is_allocated(block_to_free.get())) {
        return;
    }

    auto prev = block_to_free->prev_block;
    auto next = block_to_free->next_block;

    bool merged_prev = false;
    bool merged_next = false;
    if (prev != nullptr and not is_allocated(prev.get())) {
        prev->next_block = block_to_free->next_block;
        if (block_to_free->next_block != nullptr) {
            block_to_free->next_block->prev_block = prev;
        }
        prev->size += block_to_free->size;
        block_to_free = prev;
        merged_prev = true;
    }

    if (next != nullptr and not is_allocated(next.get())) {
        block_to_free->next_block = next->next_block;
        if (next->next_block != nullptr) {
            next->next_block->prev_block = block_to_free;
        }
        if (next == this->free_block_head_) {
            this->free_block_head_ = block_to_free ;
        }
        if (next == this->free_block_tail_) {
            this->free_block_tail_ = block_to_free ;
        }
        block_to_free->next_free = next->next_free;
        if (next->next_free != nullptr) {
            next->next_free->prev_free = block_to_free;
        }
        if (not merged_prev) {
            block_to_free->prev_free = next->prev_free;
            if (next->prev_free != nullptr) {
                next->prev_free->next_free = block_to_free;
            }
        }
        block_to_free->size += next->size;
        merged_next = true;
    }

    if (not merged_prev and not merged_next) {
        // Find where to include deallocated block in free list
        auto prev_free_block = block_to_free->prev_block;
        while (prev_free_block != nullptr and is_allocated(prev_free_block.get())) {
            prev_free_block = prev_free_block->prev_block;
        }
        auto next_free_block = block_to_free->next_block;
        while (next_free_block != nullptr and is_allocated(next_free_block.get())) {
            next_free_block = next_free_block->next_block;
        }
        block_to_free->prev_free = prev_free_block;
        if (prev_free_block != nullptr) {
            prev_free_block->next_free = block_to_free;
        } else {
            this->free_block_head_ = block_to_free;
        }

        block_to_free->next_free = next_free_block;
        if (next_free_block != nullptr) {
            next_free_block->prev_free = block_to_free ;
        } else {
            this->free_block_tail_ = block_to_free ;
        }
    }

    if (address == this->lowest_occupied_address_) {
        this->update_lowest_occupied_address();
    }
}

void FreeList::reset() {
    std::shared_ptr<Block> curr_block = this->block_head_;
    std::shared_ptr<Block> next;
    while (curr_block != nullptr) {
        next = curr_block->next_block;
        curr_block.reset();
        curr_block = next;
    }
    this->block_head_ = nullptr;
    this->free_block_head_ = nullptr;
}

void FreeList::clear() {
    this->reset();
    this->init();
}

Statistics FreeList::get_statistics() const {
    Statistics stats{
        .total_allocatable_size_bytes = this->max_size_bytes_,
        .total_allocated_bytes = 0,
        .total_free_bytes = 0,
        .largest_free_block_bytes = 0
    };

    Block *curr_block = this->block_head_.get();
    while (curr_block != nullptr) {
        if (this->is_allocated(curr_block)) {
            stats.total_allocated_bytes += curr_block->size;
        } else {
            stats.total_free_bytes += curr_block->size;
            if (curr_block->size >= stats.largest_free_block_bytes) {
                stats.largest_free_block_bytes = curr_block->size;
                stats.largest_free_block_addrs.push_back(curr_block->address + this->offset_bytes_);
            }
        }
        curr_block = curr_block->next_block.get();
    }
    if (stats.total_allocated_bytes == 0) {
        stats.total_free_bytes = this->max_size_bytes_;
        stats.largest_free_block_bytes = this->max_size_bytes_;
    }
    return stats;
}

FreeList::~FreeList() {
    // this->block_head_ and this->free_block_head_ are reset
    this->reset();
    this->block_head_.reset();
    this->free_block_head_.reset();
    this->block_tail_.reset();
    this->free_block_tail_.reset();
}

void FreeList::dump_block(const Block *block, std::ofstream &out) const {
    auto alloc_status = this->is_allocated(block) ? "Y" : "N";
    out << ",,,Address (KB):," << (block->address + this->offset_bytes_) / 1024 << "\n"
        << ",,,Size (KB):," << (block->size) / 1024 << "\n"
        << ",,,Allocated (Y/N):," << alloc_status << "\n";
}

void FreeList::dump_blocks(std::ofstream &out) const {
    out << ",,Blocks:\n";
    Block *curr_block = this->block_head_.get();
    while (curr_block != nullptr) {
        this->dump_block(curr_block, out);
        curr_block = curr_block->next_block.get();
    }
    out << "\n";
}

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
