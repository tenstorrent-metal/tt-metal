/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/common/math.hpp"

namespace tt::tt_metal::detail{

    /**
     * Returns tile size of given data format in bytes
     *
     * Return value: uint32_t
     *
     * | Argument    | Description    | Type                | Valid Range | Required |
     * |-------------|----------------|---------------------|-------------|----------|
     * | data_format | Format of data | tt::DataFormat enum |             | Yes      |
     */
    inline uint32_t TileSize(const DataFormat &data_format)
    {
        return tt::tile_size(data_format);
    }

    inline uint32_t SizeBytesPerBank(uint32_t size_bytes, uint32_t page_size_bytes, uint32_t num_banks) {
        std::cout << "Page_size_bytes " << page_size_bytes << " size bytes " << size_bytes << std::endl;
        TT_ASSERT(page_size_bytes > 0 and size_bytes % page_size_bytes == 0, "Page size {} should be divisible by buffer size {}", page_size_bytes, size_bytes);
        uint32_t num_pages = size_bytes / page_size_bytes;
        int num_equally_distributed_pages = num_pages == 1 ? 1 : 1 + ((num_pages - 1) / num_banks);
        return num_equally_distributed_pages * round_up(page_size_bytes, ADDRESS_ALIGNMENT);
    }

}
