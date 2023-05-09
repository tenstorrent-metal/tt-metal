#include <array>
#include <vector>

using std::array;
using std::vector;

typedef uint32_t u32;
typedef uint64_t u64;

constexpr u32 NR = 100;
constexpr u32 NW = 100;
constexpr u32 NIDR = 100;
constexpr u32 NILR = 100;
constexpr u32 NIDW = 100;
constexpr u32 NILW = 100;

constexpr u32 SIZE = 1000;

struct DeviceCommand {
   private:
    u32 read_ptr;
    u32 write_ptr;
    u32 interleaved_dram_read_ptr;
    u32 interleaved_l1_read_ptr;
    u32 interleaved_dram_write_ptr;
    u32 interleaved_l1_write_ptr;
    u32 launch_ptr;

    constexpr u32 size() { return (NR + 1) + (NW + 1) + (NIDR + 1) + (NILR + 1) + (NIDW + 1) + (NILW + 1) + 1; }

    array<u32, SIZE> desc;

   public:
    DeviceCommand() {
        u32 read_section_num_entries = (NR * 4 + 1);
        u32 write_section_num_entries = (NW * 4 + 1);
        u32 interleaved_dram_read_num_entries = (NIDR * 5 + 1);
        u32 interleaved_l1_read_num_entries = (NILR * 5 + 1);
        u32 interleaved_dram_write_num_entries = (NIDW * 5 + 1);
        u32 interleaved_l1_write_num_entries = (NILW * 5 + 1);

        for (u32 i = 0; i < this->size(); i++) {
            this->desc[i] = 0;
        }

        this->read_ptr = 0;
        this->write_ptr = read_section_num_entries;
        this->interleaved_dram_read_ptr = read_section_num_entries + write_section_num_entries;
        this->interleaved_l1_read_ptr =
            read_section_num_entries + write_section_num_entries + interleaved_dram_read_num_entries;
        this->interleaved_dram_write_ptr = read_section_num_entries + write_section_num_entries +
                                           interleaved_dram_read_num_entries + interleaved_l1_read_num_entries;
        this->interleaved_l1_write_ptr = read_section_num_entries + write_section_num_entries +
                                         interleaved_dram_read_num_entries + interleaved_l1_read_num_entries +
                                         interleaved_dram_write_num_entries;
        this->launch_ptr = read_section_num_entries + write_section_num_entries + interleaved_dram_read_num_entries +
                           interleaved_l1_read_num_entries + interleaved_dram_write_num_entries +
                           interleaved_l1_write_num_entries;
    }

    void add_read(u32 src, u32 src_noc, u32 dst, u32 bytes) {
        this->desc[this->read_ptr] = src;
        this->desc[this->read_ptr + 1] = src_noc;
        this->desc[this->read_ptr + 2] = dst;
        this->desc[this->read_ptr + 3] = bytes;
        this->read_ptr += 4;
    }

    void add_write(u32 src, u32 dst, u32 dst_noc, u32 bytes) {
        this->desc[this->write_ptr] = src;
        this->desc[this->write_ptr + 1] = dst;
        this->desc[this->write_ptr + 2] = dst_noc;
        this->desc[this->write_ptr + 3] = bytes;
        this->write_ptr += 4;
    }

    void add_interleaved_dram_read(u32 src, u32 src_noc, u32 dst, uint num_bytes_per_page, uint num_pages) {
        this->desc[this->interleaved_dram_read_ptr] = src;
        this->desc[this->interleaved_dram_read_ptr + 1] = src_noc;
        this->desc[this->interleaved_dram_read_ptr + 2] = dst;
        this->desc[this->interleaved_dram_read_ptr + 3] = num_bytes_per_page;
        this->desc[this->interleaved_dram_read_ptr + 4] = num_pages;
        this->interleaved_dram_read_ptr += 5;
    }

    void add_interleaved_l1_read(u32 src, u32 src_noc, u32 dst, uint num_bytes_per_page, uint num_pages) {
        this->desc[this->interleaved_l1_read_ptr] = src;
        this->desc[this->interleaved_l1_read_ptr + 1] = src_noc;
        this->desc[this->interleaved_l1_read_ptr + 2] = dst;
        this->desc[this->interleaved_l1_read_ptr + 3] = num_bytes_per_page;
        this->desc[this->interleaved_l1_read_ptr + 4] = num_pages;
        this->interleaved_l1_read_ptr += 5;
    }

    void add_interleaved_dram_write(u32 src, u32 dst, u32 dst_noc, uint num_bytes_per_page, uint num_pages) {
        this->desc[this->interleaved_dram_write_ptr] = src;
        this->desc[this->interleaved_dram_write_ptr + 1] = dst;
        this->desc[this->interleaved_dram_write_ptr + 2] = dst_noc;
        this->desc[this->interleaved_dram_write_ptr + 3] = num_bytes_per_page;
        this->desc[this->interleaved_dram_write_ptr + 4] = num_pages;
        this->interleaved_dram_write_ptr += 5;
    }

    void add_interleaved_l1_write(u32 src, u32 dst, u32 dst_noc, uint num_bytes_per_page, uint num_pages) {
        this->desc[this->interleaved_l1_write_ptr] = src;
        this->desc[this->interleaved_l1_write_ptr + 1] = dst;
        this->desc[this->interleaved_l1_write_ptr + 2] = dst_noc;
        this->desc[this->interleaved_l1_write_ptr + 3] = num_bytes_per_page;
        this->desc[this->interleaved_l1_write_ptr + 4] = num_pages;
        this->interleaved_l1_write_ptr += 5;
    }

    void launch_kernels() { this->desc[this->launch_ptr] = 1; }

    const array<u32, SIZE>& get_desc() const { return this->desc; }
};
