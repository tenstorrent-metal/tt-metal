#include <array>
#include <vector>

using std::array;
using std::vector;

typedef uint32_t u32;
typedef uint64_t u64;

static constexpr u32 NR = 100;  // Allowing up to 100 reads and writes
static constexpr u32 NW = 100;
static constexpr u32 NIDR = 1;  // We only use these on data relays, in which we only do one relay per command
static constexpr u32 NILR = 1;
static constexpr u32 NIDW = 1;
static constexpr u32 NILW = 1;

static constexpr u32 read_section_num_entries = (NR * 4 + 1);
static constexpr u32 write_section_num_entries = (NW * 4 + 1);
static constexpr u32 interleaved_dram_read_num_entries = (NIDR * 5 + 1);
static constexpr u32 interleaved_l1_read_num_entries = (NILR * 5 + 1);
static constexpr u32 interleaved_dram_write_num_entries = (NIDW * 5 + 1);
static constexpr u32 interleaved_l1_write_num_entries = (NILW * 5 + 1);

static constexpr u32 SIZE = read_section_num_entries + write_section_num_entries + interleaved_dram_read_num_entries +
                     interleaved_l1_read_num_entries + interleaved_dram_write_num_entries +
                     interleaved_l1_write_num_entries;

struct DeviceCommand {
   private:
    u32 read_ptr;
    u32 write_ptr;
    u32 interleaved_dram_read_ptr;
    u32 interleaved_l1_read_ptr;
    u32 interleaved_dram_write_ptr;
    u32 interleaved_l1_write_ptr;
    u32 launch_ptr;

    array<u32, SIZE> desc;  // Doing it this way since we may find better sizes for perf

   public:
    static constexpr u32 size() { return SIZE; }
    DeviceCommand() {
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

    /*
        Ideally, should refactor these APIs to just specify the relay address, no need to be explicit about
        the intermediate address for the dispatch core, it can figure it out based on the transfer sizes
    */

    void add_read(u32 src, u32 src_noc, u32 dst, u32 bytes) {
        TT_THROW("add_read not implemented yet");
        this->desc[this->read_ptr] = src;
        this->desc[this->read_ptr + 1] = src_noc;
        this->desc[this->read_ptr + 2] = dst;
        this->desc[this->read_ptr + 3] = bytes;
        this->read_ptr += 4;
    }

    void add_write(u32 src, u32 dst, u32 dst_noc, u32 bytes) {
        TT_THROW("add_write not implemented yet");
        this->desc[this->write_ptr] = src;
        this->desc[this->write_ptr + 1] = dst;
        this->desc[this->write_ptr + 2] = dst_noc;
        this->desc[this->write_ptr + 3] = bytes;
        this->write_ptr += 4;
    }

    void add_interleaved_dram_read(u32 src, u32 src_noc, u32 dst, uint num_bytes_per_page, uint num_pages) {
        TT_THROW("add_interleaved_dram_read not implemented yet");
        this->desc[this->interleaved_dram_read_ptr] = src;
        this->desc[this->interleaved_dram_read_ptr + 1] = src_noc;
        this->desc[this->interleaved_dram_read_ptr + 2] = dst;
        this->desc[this->interleaved_dram_read_ptr + 3] = num_bytes_per_page;
        this->desc[this->interleaved_dram_read_ptr + 4] = num_pages;
        this->interleaved_dram_read_ptr += 5;
    }

    void add_interleaved_l1_read(u32 src, u32 src_noc, u32 dst, uint num_bytes_per_page, uint num_pages) {
        TT_THROW("add_interleaved_l1_read not implemented yet");
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
        TT_THROW("add_interleaved_l1_write not implemented yet");
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
