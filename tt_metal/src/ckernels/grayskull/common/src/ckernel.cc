#include "ckernel.h"
#include "fw_debug.h"
#include "ckernel_globals.h"
#include <tensix.h>

#include "tools/profiler/kernel_profiler.hpp"

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
}

namespace ckernel
{

enum class ttRiscCores : std::uint32_t { Unpack = 0, Math = 1, Pack = 2, Brisc = 3, Nrisc = 4};

volatile uint * const reg_base = reinterpret_cast<volatile uint *>(0xFFB10000);
volatile uint * const pc_buf_base = reinterpret_cast<volatile uint *>(PC_BUF_BASE);
volatile uint * const regfile = reinterpret_cast<volatile uint *>(REGFILE_BASE);
volatile uint * const instrn_buffer = reinterpret_cast<volatile uint *>(INSTRN_BUF_BASE);
volatile uint *mailbox_base[4] = {
    reinterpret_cast<volatile uint *>(TENSIX_MAILBOX0_BASE), reinterpret_cast<volatile uint *>(TENSIX_MAILBOX1_BASE),
    reinterpret_cast<volatile uint *>(TENSIX_MAILBOX2_BASE), reinterpret_cast<volatile uint *>(TENSIX_MAILBOX3_BASE)
};
volatile uint *dbg_event_scratch = nullptr;
uint *regmem = reinterpret_cast<uint *>(REGFILE_BASE);

uint32_t cfg_state_id __attribute__((used)) = 0;  // Flip between 0 and 1 to keep state between kernel calls
uint32_t dest_offset_id __attribute__((used)) = 0; // Flip between 0 and 1 to keep dest pointer between kernel calls

uint32_t dbg_event_index = 0;
uint32_t dbg_event_end = 0;
volatile uint16_t *debug_mailbox_base = nullptr;
uint8_t mailbox_index = 0;
uint8_t mailbox_end = 32;
uint32_t op_info_offset __attribute__((used)) = 0;
volatile uint8_t *debug_buffer = nullptr;


const uint8_t thread_id = COMPILE_FOR_TRISC;

volatile uint local_mem_barrier __attribute__((used));

volatile uint * const trisc_run_mailbox = reinterpret_cast<volatile uint *>(MEM_RUN_MAILBOX_ADDRESS + PREPROCESSOR_EXPAND(MEM_MAILBOX_TRISC, COMPILE_FOR_TRISC, _OFFSET));

inline void allocate_debug_mailbox_buffer() {
   debug_mailbox_base = reinterpret_cast<volatile uint16_t *>(MEM_DEBUG_MAILBOX_ADDRESS + COMPILE_FOR_TRISC * MEM_DEBUG_MAILBOX_SIZE);
   clear_mailbox_values();
}

inline void allocate_debug_buffer() {
    // TODO(PK) reimplement debug buffer
}

} // namespace ckernel

volatile uint32_t l1_buffer[16] __attribute__ ((section ("l1_data"))) __attribute__ ((aligned (16))) __attribute__((used));

using namespace ckernel;

int main(int argc, char *argv[])
{
    uint *local_l1_start_addr = (uint *)PREPROCESSOR_EXPAND(MEM_TRISC, COMPILE_FOR_TRISC, _INIT_LOCAL_L1_BASE);
    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint*)__ldm_data_start, local_l1_start_addr, num_words);

    kernel_profiler::init_profiler();

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
    kernel_profiler::mark_time(CC_MAIN_START);
#endif
    FWEVENT("Launching production env kernels");

    // Initialize GPRs to all 0s
    for (int i = 0; i < 64; i++)
        regfile[i] = 0;

    // Init L1 buffer with 1.0f (used for reduce max)
    union {
        float f;
        uint32_t u;
    } f2u = {.f = 1.0f};

    for (uint i = 0; i < 16; i++) l1_buffer[i] = f2u.u;  // Load const into L1 buffer

    reset_cfg_state_id();

    trisc_run_mailbox_write(RESET_VAL);

    allocate_debug_mailbox_buffer();
    allocate_debug_buffer();

    //while (ready_for_next_epoch())
    {
#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
        kernel_profiler::mark_time(CC_KERNEL_MAIN_START);
#endif
        kernel_init();
#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
        kernel_profiler::mark_time(CC_KERNEL_MAIN_END);
#endif
    }

    // Signal completion
    tensix_sync();
#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
    kernel_profiler::mark_time(CC_MAIN_END);
#endif
    trisc_run_mailbox_write(KERNEL_COMPLETE);

}
