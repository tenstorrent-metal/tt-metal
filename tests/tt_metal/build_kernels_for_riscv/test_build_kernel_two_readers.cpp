#include <iostream>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"

int main(int argc, char* argv[]) {

    std::string root_dir = tt::utils::get_root_dir();

    // // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("unary","two_readers");
    std::string out_dir_path = root_dir + "/built_kernels/" + build_kernel_for_riscv_options.name;

    // data-copy has one input operand and one output operand
    build_kernel_for_riscv_options.set_hlk_operand_dataformat_all_cores(tt::HlkOperand::in0, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_hlk_operand_dataformat_all_cores(tt::HlkOperand::out0, tt::DataFormat::Float16_b);

    build_kernel_for_riscv_options.fp32_dest_acc_en = false;

    build_kernel_for_riscv_options.set_hlk_file_name_all_cores("tt_metal/kernels/compute/eltwise_copy.cpp");

    log_info(tt::LogBuildKernels, "Compiling OP: {} to {}", build_kernel_for_riscv_options.name, out_dir_path);

    // build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";; //"tt_metal/kernels/dataflow/reader_unary_push_4.cpp";
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/reader_unary_push_4.cpp";
    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/reader_unary_push_4.cpp";

    generate_binaries_params_t params = {.compute_kernel_compile_time_args = {0}};

    // generate_binary_for_risc(RISCID::BR, &build_kernel_for_riscv_options, out_dir_path, "grayskull");
    // generate_binary_for_risc(RISCID::NC, &build_kernel_for_riscv_options, out_dir_path, "grayskull");
    generate_binaries_all_riscs(&build_kernel_for_riscv_options, out_dir_path, "grayskull", params, true);

    return 0;
}
