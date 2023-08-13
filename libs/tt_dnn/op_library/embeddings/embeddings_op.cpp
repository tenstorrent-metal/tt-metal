
#include "libs/tt_dnn/op_library/embeddings/embeddings_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"


#include "tt_metal/llrt/tt_debug_print_server.hpp"


using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks embeddings_(
    const Tensor &a,
    const Tensor &weights,
    Tensor & output
){

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer *a_buffer = a.buffer();
    tt_metal::Buffer *weights_buffer = weights.buffer();
    tt_metal::Buffer *out_buffer = output.buffer();


    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device *device = a.device();
    auto dst_addr = output.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};


    uint32_t cb_id = 0;
    uint32_t num_tiles_per_cb = 1;
    bool in0_is_dram = a.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weights_is_dram = weights.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weights_dtype_is_bfloat16 = weights.dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool out_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t last_dim = 3;
    uint32_t element_size_in_bytes = 2; // size of float
    if(!weights_dtype_is_bfloat16)
        element_size_in_bytes = 1; //bfp8_b

    // row major, page size is last dim
    uint32_t single_page_size = weights.shape()[last_dim]*element_size_in_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]
    // num_pages same as num_embeddings
    uint32_t num_pages = weights.shape()[last_dim-1];

    // shape is [1,1, num_output_rows, 1] of input
    uint32_t num_output_rows = a.shape()[last_dim - 1];
    std::vector<uint32_t> reader_compile_time_args = { (std::uint32_t) weights_is_dram, (std::uint32_t)weights_dtype_is_bfloat16,
                                                (std::uint32_t) cb_id, (std::uint32_t) single_page_size, (std::uint32_t)num_pages};
    std::vector<uint32_t> writer_compile_time_args = {  (std::uint32_t) in0_is_dram, (std::uint32_t) out_is_dram, (std::uint32_t)weights_dtype_is_bfloat16,
                                                (std::uint32_t) cb_id, (std::uint32_t) single_page_size, (std::uint32_t) num_output_rows, (std::uint32_t) num_pages};


    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    //Single core implementation
    CoreRange all_cores{
        .start = {(std::size_t)start_core_x, (std::size_t)start_core_y},
        .end = {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
    };

    // Create Kernels
    auto reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_embeddings.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    auto writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_embeddings.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(weights.dtype());


    uint32_t src0_cb_index = cb_id;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program, src0_cb_index, all_cores, num_input_tiles, num_input_tiles * single_page_size, cb_data_format);


    auto dst_buffer_l1 = tt_metal::Buffer(device, sizeof(uint32_t)*512, sizeof(uint32_t)*512, tt_metal::BufferType::L1);

    std::vector<uint32_t> reader_runtime_args = {
        (std::uint32_t) weights.buffer()->address(),
    };
    tt_metal::SetRuntimeArgs(program, reader_kernel_id, all_cores, reader_runtime_args);

    std::vector<uint32_t> writer_runtime_args = {
        (std::uint32_t) a.buffer()->address(),
        (std::uint32_t) dst_buffer_l1.address(),
        (std::uint32_t) output.buffer()->address()
    };
    tt_metal::SetRuntimeArgs(program, writer_kernel_id, all_cores, writer_runtime_args);

    auto override_runtime_args_callback =
        [reader_kernel_id, writer_kernel_id, all_cores, start_core_x, start_core_y](
            const Program &program, const std::vector<Buffer *> &input_buffers, const std::vector<Buffer *> &output_buffers) {
                CoreCoord core = {(std::size_t)start_core_x, (std::size_t)start_core_y};
                auto weights_dram_buffer = input_buffers.at(1);
                {
                    auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = weights_dram_buffer->address();
                    SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
                }
                auto input_dram_buffer = input_buffers.at(0);
                auto output_dram_buffer = output_buffers.at(0);
                {
                    auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = input_dram_buffer->address();
                    runtime_args[2] = output_dram_buffer->address();
                    SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
                }
    };

    auto worker_core = device->worker_core_from_logical_core({0,0});
    tt_start_debug_print_server(device->cluster(), {0}, {worker_core});
    return {std::move(program), override_runtime_args_callback};
}



void Embeddings::validate(const std::vector<Tensor> &input_tensors) const  {
    TT_ASSERT(input_tensors.size() == 2 , "Must have between 2 input tensors");
    auto& a = input_tensors.at(0);
    const auto& weights = input_tensors.at(1);
    TT_ASSERT(a.layout() == Layout::ROW_MAJOR);
    TT_ASSERT(weights.layout() == Layout::ROW_MAJOR);
    TT_ASSERT(a.dtype() == DataType::UINT32);
    TT_ASSERT(weights.dtype() == DataType::UINT32 ||
            weights.dtype() == DataType::BFLOAT16 ||
            weights.dtype() == DataType::BFLOAT8_B);
}

std::vector<Shape> Embeddings::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    // shape is [1,1, num_output_rows, 1] of input
    Shape output_shape({1, 1, input_tensor.shape()[2],this->embedding_dim});
    return {output_shape};
}

std::vector<Tensor> Embeddings::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& weight_tensor = input_tensors.at(1);
    return operation::generic_create_output_tensors(*this, input_tensors, weight_tensor.dtype(), Layout::ROW_MAJOR, this->output_mem_config);
}

operation::ProgramWithCallbacks Embeddings::create_program(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& weights = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);
    return embeddings_(a, weights, output_tensor);
}

operation::Hash Embeddings::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& weights = input_tensors.at(1);
    return fmt::format("{}_{}_{}", *this, input_tensor, weights);
}

tt::stl::reflection::Attributes Embeddings::attributes() const {
    return {
        {"num_embeddings", this->num_embeddings},
        {"embedding_dim", this->embedding_dim},
        {"output_mem_config", this->output_mem_config},
    };
}

} // namespace tt_metal
} // namespace tt
