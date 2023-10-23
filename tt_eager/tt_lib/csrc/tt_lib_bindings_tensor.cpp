// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/conv/conv_op.hpp"
#include "tt_dnn/op_library/conv/optimized_conv_op.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_dnn/op_library/groupnorm/groupnorm_op.hpp"
#include "tt_dnn/op_library/pool/average_pool.hpp"
#include "tt_dnn/op_library/pool/max_pool.hpp"
#include "tt_dnn/op_library/fully_connected/fully_connected_op.hpp"
#include "tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_dnn/op_library/moreh_bmm/moreh_bmm_op.hpp"
#include "tt_dnn/op_library/moreh_layernorm/moreh_layernorm_op.hpp"
#include "tt_dnn/op_library/moreh_linear/moreh_linear_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/split/split_last_dim_two_chunks_tiled.hpp"
#include "tt_dnn/op_library/rotate_half/rotate_half_op.hpp"
#include "tt_dnn/op_library/rotary_embedding/rotary_embedding_op.hpp"
#include "tt_dnn/op_library/embeddings/embeddings_op.hpp"
#include "tt_dnn/op_library/update_cache/update_cache_op.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/borrowed_buffer.hpp"
#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/serialization.hpp"
#include "type_caster.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "tt_lib_bindings_tensor.hpp"

namespace tt::tt_metal{

namespace detail{
    template<class T>
    struct DataTypeToFormatType {
        using type = T;
    };

    template<>
    struct DataTypeToFormatType<bfloat16> {
        using type = uint16_t;
    };

    template<class CppType, class DataType, class PyType>
    void implement_buffer_protocol(PyType& py_buffer_t) {
        py_buffer_t
            .def(
                "__getitem__",
                [](const CppType& self, std::size_t index) {
                    return self[index];
                }
            )
            .def(
                "__len__",
                [](const CppType& self) {
                    return self.size();
                }
            )
            .def(
                "__iter__",
                [](const CppType& self) {
                    return py::make_iterator(self.begin(), self.end());
                },
                py::keep_alive<0, 1>()
            )
            .def_buffer(
                [](CppType& self) -> py::buffer_info {
                    using FormatType = typename DataTypeToFormatType<DataType>::type;
                    return py::buffer_info(
                        self.begin(),                                /* Pointer to buffer */
                        sizeof(DataType),                            /* Size of one scalar */
                        py::format_descriptor<FormatType>::format(), /* Python struct-style format descriptor */
                        1,                                           /* Number of dimensions */
                        { self.size() },                             /* Buffer dimensions */
                        { sizeof(DataType) }                         /* Strides (in bytes) for each index */
                    );
                }
            );
    };

}

void TensorModule(py::module &m_tensor) {
    // ENUM SECTION

    // layout enums
    detail::export_enum<Layout>(m_tensor);

    detail::export_enum<DataType>(m_tensor);

    detail::export_enum<StorageType>(m_tensor);

    detail::export_enum<MathFidelity>(m_tensor);

    detail::export_enum<TensorMemoryLayout>(m_tensor);

    py::enum_<BufferType>(m_tensor, "BufferType")
        .value("DRAM", BufferType::DRAM)
        .value("L1", BufferType::L1);

    // Fusible Activations
    detail::export_enum<UnaryOpType>(m_tensor, "FusibleActivation");
    py::class_<UnaryWithParam>(m_tensor, "FusibleActivationWithParam")
        .def(py::init<UnaryOpType>())
        .def(py::init<UnaryOpType, float>())
        .def(py::init<>(
            [](std::pair<UnaryOpType, float> arg) {
                return UnaryWithParam{.op_type=arg.first, .param=arg.second};
            }
        ));
    // Allow implicit construction of UnaryWithParam object without user explicitly creating it
    // Can take in just the op type, or sequence container of op type and param value
    py::implicitly_convertible<UnaryOpType, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, float>, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, int>, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, bool>, UnaryWithParam>();

    auto py_core_coord = py::class_<CoreCoord>(m_tensor, "CoreCoord", R"doc(
        Class defining core coordinate
    )doc");

    py_core_coord
        .def(py::init<std::size_t, std::size_t>())
        .def("__repr__", [](const CoreCoord& self) -> std::string {
            return self.str();
        }
        );

    auto pyMemoryConfig = py::class_<MemoryConfig>(m_tensor, "MemoryConfig", R"doc(
        Class defining memory configuration for storing tensor data on TT Accelerator device.
        There are eight DRAM memory banks on TT Accelerator device, indexed as 0, 1, 2, ..., 7.
    )doc");

    pyMemoryConfig
        .def(
            py::init<>(
                [](TensorMemoryLayout memory_layout, BufferType buffer_type) {
                    return MemoryConfig{.memory_layout=memory_layout, .buffer_type=buffer_type};
                }
            ),
            py::arg("memory_layout") = TensorMemoryLayout::INTERLEAVED,
            py::arg("buffer_type") = BufferType::DRAM, R"doc(
                Create MemoryConfig class.
                If interleaved is set to True, tensor data will be interleaved across multiple DRAM banks on TT Accelerator device.
                Otherwise, tensor data will be stored in a DRAM bank selected by dram_channel (valid values are 0, 1, ..., 7).

                Example of creating MemoryConfig specifying that tensor data should be stored in DRAM bank 3.

                .. code-block:: python

                    mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.SINGLE_BANK)
            )doc"
        )
        .def("__repr__", [](const MemoryConfig &memory_config) -> std::string {
            return fmt::format("{}", memory_config);
        }
        )
        .def("is_sharded", &MemoryConfig::is_sharded, "Whether tensor data is sharded across multiple cores in L1")
        .def_property_readonly("interleaved", [](const MemoryConfig &memory_config) {
            return memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED;
        }, "Whether tensor data is interleaved across multiple DRAM channels"
        )
        .def_readonly("buffer_type", &MemoryConfig::buffer_type, "Buffer type to store tensor data. Can be DRAM or L1")
        .def(py::self == py::self)
        .def(py::self != py::self);

    auto py_owned_buffer_for_uint32_t = py::class_<owned_buffer::Buffer<uint32_t>>(m_tensor, "owned_buffer_for_uint32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<uint32_t>, uint32_t>(py_owned_buffer_for_uint32_t);

    auto py_owned_buffer_for_float32_t = py::class_<owned_buffer::Buffer<float>>(m_tensor, "owned_buffer_for_float32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<float>, float>(py_owned_buffer_for_float32_t);

    auto py_owned_buffer_for_bfloat16_t = py::class_<owned_buffer::Buffer<bfloat16>>(m_tensor, "owned_buffer_for_bfloat16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<owned_buffer::Buffer<bfloat16>, bfloat16>(py_owned_buffer_for_bfloat16_t);

    auto py_borrowed_buffer_for_uint32_t = py::class_<borrowed_buffer::Buffer<std::uint32_t>>(m_tensor, "borrowed_buffer_for_uint32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::uint32_t>, std::uint32_t>(py_borrowed_buffer_for_uint32_t);

    auto py_borrowed_buffer_for_float32_t = py::class_<borrowed_buffer::Buffer<float>>(m_tensor, "borrowed_buffer_for_float32_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<float>, float>(py_borrowed_buffer_for_float32_t);

    auto py_borrowed_buffer_for_bfloat16_t = py::class_<borrowed_buffer::Buffer<bfloat16>>(m_tensor, "borrowed_buffer_for_bfloat16_t", py::buffer_protocol());
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<bfloat16>, bfloat16>(py_borrowed_buffer_for_bfloat16_t);

    detail::bind_unary_op(m_tensor, "mean_hw", tt::tt_metal::mean_hw, R"doc(  Returns a new tensor with the variance of the input tensor ``{0}`` on H,W axes.)doc");

    detail::bind_unary_op_with_param(
        m_tensor, "sum", &sum,
        py::arg("dim"),
        R"doc(Returns a tensor that is a sum  of input tensor with shape ``[W, Z, Y, X]`` along dimensions ``{1}``.)doc",
        R"doc("dimension to sum along", "int", "0, 1, 2, or 3")doc"
    );

    m_tensor.def("conv", &conv, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");

    py::class_<OptimizedConvParallelizationConfig>(m_tensor, "OptimizedConvParallelizationConfig")
        .def(
            py::init<>(
                [] (
                    std::tuple<std::size_t, std::size_t> grid_size,
                    uint32_t per_core_out_matrix_height_ntiles,
                    uint32_t per_core_weight_matrix_width_ntiles
                ) {
                    return OptimizedConvParallelizationConfig{
                        .grid_size={std::get<0>(grid_size), std::get<1>(grid_size)},
                        .per_core_out_matrix_height_ntiles=per_core_out_matrix_height_ntiles,
                        .per_core_weight_matrix_width_ntiles=per_core_weight_matrix_width_ntiles
                    };

                }
            ),
            py::kw_only(),
            py::arg("grid_size").noconvert(),
            py::arg("per_core_out_matrix_height_ntiles").noconvert(),
            py::arg("per_core_weight_matrix_width_ntiles").noconvert()
        );

    py::class_<OptimizedConvBlockConfig>(m_tensor, "OptimizedConvBlockConfig")
        .def(
            py::init<>(
                [] (
                    uint32_t act_block_h_ntiles,
                    uint32_t act_block_w_ntiles,
                    uint32_t weight_block_w_ntiles,
                    uint32_t out_block_h_ntiles,
                    uint32_t out_subblock_h_ntiles,
                    uint32_t out_subblock_w_ntiles
                ) {
                    return OptimizedConvBlockConfig{
                        .act_block_h_ntiles=act_block_h_ntiles,
                        .act_block_w_ntiles=act_block_w_ntiles,
                        .weight_block_w_ntiles=weight_block_w_ntiles,
                        .out_block_h_ntiles=out_block_h_ntiles,
                        .out_subblock_h_ntiles=out_subblock_h_ntiles,
                        .out_subblock_w_ntiles=out_subblock_w_ntiles
                    };
                }
            ),
            py::kw_only(),
            py::arg("act_block_h_ntiles").noconvert(),
            py::arg("act_block_w_ntiles").noconvert(),
            py::arg("weight_block_w_ntiles").noconvert(),
            py::arg("out_block_h_ntiles").noconvert(),
            py::arg("out_subblock_h_ntiles").noconvert(),
            py::arg("out_subblock_w_ntiles").noconvert()
        );

    m_tensor.def("optimized_conv", &optimized_conv,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert() = 0,
                 py::arg("output_mem_config").noconvert() = std::nullopt, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def("conv_with_fast_reader", &conv_with_fast_reader,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg("bias").noconvert() = std::nullopt,
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
                 py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
                 py::arg().noconvert(), py::arg().noconvert(), py::arg("math_fidelity").noconvert() = MathFidelity::HiFi4, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def("conv_with_address_map", &conv_with_address_map, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output
        Reader kernel uses an address map which pre-computed on the host to read activations and weights

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");

    // moreh_bmm
    m_tensor.def("moreh_bmm", &moreh_bmm,
        py::arg("input").noconvert(), py::arg("mat2").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a moreh_bmm operation.
    )doc");

    // moreh_layernorm
    m_tensor.def("moreh_layernorm", &moreh_layernorm,
        py::arg("input").noconvert(), py::arg("eps").noconvert(), py::arg("normalized_dims").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a moreh_layernorm operation.
    )doc");

    // moreh_linear
    m_tensor.def("moreh_linear", &moreh_linear,
        py::arg("input").noconvert(), py::arg("weight").noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a moreh_linear operation.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "", "Yes"
            "weight", "Input weight tensor", "Tensor", "", "Yes"
            "bias", "Input bias tensor", "Tensor", "", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    // groupnorm
    m_tensor.def("groupnorm", &groupnorm,
        py::arg("input").noconvert(), py::arg("group_size").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a groupnorm operation on the channel dimension grouped per group_size, with optional fused with post-multiplication and addition via W-bcast.
    )doc");

    // layernorm
    m_tensor.def("layernorm", &layernorm,
        py::arg("input").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a layernorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
    )doc");
    m_tensor.def("add_layernorm", &add_layernorm,
        py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a layernorm(a+b)*gamma + beta operation."
    )doc");
    m_tensor.def("rmsnorm", &rmsnorm,
        py::arg("input").noconvert(), py::arg("eps").noconvert(), py::arg("gamma").noconvert() = std::nullopt, py::arg("beta").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a rmsnorm operation on the last tensor dimension with optional fused with post-multiplication and addition via W-bcast.
    )doc");
    m_tensor.def("rotate_half", &rotate_half,
        py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs a rotate half operation used by RotaryEmbedding.
    )doc");
    m_tensor.def("rotary_embedding", &rotary_embedding,
        py::arg("input").noconvert(), py::arg("cos").noconvert(), py::arg("sin").noconvert(), py::arg("token_idx") = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        "Performs rotary embedding with a given input, cos, and sin tensors. Sequence length is inferred as the second last dim of the input tensor.
        If token_idx is passed, this assumes input is transposed to [seq_len, 1, B, head_dim], and seq_len is 1.
    )doc");
    m_tensor.def("fill_cache", &fill_cache,
         py::arg("cache").noconvert(), py::arg("input").noconvert(), py::arg("batch_idx"), R"doc(
        "Fills the cache tensor in place with the values from input at the specified batch_idx.
    )doc");
    m_tensor.def("update_cache", &update_cache,
         py::arg("cache").noconvert(), py::arg("input").noconvert(), py::arg("update_idx"), R"doc(
        "Updates the cache tensor in place with the values from input at the specified update_idx.
    )doc");


    // input embeddings
    m_tensor.def("embeddings", &embeddings,
        py::arg("input").noconvert(), py::arg("weights").noconvert(),
        py::arg("split_weights").noconvert() = false,
        py::arg("tilized").noconvert() = false,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns specific indices of the embedding table specified by the input tensor

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Tensor containing rows we want", "UInt32 Tensor", "Each element greater than 0 and less than number of embeddings in table.  Shape [batch_size, 1, num_rows, 1]", "Yes"
            "weights", "Entire embedding table", "Tensor", "Tensor shape is [1,1, num_embeddings, num_columns]. Num_columns must be divisible by 32.", "Yes"
            "split_weights", "Parallelizing over weights (instead of input). Default is false", "Bool", "", "No"
            "tilized", "Enable fused tilize on output. Default is true.", "Bool", "", "No"
    )doc");


    // FC
    m_tensor.def("fully_connected", &fully_connected,
        py::arg("act").noconvert(), py::arg("weights").noconvert(), py::arg("bias").noconvert() = std::nullopt, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Fully connected layer (linear.)

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "act", "Input activations tensor", "Tensor", "", "Yes"
            "weights", "Input weights tensor", "Tensor", "", "Yes"
            "bias", "Input bias tensor", "Tensor", "", "No"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
    )doc");

    // Pools
    m_tensor.def("average_pool_2d", &average_pool_2d,
        py::arg().noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,  R"doc(
        Average Pool 2D
        It operates on tensors whose that have channels as the last dimension

        +----------+----------------------------+------------+-------------------------------+----------+
        | Argument | Description                | Data type  | Valid range                   | Required |
        +==========+============================+============+===============================+==========+
        | act      | Input activations tensor   | Tensor     |                               | Yes      |
        +----------+----------------------------+------------+-------------------------------+----------+
    )doc");
    m_tensor.def("max_pool2d", &max_pool2d,
        py::arg("input").noconvert(),
        py::arg("in_h").noconvert(), py::arg("in_w").noconvert(),
        py::arg("kernel_h").noconvert(), py::arg("kernel_w").noconvert(),
        py::arg("stride_h") = 1, py::arg("stride_w") = 1,
        py::arg("pad_h") = 0, py::arg("pad_w") = 0,
        py::arg("dilation_h") = 1, py::arg("dilation_w") = 1,
        py::arg("output_mem_config") = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("nblocks") = 1,
        py::arg("use_multicore") = true, R"doc(
        Max Pool 2D
        +-------------------+-------------------------------+---------------+-------------+----------+
        | Argument          | Description                   | Data type     | Valid range | Required |
        +===================+===============================+===============+=============+==========+
        | input             | Input activations tensor      | Tensor        |             | Yes      |
        | in_h              | Input height                  | Tensor        |             | Yes      |
        | in_w              | Input width                   | Tensor        |             | Yes      |
        | kernel_h          | kernel window height          | uint32_t      |             | Yes      |
        | kernel_w          | kernel window width           | uint32_t      |             | Yes      |
        | stride_h          | stride in height dim          | uint32_t      |             | No       |
        | stride_w          | stride in width dim           | uint32_t      |             | No       |
        | pad_h             | padding in height dim         | uint32_t      |             | No       |
        | pad_w             | padding in width dim          | uint32_t      |             | No       |
        | dilation_h        | kernel dilation in height dim | uint32_t      |             | No       |
        | dilation_w        | kernel dilation in width dim  | uint32_t      |             | No       |
        | output_mem_config | output tensor memory config   | MemoryConfig  |             | No       |
        +-------------------+-------------------------------+---------------+-------------+----------+
    )doc");

    // TMs
    m_tensor.def("split_last_dim_two_chunks_tiled", &split_last_dim_two_chunks_tiled, py::arg("input").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Splits a tensor's last dimension in two equal sized chunks. This assumes the last dim is tile sized.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W0, Z0, Y0, X0]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"

    )doc");
    m_tensor.def("convert_conv_weight_tensor_to_tiled_layout", &convert_conv_weight_tensor_to_tiled_layout, R"doc(
       Converts convolution weights to 2d matrix tiled layout on host
       Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def("convert_conv_weight_tensor_to_special_padding_tiled_layout", &convert_conv_weight_tensor_to_special_padding_tiled_layout, R"doc(
       Converts convolution weights to 2d matrix tiled layout on host with special block height padding
       Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def(
        "format_input_tensor",
        [] (const Tensor &input, Device * device, const std::array<uint32_t, 4>& padded_shape, float pad_value, Layout target_layout, std::optional<MemoryConfig> target_mem_config = std::nullopt) {
            return AutoFormat::format_input_tensor(input, device, padded_shape, pad_value, target_layout, target_mem_config);
        },
        py::arg("input").noconvert(), py::arg("device").noconvert(), py::arg("padded_shape"), py::arg("pad_value"), py::arg("target_layout").noconvert(), py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
            Formats tensor to target layout and pads to padded shape
        )doc"
    );
    m_tensor.def(
        "format_output_tensor",
        [] (const Tensor &output, const std::array<uint32_t, 4>& shape, Device* device, Layout target_layout, std::optional<MemoryConfig> target_mem_config = std::nullopt) {
            return AutoFormat::format_output_tensor(output, shape, device, target_layout, target_mem_config);
        },
        py::arg("output").noconvert(), py::arg("shape"), py::arg("device").noconvert(), py::arg("target_layout").noconvert(), py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
            Formats tensor to target layout and unpads to shape
        )doc"
    );
    m_tensor.def(
        "pad_to_tile_shape",
        [] (const std::array<uint32_t, 4>& unpadded_shape, bool pad_c=false, bool pad_n=false, bool pad_h=true, bool pad_w=true) {
            Shape padded_shape_object = AutoFormat::pad_to_tile_shape(unpadded_shape, pad_c, pad_n, pad_h, pad_w);
            std::array<uint32_t, 4> padded_shape;
            std::copy(std::begin(padded_shape_object), std::end(padded_shape_object), std::begin(padded_shape));
            return padded_shape;
        }, R"doc(
            Returns shape padded to tile shape
        )doc"
    );

    m_tensor.def(
        "dump_tensor",
        &dump_tensor,
        R"doc(
            Dump tensor to file
        )doc"
    );

    m_tensor.def(
        "load_tensor",
        &load_tensor,
        R"doc(
            Load tensor to file
        )doc"
    );

    detail::TensorModuleCompositeOPs( m_tensor);
    detail::TensorModulePyTensor ( m_tensor);
    detail::TensorModuleDMOPs ( m_tensor);
    detail::TensorModuleCustomAndBMMOPs( m_tensor );
    detail::TensorModuleXaryOPs( m_tensor );

}

}
