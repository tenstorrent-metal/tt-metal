# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict

import ttnn.tensor as ttnn

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_composite_conv import (
    TTPyCompositeConv,
    SlidingWindowOpParams,
)


class Conv2D:
    """

    Applies a 2D convolution over an input signal composed of several input planes.

    Args:

        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel. Single int or tuple of 2 ints: (Kernel_H, Kernel_W)
        stride (int or tuple, optional): Stride of the convolution. Single int or tuple of 2 ints: (Stride_H, Stride_w) Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        dtype (DataType): Output datatype.
        weights_dtype (DataType): Weights datatype:
        device (Device): Tenstorrent device object.
        use_1d_systolic_array (bool): Specifies conv parallelization over a grid of cores on Tenstorrent device.

            The grid of cores to parallelize the conv op is determined automatically based on this flag.
            Conv is implemented as a matrix multiplication where the input, weight and output of the conv are mapped as follows to perform a 2d matrix multiplication:
            Conv as matrix multiply input shape - ```'(1, 1, N x H_out x Wout, C_in x Kernel_H x Kernel_W)'```,
            Conv as matrix multiply weight shape - ```'(1, 1, C_in x Kernel_H x Kernel_W, C_out)'```,
            Conv as matrix multiply output shape - ```'(1, 1, N x H_out x Wout, C_out)'```
            if ``use_1d_systolic_array`` is ``True``, conv output tensor (with the above output shape) is sliced by the y-dim across cores.
            if it is ``False``, conv output tensor is sliced by the y-dim across columns of the grid and is sliced by the x-dim across the rows of the grid.

        conv_blocking_and_parallelization_config_override (Dict, optional): Default: None. One or more config parameters to override the automatically determined parameters.

            Valid config parameter names and description:

            ``'num_cores_nhw'``: Number of cores to slice the y-dim of conv as matrix multiply input shape i.e. ```'(1, 1, N x H_out x Wout, C_in x Kernel_H x Kernel_W)'```
            ``'grid_size'``: Tuple of two ints: (number of columns, number of rows)
            ``'per_core_out_matrix_height'``: Per core (sliced) y-dim conv as matrix multiply output shape. Must be divisible by 32.
            ``'per_core_out_matrix_width'``: Per core (sliced) x-dim conv as matrix multiply output shape. Must be divisible by 32.
            Other per core config parameters that specify how to perform the matrix multiply in blocks
            ``'act_block_h'``: Block height of activation matrix. Must be divisible by 32.
            ``'act_block_w'``: Block width of activation matrix. Must be divisible by 32. Must be either equal to in_channels or in_channels x Kernel_H x Kernel_W.
            ``'act_c_num_blocks'``: Number of blocks of input and output channels. Must evenly divide both input and output channels. Must be equal to 1 if use_1d_stolic_array is True.
            ``'weight_block_w'``: Block width of weight matrix. Must be divisible by 32.
            ``'out_block_h'``: Block height of output matrix. Must be divisible by 32. Must be equal to per_core_out_matrix_height if it is set.
            ``'out_block_w'``: Block width of output matrix. Must be divisible by 32. Must be equal to per_core_out_matrix_width if it is set.
            ``'out_subblock_h'``: Sub block height of output matrix. Must be divisible by 32. Both sublock height and width must be set if one of them is set.
            ``'out_subblock_w'``: Sub block width of output matrix. Must be divisible by 32. Both sublock height and width must be set if one of them is set.

        batch_size (int): Batch size of input
        input_height (int): Height of the input image
        input_width (int): Width of the input image
        reader_patterns_cache (Dict): Special config tensors are generated for conv reader kernels that are added to this dictionary. These tensors can be reused across different conv ops with same parameters.

            Provide an empty dictionary to the first conv op and then, provide the same cache dictionary to subsequent conv ops.

        math_fidelity (MathFidelity): ``'MathFidelity.HiFi4'`` or ``'MathFidelity.LoFi'``.
        weight (Tensor): Weight TT Tensor in row major layout with shape - ```'(C_out, C_in, Kernel_H, Kernel_W)'```
        bias (Tensor, optional): Bias TT Tensor in row major layout with shape - ```'(1, 1, 1, C_out)'```. Default: None.
        activation (str, optional): Specifies if there is an activation op to be fused with conv op.

            Only relu activation fuction is supported: ```RELU```. Default: None.

        reallocate_halo_output (bool, optional): flag to reallocate halo op's output before conv op to reduce memory defragmentation.

    Note:

        Only `dilation=1` or `dilation=(1,1)` is supported

    Note:

        Only `groups=1` is supported

    Note:

        Only `padding_mode=zeros` is supported

    Examples:

        # With square kernels and equal stride
        m = nn.Conv2d(16, 33, 3, stride=2)
        # non-square kernels and unequal stride and with padding
        m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        # non-square kernels and unequal stride and with padding and dilation
        m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        input = torch.randn(20, 16, 50, 100)
        output = m(input)

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        dtype: ttnn.DataType = None,
        *,
        device: ttnn.Device,
        use_1d_systolic_array: bool,
        batch_size: int,
        input_height: int,
        input_width: int,
        reader_patterns_cache: Dict,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor = None,
        math_fidelity: ttnn.MathFidelity = None,
        weights_dtype: ttnn.DataType = None,
        activation: str = None,
        conv_blocking_and_parallelization_config_override: Dict = None,
        reallocate_halo_output: bool = False,
    ):
        assert (
            padding_mode == "zeros"
        ), f"Only convs with padding_mode=zeroes supported. Found padding_mode set to {padding_mode}."
        if isinstance(kernel_size, int):
            window_h = kernel_size
            window_w = kernel_size
        else:
            window_h, window_w = kernel_size

        if isinstance(stride, int):
            stride_h = stride
            stride_w = stride
        else:
            stride_h, stride_w = stride

        if isinstance(padding, int):
            pad_h = padding
            pad_w = padding
        else:
            pad_h, pad_w = padding

        if isinstance(dilation, int):
            dilation_h = dilation
            dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        assert dilation_h == 1, f"Only convs with dilation == 1 supported. Found dilation_h={dilation_h}"
        assert dilation_w == 1, f"Only convs with dilation == 1 supported. Found dilation_w={dilation_w}"
        assert groups == 1, "Only convs with groups == 1 supported"
        sliding_window_op_params = SlidingWindowOpParams(
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            window_h=window_h,
            window_w=window_w,
            batch_size=batch_size,
            input_h=input_height,
            input_w=input_width,
        )
        fuse_relu = False
        if activation is not None:
            assert activation == "RELU", f"Only support relu fusion with conv. Got activation={activation}."
            fuse_relu = True
        if bias is not None:
            bias = bias.value
        weight = weight.value
        self.conv = TTPyCompositeConv(
            sliding_window_op_params,
            weight,
            out_channels,
            in_channels,
            device,
            use_1d_systolic_array,
            reader_patterns_cache,
            bias=bias,
            conv_blocking_and_parallelization_config_override=conv_blocking_and_parallelization_config_override,
            fuse_relu=fuse_relu,
            output_dtype=dtype,
            weights_dtype=weights_dtype,
            math_fidelity=math_fidelity,
            move_utwh_output=reallocate_halo_output,
        )

    def __call__(self, activation: ttnn.Tensor):
        return ttnn.Tensor(self.conv(activation.value))

    def copy_input_to_device(self, input: ttnn.Tensor):
        return ttnn.Tensor(self.conv.copy_input_to_device(input.value))

    def copy_output_from_device(self, output: ttnn.Tensor):
        return ttnn.Tensor(self.conv.copy_output_from_device(output.value))


__all__ = []
