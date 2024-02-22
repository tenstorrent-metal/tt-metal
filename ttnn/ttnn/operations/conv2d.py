# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict, Optional

import ttnn

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_composite_conv import (
    TTPyCompositeConv,
    SlidingWindowOpParams,
)


class Conv2d:
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
        reader_patterns_cache: Optional[Dict],
        weight: ttnn.Tensor,
        bias: ttnn.Tensor = None,
        math_fidelity: ttnn.MathFidelity = None,
        weights_dtype: ttnn.DataType = None,
        activation: str = None,
        conv_blocking_and_parallelization_config_override: Dict = None,
        reallocate_halo_output: bool = False,
        using_parameters_cache: bool = False,
        move_weights_to_device: bool = True,
        use_shallow_conv_variant: bool = False,
        enable_auto_formatting: bool = False,
        deallocate_activation: bool = False,
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
            activation = activation.lower()
            assert activation == "relu", f"Only support relu fusion with conv. Got activation={activation}."
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
            using_parameters_cache=using_parameters_cache,
            move_weights_to_device=move_weights_to_device,
            use_shallow_conv_variant=use_shallow_conv_variant,
            enable_auto_formatting=enable_auto_formatting,
            deallocate_activation=deallocate_activation,
        )

    @ttnn.register_operation(name="ttnn.Conv2d.__call__", validate_input_tensors=lambda *args, **kwargs: True)
    def __call__(self, activation: ttnn.Tensor):
        return ttnn.Tensor(self.conv(activation.value))

    @ttnn.register_operation(
        name="ttnn.Conv2d.copy_input_to_device", validate_input_tensors=lambda *args, **kwargs: True
    )
    def copy_input_to_device(self, input: ttnn.Tensor):
        return ttnn.Tensor(self.conv.copy_input_to_device(input.value))

    @ttnn.register_operation(
        name="ttnn.Conv2d.copy_output_from_device", validate_input_tensors=lambda *args, **kwargs: True
    )
    def copy_output_from_device(self, output: ttnn.Tensor):
        return ttnn.Tensor(self.conv.copy_output_from_device(output.value))

    @ttnn.register_operation(name="ttnn.Conv2d.get_num_cores_nhw", validate_input_tensors=lambda *args, **kwargs: True)
    def get_num_cores_nhw(self):
        return self.conv.get_num_cores_nhw()

    @ttnn.register_operation(
        name="ttnn.Conv2d.get_parallel_config", validate_input_tensors=lambda *args, **kwargs: True
    )
    def get_parallel_config(self):
        return self.conv.get_parallel_config()


__all__ = []
