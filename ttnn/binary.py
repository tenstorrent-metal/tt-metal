# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import sys
import tt_lib as ttl
from ttnn.core import reshape, _reshape_to_4D
from ttnn.decorators import decorate_operation
from typing import Union
from ttnn.tensor import (
    Tensor,
    has_storage_type_of,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
    DEVICE_STORAGE_TYPE,
    LossMode,
    LOSS_MODE_NONE,
    LOSS_MODE_SUM,
    LOSS_MODE_MEAN,
)
from ttnn.core import reshape, _reshape_to_4D
from ttnn.decorators import decorate_operation
import torch
import torch.nn.functional as F

THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_binary_function(name, ttl_binary_function, torch_function):
    def _torch_binary(input_tensor_a: Tensor, input_tensor_b: Tensor, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        assert torch_function, f"Torch function not implemented for {str(ttl_binary_function)}"
        return torch_function(input_tensor_a, input_tensor_b)

    @decorate_operation(torch_function=_torch_binary, name=name)
    def binary_function(
        input_tensor_a: Tensor, input_tensor_b: Tensor, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG
    ) -> Tensor:
        if not (input_tensor_a.shape == input_tensor_b.shape):
            raise RuntimeError("input_tensors must be of same size!")

        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)
        input_tensor_b = _reshape_to_4D(input_tensor_b)

        if not isinstance(input_tensor_a, Tensor) or not isinstance(input_tensor_b, Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE) or not has_storage_type_of(
            input_tensor_b, DEVICE_STORAGE_TYPE
        ):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value

        ttl_output_tensor = ttl_binary_function(ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config)

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    binary_function.__name__ = f"ttnn.{name}"
    binary_function.__doc__ = f"""{name}(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor_a` and  :attr:`input_tensor_b` element-wise.

        .. math::
             \\texttt{{{name}}}(\\mathrm{{input\\_tensor_a}}_i, {{input\\_tensor_b}}_i)

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`

        Example::

            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b)

        """
    setattr(THIS_MODULE, name, binary_function)
    __all__.append(name)
    return binary_function


def register_ttl_binary_function_with_float_parameter(name, ttl_binary_function, torch_function):
    def _torch_binary(input_tensor_a: Tensor, input_tensor_b: Tensor, parameter, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        assert torch_function, f"Torch function not implemented for {str(ttl_binary_function)}"
        return torch_function(input_tensor_a, input_tensor_b, parameter)

    @decorate_operation(torch_function=_torch_binary, name=name)
    def binary_function(
        input_tensor_a: Tensor,
        input_tensor_b: Tensor,
        parameter: float,
        *,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> Tensor:
        if not (input_tensor_a.shape == input_tensor_b.shape):
            raise RuntimeError("input_tensors must be of same size!")

        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)
        input_tensor_b = _reshape_to_4D(input_tensor_b)

        if not isinstance(input_tensor_a, Tensor) or not isinstance(input_tensor_b, Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE) or not has_storage_type_of(
            input_tensor_b, DEVICE_STORAGE_TYPE
        ):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value

        ttl_output_tensor = ttl_binary_function(
            ttl_input_tensor_a, ttl_input_tensor_b, parameter, output_mem_config=memory_config
        )

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    binary_function.__name__ = f"ttnn.{name}"
    binary_function.__doc__ = f"""{name}(input_tensor_a: Tensor, input_tensor_b: Tensor, parameter: float) -> Tensor

        Applies {name} to :attr:`input_tensor_a`  and  :attr:`input_tensor_b` element-wise.

        .. math::
             \\texttt{{{name}}}(\\mathrm{{input\\_tensor_a}}_i, {{input\\_tensor_b}}_i, parameter)

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`

        Example::

            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b, 2.0)
        """

    setattr(THIS_MODULE, name, binary_function)
    __all__.append(name)
    return binary_function


def register_ttl_assign_function(name, ttl_binary_function, torch_function):
    def _torch_binary(input_tensor_a: Tensor, input_tensor_b: Tensor, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        assert torch_function, f"Torch function not implemented for {str(ttl_binary_function)}"
        return torch_function(input_tensor_a, input_tensor_b)

    @decorate_operation(torch_function=_torch_binary, name=name)
    def binary_function(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor:
        if not (input_tensor_a.shape == input_tensor_b.shape):
            raise RuntimeError("input_tensors must be of same size!")

        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)
        input_tensor_b = _reshape_to_4D(input_tensor_b)

        if not isinstance(input_tensor_a, Tensor) or not isinstance(input_tensor_b, Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE) or not has_storage_type_of(
            input_tensor_b, DEVICE_STORAGE_TYPE
        ):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value

        ttl_output_tensor = ttl_binary_function(ttl_input_tensor_a, ttl_input_tensor_b)

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    binary_function.__name__ = f"ttnn.{name}"
    binary_function.__doc__ = f"""{name}(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor_a` and  :attr:`input_tensor_b` element-wise.

        .. math::
             \\texttt{{{name}}}(\\mathrm{{input\\_tensor_a}}_i, {{input\\_tensor_b}}_i)

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`

        Example::

            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b)

        """
    setattr(THIS_MODULE, name, binary_function)
    __all__.append(name)
    return binary_function


def register_ttl_binary_function_with_multiple_parameter(name, ttl_binary_function, torch_function):
    def _torch_binary(input_tensor_a: Tensor, input_tensor_b: Tensor, parameter_1, parameter_2, equal_nan, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        assert torch_function, f"Torch function not implemented for {str(ttl_binary_function)}"
        return torch_function(input_tensor_a, input_tensor_b, parameter_1, parameter_2, equal_nan)

    @decorate_operation(torch_function=_torch_binary, name=name)
    def binary_function(
        input_tensor_a: Tensor,
        input_tensor_b: Tensor,
        parameter_1: float,
        parameter_2: float,
        equal_nan: bool,
        *,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> Tensor:
        if not (input_tensor_a.shape == input_tensor_b.shape):
            raise RuntimeError("input_tensors must be of same size!")

        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)
        input_tensor_b = _reshape_to_4D(input_tensor_b)

        if not isinstance(input_tensor_a, Tensor) or not isinstance(input_tensor_b, Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE) or not has_storage_type_of(
            input_tensor_b, DEVICE_STORAGE_TYPE
        ):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value

        ttl_output_tensor = ttl_binary_function(
            ttl_input_tensor_a, ttl_input_tensor_b, parameter_1, parameter_2, equal_nan, output_mem_config=memory_config
        )

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    binary_function.__name__ = f"ttnn.{name}"
    binary_function.__doc__ = f"""{name}(input_tensor_a: Tensor, input_tensor_b: Tensor, rtol: float, atol: float, equal_nan: bool ) -> Tensor

        Applies {name} to :attr:`input_tensor_a`  and  :attr:`input_tensor_b` element-wise.

        .. math::
             \\texttt{{{name}}}(\\mathrm{{input\\_tensor_a}}_i, {{input\\_tensor_b}}_i)

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`

        Example::

            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b, 0.4 , 0.1, True)

        """
    setattr(THIS_MODULE, name, binary_function)
    __all__.append(name)
    return binary_function


def register_ttl_binary_loss_function(name, ttl_binary_function, torch_function):
    def _torch_binary(input_tensor_a: Tensor, input_tensor_b: Tensor, reduction_mode, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        assert torch_function, f"Torch function not implemented for {str(ttl_binary_function)}"
        return torch_function(input_tensor_a, input_tensor_b, reduction_mode)

    @decorate_operation(torch_function=_torch_binary, name=name)
    def binary_function(
        input_tensor_a: Tensor,
        input_tensor_b: Tensor,
        loss_mode: LossMode = LOSS_MODE_NONE,
        *,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> Tensor:
        if not (input_tensor_a.shape == input_tensor_b.shape):
            raise RuntimeError("input_tensors must be of same size!")

        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)
        input_tensor_b = _reshape_to_4D(input_tensor_b)

        if not isinstance(input_tensor_a, Tensor) or not isinstance(input_tensor_b, Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE) or not has_storage_type_of(
            input_tensor_b, DEVICE_STORAGE_TYPE
        ):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value

        ttl_output_tensor = ttl_binary_function(
            ttl_input_tensor_a, ttl_input_tensor_b, loss_mode, output_mem_config=memory_config
        )

        output_tensor = Tensor(ttl_output_tensor)
        return output_tensor

    binary_function.__name__ = f"ttnn.{name}"
    binary_function.__doc__ = f"""{name}(input_tensor_a: Tensor, input_tensor_b: Tensor, mode: LossMode) -> Tensor

        Applies {name} to :attr:`input_tensor_a`  and  :attr:`input_tensor_b` element-wise.

        .. math::
             \\texttt{{{name}}}(\\mathrm{{input\\_tensor_a}}_i, {{input\\_tensor_b}}_i)

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`

        Example::

            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b, mode)
        """

    setattr(THIS_MODULE, name, binary_function)
    __all__.append(name)
    return binary_function


def register_ttl_outer_function(name, ttl_binary_function, torch_function):
    def _torch_binary(input_tensor_a: Tensor, input_tensor_b: Tensor, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        assert torch_function, f"Torch function not implemented for {str(ttl_binary_function)}"
        return torch_function(input_tensor_a, input_tensor_b)

    @decorate_operation(torch_function=_torch_binary, name=name)
    def binary_function(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor:
        shape_a = [input_tensor_a.shape[0], input_tensor_a.shape[1], input_tensor_a.shape[2], input_tensor_a.shape[3]]
        shape_b = [input_tensor_b.shape[0], input_tensor_b.shape[1], input_tensor_b.shape[2], input_tensor_b.shape[3]]

        if not (shape_a.count(1) == 3) or not (shape_b.count(1) == 3):
            raise RuntimeError("both input_tensors, 3 dimensions are required to be 1 for outer product!")

        if not isinstance(input_tensor_a, Tensor) or not isinstance(input_tensor_b, Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE) or has_storage_type_of(
            input_tensor_b, DEVICE_STORAGE_TYPE
        ):
            raise RuntimeError("Outer op: input_tensors should not be on device!")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value

        ttl_output_tensor = ttl_binary_function(ttl_input_tensor_a, ttl_input_tensor_b)

        output_tensor = Tensor(ttl_output_tensor)
        return output_tensor

    binary_function.__name__ = f"ttnn.{name}"
    binary_function.__doc__ = f"""{name}(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor_a` and  :attr:`input_tensor_b` element-wise.

        .. math::
             \\texttt{{{name}}}(\\mathrm{{input\\_tensor_a}}_i, {{input\\_tensor_b}}_i)

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`

        Example::

            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b)

        """

    setattr(THIS_MODULE, name, binary_function)
    __all__.append(name)
    return binary_function


def register_ttl_concat_function(name, ttl_binary_function, torch_function):
    def _torch_binary(input_tensor_a: Tensor, input_tensor_b: Tensor, dim, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        assert torch_function, f"Torch function not implemented for {str(ttl_binary_function)}"
        return torch_function([input_tensor_a, input_tensor_b], dim)

    @decorate_operation(torch_function=_torch_binary, name=name)
    def binary_function(
        input_tensor_a: Tensor,
        input_tensor_b: Tensor,
        dim: int,
        *,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> Tensor:
        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)
        input_tensor_b = _reshape_to_4D(input_tensor_b)

        if not isinstance(input_tensor_a, Tensor) or not isinstance(input_tensor_b, Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE) or not has_storage_type_of(
            input_tensor_b, DEVICE_STORAGE_TYPE
        ):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value

        ttl_output_tensor = ttl_binary_function(
            [ttl_input_tensor_a, ttl_input_tensor_b], dim, output_mem_config=memory_config
        )

        output_tensor = Tensor(ttl_output_tensor)
        return output_tensor

    binary_function.__name__ = f"ttnn.{name}"
    binary_function.__doc__ = f"""{name}(input_tensor_a: Tensor, input_tensor_b: Tensor, dim: int) -> Tensor

        Applies {name} to :attr:`input_tensor_a` and  :attr:`input_tensor_b` element-wise.

        .. math::
             \\texttt{{{name}}}(\\mathrm{{input\\_tensor_a}}_i, {{input\\_tensor_b}}_i)

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`

        Example::

            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b, dim)

        """
    setattr(THIS_MODULE, name, binary_function)
    __all__.append(name)
    return binary_function


# register functions


def torch_squared_difference(x, y, *args, **kwargs):
    t_diff = torch.sub(x, y)
    return torch.square(t_diff)


def torch_assign_binary(x, y, *args, **kwargs):
    y.copy_(x)
    return y


TTL_BINARY_FUNCTIONS = [
    ("atan2", ttl.tensor.atan2, torch.atan2),
    ("hypot", ttl.tensor.hypot, torch.hypot),
    ("ldexp", ttl.tensor.ldexp, torch.ldexp),
    ("logaddexp", ttl.tensor.logaddexp, torch.logaddexp),
    ("logaddexp2", ttl.tensor.logaddexp2, torch.logaddexp2),
    ("logical_and", ttl.tensor.logical_and, torch.logical_and),
    ("logical_or", ttl.tensor.logical_or, torch.logical_or),
    ("logical_xor", ttl.tensor.logical_xor, torch.logical_xor),
    ("max", ttl.tensor.max, torch.max),
    ("min", ttl.tensor.min, torch.min),
    ("nextafter", ttl.tensor.nextafter, torch.nextafter),
    ("squared_difference", ttl.tensor.squared_difference, torch_squared_difference),
    ("xlogy", ttl.tensor.xlogy, torch.xlogy),
]


for binary_function_name, ttl_binary_function, torch_function in TTL_BINARY_FUNCTIONS:
    register_ttl_binary_function(binary_function_name, ttl_binary_function, torch_function)


TTL_BINARY_FUNCTIONS_WITH_FLOAT_PARAMETER = [
    ("addalpha", ttl.tensor.addalpha, torch.add),
    ("subalpha", ttl.tensor.subalpha, torch.sub),
]

for binary_function_name, ttl_binary_function, torch_function in TTL_BINARY_FUNCTIONS_WITH_FLOAT_PARAMETER:
    register_ttl_binary_function_with_float_parameter(binary_function_name, ttl_binary_function, torch_function)


TTL_FUNCTION_ASSIGN = [
    ("assign", ttl.tensor.assign, torch_assign_binary),
]

for binary_function_name, ttl_binary_function, torch_function in TTL_FUNCTION_ASSIGN:
    register_ttl_assign_function(binary_function_name, ttl_binary_function, torch_function)


TTL_FUNCTION_WITH_MULTIPLE_PARAMETER = [
    ("isclose", ttl.tensor.isclose, torch.isclose),
]

for binary_function_name, ttl_binary_function, torch_function in TTL_FUNCTION_WITH_MULTIPLE_PARAMETER:
    register_ttl_binary_function_with_multiple_parameter(binary_function_name, ttl_binary_function, torch_function)


TTL_FUNCTION_LOSS = [
    ("maeloss", ttl.tensor.maeloss, torch.nn.L1Loss),
    ("mseloss", ttl.tensor.mseloss, torch.nn.MSELoss),
]

for binary_function_name, ttl_binary_function, torch_function in TTL_FUNCTION_LOSS:
    register_ttl_binary_loss_function(binary_function_name, ttl_binary_function, torch_function)


TTL_FUNCTION_OUTER = [
    ("outer", ttl.tensor.outer, torch.outer),
]

for binary_function_name, ttl_binary_function, torch_function in TTL_FUNCTION_OUTER:
    register_ttl_outer_function(binary_function_name, ttl_binary_function, torch_function)


TTL_FUNCTION_CONCAT = [
    ("concat", ttl.tensor.concat, torch.cat),
]

for binary_function_name, ttl_binary_function, torch_function in TTL_FUNCTION_CONCAT:
    register_ttl_concat_function(binary_function_name, ttl_binary_function, torch_function)

Tensor.atan2 = atan2
Tensor.hypot = hypot
Tensor.ldexp = ldexp
Tensor.logaddexp = logaddexp
Tensor.logaddexp2 = logaddexp2
Tensor.logical_and = logical_and
Tensor.logical_or = logical_or
Tensor.logical_xor = logical_xor
Tensor.max = max
Tensor.min = min
Tensor.nextafter = nextafter
Tensor.squared_difference = squared_difference
Tensor.xlogy = xlogy
Tensor.addalpha = addalpha
Tensor.subalpha = subalpha
Tensor.assign = assign
Tensor.isclose = isclose
Tensor.maeloss = maeloss
Tensor.mseloss = mseloss
Tensor.outer = outer
Tensor.concat = concat
