# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import tt_lib as ttl
import math
from typing import Union
import ttnn

import torch


THIS_MODULE = sys.modules[__name__]

__all__ = []


def torch_cbrt(x, *args, **kwargs):
    return torch.sgn(x) * torch.pow(torch.abs(x), 1.0 / 3)

def torch_multigammaln(x, *args, **kwargs):
    result = torch.lgamma(x)
    result += torch.lgamma(x - 0.5)
    result += torch.lgamma(x - 1.0)
    result += torch.lgamma(x - 1.5)
    result += 3.434189657547
    return result

def register_ttl_math_function(name, ttl_unary_function, op_name):
    def _torch_math(input_tensor: ttnn.Tensor, **_):
        name_to_torch_function = {
            "abs": torch.abs,
            "cbrt": torch_cbrt,
            "deg2rad": torch.deg2rad,
            "digamma": torch.digamma,
            "erf": torch.erf,
            "erfc": torch.erfc,
            "erfinv": torch.erfinv,
            "exp2": torch.exp2,
            "expm1": torch.expm1,
            "i0": torch.i0,
            "isfinite": torch.isfinite,
            "isinf": torch.inf,
            "isnan": torch.isnan,
            "isneginf": torch.isneginf,
            "isposinf": torch.isposinf,
            "lgamma": torch.lgamma,
            "log10": torch.log10,
            "log1p": torch.log1p,
            "log2": torch.log2,
            "multigammaln": torch_multigammaln,
            "neg": torch.neg,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)
        return torch_function(input_tensor)

    def _math_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_math_validate_input_tensors,
        torch_function=_torch_math,
    )
    def math_function(
        input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        ttl_input_tensor = input_tensor.value

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensor must be on device!")
        ttl_input_tensor = input_tensor.value

        ttl_output_tensor = ttl_unary_function(ttl_input_tensor, output_mem_config=memory_config)

        output_tensor = ttnn.Tensor(ttl_output_tensor)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    math_function.__name__ = f"ttnn.{name}"
    math_function.__doc__ = f"""{name}(input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Applies {name} to :attr:`input_tensor` element-wise.

        .. math::
            {(op_name)}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{name}(tensor)

        {math_function.__doc__}

        """
    setattr(THIS_MODULE, name, math_function)


TTL_MATH_FUNCTIONS = [
    ("abs", ttl.tensor.abs,"abs"),
    ("cbrt", ttl.tensor.cbrt,"cbrt"),
    ("deg2rad", ttl.tensor.deg2rad,"deg2rad"),
    ("digamma", ttl.tensor.digamma,"digamma"),
    ("erf", ttl.tensor.erf,"erf"),
    ("erfc", ttl.tensor.erfc,"erfc"),
    ("erfinv", ttl.tensor.erfinv,"erfinv"),
    ("exp2", ttl.tensor.exp2,"exp2"),
    ("expm1", ttl.tensor.expm1,"expm1"),
    ("i0", ttl.tensor.i0, "i0"),
    ("isfinite", ttl.tensor.isfinite, "isfinite"),
    ("isinf", ttl.tensor.isinf, "isinf"),
    ("isnan", ttl.tensor.isnan, "isnan"),
    ("isneginf", ttl.tensor.isneginf, "isneginf"),
    ("isposinf", ttl.tensor.isposinf, "isposinf"),
    ("lgamma", ttl.tensor.lgamma, "lgamma"),
    ("log10", ttl.tensor.log10, "log10"),
    ("log1p", ttl.tensor.log1p, "log1p"),
    ("log2", ttl.tensor.log2, "log2"),
    ("multigammaln", ttl.tensor.multigammaln, "multigammaln"),
    ("neg", ttl.tensor.neg, "neg"),
]

for math_function_name, ttl_math_function, op_name in TTL_MATH_FUNCTIONS:
    register_ttl_math_function(math_function_name, ttl_math_function, op_name)

__all__ = []
