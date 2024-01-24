API
###

Tensor
******

.. toctree::
   :maxdepth: 1

   ttnn/from_torch
   ttnn/to_torch
   ttnn/to_device
   ttnn/from_device
   ttnn/to_layout
   ttnn/dump_tensor
   ttnn/load_tensor
   ttnn/deallocate
   ttnn/reallocate
   ttnn/to_memory_config


Memory Config
*************

.. toctree::
   :maxdepth: 1

   ttnn/create_sharded_memory_config


Operations
**********

Matrix Multiplication
=====================

.. toctree::
   :maxdepth: 1

   ttnn/matmul
   ttnn/linear

Convolution
=====================

.. toctree::
   :maxdepth: 1

   ttnn/Conv2D

Pointwise Unary
================

.. toctree::
   :maxdepth: 1

   ttnn/exp
   ttnn/gelu
   ttnn/log
   ttnn/relu
   ttnn/rsqrt
   ttnn/silu
   ttnn/softmax
   ttnn/tanh

Pointwise Binary
================

.. toctree::
   :maxdepth: 1

   ttnn/add
   ttnn/mul
   ttnn/sub
   ttnn/pow

Reduction
=========

.. toctree::
   :maxdepth: 1

   ttnn/mean

Data Movement
=============

.. toctree::
   :maxdepth: 1

   ttnn/concat
   ttnn/pad
   ttnn/permute
   ttnn/reshape
   ttnn/split
   ttnn/repeat_interleave

Normalization
=============

.. toctree::
   :maxdepth: 1

   ttnn/group_norm
   ttnn/layer_norm
   ttnn/rms_norm

Transformer
===========

.. toctree::
   :maxdepth: 1

   ttnn/transformer/split_query_key_value_and_split_heads
   ttnn/transformer/concatenate_heads
   ttnn/transformer/attention_softmax
   ttnn/transformer/attention_softmax_

Embedding
=========

.. toctree::
   :maxdepth: 1

   ttnn/embedding


Model Conversion
****************

.. toctree::
   :maxdepth: 1

   ttnn/model_preprocessing/preprocess_model_parameters
