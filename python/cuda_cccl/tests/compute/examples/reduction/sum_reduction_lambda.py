# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Sum all values in an array using reduction with a lambda function.

This example demonstrates that lambda functions can be used directly
as reduction operators, providing a concise alternative to defining
named functions.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Prepare the input and output arrays.
dtype = np.int32
h_init = np.array([0], dtype=dtype)
d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
d_output = cp.empty(1, dtype=dtype)

# Perform the reduction using a lambda function.
def add_op(a, b):
    return a + b
reducer = cuda.compute.make_reduce_into(d_input, d_output, add_op, h_init)
temp_storage_bytes = int(
    reducer(None, d_input, d_output, add_op, len(d_input), h_init, None)
)
d_temp_storage = cp.empty(
    temp_storage_bytes if temp_storage_bytes > 0 else 0, dtype=np.uint8
)
reducer(d_temp_storage, d_input, d_output, add_op, len(d_input), h_init, None)

# Verify the result.
expected_output = 15
assert (d_output == expected_output).all()
result = d_output[0]
print(f"Sum reduction with lambda result: {result}")
