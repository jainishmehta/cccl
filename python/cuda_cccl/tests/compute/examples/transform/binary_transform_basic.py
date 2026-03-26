# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use binary_transform to perform elementwise addition.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
)

# Prepare the input and output arrays.
input1_data = np.array([1, 2, 3, 4], dtype=np.int32)
input2_data = np.array([10, 20, 30, 40], dtype=np.int32)
d_in1 = cp.asarray(input1_data)
d_in2 = cp.asarray(input2_data)
d_out = cp.empty_like(d_in1)

transformer = cuda.compute.make_binary_transform(d_in1, d_in2, d_out, OpKind.PLUS)
get_bytes = getattr(transformer, "get_temp_storage_bytes", None)
compute = getattr(transformer, "compute", None)
if get_bytes is not None and compute is not None:
    temp_storage_bytes = int(
        get_bytes(
            d_in1,
            d_in2,
            d_out,
            OpKind.PLUS,
            len(d_in1),
        )
    )
    d_temp_storage = (
        None
        if temp_storage_bytes == 0
        else cp.empty(temp_storage_bytes, dtype=np.uint8)
    )
    compute(
        d_temp_storage,
        d_in1,
        d_in2,
        d_out,
        OpKind.PLUS,
        len(d_in1),
    )
else:
    transformer(d_in1, d_in2, d_out, OpKind.PLUS, len(d_in1))

# Verify the result.
result = d_out.get()
expected = input1_data + input2_data

np.testing.assert_array_equal(result, expected)
print(f"Binary transform result: {result}")
