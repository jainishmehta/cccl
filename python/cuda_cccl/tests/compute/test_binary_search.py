# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import cupy as cp
import numpy as np
import pytest

import cuda.compute

DTYPE_LIST = [
    np.int32,
    np.int64,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
]


def random_sorted_array(size, dtype, max_value=1000):
    rng = np.random.default_rng()
    if np.isdtype(dtype, "integral"):
        data = rng.integers(max_value, size=size, dtype=dtype)
    else:
        if dtype == np.float16:  # pragma: no cover - float16 not used here
            data = rng.random(size=size, dtype=np.float32).astype(dtype)
        else:
            data = rng.random(size=size, dtype=dtype)
    data.sort()
    return data


@pytest.fixture(scope="function", autouse=True)
def disable_sass_check(monkeypatch):
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )


def binary_search_object_api(factory, d_data, d_values, d_out, *, num_items, num_values, comp, stream=None):
    searcher = factory(d_data, d_values, d_out, comp)
    nbytes = int(
        searcher(None, d_data, d_values, d_out, comp, num_items, num_values, stream)
    )
    d_temp = cp.empty(nbytes if nbytes > 0 else 0, dtype=np.uint8)
    searcher(d_temp, d_data, d_values, d_out, comp, num_items, num_values, stream)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize(
    "num_items,num_values", [(0, 0), (0, 128), (128, 0), (512, 128)]
)
def test_lower_bound_basic(dtype, num_items, num_values):
    h_data = random_sorted_array(num_items, dtype)
    h_values = random_sorted_array(num_values, dtype)

    d_data = cp.asarray(h_data)
    d_values = cp.asarray(h_values)
    d_out = cp.empty(num_values, dtype=np.uintp)

    binary_search_object_api(
        cuda.compute.make_lower_bound,
        d_data,
        d_values,
        d_out,
        num_items=num_items,
        num_values=num_values,
        comp=None,
    )

    expected = np.searchsorted(h_data, h_values, side="left").astype(np.uintp)
    got = cp.asnumpy(d_out)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize(
    "num_items,num_values", [(0, 0), (0, 128), (128, 0), (512, 128)]
)
def test_upper_bound_basic(dtype, num_items, num_values):
    h_data = random_sorted_array(num_items, dtype)
    h_values = random_sorted_array(num_values, dtype)

    d_data = cp.asarray(h_data)
    d_values = cp.asarray(h_values)
    d_out = cp.empty(num_values, dtype=np.uintp)

    binary_search_object_api(
        cuda.compute.make_upper_bound,
        d_data,
        d_values,
        d_out,
        num_items=num_items,
        num_values=num_values,
        comp=None,
    )

    expected = np.searchsorted(h_data, h_values, side="right").astype(np.uintp)
    got = cp.asnumpy(d_out)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_binary_search_with_duplicates(dtype):
    rng = np.random.default_rng()
    h_data = (
        rng.integers(10, size=1024, dtype=dtype)
        if np.isdtype(dtype, "integral")
        else rng.random(1024, dtype=dtype)
    )
    h_data.sort()
    h_values = (
        rng.integers(10, size=128, dtype=dtype)
        if np.isdtype(dtype, "integral")
        else rng.random(128, dtype=dtype)
    )

    d_data = cp.asarray(h_data)
    d_values = cp.asarray(h_values)
    d_out = cp.empty(len(h_values), dtype=np.uintp)
    n_items = len(h_data)
    n_values = len(h_values)

    binary_search_object_api(
        cuda.compute.make_lower_bound,
        d_data,
        d_values,
        d_out,
        num_items=n_items,
        num_values=n_values,
        comp=None,
    )
    expected = np.searchsorted(h_data, h_values, side="left").astype(np.uintp)
    got = cp.asnumpy(d_out)
    assert np.array_equal(got, expected)

    binary_search_object_api(
        cuda.compute.make_upper_bound,
        d_data,
        d_values,
        d_out,
        num_items=n_items,
        num_values=n_values,
        comp=None,
    )
    expected = np.searchsorted(h_data, h_values, side="right").astype(np.uintp)
    got = cp.asnumpy(d_out)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("factory", [cuda.compute.make_lower_bound, cuda.compute.make_upper_bound])
def test_binary_search_requires_unsigned_output(factory):
    """Output must be unsigned integer dtype for indices."""
    d_data = cp.asarray(np.array([1, 2, 3, 4], dtype=np.int32))
    d_values = cp.asarray(np.array([2, 3], dtype=np.int32))
    d_out = cp.empty(len(d_values), dtype=np.int32)
    num_items, num_values = 4, 2

    with pytest.raises(TypeError, match="unsigned integer"):
        binary_search_object_api(
            factory,
            d_data,
            d_values,
            d_out,
            num_items=num_items,
            num_values=num_values,
            comp=None,
        )


@pytest.mark.parametrize("factory", [cuda.compute.make_lower_bound, cuda.compute.make_upper_bound])
def test_binary_search_requires_pointer_sized_output(factory, bad_dtype):
    """Output must be pointer-sized (np.uintp) to hold any valid index."""
    d_data = cp.asarray(np.array([1, 2, 3, 4], dtype=np.int32))
    d_values = cp.asarray(np.array([2, 3], dtype=np.int32))
    d_out = cp.empty(len(d_values), dtype=bad_dtype)
    num_items, num_values = 4, 2

    with pytest.raises(ValueError, match="pointer-sized"):
        binary_search_object_api(
            factory,
            d_data,
            d_values,
            d_out,
            num_items=num_items,
            num_values=num_values,
            comp=None,
        )
