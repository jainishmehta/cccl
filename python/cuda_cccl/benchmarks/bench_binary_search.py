# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import cupy as cp
import numpy as np
import pytest

import cuda.compute

def lower_bound_run(d_data, d_values, d_out, build_only):
    searcher = cuda.compute.make_lower_bound(d_data, d_values, d_out)
    if not build_only:
        comp = None
        stream = None
        num_items = len(d_data)
        num_values = len(d_values)
        try:
            nbytes = int(
                searcher(
                    None,
                    d_data,
                    d_values,
                    d_out,
                    comp,
                    num_items,
                    num_values,
                    stream,
                )
            )
            d_temp = cp.empty(nbytes if nbytes > 0 else 0, dtype=np.uint8)
            searcher(
                d_temp, d_data, d_values, d_out, comp, num_items, num_values, stream
            )
        except TypeError:
            searcher(d_data, d_values, d_out, comp, num_items, num_values, stream)
    cp.cuda.runtime.deviceSynchronize()


def upper_bound_run(d_data, d_values, d_out, build_only):
    searcher = cuda.compute.make_upper_bound(d_data, d_values, d_out)
    if not build_only:
        comp = None
        stream = None
        num_items = len(d_data)
        num_values = len(d_values)
        get_bytes = getattr(searcher, "get_temp_storage_bytes", None)
        if get_bytes is not None:
            try:
                temp_storage_bytes = int(
                    get_bytes(
                        d_data,
                        d_values,
                        d_out,
                        comp=comp,
                        num_items=num_items,
                        num_values=num_values,
                    )
                )
                d_temp_storage = cp.empty(
                    temp_storage_bytes if temp_storage_bytes > 0 else 0, dtype=np.uint8
                )
                searcher(
                    d_temp_storage,
                    d_data,
                    d_values,
                    d_out,
                    comp,
                    num_items,
                    num_values,
                )
            except (TypeError, AttributeError):
                # Older wheels can expose different probing signatures and may crash
                # if passed None as temp storage. Fall back to one-shot calls.
                pass
            else:
                cp.cuda.runtime.deviceSynchronize()
                return

        try:
            searcher(d_data, d_values, d_out, comp, num_items, num_values, stream)
        except TypeError:
            try:
                searcher(d_data, d_values, d_out, comp, num_items, num_values)
            except TypeError:
                searcher(d_data, d_values, d_out, comp)
    cp.cuda.runtime.deviceSynchronize()


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_lower_bound(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    d_data = cp.sort(cp.random.randint(0, 1000, actual_size, dtype=np.int32))
    d_values = cp.random.randint(0, 1000, actual_size, dtype=np.int32)
    d_out = cp.empty_like(d_values, dtype=np.uintp)

    def run():
        lower_bound_run(
            d_data, d_values, d_out, build_only=(bench_fixture == "compile_benchmark")
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_upper_bound(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    d_data = cp.sort(cp.random.randint(0, 1000, actual_size, dtype=np.int32))
    d_values = cp.random.randint(0, 1000, actual_size, dtype=np.int32)
    d_out = cp.empty_like(d_values, dtype=np.uintp)

    def run():
        upper_bound_run(
            d_data, d_values, d_out, build_only=(bench_fixture == "compile_benchmark")
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)
