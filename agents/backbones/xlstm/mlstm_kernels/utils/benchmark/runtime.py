#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import gc
import logging
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

LOGGER = logging.getLogger(__name__)


@dataclass
class RuntimeResult:
    runtime: float  # in milliseconds

    peak_memory_allocated: int  # in bytes

    runtime_stats: dict[str, float] = None

    peak_memory_allocated_stats: dict[str, int] = None

    runtime_quantiles: list[float] = None


ReturnModes = Literal["min", "max", "mean", "median"]


def measure_runtime(
    fn: Callable,
    warmup: int = 25,
    rep: int = 100,
    warmup_and_rep_in_ms: bool = True,
    grad_to_none: tuple = None,
    quantiles: list[float] = None,
    fast_flush: bool = True,
    return_mode: ReturnModes = "mean",
    device: str = None,
    free_memory: bool = True,
    profiler: Any = None,
) -> RuntimeResult:
    """
    Benchmark the runtime of the provided function. By default, return the mean.

    Copy with minor adaptations. of the original function from triton.testing.do_bench
    (https://github.com/triton-lang/triton/blob/main/python/triton/testing.py).

    """
    assert return_mode in typing.get_args(
        ReturnModes
    ), f"return_mode must be one of {typing.get_args(ReturnModes)}"
    import torch
    from torch.profiler import record_function
    if profiler is not None:
        assert isinstance(profiler, torch.profiler.profiler.profile), (
            f"profiler must be a torch.profiler.profile object, but got {type(profiler)}."
        )

    device = torch.device(device) if device is not None else None

    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    # compute number of warmup and repeat
    if warmup_and_rep_in_ms:
        # Estimate the runtime of the function
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            cache.zero_()
            fn()
            if profiler is not None:
                profiler.step()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))
    else:
        n_warmup = int(warmup)
        n_repeat = int(rep)

    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

    peak_memory_allocated = []
    # Warm-up
    for i in range(n_warmup):
        LOGGER.debug(f"Warmup iteration {i}")
        with record_function(f"warmup_iter_{i}"):
            fn()
            if profiler is not None:
                profiler.step()
    # Benchmark
    for i in range(n_repeat):
        LOGGER.debug(f"Benchmark iteration {i}")
        if profiler is not None:
            LOGGER.debug(f"Profiler step {profiler.step_num}, Profiler Action {profiler.current_action}")
        with record_function(f"main_iter_{i}"):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if grad_to_none is not None:
                for x in grad_to_none:
                    if hasattr(x, "grad"):
                        x.grad = None
            # we clear the L2 cache before each run
            cache.zero_()
            # reset memory allocated counter
            torch.cuda.reset_peak_memory_stats(device=device)
            # record time of `fn`
            start_event[i].record()
            fn()
            end_event[i].record()
            peak_memory_allocated_iter = torch.cuda.max_memory_allocated(device=device)
            peak_memory_allocated.append(peak_memory_allocated_iter)
            torch.cuda.reset_peak_memory_stats(device=device)
            if free_memory:
                gc.collect()
                torch.cuda.empty_cache()
            if profiler is not None:
                profiler.step()

    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    peak_memory_allocated = torch.tensor(peak_memory_allocated, dtype=torch.int64)
    LOGGER.debug(f"Times: {times}")
    LOGGER.debug(f"Peak memory allocated: {peak_memory_allocated}")

    def get_stat(x, mode):
        return getattr(torch, mode)(x.to(dtype=torch.float64)).item()

    if quantiles is not None:
        runtime_quantiles = torch.quantile(
            times, torch.tensor(quantiles, dtype=torch.float)
        ).tolist()
    else:
        runtime_quantiles = None

    runtime_result = RuntimeResult(
        runtime=get_stat(times, return_mode),
        peak_memory_allocated=int(get_stat(peak_memory_allocated, return_mode)),
        runtime_stats={
            mode: get_stat(times, mode) for mode in typing.get_args(ReturnModes)
        },
        peak_memory_allocated_stats={
            mode: int(get_stat(peak_memory_allocated, mode))
            for mode in typing.get_args(ReturnModes)
        },
        runtime_quantiles=runtime_quantiles,
    )

    return runtime_result
