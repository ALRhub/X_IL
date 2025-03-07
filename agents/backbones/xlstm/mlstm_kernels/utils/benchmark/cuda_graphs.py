#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from typing import Any, Callable

import torch

LOGGER = logging.getLogger(__name__)


def compile_with_cuda_graphs(fn: Callable[..., Any], warmups: int = 3) -> torch.cuda.CUDAGraph:
    """
    Compile the provided function with CUDA graphs.

    Args:
        fn: The function to compile. Should take no arguments.
        warmups: The number of warmup iterations to run.
    
    Returns:
        The compiled CUDA graph. Can be executed with `graph.replay()`.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for idx in range(warmups):
            LOGGER.debug(f"Running CUDA Graph Warmup Step {idx + 1}/{warmups}")
            fn()
        s.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)

    LOGGER.debug("Tracing CUDA Graph for benchmark function.")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    LOGGER.debug("CUDA Graph traced.")

    return graph


def compile_kwargs_with_cuda_graphs(fn: Callable[[Any], Any], inputs: dict, warmups: int = 3, clone_outputs: bool = False) -> tuple[torch.cuda.CUDAGraph, Callable[[Any], Any]]:
    """
    Compile the provided function with CUDA graphs.

    Args:
        fn: The function to compile. Should take no arguments.
        warmups: The number of warmup iterations to run.
    
    Returns:
        The compiled CUDA graph. Can be executed with `graph.replay()`.
    """
    # Warmup.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for idx in range(warmups):
            LOGGER.debug(f"Running CUDA Graph Warmup Step {idx + 1}/{warmups}")
            _ = fn(**inputs)
        s.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)

    # Trace the CUDA graph.
    LOGGER.debug("Tracing CUDA Graph for benchmark function.")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn(**inputs)
    LOGGER.debug("CUDA Graph traced.")

    # Create a replay function, using the input/output buffers.
    def fn_replay(**new_inputs):
        tree_map(
            lambda x, y: x.copy_(y) if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) else None,
            inputs,
            new_inputs,
        )
        graph.replay()
        if clone_outputs:
            return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, outputs)
        else:
            return outputs

    return graph, fn_replay


def tree_map(fn, tree, *rest):
    import jax
    return jax.tree_map(fn, tree, *rest)