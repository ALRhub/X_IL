#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from ..param_handling import KernelSpec, ModelSpec
from ..runtime import RuntimeResult, measure_runtime

LOGGER = logging.getLogger(__name__)

BenchmarkCreator = Callable[[KernelSpec, dict[str, Any]], "BenchmarkInterface"]
ModelBenchmarkCreator = Callable[[ModelSpec, dict[str, Any]], "ModelBenchmarkInterface"]


@dataclass
class BenchmarkInterface(ABC):
    warmup: int = 25
    """Warmup time (in ms) or warmup iterations."""
    rep: int = 1000
    """Repetition time (in ms) or repetition iterations."""

    warmup_and_rep_in_ms: bool = False
    """If true, the warmup and rep are in milliseconds, otherwise they are iterations."""

    device: str = "cuda"
    """The device to run the benchmark on."""

    dtype: Literal["float16", "float32", "float64", "bfloat16"] = "bfloat16"
    """The data type to use for the benchmark."""

    benchmark_fn: Callable = None
    """The benchmark function to run."""

    def set_params(self, param_dict: dict) -> None:
        """Used to set all or multiple parameters of the benchmark at once."""
        if param_dict is None:
            return
        for k, v in param_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown parameter: {k}")

    @abstractmethod
    def setup_benchmark(self) -> None:
        """Sets up the benchmark function to run."""
        raise NotImplementedError

    def run_benchmark(
        self,
        return_mode: Literal["mean", "median"] = "mean",
        grad_to_none: tuple[torch.Tensor, ...] | None = None,
        profiler=None,
    ) -> RuntimeResult:
        """Runs the benchmark and returns the runtime in milliseconds."""

        if self.benchmark_fn is None:
            raise RuntimeError("The benchmark function has not been set up.")
        try:
            runtime = measure_runtime(
                self.benchmark_fn,
                warmup=self.warmup,
                rep=self.rep,
                warmup_and_rep_in_ms=self.warmup_and_rep_in_ms,
                return_mode=return_mode,
                grad_to_none=grad_to_none,
                device=self.device,
                profiler=profiler,
            )
        except (Exception, torch.OutOfMemoryError, RuntimeError) as e:
            LOGGER.warning(f"Error: {e}")
            LOGGER.warning(traceback.format_exc())
            runtime = RuntimeResult(runtime=float("nan"), peak_memory_allocated=-1)
        return runtime


BenchmarkFnContextManagerCfgType = Literal["none", "no_grad", "inference_mode"]


@dataclass
class ModelBenchmarkInterface(BenchmarkInterface):
    model: nn.Module = None
    """The model to benchmark."""
    use_torch_compile_model: bool = True
    """If true, the model will be wrapped with torch.compile."""
    benchmark_fn_context_manager: BenchmarkFnContextManagerCfgType = "none"

    def setup_model(self) -> None:
        """Use this method to setup (i.e. load & initialize) the model.
        Can improve the runtime of the benchmark.
        Note that this method is optional and will be called once before the varying
        parameters are set and the benchmark is run.
        CAREFUL: If there is a model parameter in the varying parameters, the
        benchmark will not run the correct model.
        """
        pass


@dataclass
class KernelBenchmarkInterface(BenchmarkInterface):
    fwbw: bool = True
    """If true, the benchmark will run the forward and backward pass."""

    kernel_inputs: tuple[torch.Tensor, ...] = None
    """The input tensors to the benchmark function."""

    kernel_name: str = None
    """The name of the kernel to benchmark."""

    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        """Returns the kernel function to benchmark.
        The inputs to the kernel function are the tensors returned by `_get_input_tensors`."""
        raise NotImplementedError

    def _get_loss_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """By default use the sum of the output as loss."""

        def loss_fn(output: torch.Tensor) -> torch.Tensor:
            return torch.sum(output)

        return loss_fn

    def available_kernels(self) -> list[str]:
        """Returns the available kernel names for the benchmark."""
        raise NotImplementedError

    def setup_benchmark(self) -> None:
        torch_dtype = getattr(torch, self.dtype)

        inputs = self._get_input_tensors()

        inputs = [
            x.to(device=self.device, dtype=torch_dtype).requires_grad_(self.fwbw)
            if isinstance(x, torch.Tensor)
            else x
            for x in inputs
        ]
        self.kernel_inputs = inputs

        kernel_fn = self._get_kernel_fn()

        loss_fn = self._get_loss_fn()

        def benchmark_fn() -> None:
            output = kernel_fn(*self.kernel_inputs)
            if self.fwbw:
                loss = loss_fn(output)
                loss.backward()

        self.benchmark_fn = benchmark_fn

    def run_benchmark(
        self,
        return_mode: Literal["mean"] | Literal["median"] = "mean",
        grad_to_none: tuple[torch.Tensor, ...] | None = None,
    ) -> RuntimeResult:
        return super().run_benchmark(return_mode, grad_to_none)
