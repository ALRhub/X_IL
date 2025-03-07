#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from ..param_handling import KernelSpec
from .interface import KernelBenchmarkInterface


@dataclass
class mLSTMStepKernelBenchmark(KernelBenchmarkInterface):
    batch_size: int = None
    num_heads: int = None
    head_dim_qk: int = None
    head_dim_v: int = None

    siz_b_DHQK: int = None
    siz_b_DHHV: int = None
    num_warps: int = None
    num_stages: int = None

    use_torch_compile: bool = False

    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        c_old = torch.zeros(
            (self.batch_size, self.num_heads, self.head_dim_qk, self.head_dim_v),
            dtype=torch.float32,
        )
        n_old = torch.zeros(
            (self.batch_size, self.num_heads, self.head_dim_qk), dtype=torch.float32
        )
        m_old = torch.zeros((self.batch_size, self.num_heads, 1), dtype=torch.float32)

        q = torch.randn(
            (self.batch_size, self.num_heads, self.head_dim_qk), dtype=torch.float32
        )
        k = torch.randn(
            (self.batch_size, self.num_heads, self.head_dim_qk), dtype=torch.float32
        )
        v = torch.randn(
            (self.batch_size, self.num_heads, self.head_dim_v), dtype=torch.float32
        )
        i = torch.randn((self.batch_size, self.num_heads, 1), dtype=torch.float32)
        f = torch.randn((self.batch_size, self.num_heads, 1), dtype=torch.float32) + 4.5

        return q, k, v, i, f, c_old, n_old, m_old

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from mlstm_kernels.torch import get_mlstm_step_kernel

        kernel_fn = get_mlstm_step_kernel(self.kernel_name)
        if self.use_torch_compile:
            kernel_fn = torch.compile(kernel_fn)
        return kernel_fn

    def setup_benchmark(self) -> None:
        torch_dtype = getattr(torch, self.dtype)

        inputs = self._get_input_tensors()

        inputs = [x.to(device=self.device, dtype=torch_dtype) for x in inputs]
        self.kernel_inputs = inputs

        kernel_fn = self._get_kernel_fn()
        kernel_fn = partial(
            kernel_fn,
            num_warps=self.num_warps,
            siz_b_DHQK=self.siz_b_DHQK,
            siz_b_DHHV=self.siz_b_DHHV,
            num_stages=self.num_stages,
        )

        def benchmark_fn() -> None:
            output = kernel_fn(*self.kernel_inputs)

        self.benchmark_fn = benchmark_fn

    def available_kernels(self) -> list[str]:
        from mlstm_kernels.torch import get_available_mlstm_step_kernels

        return get_available_mlstm_step_kernels()



@dataclass
class MambaStepKernelBenchmark(KernelBenchmarkInterface):
    batch_size: int = None
    num_heads: int = None
    head_dim_qk: int = None
    head_dim_v: int = None

    use_torch_compile: bool = False

    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        # see: https://github.com/state-spaces/mamba/blob/
        # 442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/ops/triton/selective_state_update.py#L204
        if self.kernel_name == "mamba2":
            state = torch.randn(self.batch_size, self.num_heads, self.head_dim_v, self.head_dim_qk)
            x = torch.randn(self.batch_size, self.num_heads, self.head_dim_v)
            dt = torch.randn(self.batch_size, self.num_heads, self.head_dim_v)
            A = torch.randn(self.num_heads, self.head_dim_v, self.head_dim_qk)
            B = torch.randn(self.batch_size, self.head_dim_qk)
            C = torch.randn(self.batch_size, self.head_dim_qk)
            D = torch.randn(self.num_heads, self.head_dim_v)
            z = torch.randn(self.batch_size, self.num_heads, self.head_dim_v)
            dt_bias = torch.randn(self.num_heads, self.head_dim_v)
        elif self.kernel_name == "mamba":
            state = torch.randn(self.batch_size, self.num_heads*self.head_dim_v, self.head_dim_qk)
            x = torch.randn(self.batch_size, self.num_heads * self.head_dim_v)
            dt = torch.randn(self.batch_size, self.num_heads * self.head_dim_v)
            A = torch.randn(self.num_heads * self.head_dim_v, self.head_dim_qk)
            B = torch.randn(self.batch_size, self.head_dim_qk)
            C = torch.randn(self.batch_size, self.head_dim_qk)
            D = torch.randn(self.num_heads * self.head_dim_v)
            z = torch.randn(self.batch_size, self.num_heads * self.head_dim_v)
            dt_bias = torch.randn(self.num_heads * self.head_dim_v)
        else:
            raise ValueError(f"Bad kernel name {self.kernel_name} not in {self.available_kernels()}")
        return state, x, dt, A, B, C, D, z, dt_bias

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update

        kernel_fn = selective_state_update
        if self.use_torch_compile:
            kernel_fn = torch.compile(kernel_fn)
        return kernel_fn

    def setup_benchmark(self) -> None:
        torch_dtype = getattr(torch, self.dtype)

        inputs = self._get_input_tensors()

        inputs = [
            x.to(device=self.device, dtype=torch_dtype) if i not in [0, 3, 6]
            else x.to(device=self.device)
            for i, x in enumerate(inputs)]
        self.kernel_inputs = inputs

        kernel_fn = self._get_kernel_fn()
        
        def benchmark_fn() -> None:
            output = kernel_fn(*self.kernel_inputs)

        self.benchmark_fn = benchmark_fn

    def available_kernels(self) -> list[str]:
        
        return ["mamba", "mamba2"]



@dataclass
class FlashLinearAttentionStepKernelBenchmark(KernelBenchmarkInterface):
    batch_size: int = None
    num_heads: int = None
    head_dim_qk: int = None
    head_dim_v: int = None

    use_torch_compile: bool = False
    kernel_name: str = "fused_recurrent_gla"

    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        c_old = torch.zeros(
            (self.batch_size, self.num_heads, self.head_dim_qk, self.head_dim_v),
            dtype=torch.float32,
        )
        
        q = torch.randn(
            (self.batch_size, 1, self.num_heads, self.head_dim_qk), dtype=torch.float32
        )
        k = torch.randn(
            (self.batch_size, 1, self.num_heads, self.head_dim_qk), dtype=torch.float32
        )
        v = torch.randn(
            (self.batch_size, 1, self.num_heads, self.head_dim_v), dtype=torch.float32
        )
        if self.kernel_name == "fused_recurrent_gla":
            g = torch.randn((self.batch_size, 1, self.num_heads, self.head_dim_qk), dtype=torch.float32) + 4.5
        elif self.kernel_name == "fused_recurrent_simple_gla":
            g = torch.randn((self.batch_size, 1, self.num_heads), dtype=torch.float32) + 4.5
        else:
            raise ValueError(f"Bad kernel name {self.kernel_name} not in {self.available_kernels()}")

        return c_old, q, k, v, g

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from functools import partial

        from fla.ops.gla import fused_recurrent_gla
        from fla.ops.simple_gla import fused_recurrent_simple_gla

        def kernel_fn(q, k, v, f, initial_state):
            if self.kernel_name == "fused_recurrent_gla":
                return fused_recurrent_gla(
                    q, k, v, f, gv=None, scale=None,
                    initial_state=initial_state, output_final_state=True)
            elif self.kernel_name == "fused_recurrent_simple_gla":
                return fused_recurrent_simple_gla(
                    q, k, v, g=f, scale=None,
                    initial_state=initial_state, output_final_state=True)
            else:
                raise ValueError(f"Bad kernel name {self.kernel_name} not in {self.available_kernels()}")
        
        if self.use_torch_compile:
            kernel_fn = torch.compile(kernel_fn)
        return kernel_fn

    def setup_benchmark(self) -> None:
        torch_dtype = getattr(torch, self.dtype)

        inputs = self._get_input_tensors()

        inputs = [x.to(device=self.device, dtype=torch_dtype) for x in inputs]
        self.kernel_inputs = inputs

        kernel_fn = self._get_kernel_fn()
        
        def benchmark_fn() -> None:
            output = kernel_fn(*self.kernel_inputs)

        self.benchmark_fn = benchmark_fn

    def available_kernels(self) -> list[str]:
        return ["fused_recurrent_gla", "fused_recurrent_simple_gla"]



def create_inference_kernel_benchmark(
    kernel_spec: KernelSpec, param_dict: dict[str, Any]
) -> KernelBenchmarkInterface:
    mlstm_step_kernel_benchmark = mLSTMStepKernelBenchmark()
    fla_step_kernel_benchmark = FlashLinearAttentionStepKernelBenchmark()
    mamba_step_kernel_benchmark = MambaStepKernelBenchmark()


    if kernel_spec.kernel_name in mlstm_step_kernel_benchmark.available_kernels():
        benchmark = mlstm_step_kernel_benchmark
    elif kernel_spec.kernel_name in fla_step_kernel_benchmark.available_kernels():
        benchmark = fla_step_kernel_benchmark
    elif kernel_spec.kernel_name in mamba_step_kernel_benchmark.available_kernels():
        benchmark = mamba_step_kernel_benchmark
    else:
        raise ValueError(f"Unknown kernel name: {kernel_spec.kernel_name}")

    benchmark.kernel_name = kernel_spec.kernel_name
    benchmark.dtype = kernel_spec.dtype
    benchmark.fwbw = kernel_spec.fwbw
    benchmark.use_torch_compile = kernel_spec.use_torch_compile
    benchmark.set_params(param_dict)
    return benchmark
