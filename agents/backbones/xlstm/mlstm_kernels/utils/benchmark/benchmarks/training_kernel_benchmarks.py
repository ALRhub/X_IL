#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..param_handling import KernelSpec
from .interface import KernelBenchmarkInterface

LOGGER = logging.getLogger(__name__)


@dataclass
class mLSTMBenchmark(KernelBenchmarkInterface):
    batch_size: int = None
    sequence_length: int = None
    num_heads: int = None
    head_dim_qk: int = None
    head_dim_v: int = None

    chunk_size: int = None

    kernel_name: str = None

    use_torch_compile: bool = False

    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        q = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
            dtype=torch.float32,
        )
        k = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
            dtype=torch.float32,
        )
        v = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_v),
            dtype=torch.float32,
        )
        ig = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length),
            dtype=torch.float32,
        )
        fg = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length),
            dtype=torch.float32,
        )
        return q, k, v, ig, fg

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from functools import partial

        from mlstm_kernels.torch import get_mlstm_kernel

        kernel_fn = get_mlstm_kernel(self.kernel_name)
        kernel_fn = partial(kernel_fn, chunk_size=self.chunk_size)
        if self.use_torch_compile:
            kernel_fn = torch.compile(kernel_fn)
        return kernel_fn

    def available_kernels(self) -> list[str]:
        from mlstm_kernels.torch import get_available_mlstm_kernels

        return get_available_mlstm_kernels()


@dataclass
class FlashAttentionBenchmark(mLSTMBenchmark):
    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        # possibly sequence_length and num_heads is transposed for PyTorch
        # versions < 2.5
        q = torch.randn(
            (
                self.batch_size,
                self.num_heads,
                self.sequence_length,
                self.head_dim_qk,
            ),
            dtype=torch.float32,
        )
        k = torch.randn(
            (
                self.batch_size,
                self.num_heads,
                self.sequence_length,
                self.head_dim_qk,
            ),
            dtype=torch.float32,
        )
        v = torch.randn(
            (
                self.batch_size,
                self.num_heads,
                self.sequence_length,
                self.head_dim_v,
            ),
            dtype=torch.float32,
        )
        return q, k, v

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from mlstm_kernels.baselines.flash_attention import (
            registry as flash_attention_registry,
        )

        kernel_fn = flash_attention_registry[self.kernel_name]

        return kernel_fn

    def available_kernels(self) -> list[str]:
        from mlstm_kernels.baselines.flash_attention import (
            registry as flash_attention_registry,
        )

        return list(flash_attention_registry.keys())


@dataclass
class mLSTMXLChunkSizeTuneBenchmark(mLSTMBenchmark):
    chunk_size_inter: int = None
    chunk_size_intra: int = None
    siz_b_L_parallel: int = None
    siz_b_L_loop: int = None
    siz_b_DH_parallel: int = None
    siz_b_DH_loop: int = None
    num_warps_intra: int = None
    num_warps_inter: int = None
    num_stages_intra: int = None
    num_stages_inter: int = None
    recompute_states_in_bw: bool = True
    output_dtype: Literal["bfloat16", "float32"] = "float32"

    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        q = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
            dtype=torch.float32,
        )
        k = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
            dtype=torch.float32,
        )
        v = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_v),
            dtype=torch.float32,
        )
        ig = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length),
            dtype=torch.float32,
        )
        fg = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length),
            dtype=torch.float32,
        )
        return q, k, v, ig, fg

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from functools import partial

        assert (
            self.kernel_name == "mlstm_chunkwise__xl_chunk"
        ), "Only supports mlstm_chunkwise__xl_chunk kernel"

        from mlstm_kernels.torch.chunkwise.triton_xl_chunk import (
            mlstm_chunkwise__xl_chunk,
        )

        kernel_fn = mlstm_chunkwise__xl_chunk
        kernel_fn = partial(
            kernel_fn,
            chunk_size_inter=self.chunk_size_inter,
            chunk_size_intra=self.chunk_size_intra,
            siz_b_L_parallel=self.siz_b_L_parallel,
            siz_b_L_loop=self.siz_b_L_loop
            if self.siz_b_L_loop is not None
            else self.siz_b_L_parallel,
            siz_b_DH_parallel=self.siz_b_DH_parallel,
            siz_b_DH_loop=self.siz_b_DH_loop,
            num_warps_intra=self.num_warps_intra,
            num_warps_inter=self.num_warps_inter,
            num_stages_intra=self.num_stages_intra,
            num_stages_inter=self.num_stages_inter,
            recompute_states_in_bw=self.recompute_states_in_bw,
        )

        return kernel_fn

    def available_kernels(self) -> list[str]:
        return ["mlstm_chunkwise__xl_chunk"]


@dataclass
class FlashLinearAttentionKernelBenchmark(KernelBenchmarkInterface):
    batch_size: int = None
    num_heads: int = None
    sequence_length: int = None
    head_dim_qk: int = None
    head_dim_v: int = None

    use_torch_compile: bool = False

    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        q = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
            dtype=torch.float32,
        )
        k = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
            dtype=torch.float32,
        )
        v = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_v),
            dtype=torch.float32,
        )
        if "simple_gla" in self.kernel_name:
            fg = torch.randn(
                (self.batch_size, self.num_heads, self.sequence_length),
                dtype=torch.float32,
            )
            return q, k, v, fg
        if "gla" in self.kernel_name:
            fg = torch.randn(
                (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
                dtype=torch.float32,
            )
            return q, k, v, fg
        return q, k, v

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from fla.ops.gla import chunk_gla, fused_chunk_gla
        from fla.ops.retention import chunk_retention, parallel_retention
        from fla.ops.retention.naive import naive_retention
        from fla.ops.simple_gla import chunk_simple_gla

        if self.kernel_name == "chunk_gla":
            kernel_pre_fn = chunk_gla
        elif self.kernel_name == "fused_chunk_gla":
            kernel_pre_fn = fused_chunk_gla
        elif self.kernel_name == "chunk_simple_gla":
            kernel_pre_fn = chunk_simple_gla
        elif self.kernel_name == "chunk_retention":
            kernel_pre_fn = chunk_retention
        elif self.kernel_name == "parallel_retention":
            kernel_pre_fn = parallel_retention
        elif self.kernel_name == "naive_retention":
            kernel_pre_fn = naive_retention
        else:
            raise ValueError(f"Bad kernel name {self.kernel_name} not in {self.available_kernels()}")

        # only take first output of tuple
        def kernel_fn(*args, **kwargs):
            return kernel_pre_fn(*args, **kwargs)[0]

        if self.use_torch_compile:
            kernel_fn = torch.compile(kernel_fn)
        return kernel_fn

    def available_kernels(self) -> list[str]:
        return [
            "chunk_gla", "fused_chunk_gla", "chunk_simple_gla",
            "chunk_retention", "parallel_retention", "naive_retention",]


@dataclass
class MambaKernelBenchmark(KernelBenchmarkInterface):
    batch_size: int = None
    num_heads: int = None
    sequence_length: int = None
    head_dim_qk: int = None
    head_dim_v: int = None
    # num groups in Mamba is similar to an additional head dimension within the keys / queries
    # it has to divide the number of heads and means that the same queries and keys are re-used for 
    # nheads/ngroups values
    # See: https://github.com/state-spaces/mamba/blob/
    # 442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/ops/triton/selective_state_update.py#L255
    # it is probably inspired by `arXiV:2305.13245`
    num_groups: int = 1
    width: int = 4  # convolution kernel size

    use_torch_compile: bool = False

    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        # equivalence to Mamba notation:
        # head_dim_qk === state_dim
        # head_dim_v === inner_dim
        # num_heads === num_heads
        # For Mamba (v1): inner_dim = self.num_heads * self.head_dim_v

        # see: https://github.com/state-spaces/mamba/blob/
        # 442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/ops/selective_scan_interface.py#L91
        if "mamba" == self.kernel_name:
            x = torch.randn(
                (self.batch_size, self.num_heads*self.head_dim_v, self.sequence_length,),
                dtype=torch.float32,
            )
            dt = torch.randn(
                (self.batch_size, self.num_heads*self.head_dim_v, self.sequence_length,),
                dtype=torch.float32,
            )
            A = torch.randn(
                (self.num_heads*self.head_dim_v, self.head_dim_qk),
                dtype=torch.float32,
            )
            B = torch.randn(
                (self.batch_size, self.head_dim_qk, self.sequence_length),
                dtype=torch.float32,
            )
            C = torch.randn(
                (self.batch_size, self.head_dim_qk, self.sequence_length),
                dtype=torch.float32,
            )
            D = torch.randn(
                (self.num_heads*self.head_dim_v),
                dtype=torch.float32,
            )
            z = torch.randn(
                (self.batch_size, self.num_heads*self.head_dim_v, self.sequence_length,),
                dtype=torch.float32,
            )
            return x, dt, A, B, C, D, z
        # see: https://github.com/state-spaces/mamba/blob/
        # 442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/ops/triton/ssd_combined.py#L933
        if "mamba2_noconv" == self.kernel_name:
            x = torch.randn(
                (self.batch_size, self.sequence_length,  self.num_heads, self.head_dim_v),
                dtype=torch.float32,
            )
            dt = torch.randn(
                (self.batch_size, self.sequence_length,  self.num_heads),
                dtype=torch.float32,
            )
            A = torch.randn(
                (self.num_heads),
                dtype=torch.float32,
            )
            B = torch.randn(
                (self.batch_size, self.sequence_length, self.num_heads, self.head_dim_qk),
                dtype=torch.float32,
            )
            C = torch.randn(
                (self.batch_size, self.sequence_length, self.num_heads, self.head_dim_qk),
                dtype=torch.float32,
            )
            chunk_size = 256

            return x, dt, A, B, C, chunk_size
        # see: https://github.com/state-spaces/mamba/blob/
        # 442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/ops/triton/ssd_combined.py#L933
        if "mamba2" == self.kernel_name:
            zxbcdt = torch.randn(
                (self.batch_size, self.sequence_length,  
                2*self.num_heads *self.head_dim_v + 2*self.head_dim_qk*self.num_groups + self.num_heads),
                dtype=torch.float32,
            )
            conv1d_weight = torch.randn(
                (self.num_heads*self.head_dim_v + 2*self.num_groups*self.head_dim_qk, self.width),
                dtype=torch.float32,
            )
            conv1d_bias = torch.randn(
                (self.num_heads*self.head_dim_v + 2*self.num_groups*self.head_dim_qk),
                dtype=torch.float32,
            )
            dt_bias = torch.randn(
                (self.num_heads),
                dtype=torch.float32,
            )
            A = torch.randn(
                (self.num_heads),
                dtype=torch.float32,
            )
            D = torch.randn(
                (self.num_heads, self.head_dim_v),
                dtype=torch.float32,
            )
            chunk_size = 256
            return zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size
        raise ValueError(f"Bad kernel name {self.kernel_name} not in {self.available_kernels()}")

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        from mamba_ssm.ops.triton.ssd_combined import (
            mamba_chunk_scan_combined,
            mamba_split_conv1d_scan_combined,
        )
        
        if self.kernel_name == "mamba":
            kernel_pre_fn = selective_scan_fn
        elif self.kernel_name == "mamba2_noconv":
            kernel_pre_fn = mamba_chunk_scan_combined
        elif self.kernel_name == "mamba2":
            kernel_pre_fn = mamba_split_conv1d_scan_combined
        else:
            raise ValueError(f"Bad kernel name {self.kernel_name} not in {self.available_kernels()}")

        # only take first output of tuple
        def kernel_fn(*args, **kwargs):
            return kernel_pre_fn(*args, **kwargs)[0]

        if self.use_torch_compile:
            kernel_fn = torch.compile(kernel_fn)
        return kernel_fn


    def setup_benchmark(self) -> None:
        torch_dtype = getattr(torch, self.dtype)

        inputs = self._get_input_tensors()

        inputs = [
            x.to(device=self.device).requires_grad_(self.fwbw) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]

        # see 

        if self.kernel_name == "mamba":
            x, dt, A, B, C, D, z = inputs
            x, dt, B, C, z = (
                x.to(dtype=torch_dtype), dt.to(dtype=torch_dtype),
                B.to(dtype=torch_dtype), C.to(dtype=torch_dtype),
                z.to(dtype=torch_dtype)
            )
            inputs = x, dt, A, B, C, D, z
        elif self.kernel_name == "mamba2_noconv":
            x, dt, A, B, C, chunk_size = inputs
            x, dt, B, C = (
                x.to(dtype=torch_dtype),
                dt.to(dtype=torch_dtype),
                B.to(dtype=torch_dtype),
                C.to(dtype=torch_dtype)
            )
            inputs = x, dt, A, B, C, chunk_size
        elif self.kernel_name == "mamba2":
            zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size = inputs
            zxbcdt, conv1d_weight, conv1d_bias, dt_bias, D = (
                zxbcdt.to(dtype=torch_dtype),
                conv1d_weight.to(dtype=torch_dtype),
                conv1d_bias.to(dtype=torch_dtype),
                dt_bias.to(dtype=torch_dtype),
                D.to(dtype=torch_dtype),
            )
            inputs = zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size
        self.kernel_inputs = inputs

        kernel_fn = self._get_kernel_fn()

        loss_fn = self._get_loss_fn()

        def benchmark_fn() -> None:
            output = kernel_fn(*self.kernel_inputs)
            if self.fwbw:
                loss = loss_fn(output)
                loss.backward()

        self.benchmark_fn = benchmark_fn


    def available_kernels(self) -> list[str]:
        return [
            "mamba", "mamba2", "mamba2_noconv"]




@dataclass
class FlashAttention3Benchmark(mLSTMBenchmark):
    batch_size: int = None
    sequence_length: int = None
    num_heads: int = None
    head_dim_qk: int = None
    head_dim_v: int = None

    chunk_size: int = None

    kernel_name: str = None

    use_torch_compile: bool = False
    
    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        q = torch.randn(
            (self.batch_size, self.sequence_length, self.num_heads, self.head_dim_qk),
            dtype=torch.float32,
        )
        k = torch.randn(
            (self.batch_size, self.sequence_length, self.num_heads, self.head_dim_qk),
            dtype=torch.float32,
        )
        v = torch.randn(
            (self.batch_size, self.sequence_length, self.num_heads, self.head_dim_v),
            dtype=torch.float32,
        )
        return q, k, v

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from functools import partial

        assert (
            self.kernel_name == "flashattn3"
        ), "Only supports flashattn3"

        import os
        import sys
        from pathlib import Path
        sys.path.append(str(Path(os.path.abspath(__file__)).parents[5] / "flash-attention" / "hopper"))
        from flash_attn_interface import flash_attn_func

        kernel_fn = partial(
            flash_attn_func,
            causal=True,
            deterministic=False,
            gqa_parallel=False,
            softmax_scale=None,
            window_size=(-1, -1),
            descale_q=None,
            descale_k=None,
            descale_v=None,
        )

        def kernel_fn2(q, k, v):
            return kernel_fn(q, k, v)[0]
        
        return kernel_fn2

    def available_kernels(self) -> list[str]:
        return ["flashattn3"]



def create_training_kernel_benchmark(
    kernel_spec: KernelSpec, param_dict: dict[str, Any]
) -> KernelBenchmarkInterface:
    mlstm_benchmark = mLSTMBenchmark()
    flashattention_benchmark = FlashAttentionBenchmark()
    mlstm_xl_chunk_size_tune_benchmark = mLSTMXLChunkSizeTuneBenchmark()
    flashlinearattention_benchmark = FlashLinearAttentionKernelBenchmark()
    mamba_benchmark = MambaKernelBenchmark()
    flashattn3_benchmark = FlashAttention3Benchmark()

    all_available_kernels = (
        mlstm_benchmark.available_kernels()
        + flashattention_benchmark.available_kernels()
        + mlstm_xl_chunk_size_tune_benchmark.available_kernels()
        + flashlinearattention_benchmark.available_kernels()
        + mamba_benchmark.available_kernels()
        + flashattn3_benchmark.available_kernels()
    )

    if kernel_spec.kernel_name in mlstm_benchmark.available_kernels():
        benchmark = mlstm_benchmark
    elif kernel_spec.kernel_name in flashattention_benchmark.available_kernels():
        benchmark = flashattention_benchmark
    elif (
        kernel_spec.kernel_name
        in mlstm_xl_chunk_size_tune_benchmark.available_kernels()
    ):
        benchmark = mlstm_xl_chunk_size_tune_benchmark
    elif (
        kernel_spec.kernel_name
        in flashlinearattention_benchmark.available_kernels()
    ):
        benchmark = flashlinearattention_benchmark
    
    elif (
        kernel_spec.kernel_name
        in mamba_benchmark.available_kernels()
    ):
        benchmark = mamba_benchmark
    elif (
        kernel_spec.kernel_name
        in flashattn3_benchmark.available_kernels()
    ):
        benchmark = flashattn3_benchmark
    else:
        raise ValueError(
            f"Unknown kernel name: {kernel_spec.kernel_name}, available kernels: {all_available_kernels}"
        )

    benchmark.kernel_name = kernel_spec.kernel_name
    benchmark.dtype = kernel_spec.dtype
    benchmark.fwbw = kernel_spec.fwbw
    benchmark.use_torch_compile = kernel_spec.use_torch_compile
    benchmark.set_params(param_dict)
    return benchmark
