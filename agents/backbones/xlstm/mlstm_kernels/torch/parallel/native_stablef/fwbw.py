#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
PyTorch

mLSTM forward and backward pass. Parallel formulation.
"""

from collections.abc import Callable

import torch
from torch.amp import custom_bwd, custom_fwd

from ...utils import contiguous
from .bw import mlstm_parallel_bw
from .fw import mlstm_parallel_fw


def _mlstm_parallel_fwbw_generator(autocast_kernel_dtype=torch.float32) -> Callable:
    class _mlstm_parallel_fwbw(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda", cast_inputs=autocast_kernel_dtype)
        @contiguous
        def forward(
            ctx,
            matQ: torch.Tensor,
            matK: torch.Tensor,
            matV: torch.Tensor,
            vecI: torch.Tensor,
            vecF: torch.Tensor,
            eps: float = 1e-6,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            matH, vecN, vecM = mlstm_parallel_fw(
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                eps=eps,
            )
            ctx.save_for_backward(matQ, matK, matV, vecI, vecF, vecN, vecM, torch.tensor(eps))
            return matH, vecN, vecM

        @staticmethod
        @custom_bwd(device_type="cuda")
        @contiguous
        def backward(
            ctx,
            matDeltaHtilde: torch.Tensor,
            vecDeltaN_unused: torch.Tensor,
            vecDeltaM_unused: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            (matQ, matK, matV, vecI, vecF, vecN, vecM, eps) = ctx.saved_tensors
            matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF = mlstm_parallel_bw(
                matDeltaHtilde=matDeltaHtilde,
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                vecN=vecN,
                vecM=vecM,
                eps=float(eps),
            )
            return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF, None

    return _mlstm_parallel_fwbw


_mlstm_parallel_fwbw_float32 = _mlstm_parallel_fwbw_generator(autocast_kernel_dtype=torch.float32)
_mlstm_parallel_fwbw_float16 = _mlstm_parallel_fwbw_generator(autocast_kernel_dtype=torch.float16)
_mlstm_parallel_fwbw_bfloat16 = _mlstm_parallel_fwbw_generator(autocast_kernel_dtype=torch.bfloat16)


def _get_parallel_fwbw_kernel(autocast_kernel_dtype: torch.dtype) -> Callable:
    if autocast_kernel_dtype == torch.float32:
        return _mlstm_parallel_fwbw_float32
    elif autocast_kernel_dtype == torch.float16:
        return _mlstm_parallel_fwbw_float16
    elif autocast_kernel_dtype == torch.bfloat16:
        return _mlstm_parallel_fwbw_bfloat16
    else:
        raise ValueError(f"Unsupported autocast_kernel_dtype: {autocast_kernel_dtype}")


def mlstm_parallel__native_stablef_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    assert c_initial is None, "c_initial is not supported"
    assert n_initial is None, "n_initial is not supported"
    assert m_initial is None, "m_initial is not supported"
    assert not return_last_states, "return_last_states is not supported"

    matH, _, _ = mlstm_parallel_fw(
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        eps=eps,
    )
    return matH


def mlstm_parallel__native_stablef_custbw(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    assert c_initial is None, "c_initial is not supported"
    assert n_initial is None, "n_initial is not supported"
    assert m_initial is None, "m_initial is not supported"
    assert return_last_states is False, "return_last_states is not supported"

    _mlstm_parallel_fwbw = _get_parallel_fwbw_kernel(autocast_kernel_dtype=autocast_kernel_dtype)

    matH, _, _ = _mlstm_parallel_fwbw.apply(q, k, v, i, f, eps)
    return matH
