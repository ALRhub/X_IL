#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import triton

from ...triton.recurrent.fw_step_fused import recurrent_step_fw_kernel
from ...utils.kernels import is_power_of_2
from ..utils import contiguous_noctx, torch2triton_dtype

# NOTE: This kernel fails in the tests. Therefore, it should not be used.

@contiguous_noctx
def mlstm_recurrent_step__triton_fw(
    matC_old: torch.Tensor,  # (B, NH, DHQK, DHHV)
    vecN_old: torch.Tensor,  # (B, NH, DHQK)
    scaM_old: torch.Tensor,  # (B, NH, 1)
    vecQ: torch.Tensor,  # (B, NH, DHQK)
    vecK: torch.Tensor,  # (B, NH, DHQK)
    vecV: torch.Tensor,  # (B, NH, DHHV)
    scaI: torch.Tensor,  # (B, NH, 1)
    scaF: torch.Tensor,  # (B, NH, 1)
    matC_new: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_new: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_new: torch.Tensor = None,  # (B, NH, 1)
    qk_scale: float = None,
    eps: float = 1e-6,
    siz_b_DHQK: int | None = None,
    siz_b_DHHV: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    dtype_state: torch.dtype = torch.float32,
):
    B, NH, DHQK = vecQ.shape
    _, _, DHHV = vecV.shape
    assert vecQ.shape == vecK.shape, "q and k must have the same shape"
    assert matC_old.shape == (
        B,
        NH,
        DHQK,
        DHHV,
    ), f"matC_old has wrong shape, got {matC_old.shape}"
    assert vecN_old.shape == (
        B,
        NH,
        DHQK,
    ), f"vecN_old has wrong shape, got {vecN_old.shape}"
    assert scaM_old.shape == (
        B,
        NH,
        1,
    ), f"scaM_old has wrong shape, got {scaM_old.shape}"
    assert scaI.shape == (B, NH, 1), f"scaI has wrong shape, got {scaI.shape}"
    assert scaF.shape == (B, NH, 1), f"scaF has wrong shape, got {scaF.shape}"

    DTYPE = vecQ.dtype

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    if matC_new is None:
        assert (
            vecN_new is None and scaM_new is None
        ), "Initial states must be provided together."
        matC_new = torch.empty_like(matC_old, dtype=dtype_state)
        vecN_new = torch.empty_like(vecN_old, dtype=dtype_state)
        scaM_new = torch.empty_like(scaM_old, dtype=dtype_state)
    else:
        assert (
            vecN_new is not None and scaM_new is not None
        ), "Initial states must be provided together."

    min_siz_b_DHQK = 64
    min_siz_b_DHHV = 64

    assert (
        is_power_of_2(DHQK) or DHQK % min_siz_b_DHQK == 0
    ), f"DHQK must be a power of 2 or multiple of {min_siz_b_DHQK}. Got {DHQK}."
    assert (
        is_power_of_2(DHHV) or DHHV % min_siz_b_DHHV == 0
    ), f"DHHV must be a power of 2 or multiple of {min_siz_b_DHHV}. Got {DHHV}."

    siz_b_DHQK = min(min_siz_b_DHQK, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(min_siz_b_DHHV, triton.next_power_of_2(DHHV))

    # num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    grid = (1, num_b_DHHV, B * NH)
    if num_warps is None:
        num_warps = 4 if siz_b_DHQK >= 64 else 2

    num_stages = 1 if num_stages is None else num_stages

    # create output tensors
    vecH = torch.empty_like(vecV)

    recurrent_step_fw_kernel[grid](
        matC_old=matC_old,
        vecN_old=vecN_old,
        scaM_old=scaM_old,
        vecQ=vecQ,
        vecK=vecK,
        vecV=vecV,
        scaI=scaI,
        scaF=scaF,
        vecH=vecH,
        matC_new=matC_new,
        vecN_new=vecN_new,
        scaM_new=scaM_new,
        qk_scale=qk_scale,
        str_matC_B_NH=matC_old.stride(1),
        str_matC_DHQK=matC_old.stride(2),
        str_matC_DHHV=matC_old.stride(3),
        str_vecN_B_NH=vecN_old.stride(1),
        str_vecN_DHQK=vecN_old.stride(2),
        str_scaM_B_NH=scaM_old.stride(1),
        str_vecQK_NH=vecQ.stride(1),
        str_vecQK_DHQK=vecQ.stride(2),
        str_vecVH_B_NH=vecV.stride(1),
        str_vecVH_DHHV=vecV.stride(2),
        str_scaIF_B_NH=scaI.stride(1),
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        EPS=eps,
        DTYPE=torch2triton_dtype(DTYPE),
        DTYPE_STATE=torch2triton_dtype(dtype_state),
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return vecH, (matC_new, vecN_new, scaM_new)


def mlstm_recurrent_step__triton(
    q: torch.Tensor,  # (B, NH, DHQK)
    k: torch.Tensor,  # (B, NH, DHQK)
    v: torch.Tensor,  # (B, NH, DHV)
    i: torch.Tensor,  # (B, NH, 1)
    f: torch.Tensor,  # (B, NH, 1)
    c: torch.Tensor,  # (B, NH, DHQK, DHV)
    n: torch.Tensor,  # (B, NH, DHQK)
    m: torch.Tensor,  # (B, NH, 1)
    eps: float = 1e-6,
    dtype_state: torch.dtype = torch.float32,
    **kwargs,
) -> tuple[
    torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:  # vecH, (matC_state_new (B, NH, DHQK, DHV), vecN_state_new (B, NH, DHQK), vecM_state_new (B, NH, 1))
    """This is a single step of the mLSTM operation in recurrent form."""
    return mlstm_recurrent_step__triton_fw(
        matC_old=c,
        vecN_old=n,
        scaM_old=m,
        vecQ=q,
        vecK=k,
        vecV=v,
        scaI=i,
        scaF=f,
        eps=eps,
        dtype_state=dtype_state,
        **kwargs,
    )
