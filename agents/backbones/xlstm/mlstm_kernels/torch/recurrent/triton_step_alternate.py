#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck

import torch
import triton

from ...triton.recurrent.fw_step_alternate import (
    recurrent_step_fw_kernel_C,
    recurrent_step_fw_kernel_H,
)
from ..utils import contiguous_noctx

# NOTE: This kernel fails in the tests. Therefore, it should not be used.

@contiguous_noctx
def mlstm_recurrent_step__triton_alternate_fw(
    matC_old: torch.Tensor,  # (B, NH, DHQK, DHV)
    vecN_old: torch.Tensor,  # (B, NH, DHQK)
    scaM_old: torch.Tensor,  # (B, NH, 1)
    vecQ: torch.Tensor,  # (B, NH, DHQK)
    vecK: torch.Tensor,  # (B, NH, DHQK)
    vecV: torch.Tensor,  # (B, NH, DHV)
    scaI: torch.Tensor,  # (B, NH, 1)
    scaF: torch.Tensor,  # (B, NH, 1)
    matC_new: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_new: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_new: torch.Tensor = None,  # (B, NH, 1)
    qk_scale: float = None,
    eps: float = 1e-6,
    # BLOCK_DQK: int = 16,
    # BLOCK_DV: int = 16,
    # BLOCK_DQK_H: int = 16,
    # BLOCK_DV_H: int = 16,
):
    B, NH, DHQK = vecQ.shape
    _, _, DHV = vecV.shape
    assert vecQ.shape == vecK.shape, "q and k must have the same shape"
    assert matC_old.shape == (
        B,
        NH,
        DHQK,
        DHV,
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

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    if matC_new is None:
        assert (
            vecN_new is None and scaM_new is None
        ), "Initial states must be provided together."
        matC_new = torch.empty_like(matC_old)
        vecN_new = torch.empty_like(vecN_old)
        scaM_new = torch.empty_like(scaM_old)

    def grid_fn_C(args):
        NUM_BLOCKS_DQK = triton.cdiv(DHQK, args["BLOCK_DQK"])
        NUM_BLOCKS_DV = triton.cdiv(DHV, args["BLOCK_DV"])
        NUM_BATCH_HEAD = B * NH
        grid = (NUM_BLOCKS_DQK, NUM_BLOCKS_DV, NUM_BATCH_HEAD)
        return grid

    # DEBUG ONLY
    # def grid_fn_C(*args):
    #     NUM_BLOCKS_DQK = triton.cdiv(DHQK, BLOCK_DQK)
    #     NUM_BLOCKS_DV = triton.cdiv(DHV, BLOCK_DV)
    #     NUM_BATCH_HEAD = B * NH
    #     grid = (NUM_BLOCKS_DQK, NUM_BLOCKS_DV, NUM_BATCH_HEAD)
    #     print(grid)
    #     return grid

    grid_C = grid_fn_C

    # create output tensors
    vecH = torch.empty_like(vecV)

    recurrent_step_fw_kernel_C[grid_C](
        matC_old=matC_old,
        vecN_old=vecN_old,
        scaM_old=scaM_old,
        vecK=vecK,
        vecV=vecV,
        scaI=scaI,
        scaF=scaF,
        matC_new=matC_new,
        vecN_new=vecN_new,
        scaM_new=scaM_new,
        qk_scale=qk_scale,
        s_matC_b=matC_old.stride(0),
        s_matC_nh=matC_old.stride(1),
        s_matC_dhqk=matC_old.stride(2),
        s_matC_dhv=matC_old.stride(3),
        s_vecN_b=vecN_old.stride(0),
        s_vecN_nh=vecN_old.stride(1),
        s_vecN_dhqk=vecN_old.stride(2),
        s_scaM_b=scaM_old.stride(0),
        s_scaM_nh=scaM_old.stride(1),
        s_vecQK_b=vecQ.stride(0),
        s_vecQK_nh=vecQ.stride(1),
        s_vecQK_dhqk=vecQ.stride(2),
        s_vecVH_b=vecV.stride(0),
        s_vecVH_nh=vecV.stride(1),
        s_vecVH_dhv=vecV.stride(2),
        s_scaIF_b=scaI.stride(0),
        s_scaIF_nh=scaI.stride(1),
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHV=DHV,
        # BLOCK_DQK,
        # BLOCK_DV,
        EPS=eps,
    )

    def grid_fn_h(args):
        NUM_BLOCKS_DV_H = triton.cdiv(DHV, args["BLOCK_DV"])
        NUM_BATCH_HEAD = B * NH
        grid = (1, NUM_BLOCKS_DV_H, NUM_BATCH_HEAD)
        return grid

    # DEBUG ONLY
    # def grid_fn_h(*args):
    #     NUM_BLOCKS_DV_H = triton.cdiv(DHV, BLOCK_DV_H)
    #     NUM_BATCH_HEAD = B * NH
    #     grid = (1, NUM_BLOCKS_DV_H, NUM_BATCH_HEAD)
    #     print(grid)
    #     return grid

    grid_h = grid_fn_h

    recurrent_step_fw_kernel_H[grid_h](
        vecQ=vecQ,
        vecH=vecH,
        matC_new=matC_new,
        vecN_new=vecN_new,
        scaM_new=scaM_new,
        qk_scale=qk_scale,
        s_matC_b=matC_old.stride(0),
        s_matC_nh=matC_old.stride(1),
        s_matC_dhqk=matC_old.stride(2),
        s_matC_dhv=matC_old.stride(3),
        s_vecN_b=vecN_old.stride(0),
        s_vecN_nh=vecN_old.stride(1),
        s_vecN_dhqk=vecN_old.stride(2),
        s_scaM_b=scaM_old.stride(0),
        s_scaM_nh=scaM_old.stride(1),
        s_vecQK_b=vecQ.stride(0),
        s_vecQK_nh=vecQ.stride(1),
        s_vecQK_dhqk=vecQ.stride(2),
        s_vecVH_b=vecV.stride(0),
        s_vecVH_nh=vecV.stride(1),
        s_vecVH_dhv=vecV.stride(2),
        s_scaIF_b=scaI.stride(0),
        s_scaIF_nh=scaI.stride(1),
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHV=DHV,
        # BLOCK_DQK_H,
        # BLOCK_DV_H,
        EPS=eps,
    )

    return vecH, (matC_new, vecN_new, scaM_new)


def mlstm_recurrent_step__triton_alternate(
    q: torch.Tensor,  # (B, NH, DHQK)
    k: torch.Tensor,  # (B, NH, DHQK)
    v: torch.Tensor,  # (B, NH, DHV)
    i: torch.Tensor,  # (B, NH, 1)
    f: torch.Tensor,  # (B, NH, 1)
    c: torch.Tensor,  # (B, NH, DHQK, DHV)
    n: torch.Tensor,  # (B, NH, DHQK)
    m: torch.Tensor,  # (B, NH, 1)
    eps: float = 1e-6,
    **kwargs,
) -> tuple[
    torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:  # vecH, (matC_state_new (B, NH, DHQK, DHV), vecN_state_new (B, NH, DHQK), vecM_state_new (B, NH, 1))
    """This is a single step of the mLSTM operation in recurrent form."""
    return mlstm_recurrent_step__triton_alternate_fw(
        matC_old=c,
        vecN_old=n,
        scaM_old=m,
        vecQ=q,
        vecK=k,
        vecV=v,
        scaI=i,
        scaF=f,
        eps=eps,
        **kwargs,
    )
