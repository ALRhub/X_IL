#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import triton

from ....triton.chunkwise.limit_chunk import mlstm_chunkwise__parallel_fw_H_kernel
from ....utils.kernels import is_power_of_2
from ...utils import torch2triton_dtype


def mlstm_chunkwise__parallel_fw_H(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    # these states must be all states up to the last chunk, i.e. :-1
    matC_states: torch.Tensor,  # (B, NH, NC * DHQK, DHHV)
    vecN_states: torch.Tensor,  # (B, NH, NC * DHQK)
    scaMinter_states: torch.Tensor,  # (B, NH, NC)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    qk_scale: float = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:  # matH_out (B, NH, S, DHHV), vecN_out (B, NH, S)
    """This function defines the grid and block sizes for the kernel launch and calls the kernel."""
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]

    NC = NUM_CHUNKS
    L = CHUNK_SIZE

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    matH_out = torch.empty(B, NH, S, DHHV, device=matQ.device, dtype=matQ.dtype)
    vecN_out = torch.empty(B, NH, S, device=matQ.device, dtype=torch.float32)
    vecM_out = torch.empty(B, NH, S, device=matQ.device, dtype=torch.float32)

    grid = (num_b_DHHV, NC, B * NH)
    mlstm_chunkwise__parallel_fw_H_kernel[grid](
        matQ=matQ,
        matK=matK,
        matV=matV,
        matC_states=matC_states,
        vecN_states=vecN_states,
        scaMinter_states=scaMinter_states,
        vecI=vecI,
        vecB=vecB,
        matHout=matH_out,
        vecNout=vecN_out,
        vecMout=vecM_out,
        qk_scale=qk_scale,
        str_matQK_B_NH=matQ.stride(1),
        str_matQK_S=matQ.stride(2),
        str_matQK_DHQK=matQ.stride(3),
        str_matHV_B_NH=matV.stride(1),
        str_matHV_S=matV.stride(2),
        str_matHV_DHHV=matV.stride(3),
        str_matCstates_B_NH=matC_states.stride(1),
        str_matCstates_NCDHQK=matC_states.stride(2),
        str_matCstates_DHHV=matC_states.stride(3),
        str_vecNstates_B_NH=vecN_states.stride(1),
        str_vecNstates_NCDHQK=vecN_states.stride(2),
        str_scaMinterstates_B_NH=scaMinter_states.stride(1),
        str_vecBI_B_NH=vecB.stride(1),
        str_vecBI_NC=vecB.stride(2),
        str_vecBI_L=vecB.stride(3),
        str_vecMN_B_NH=vecN_out.stride(1),
        str_vecMN_S=vecN_out.stride(2),
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        DTYPE=torch2triton_dtype(matQ.dtype),
        EPS=EPS,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matH_out, vecN_out, vecM_out
