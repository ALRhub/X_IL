#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import triton

from ....triton.parallel.limit_headdim import mlstm_parallel_bw_dKdV_kernel, mlstm_parallel_bw_dQ_kernel


def mlstm_parallel_bw(
    matDeltaHtilde: torch.Tensor,
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    vecM: torch.Tensor,
    vecN: torch.Tensor,
    eps: float = 1e-6,
    # BLOCK_Q_dKdV: int = BLOCK_Q,
    # BLOCK_KV_dKdV: int = BLOCK_KV,
    # BLOCK_Q_dQ: int = BLOCK_Q,
    # BLOCK_KV_dQ: int = BLOCK_KV,
) -> torch.Tensor:
    # batch size, number of heads, sequence length, head dimension
    BS, NH, SL, DH = matQ.shape
    assert vecI.shape == (BS, NH, SL)
    assert vecF.shape == (BS, NH, SL)
    assert vecM.shape == (BS, NH, SL)
    assert vecN.shape == (BS, NH, SL)
    assert matDeltaHtilde.shape == (BS, NH, SL, DH)
    assert matQ.shape == matK.shape == matV.shape

    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = matQ.shape[-1], matK.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = matV.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {
        16,
        32,
        64,
        128,
        256,
    }, f"Only support HEAD_DIM in [16, 32, 64, 128, 256], got {HEAD_DIM_K}"

    ## ? preprocessing, initialization
    matDeltaQ = torch.empty_like(matQ)
    matDeltaK = torch.empty_like(matK)
    matDeltaV = torch.empty_like(matV)

    vecDeltaI = torch.zeros_like(vecI)

    # Note we want to compute the forget gate cumsum in float32.
    # This results in a more accurate cumsum and lower numerical precision errors.
    vecF_cs = torch.nn.functional.logsigmoid(vecF.to(dtype=torch.float32)).cumsum(-1)
    ## ? end preprocessing

    grid_dKdV = lambda args: (
        triton.cdiv(SL, args["BLOCK_KV"]),
        BS * NH,
        1,
    )
    # fix grid for debugging
    # grid_dKdV = lambda args: (
    #     triton.cdiv(SL, BLOCK_KV_dKdV),
    #     BS * NH,
    #     1,
    # )
    # print(f"Triton grid: {grid(None)}, BLOCK_Q: {BLOCK_Q}, BLOCK_KV: {BLOCK_KV}")

    # strides for matQ, matK, matV are same as matDeltaQ, matDeltaK, matDeltaV
    mlstm_parallel_bw_dKdV_kernel[grid_dKdV](
        matDeltaHtilde=matDeltaHtilde.contiguous(),
        matQ=matQ.contiguous(),
        matK=matK.contiguous(),
        matV=matV.contiguous(),
        vecI=vecI.contiguous(),
        vecF_cs=vecF_cs.contiguous(),
        vecM=vecM.contiguous(),
        vecN=vecN.contiguous(),
        qk_scale=HEAD_DIM_Q**0.5,
        matDeltaQ=matDeltaQ,
        matDeltaK=matDeltaK,
        matDeltaV=matDeltaV,
        vecDeltaI=vecDeltaI,
        stride_dhtz=matDeltaHtilde.stride(0),
        stride_dhth=matDeltaHtilde.stride(1),
        stride_dhts=matDeltaHtilde.stride(2),
        stride_dhtd=matDeltaHtilde.stride(3),
        stride_qz=matQ.stride(0),
        stride_qh=matQ.stride(1),
        stride_qs=matQ.stride(2),
        stride_qd=matQ.stride(3),
        stride_kz=matK.stride(0),
        stride_kh=matK.stride(1),
        stride_ks=matK.stride(2),
        stride_kd=matK.stride(3),
        stride_vz=matV.stride(0),
        stride_vh=matV.stride(1),
        stride_vs=matV.stride(2),
        stride_vd=matV.stride(3),
        stride_ifmn_z=vecF_cs.stride(0),
        stride_ifmn_h=vecF_cs.stride(1),
        stride_ifmn_s=vecF_cs.stride(2),
        Z=BS,
        H=NH,
        N_CTX=SL,
        HEAD_DIM=HEAD_DIM_K,
        EPS=eps,
        # BLOCK_Q=BLOCK_Q_dKdV,
        # BLOCK_KV=BLOCK_KV_dKdV,
    )

    grid_dQ = lambda args: (
        triton.cdiv(SL, args["BLOCK_Q"]),
        BS * NH,
        1,
    )
    # fix grid for debugging
    # grid_dQ = lambda args: (
    #     triton.cdiv(SL, BLOCK_Q_dQ),
    #     BS * NH,
    #     1,
    # )
    # print(f"Triton grid: {grid(None)}, BLOCK_Q: {BLOCK_Q}, BLOCK_KV: {BLOCK_KV}")

    mlstm_parallel_bw_dQ_kernel[grid_dQ](
        matDeltaHtilde=matDeltaHtilde.contiguous(),
        matQ=matQ.contiguous(),
        matK=matK.contiguous(),
        matV=matV.contiguous(),
        vecI=vecI.contiguous(),
        vecF_cs=vecF_cs.contiguous(),
        vecM=vecM.contiguous(),
        vecN=vecN.contiguous(),
        qk_scale=HEAD_DIM_Q**0.5,
        matDeltaQ=matDeltaQ,
        matDeltaK=matDeltaK,
        matDeltaV=matDeltaV,
        vecDeltaI=vecDeltaI,
        stride_dhtz=matDeltaHtilde.stride(0),
        stride_dhth=matDeltaHtilde.stride(1),
        stride_dhts=matDeltaHtilde.stride(2),
        stride_dhtd=matDeltaHtilde.stride(3),
        stride_qz=matQ.stride(0),
        stride_qh=matQ.stride(1),
        stride_qs=matQ.stride(2),
        stride_qd=matQ.stride(3),
        stride_kz=matK.stride(0),
        stride_kh=matK.stride(1),
        stride_ks=matK.stride(2),
        stride_kd=matK.stride(3),
        stride_vz=matV.stride(0),
        stride_vh=matV.stride(1),
        stride_vs=matV.stride(2),
        stride_vd=matV.stride(3),
        stride_ifmn_z=vecF_cs.stride(0),
        stride_ifmn_h=vecF_cs.stride(1),
        stride_ifmn_s=vecF_cs.stride(2),
        Z=BS,
        H=NH,
        N_CTX=SL,
        HEAD_DIM=HEAD_DIM_K,
        EPS=eps,
        # BLOCK_Q=BLOCK_Q_dQ,
        # BLOCK_KV=BLOCK_KV_dQ,
    )

    ## ? postprocessing
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    # should we cast to float32 here?
    # No! this causes loading and casting all the data again.
    vecDeltaFbar_acc = (matQ * matDeltaQ - matK * matDeltaK).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)
    ## ? end postprocessing

    return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF
