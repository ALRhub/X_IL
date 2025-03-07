#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import triton

from ....triton.parallel.limit_headdim import mlstm_parallel_fw_kernel

MINIMUM_MAX_VAL = -10  # -float("inf")  # -10.0


def mlstm_parallel_fw(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
    # BLOCK_Q: int = BLOCK_Q,
    # BLOCK_KV: int = BLOCK_KV,
) -> torch.Tensor:
    # batch size, number of heads, sequence length, head dimension
    BS, NH, SL, DH = matQ.shape
    assert vecI.shape == (BS, NH, SL)
    assert vecF.shape == (BS, NH, SL)

    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = matQ.shape[-1], matK.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = matV.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V, f"Q, K, V must have the same head dimension"
    assert HEAD_DIM_K in {
        16,
        32,
        64,
        128,
        256,
    }, f"Only head dimensions 16, 32, 64, 128, 256 are supported, got {HEAD_DIM_K}"

    def grid(args):
        return triton.cdiv(matQ.shape[2], args["BLOCK_Q"]), matQ.shape[0] * matQ.shape[1], 1

    # fix grid for debugging
    # def grid(args):
    #     return triton.cdiv(matQ.shape[2], BLOCK_Q), matQ.shape[0] * matQ.shape[1], 1

    # print(f"Triton grid: {grid(None)}, BLOCK_Q: {BLOCK_Q}, BLOCK_KV: {BLOCK_KV}")

    matH = torch.empty_like(matQ)

    vecN = torch.zeros(
        (matQ.shape[0], matQ.shape[1], matQ.shape[2]),
        device=matQ.device,
        dtype=torch.float32,
    )
    vecM = torch.zeros(
        (matQ.shape[0], matQ.shape[1], matQ.shape[2]),
        device=matQ.device,
        dtype=torch.float32,
    )

    # Note we want to compute the forget gate cumsum in float32.
    # This results in a more accurate cumsum and lower numerical precision errors.
    vecF_cs = torch.nn.functional.logsigmoid(vecF.to(dtype=torch.float32)).cumsum(-1)

    mlstm_parallel_fw_kernel[grid](
        matQ=matQ.contiguous(),
        matK=matK.contiguous(),
        matV=matV.contiguous(),
        vecI=vecI.contiguous(),
        vecF_cs=vecF_cs.contiguous(),
        qk_scale=HEAD_DIM_Q**0.5,
        matH=matH,
        vecN=vecN,
        vecM=vecM,
        stride_qz=matQ.stride(0),
        stride_qh=matQ.stride(1),
        stride_qm=matQ.stride(2),
        stride_qk=matQ.stride(3),
        stride_kz=matK.stride(0),
        stride_kh=matK.stride(1),
        stride_kn=matK.stride(2),
        stride_kk=matK.stride(3),
        stride_vz=matV.stride(0),
        stride_vh=matV.stride(1),
        stride_vk=matV.stride(2),
        stride_vn=matV.stride(3),
        stride_hz=matH.stride(0),
        stride_hh=matH.stride(1),
        stride_hm=matH.stride(2),
        stride_hn=matH.stride(3),
        stride_ifmn_z=vecF_cs.stride(0),
        stride_ifmn_h=vecF_cs.stride(1),
        stride_ifmn_m=vecF_cs.stride(2),
        Z=BS,
        H=NH,
        N_CTX=SL,
        HEAD_DIM=HEAD_DIM_K,
        # BLOCK_Q=BLOCK_Q,
        # BLOCK_KV=BLOCK_KV,
        MINIMUM_MAX_VAL=MINIMUM_MAX_VAL,
        EPS=eps,
    )

    return matH, vecM, vecN
