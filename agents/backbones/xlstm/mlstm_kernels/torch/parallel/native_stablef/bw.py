#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
from torch.nn.functional import logsigmoid


def mlstm_parallel_bw(
    matDeltaHtilde: torch.Tensor,
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    vecN: torch.Tensor,
    vecM: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, ...]:
    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = logsigmoid(vecF)  # (B, NH, S)

    matLogSigF_tril = vecLogSigF[:, :, :, None].repeat(1, 1, 1, S).tril(-1)
    matLogSigF_cum = matLogSigF_tril.cumsum(-2)

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF_cum, -float("inf"))

    matLogD = matLogSigF_mask + vecI[:, :, None, :]

    matLogD_stabilized = matLogD - vecM[:, :, :, None]

    matD = torch.exp(matLogD_stabilized)  # (B, NH, S, S)

    # intermediate delta-errors
    matDeltaC = matDeltaHtilde @ matV.transpose(-2, -1) / (vecN[:, :, :, None] + eps)

    matS = (matQ @ matK.transpose(-2, -1)) * (DHQK**-0.5)

    matDeltaDtilde = matDeltaC * matD * matS

    vecDeltaI = torch.sum(matDeltaDtilde, dim=-2)

    # output delta-errors / gradients
    matP = matDeltaC * matD

    matDeltaQ = (matP @ matK) * (DHQK**-0.5)
    matDeltaK = (matP.transpose(-2, -1) @ matQ) * (DHQK**-0.5)

    matCtilde = matS * matD
    matDeltaV = matCtilde.transpose(-2, -1) @ (matDeltaHtilde / (vecN[:, :, :, None] + eps))

    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    vecDeltaFbar_acc = (matQ * matDeltaQ - matK * matDeltaK).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)

    return (
        matDeltaQ,
        matDeltaK,
        matDeltaV,
        vecDeltaI,
        vecDeltaF,
    )
