#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import jax.numpy as jnp


def mlstm_parallel_bw(
    matDeltaHtilde: jax.Array,
    matQ: jax.Array,
    matK: jax.Array,
    matV: jax.Array,
    vecI: jax.Array,
    vecF: jax.Array,
    vecN: jax.Array,
    vecM: jax.Array,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    vecLogSigF = jax.nn.log_sigmoid(vecF)  # (B, NH, S)
    vecLogSigF_cumsum = jnp.cumsum(vecLogSigF, axis=-1)

    matLogSigF = vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]

    ltr = jnp.tril(
        jnp.ones(
            (S, S),
            dtype=jnp.bool_,
        )
    )

    matLogSigF_mask = jnp.where(ltr, matLogSigF, -float("inf"))

    matLogD = matLogSigF_mask + vecI[:, :, None, :]

    matLogD_stabilized = matLogD - vecM[:, :, :, None]

    matD = jnp.exp(matLogD_stabilized)  # (B, NH, S, S)

    # intermediate delta-errors
    matDeltaC = matDeltaHtilde @ matV.swapaxes(-2, -1) / (vecN[:, :, :, None] + eps)

    matS = (matQ @ matK.swapaxes(-2, -1)) * (DHQK**-0.5)

    matDeltaDtilde = matDeltaC * matD * matS

    vecDeltaI = jnp.sum(matDeltaDtilde, axis=-2)

    # output delta-errors / gradients
    matP = matDeltaC * matD

    matDeltaQ = (matP @ matK) * (DHQK**-0.5)
    matDeltaK = (matP.swapaxes(-2, -1) @ matQ) * (DHQK**-0.5)

    matCtilde = matS * matD
    matDeltaV = matCtilde.swapaxes(-2, -1) @ (matDeltaHtilde / (vecN[:, :, :, None] + eps))

    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    vecDeltaFbar_acc = jnp.sum((matQ * matDeltaQ - matK * matDeltaK), axis=-1)
    vecDeltaFbar = jnp.flip(jnp.cumsum(jnp.flip(vecDeltaFbar_acc, axis=-1), axis=-1), axis=-1)
    vecDeltaF = vecDeltaFbar * jax.nn.sigmoid(-vecF)

    return (
        matDeltaQ,
        matDeltaK,
        matDeltaV,
        vecDeltaI,
        vecDeltaF,
    )
