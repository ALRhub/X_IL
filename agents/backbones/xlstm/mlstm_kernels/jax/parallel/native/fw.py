#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
Jax.

mLSTM forward and backward pass. Parallel formulation.
"""

import jax
import jax.numpy as jnp


def mlstm_parallel_fw(
    matQ: jax.Array,
    matK: jax.Array,
    matV: jax.Array,
    vecI: jax.Array,
    vecF: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    vecLogSigF = jax.nn.log_sigmoid(vecF)  # (B, NH, S)
    vecLogSigF_cumsum = jnp.cumsum(vecLogSigF, axis=-1)

    matLogSigF = vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]

    ltr = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))

    matLogSigF_mask = jnp.where(ltr, matLogSigF, -float("inf"))

    matLogD = matLogSigF_mask + vecI[:, :, None, :]

    vecM = jnp.max(matLogD, axis=-1, keepdims=True)  # (B, NH, S, 1)
    matLogD_stabilized = matLogD - vecM

    matD = jnp.exp(matLogD_stabilized)  # (B, NH, S, S)

    matS = (matQ @ matK.swapaxes(-2, -1)) * (DHQK**-0.5)  # (B, NH, S, S)

    matCtilde = matS * matD  # (B, NH, S, S)
    vecN = jnp.maximum(jnp.abs(jnp.sum(matCtilde, axis=-1, keepdims=True)), jnp.exp(-vecM))  # (B, NH, S, 1)
    # (B, NH, S, S)
    matC = matCtilde / (vecN + eps)

    matH = matC @ matV  # (B, NH, S, DH)

    vecN = vecN.squeeze(-1)
    vecM = vecM.squeeze(-1)

    return (matH, vecN, vecM)
