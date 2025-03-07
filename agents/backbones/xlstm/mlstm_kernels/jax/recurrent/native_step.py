#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck

"""
Jax.

This module contains the recurrent step implementation of the mLSTM.
"""

import jax
import jax.numpy as jnp


def mlstm_recurrent_step__native_fw(
    matC_state: jax.Array,  # (B, NH, DHQK, DHV)
    vecN_state: jax.Array,  # (B, NH, DHQK)
    scaM_state: jax.Array,  # (B, NH, 1)
    vecQ: jax.Array,  # (B, NH, DHQK)
    vecK: jax.Array,  # (B, NH, DHQK)
    vecV: jax.Array,  # (B, NH, DHV)
    scaI: jax.Array,  # (B, NH, 1)
    scaF: jax.Array,  # (B, NH, 1)
    eps: float = 1e-6,
    **kwargs,
) -> tuple[
    jax.Array, tuple[jax.Array, jax.Array, jax.Array]
]:  # vecH, (matC_state_new (B, NH, DHQK, DHV), vecN_state_new (B, NH, DHQK), vecM_state_new (B, NH, 1))
    """This is a single step of the mLSTM operation in recurrent form.

    Args:
        matC_state (jax.Array): (B, NH, DHQK, DHV)
        vecN_state (jax.Array): (B, NH, DHQK)
        scaM_state (jax.Array): (B, NH, 1)
        vecQ (jax.Array): (B, NH, DHQK)
        vecK (jax.Array): (B, NH, DHQK)
        vecV (jax.Array): (B, NH, DHV)
        scaI (jax.Array): (B, NH, 1)
        scaF (jax.Array): (B, NH, 1)
        eps (float, optional): Used for building the forgetgate matrix. Defaults to 1e-6.

    Returns:
        tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            (hidden_state [B, NH, DHV], (c_state_new [B, NH, DHQK, DHV], n_state_new [B, NH, DHQK]], m_state_new [B, NH, 1]))
    """
    B, NH, DHQK = vecQ.shape

    # gates
    scaF_log = jax.nn.log_sigmoid(scaF)

    # update rule
    scaM_state_new = jnp.maximum(scaF_log + scaM_state, scaI)  # (B, NH, 1)

    scaF_act = jnp.exp(scaF_log + scaM_state - scaM_state_new)  # (B, NH, 1)
    scaI_act = jnp.exp(scaI - scaM_state_new)  # (B, NH, 1)

    vecQ_scaled = vecQ * (DHQK ** (-0.5))  # (B, NH, DHQK)

    matC_state_new = scaF_act[:, :, :, None] * matC_state + scaI_act[:, :, :, None] * (
        vecK[:, :, :, None] @ vecV[:, :, None, :]
    )  # (B, NH, DHQK, DHV)
    vecN_state_new = scaF_act * vecN_state + scaI_act * vecK  # (B, NH, DHQK)

    h_num = vecQ_scaled[:, :, None, :] @ matC_state_new  # (B, NH, 1, DHV)
    h_num = h_num.squeeze(2)  # (B, NH, DHV)

    qn_dotproduct = (
        vecQ_scaled[:, :, None, :] @ vecN_state_new[:, :, :, None]
    )  # (B, NH, 1, 1)
    qn_dotproduct = qn_dotproduct.squeeze(2)  # (B, NH, 1)
    max_val = jnp.exp(-scaM_state_new)  # (B, NH, 1)
    h_denom = jnp.maximum(jnp.abs(qn_dotproduct), max_val) + eps  # (B, NH, 1)
    h = h_num / h_denom  # (B, NH, DHV) / (B, NH, 1) = (B, NH, DHV)

    return h, (matC_state_new, vecN_state_new, scaM_state_new)


def mlstm_recurrent_step__native(
    q: jax.Array,  # (B, NH, DHQK)
    k: jax.Array,  # (B, NH, DHQK)
    v: jax.Array,  # (B, NH, DHV)
    i: jax.Array,  # (B, NH, 1)
    f: jax.Array,  # (B, NH, 1)
    c: jax.Array,  # (B, NH, DHQK, DHV)
    n: jax.Array,  # (B, NH, DHQK)
    m: jax.Array,  # (B, NH, 1)
    eps: float = 1e-6,
    **kwargs,
) -> tuple[
    jax.Array, tuple[jax.Array, jax.Array, jax.Array]
]:  # vecH, (matC_state_new (B, NH, DHQK, DHV), vecN_state_new (B, NH, DHQK), vecM_state_new (B, NH, 1))
    """This is a single step of the mLSTM operation in recurrent form."""
    return mlstm_recurrent_step__native_fw(
        matC_state=c,
        vecN_state=n,
        scaM_state=m,
        vecQ=q,
        vecK=k,
        vecV=v,
        scaI=i,
        scaF=f,
        eps=eps,
        **kwargs,
    )
