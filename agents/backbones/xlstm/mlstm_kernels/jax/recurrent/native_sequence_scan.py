#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Callable

import jax
import jax.numpy as jnp

from .native_step import mlstm_recurrent_step__native_fw
from .triton_step import mlstm_recurrent_step__triton_fw


def _mlstm_recurrent_sequence_loop_scan_fw(
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHV)
    vecI: jax.Array,  # (B, NH, S)
    vecF: jax.Array,  # (B, NH, S)
    matC_initial: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    vecN_initial: jax.Array | None = None,  # (B, NH, DHQK)
    scaM_initial: jax.Array | None = None,  # (B, NH)
    return_last_states: bool = False,
    eps: float = 1e-6,
    state_dtype: jnp.dtype | None = None,
    mlstm_step_fn: Callable = mlstm_recurrent_step__native_fw,
    **kwargs,
) -> (
    jax.Array
    | tuple[
        jax.Array,  # (B, NH, S, DHV)
        tuple[jax.Array, jax.Array, jax.Array],
        # (matC_state_last (B, NH, DHQK, DHV), vecN_state_last (B, NH, DHQK), vecM_state_last (B, NH, 1))
    ]
):
    """
    Forward pass of the mLSTM cell in recurrent form on a full sequence.
    This function uses jax.lax.scan to loop over the sequence.

    Args:
        matQ: Queries tensor of shape (B, NH, S, DHQK).
        matK: Keys tensor of shape (B, NH, S, DHQK).
        matV: Values tensor of shape (B, NH, S, DHV).
        vecI: Input gate pre-activation tensor of shape (B, NH, S).
        vecF: Forget gate pre-activation tensor of shape (B, NH, S).
        matC_initial: Initial memory state tensor of shape (B, NH, DHQK, DHV).
        vecN_initial: Initial normalizer state tensor of shape (B, NH, DHQK).
        scaM_initial: Initial max state tensor of shape (B, NH).
        return_last_states: Whether to return the last states. Defaults to False.
        eps: Epsilon value for numerical stability. Applied when dividing C by N. Defaults to 1e-6.
        state_dtype: Dtype of the states. If None, uses the dtype of the initial states if provided, or other
            the dtypes of the pre-activations. If initial states are provided, the return dtype will be the same as the
            initial states. Defaults to None.
        mlstm_step_fn: Callable: The step function to use.

    Returns:
        Hidden states tensor of shape (B, NH, S, DHV) if `return_last_states` is False.
        Tuple of hidden states tensor and tuple of last states tensors if `return_last_states` is True.
    """
    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]
    if vecI.ndim == 3:
        vecI = vecI[:, :, :, None]
    if vecF.ndim == 3:
        vecF = vecF[:, :, :, None]

    if matC_initial is not None:
        assert vecN_initial is not None and scaM_initial is not None, "Initial states must be provided together."
        assert scaM_initial.axis() == 2, "Initial states must be 2D."
        matC_state, vecN_state, scaM_state = (
            matC_initial,
            vecN_initial,
            scaM_initial[:, :, None],
        )
        if state_dtype is not None:
            matC_state = matC_state.astype(state_dtype)
            vecN_state = vecN_state.astype(state_dtype)
            scaM_initial = scaM_initial.astype(state_dtype)
    else:
        if state_dtype is None:
            state_dtype = vecF.dtype
        # memory state
        matC_state = jnp.zeros((B, NH, DHQK, DHV), dtype=state_dtype)
        # normalizer state
        vecN_state = jnp.zeros((B, NH, DHQK), dtype=state_dtype)
        # max state
        scaM_state = jnp.zeros((B, NH, 1), dtype=state_dtype)

    # for jax.lax.scan, time dimension must be the first dimension
    def scan_fn(carry, inputs):
        matC_state, vecN_state, scaM_state = carry
        vecQ, vecK, vecV, scaI, scaF = inputs
        matH, carry = mlstm_step_fn(matC_state, vecN_state, scaM_state, vecQ, vecK, vecV, scaI, scaF, eps=eps)
        return carry, matH

    inputs = jax.tree.map(lambda x: jnp.moveaxis(x, 2, 0), (matQ, matK, matV, vecI, vecF))
    (matC_state, vecN_state, scaM_state), matH = jax.lax.scan(
        f=scan_fn, init=(matC_state, vecN_state, scaM_state), xs=inputs
    )
    matH = jnp.moveaxis(matH, 0, 2)  # Scan returns time as the first dimension.

    if return_last_states:
        if matC_initial is not None:
            matC_state = matC_state.astype(matC_initial.dtype)
        if vecN_initial is not None:
            vecN_state = vecN_state.astype(vecN_initial.dtype)
        if scaM_initial is not None:
            scaM_state = scaM_state.astype(scaM_initial.dtype)
        return matH, (matC_state, vecN_state, scaM_state)
    else:
        return matH


def mlstm_recurrent_sequence__native_fw(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    i: jax.Array,
    f: jax.Array,
    c_initial: jax.Array | None = None,
    n_initial: jax.Array | None = None,
    m_initial: jax.Array | None = None,
    return_last_states: bool = False,
    state_dtype: jnp.dtype | None = None,
    eps: float = 1e-6,
    **kwargs,
) -> jax.Array | tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """
    Forward pass of the mLSTM cell in recurrent form on a full sequence using native JAX implementation.

    Args:
        q: Queries tensor of shape (B, NH, S, DHQK).
        k: Keys tensor of shape (B, NH, S, DHQK).
        v: Values tensor of shape (B, NH, S, DHV).
        i: Input gate pre-activation tensor of shape (B, NH, S).
        f: Forget gate pre-activation tensor of shape (B, NH, S).
        c_initial: Initial memory state tensor of shape (B, NH, DHQK, DHV).
        n_initial: Initial normalizer state tensor of shape (B, NH, DHQK).
        m_initial: Initial max state tensor of shape (B, NH).
        return_last_states: Whether to return the last states. Defaults to False.
        eps: Epsilon value for numerical stability. Applied when dividing C by N. Defaults to 1e-6.
        state_dtype: Dtype of the states. If None, uses the dtype of the initial states if provided, or other
            the dtypes of the pre-activations. If initial states are provided, the return dtype will be the same as the
            initial states. Defaults to None.

    Returns:
        Hidden states tensor of shape (B, NH, S, DHV) if `return_last_states` is False.
        Tuple of hidden states tensor and tuple of last states tensors if `return_last_states` is True.
    """

    return _mlstm_recurrent_sequence_loop_scan_fw(
        mlstm_step_fn=mlstm_recurrent_step__native_fw,
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        matC_initial=c_initial,
        vecN_initial=n_initial,
        scaM_initial=m_initial,
        return_last_states=return_last_states,
        eps=eps,
        state_dtype=state_dtype,
    )
    


def mlstm_recurrent_sequence__triton_step_fused_fw(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    i: jax.Array,
    f: jax.Array,
    c_initial: jax.Array | None = None,
    n_initial: jax.Array | None = None,
    m_initial: jax.Array | None = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    state_dtype: jnp.dtype | None = None,
    **kwargs,
) -> jax.Array | tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """
    Forward pass of the mLSTM cell in recurrent form on a full sequence using the fused Triton step
    kernel.

    Args:
        q: Queries tensor of shape (B, NH, S, DHQK).
        k: Keys tensor of shape (B, NH, S, DHQK).
        v: Values tensor of shape (B, NH, S, DHV).
        i: Input gate pre-activation tensor of shape (B, NH, S).
        f: Forget gate pre-activation tensor of shape (B, NH, S).
        c_initial: Initial memory state tensor of shape (B, NH, DHQK, DHV).
        n_initial: Initial normalizer state tensor of shape (B, NH, DHQK).
        m_initial: Initial max state tensor of shape (B, NH).
        return_last_states: Whether to return the last states. Defaults to False.
        eps: Epsilon value for numerical stability. Applied when dividing C by N. Defaults to 1e-6.
        state_dtype: Dtype of the states. If None, uses the dtype of the initial states if provided, or other
            the dtypes of the pre-activations. If initial states are provided, the return dtype will be the same as the
            initial states. Defaults to None.

    Returns:
        Hidden states tensor of shape (B, NH, S, DHV) if `return_last_states` is False.
        Tuple of hidden states tensor and tuple of last states tensors if `return_last_states` is True.
    """
    return _mlstm_recurrent_sequence_loop_scan_fw(
        mlstm_step_fn=mlstm_recurrent_step__triton_fw,
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        matC_initial=c_initial,
        vecN_initial=n_initial,
        scaM_initial=m_initial,
        return_last_states=return_last_states,
        eps=eps,
        state_dtype=state_dtype,        
    )