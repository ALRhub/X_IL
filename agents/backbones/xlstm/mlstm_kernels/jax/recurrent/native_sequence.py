#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Callable

import jax
import jax.numpy as jnp

from .native_step import mlstm_recurrent_step__native_fw
from .triton_step import mlstm_recurrent_step__triton_fw


def _mlstm_recurrent_sequence_loop_fw(
    mlstm_step_fn: Callable,
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHV)
    vecI: jax.Array,  # (B, NH, S)
    vecF: jax.Array,  # (B, NH, S)
    matC_initial: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    vecN_initial: jax.Array | None = None,  # (B, NH, DHQK)
    scaM_initial: jax.Array | None = None,  # (B, NH)
    return_last_states: bool = False,
    return_all_states: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[
    jax.Array,  # (B, NH, S, DHV)
    jax.Array,  # (B, NH, S, DHQK)
    jax.Array,  # (B, NH, S)
    None
    | (
        tuple[jax.Array, jax.Array, jax.Array]
    ),  # (matC_state_last (B, NH, DHQK, DHV), vecN_state_last (B, NH, DHQK), vecM_state_last (B, NH, 1))
    None
    | (
        tuple[jax.Array, jax.Array, jax.Array]
    ),  # (matC_states (B, NH, S, DHQK, DHV), vecN_states (B, NH, S, DHQK), vecM_states (B, NH, S))
]:
    """
    Forward pass of the mLSTM cell in recurrent form on a full sequence.
    The recurrent loop is implemented using a for loop.

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
    dtype = matQ.dtype

    if matC_initial is not None:
        assert vecN_initial is not None and scaM_initial is not None, "Initial states must be provided together."
        assert scaM_initial.axis() == 2, "Initial states must be 2D."
        matC_state, vecN_state, vecM_state = (
            matC_initial,
            vecN_initial,
            scaM_initial[:, :, None],
        )
    else:
        # memory state
        matC_state = jnp.zeros((B, NH, DHQK, DHV), dtype=dtype)
        # normalizer state
        vecN_state = jnp.zeros((B, NH, DHQK), dtype=dtype)
        # max state
        vecM_state = jnp.zeros((B, NH, 1), dtype=dtype)

    if return_all_states:
        matC_list = []
        matC_list.append(matC_state)

    vecH_list = []
    vecN_list, vecM_list = [], []
    vecN_list.append(vecN_state)
    vecM_list.append(vecM_state)
    for t in range(S):
        # gates
        vecF_t, vecI_t = vecF[:, :, t, None], vecI[:, :, t, None]  # (B, NH, 1)

        # projections
        vecQ_t, vecK_t, vecV_t = (
            matQ[:, :, t, :],  # (B, NH, DHQK)
            matK[:, :, t, :],  # (B, NH, DHQK)
            matV[:, :, t, :],  # (B, NH, DHV)
        )

        # step
        vecH, (matC_state, vecN_state, vecM_state) = mlstm_step_fn(
            matC_state=matC_state,
            vecN_state=vecN_state,
            scaM_state=vecM_state,
            vecQ=vecQ_t,
            vecK=vecK_t,
            vecV=vecV_t,
            scaI=vecI_t,
            scaF=vecF_t,
            eps=eps,
        )
        vecH_list.append(vecH)
        vecN_list.append(vecN_state)
        vecM_list.append(vecM_state)

        if return_all_states:
            matC_list.append(matC_state)

    matH = jnp.stack(vecH_list, axis=-2)  # (B, NH, S, DHV)
    vecN_states = jnp.stack(vecN_list, axis=-2)  # (B, NH, S, DHQK)
    vecM_states = jnp.concatenate(vecM_list, axis=-1)  # (B, NH, S)

    ret_tuple = (matH, vecN_states, vecM_states)

    if return_last_states:
        ret_tuple += ((matC_state, vecN_state, vecM_state),)
    else:
        ret_tuple += (None,)

    if return_all_states:
        matC_states = jnp.stack(matC_list, axis=-3)  # (B, NH, S, DHQK, DHV)
        ret_tuple += ((matC_states, vecN_states, vecM_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple


def mlstm_recurrent_sequence__native_fw(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    i: jax.Array,
    f: jax.Array,
    c_initial: jax.Array = None,
    n_initial: jax.Array = None,
    m_initial: jax.Array = None,
    return_last_states: bool = False,
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
    ret_tuple = _mlstm_recurrent_sequence_loop_fw(
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
        return_all_states=False,
    )
    if return_last_states:
        return ret_tuple[0], ret_tuple[3]
    else:
        return ret_tuple[0]


def mlstm_recurrent_sequence__triton_step_fused_fw(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    i: jax.Array,
    f: jax.Array,
    c_initial: jax.Array = None,
    n_initial: jax.Array = None,
    m_initial: jax.Array = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
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

    ret_tuple = _mlstm_recurrent_sequence_loop_fw(
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
        return_all_states=False,
    )
    if return_last_states:
        return ret_tuple[0], ret_tuple[3]
    else:
        return ret_tuple[0]
