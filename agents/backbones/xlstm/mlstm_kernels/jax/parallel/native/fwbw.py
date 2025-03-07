#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
Jax.

mLSTM forward and backward pass. Parallel formulation.
"""
from collections.abc import Callable

import jax
import jax.numpy as jnp

from .bw import mlstm_parallel_bw
from .fw import mlstm_parallel_fw


def mlstm_parallel__native_autograd(
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
) -> jax.Array:
    """
    Apply the mLSTM parallel formulation in native JAX.
    Gradients are computed through autograd.
    This function does not use stable forget gate matrix computation.

    Args:
        q: The query tensor of shape (B, NH, S, DHQK).
        k: The key tensor of shape (B, NH, S, DHQK).
        v: The value tensor of shape (B, NH, S, DHV).
        i: The input gate preactivation tensor of shape (B, NH, S).
        f: The forget gate preactivation tensor of shape (B, NH, S).
        c_initial: The initial chunk state tensor of shape (B, NH, DHQK, DHV).
        n_initial: The initial chunk state tensor of shape (B, NH, DHQK).
        m_initial: The initial chunk state tensor of shape (B, NH).
        return_last_states: Whether to return the last states of the mLSTM.
        eps: The epsilon value to use for numerical stability.
    Returns:
        The output of the mLSTM computation. 
    """

    assert c_initial is None, "c_initial is not supported"
    assert n_initial is None, "n_initial is not supported"
    assert m_initial is None, "m_initial is not supported"
    assert not return_last_states, "return_last_states is not supported"

    matH, _, _ = mlstm_parallel_fw(
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        eps=eps,
    )
    return matH

def _mlstm_parallel_fwbw_generator(
    autocast_kernel_dtype: jnp.dtype = jnp.bfloat16,
    eps: float = 1e-6,
) -> Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
]:
    """
    Generate a forward and backward pass function for the mLSTM parallel formulation.

    Args:
        autocast_kernel_dtype: The dtype to use for the kernel computation. All inputs arguments up to vecF
            are cast to this dtype. vecF is automatically casted to float32 in the kernels.
        eps: The epsilon value to use for numerical stability.

    Returns:
        A function that computes the forward pass of the mLSTM chunkwise formulation, which custom gradients for the
        backward pass. The function input signatures is:

            forward(
                matQ: jax.Array,  # (B, NH, S, DHQK)
                matK: jax.Array,  # (B, NH, S, DHQK)
                matV: jax.Array,  # (B, NH, S, DHV)
                vecI: jax.Array,  # (B, NH, S)
                vecF: jax.Array,  # (B, NH, S)
            ) -> jax.Array:
        The function returns the output of the mLSTM computation.
    """

    @jax.custom_gradient
    def forward(
        matQ: jax.Array,  # (B, NH, S, DHQK)
        matK: jax.Array,  # (B, NH, S, DHQK)
        matV: jax.Array,  # (B, NH, S, DHV)
        vecI: jax.Array,  # (B, NH, S)
        vecF: jax.Array,  # (B, NH, S)
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        B, NH, S, DHQK = matQ.shape
        # Verify shapes to prevent errors in the kernels.
        assert matK.shape == (B, NH, S, DHQK), f"matK shape {matK.shape} does not match matQ shape {matQ.shape}."
        assert matV.shape[:-1] == (B, NH, S), f"matV shape {matV.shape} does not match matQ shape {matQ.shape}."
        assert vecI.shape == (B, NH, S), f"vecI shape {vecI.shape} does not match matQ shape {matQ.shape}."
        assert vecF.shape == (B, NH, S), f"vecF shape {vecF.shape} does not match matQ shape {matQ.shape}."
        # Cast to autocast_kernel_dtype. Exclude vecF as it is automatically upcasted to float32 in kernels.
        orig_dtypes = {"q": matQ.dtype, "k": matK.dtype, "v": matV.dtype, "i": vecI.dtype, "f": vecF.dtype}
        matQ = matQ.astype(autocast_kernel_dtype)
        matK = matK.astype(autocast_kernel_dtype)
        matV = matV.astype(autocast_kernel_dtype)
        vecI = vecI.astype(autocast_kernel_dtype)

        # Call the forward parallel jax implementation for the mLSTM.
        matH_out, vecN, vecM = mlstm_parallel_fw(
            matQ=matQ,
            matK=matK,
            matV=matV,
            vecI=vecI,
            vecF=vecF,
            eps=eps,
        )

        def backward(
            grad_list: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[
            jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None, jax.Array | None
        ]:
            """Backward function with reverse function signature of forward."""
            # Unpack the gradients.
            matDeltaH, _, _ = grad_list
            # Call the backward triton kernels for the mLSTM.
            (
                matDeltaQ,
                matDeltaK,
                matDeltaV,
                vecDeltaI,
                vecDeltaF,
            ) = mlstm_parallel_bw(
                matDeltaHtilde=matDeltaH,
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                vecN=vecN,
                vecM=vecM,
                eps=eps,
            )
            # Cast back to original dtypes.
            matDeltaQ = matDeltaQ.astype(orig_dtypes["q"])
            matDeltaK = matDeltaK.astype(orig_dtypes["k"])
            matDeltaV = matDeltaV.astype(orig_dtypes["v"])
            vecDeltaI = vecDeltaI.astype(orig_dtypes["i"])
            vecDeltaF = vecDeltaF.astype(orig_dtypes["f"])
            # Return gradients.
            return (
                matDeltaQ,
                matDeltaK,
                matDeltaV,
                vecDeltaI,
                vecDeltaF,
            )

        return (matH_out, vecN, vecM), backward

    return forward


def _get_parallel_fwbw_kernel(autocast_kernel_dtype: jnp.dtype, **kwargs) -> Callable:
    """
    Get the forward and backward pass function for the mLSTM parallel formulation.

    Args:
        autocast_kernel_dtype: The dtype to use for the kernel computation. All inputs arguments up to vecF
            are cast to this dtype. vecF is automatically casted to float32 in the kernels.
        **kwargs: Additional keyword arguments to pass to the kernel function.

    Returns:
        A function that computes the forward pass of the mLSTM chunkwise formulation, which custom gradients for the
        backward pass. See _mlstm_parallel_fwbw_generator for the function signature.
    """
    if autocast_kernel_dtype in ["float32", "float16", "bfloat16", jnp.float32, jnp.float16, jnp.bfloat16]:
        return _mlstm_parallel_fwbw_generator(autocast_kernel_dtype, **kwargs)
    else:
        raise ValueError(f"Unsupported kernel dtype {autocast_kernel_dtype}.")


def mlstm_parallel__native_custbw(
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
    autocast_kernel_dtype: jnp.dtype = jnp.float32,
    **kwargs,
) -> jax.Array:
    """
    Apply the mLSTM parallel formulation in native JAX.
    Gradients are computed through autograd.
    This function does not use stable forget gate matrix computation.


    Args:
        q: The query tensor of shape (B, NH, S, DHQK).
        k: The key tensor of shape (B, NH, S, DHQK).
        v: The value tensor of shape (B, NH, S, DHV).
        i: The input gate preactivation tensor of shape (B, NH, S).
        f: The forget gate preactivation tensor of shape (B, NH, S).
        c_initial: The initial chunk state tensor of shape (B, NH, DHQK, DHV).
        n_initial: The initial chunk state tensor of shape (B, NH, DHQK).
        m_initial: The initial chunk state tensor of shape (B, NH).
        return_last_states: Whether to return the last states of the mLSTM.
        eps: The epsilon value to use for numerical stability.
        autocast_kernel_dtype: The dtype to use for the mLSTM computation. All inputs arguments up
            to vecF are cast to this dtype. vecF is automatically casted to float32 in the mLSTM computation.


    Returns:
        The output of the mLSTM computation. 
    """

    assert c_initial is None, "c_initial is not supported"
    assert n_initial is None, "n_initial is not supported"
    assert m_initial is None, "m_initial is not supported"
    assert return_last_states is False, "return_last_states is not supported"

    _mlstm_parallel_fwbw = _get_parallel_fwbw_kernel(autocast_kernel_dtype=autocast_kernel_dtype, eps=eps)

    matH, _, _ = _mlstm_parallel_fwbw(q, k, v, i, f)
    return matH
