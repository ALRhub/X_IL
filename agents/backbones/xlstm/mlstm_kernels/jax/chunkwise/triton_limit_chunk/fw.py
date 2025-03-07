#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import jax.numpy as jnp

from .fw_parallel import mlstm_chunkwise__parallel_fw_H
from .fw_recurrent import mlstm_chunkwise__recurrent_fw_C


def mlstm_chunkwise_fw(
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHV)
    vecI: jax.Array,  # (B, NH, S)
    vecF: jax.Array,  # (B, NH, S)
    matC_initial: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    vecN_initial: jax.Array | None = None,  # (B, NH, DHQK)
    scaM_initial: jax.Array | None = None,  # (B, NH)
    qk_scale: float = None,
    return_last_states: bool = False,
    return_all_states: bool = False,
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
) -> tuple[
    jax.Array,  # matH_out (B, NH, S, DHV)
    jax.Array,  # vecN_out (B, NH, S)
    jax.Array,  # vecM_out (B, NH, S)
    None
    | (
        tuple[jax.Array, jax.Array, jax.Array]
    ),  # last_states (matC_states (B, NH, DHQK, DHV), vecN_states (B, NH, DHQK), scaMinter_states (B, NH))
    None
    | (
        tuple[jax.Array, jax.Array, jax.Array]
    ),  # all_states (matC_states (B, NH, (NC+1) * DHQK, DHV), vecN_states (B, NH, (NC+1) * DHQK),
    # scaMinter_states (B, NH, (NC+1)))
]:
    """
    Execute the forward pass of the mLSTM chunkwise formulation.

    Args:
        matQ: Tensor containing the queries. Shape (B, NH, S, DHQK).
        matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
        matV: Tensor containing the values. Shape (B, NH, S, DHV).
        vecI: Tensor containing the input gate. Shape (B, NH, S).
        vecF: Tensor containing the forget gate. Shape (B, NH, S).
        matC_initial: Initial state of the C matrix. Shape (B, NH, DHQK, DHV).
            Defaults to None.
        vecN_initial: Initial state of the N vector. Shape (B, NH, DHQK).
            Defaults to None.
        scaM_initial: Initial state of the M scalar. Shape (B, NH).
            Defaults to None.
        qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
        return_last_states: Whether to return the last states. Defaults to False.
        return_all_states: Whether to return all states. Defaults to False.
        CHUNK_SIZE: Chunk size for the kernel. Defaults to 64.
        EPS: Small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        Tuple containing the output matrix H (shape (B, NH, S, DHV)), the N vector (shape (B, NH, S)),
        the M scalar (shape (B, NH)). Optionally, it might contain last states (matC_states,
        vecN_states, scaMinter_states) and optional all states (matC_states, vecN_states,
        scaMinter_states).
    """
    B, NH, S, DHQK = matQ.shape
    assert S % CHUNK_SIZE == 0, f"Sequence length {S} is not divisible by chunk size {CHUNK_SIZE}."
    NC = S // CHUNK_SIZE

    vecI = vecI.reshape(B, NH, NC, CHUNK_SIZE)
    vecF = vecF.reshape(B, NH, NC, CHUNK_SIZE).astype(jnp.float32)

    # Compute the gates, the g and the a and b vectors.
    vecF_logsig: jax.Array = jax.nn.log_sigmoid(vecF)
    vecB = vecF_logsig.cumsum(axis=-1)

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    # Materialize the C_k, n_k, m_k states for each chunk.
    matC_k_states, vecN_k_states, scaMinter_k_states = mlstm_chunkwise__recurrent_fw_C(
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaMinter_initial=scaM_initial,
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
    )

    # Compute the outputs within each chunk.
    matH_out, vecN_out, vecM_out = mlstm_chunkwise__parallel_fw_H(
        matQ=matQ,
        matK=matK,
        matV=matV,
        # These slices are not needed in the kernel and introduces considerable overhead.
        matC_states=matC_k_states,
        vecN_states=vecN_k_states,
        scaMinter_states=scaMinter_k_states,
        vecI=vecI,
        vecB=vecB,
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )

    # Return the outputs and optionally the states.
    ret_tuple = (
        matH_out,
        vecN_out,
        vecM_out,
    )
    if return_last_states:
        ret_tuple += (
            (
                matC_k_states[:, :, -DHQK:, :],
                vecN_k_states[:, :, -DHQK:],
                scaMinter_k_states[:, :, -1],
            ),
        )
    else:
        ret_tuple += (None,)
    if return_all_states:
        ret_tuple += ((matC_k_states, vecN_k_states, scaMinter_k_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple  # (matH_out, vecN_out, vecM_out, optional(last_states), optional(all_states))
