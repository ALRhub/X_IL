#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import jax.numpy as jnp

from .bw_parallel import mlstm_chunkwise__parallel_bw_dQKV
from .bw_recurrent import mlstm_chunkwise__recurrent_bw_dC
from .fw_recurrent import mlstm_chunkwise__recurrent_fw_C


def mlstm_chunkwise_bw(
    # Forward arguments
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHV)
    vecI: jax.Array,  # (B, NH, S)
    vecF: jax.Array,  # (B, NH, S)
    matC_initial: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    vecN_initial: jax.Array | None = None,  # (B, NH, DHQK)
    scaM_initial: jax.Array | None = None,  # (B, NH)
    qk_scale: float = None,
    # Backward arguments
    matC_all: jax.Array | None = None,  # (B, NH, NC * DHQK, DHV)
    vecN_all: jax.Array | None = None,  # (B, NH, NC * DHQK)
    scaM_all: jax.Array | None = None,  # (B, NH, NC)
    vecN_out: jax.Array | None = None,  # (B, NH, NC * L) = (B, NH, S)
    vecM_out: jax.Array | None = None,  # (B, NH, NC * L) = (B, NH, S)
    matDeltaH: jax.Array | None = None,  # (B, NH, S, DHV)
    matDeltaC_last: jax.Array | None = None,  # (B, NH, DHQK, DHV)
    # Common arguments
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
    reduce_slicing: bool = False,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None, jax.Array | None
]:  # matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF, matDeltaC_initial, vecDeltaN_initial, scaDeltaM_initial
    """
    Computes the backward pass of the mLSTM chunkwise formulation.

    Args:
        matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
        matK: Tensor containing the key vectors. Shape (B, NH, S, DHQK).
        matV: Tensor containing the value vectors. Shape (B, NH, S, DHV).
        vecI: Tensor containing the input gate pre-activations. Shape (B, NH, S).
        vecF: Tensor containing the forget gate pre-activations. Shape (B, NH, S).
        matC_initial: Tensor containing the initial C states. Shape (B, NH, DHQK, DHV).
            Defaults to None.
        vecN_initial: Tensor containing the initial N states. Shape (B, NH, DHQK).
            Defaults to None.
        scaM_initial: Tensor containing the initial M states. Shape (B, NH).
            Defaults to None.
        qk_scale: Scale factor for the QK matrix. Defaults to None.
        matC_all: Tensor containing all C states. Shape (B, NH, NC * DHQK, DHV).
            Defaults to None.
        vecN_all: Tensor containing all N states. Shape (B, NH, NC * DHQK).
            Defaults to None.
        scaM_all: Tensor containing all M states. Shape (B, NH, NC).
            Defaults to None.
        vecN_out: Tensor containing the N states for the output. Shape (B, NH, S).
            Defaults to None.
        vecM_out: Tensor containing the M states for the output. Shape (B, NH, S).
            Defaults to None.
        matDeltaH: Tensor containing the H gradients. Shape (B, NH, S, DHV).
            Defaults to None.
        matDeltaC_last: Tensor containing the last C gradients. Shape (B, NH, DHQK, DHV).
            Defaults to None.
        CHUNK_SIZE: Chunk size. Defaults to 64.
        EPS: Epsilon value. Defaults to 1e-6.
        reduce_slicing: If True, reduces the slicing operations taken in the preprocessing to
            the kernel. This leads to performance improvements during training while returning
            the same results. Defaults to False.

    Returns:
        Gradients for the query, key, value, vecI and vecF matrices. Shapes (B, NH, S, DHQK),
        (B, NH, S, DHQK), (B, NH, S, DHV), (B, NH, S), (B, NH, S). If initial states are provided,
        the function also returns the gradients for the initial C, N and M states.
    """
    B, NH, S, DHQK = matQ.shape

    assert S % CHUNK_SIZE == 0, f"Sequence length {S} is not divisible by chunk size {CHUNK_SIZE}."

    NC = S // CHUNK_SIZE

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    vecI = vecI.reshape(B, NH, NC, CHUNK_SIZE)
    vecF = vecF.reshape(B, NH, NC, CHUNK_SIZE).astype(jnp.float32)

    # Compute the gates, the g and the a and b vectors.
    vecF_logsig: jax.Array = jax.nn.log_sigmoid(vecF)
    vecB = vecF_logsig.cumsum(axis=-1)

    # Recompute the "all" states if needed.
    if matC_all is None:
        assert (
            (matC_all is None) and (vecN_all is None) and (scaM_all is None)
        ), "Either all or none of the states must be provided."
        matC_all, vecN_all, scaM_all = mlstm_chunkwise__recurrent_fw_C(
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

    # Recurrent backward: compute the deltaC gradients.
    matDeltaC_states = mlstm_chunkwise__recurrent_bw_dC(
        matQ=matQ,  # (B, NH, S, DHQK)
        vecB=vecB,  # (B, NH, NC, L)
        scaM_inter=scaM_all,  # (B, NH, NC+1)
        vecM_combine=vecM_out,  # (B, NH, S)
        matDeltaH=matDeltaH,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_last=matDeltaC_last,  # (B, NH, DHQK, DHV)
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )  # (B, NH, NC * DHQK, DHV)

    # Parallel backward: compute the deltaQ, deltaK, deltaV, deltaI gradients

    matDeltaQ, matDeltaK, matDeltaV = mlstm_chunkwise__parallel_bw_dQKV(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        vecM_combine=vecM_out,
        scaM_inter=scaM_all,  # (B, NH, NC)
        matC_states=matC_all,  # (B, NH, (NC+1) * DHQK, DHV)
        matDeltaH=matDeltaH,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_states=matDeltaC_states,  # (B, NH, (NC+1) * DHQK, DHV)
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )

    # Postprocessing: compute deltaF and deltaI gradients.
    vecF = vecF.reshape(B, NH, S)
    # Compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1)).
    matQ = matQ.astype(jnp.float32)
    matK = matK.astype(jnp.float32)
    matDeltaQ = matDeltaQ.astype(jnp.float32)
    matDeltaK = matDeltaK.astype(jnp.float32)
    vecDeltaFbar_acc = ((matQ * matDeltaQ) - (matK * matDeltaK)).sum(axis=-1)
    vecDeltaFbar = jnp.flip(vecDeltaFbar_acc, axis=-1).astype(jnp.float32)
    vecDeltaFbar = jnp.flip(vecDeltaFbar.cumsum(axis=-1), axis=-1)
    vecDeltaF = vecDeltaFbar * jax.nn.sigmoid(-vecF)
    # Compute deltaI.
    # Both are equivalent:
    # vecDeltaI = (matV * matDeltaV).sum(-1)
    vecDeltaI = (matK * matDeltaK).sum(axis=-1)

    matDeltaC_initial = matDeltaC_states[:, :, :DHQK, :] if matC_initial is not None else None
    vecDeltaN_initial = jnp.zeros_like(vecN_initial) if vecN_initial is not None else None
    scaDeltaM_initial = jnp.zeros_like(scaM_initial) if scaM_initial is not None else None

    return (
        matDeltaQ,
        matDeltaK,
        matDeltaV,
        vecDeltaI,
        vecDeltaF,
        matDeltaC_initial,
        vecDeltaN_initial,
        scaDeltaM_initial,
    )
