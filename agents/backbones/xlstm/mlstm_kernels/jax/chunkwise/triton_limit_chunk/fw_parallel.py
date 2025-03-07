#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from ....triton.chunkwise.limit_chunk.fw_kernel_parallel import (
    mlstm_chunkwise__parallel_fw_H_kernel,
)
from ....utils.kernels import is_power_of_2
from ...stride_utils import get_stride
from ...utils import jax2triton_dtype


def mlstm_chunkwise__parallel_fw_H(
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHHV)
    matC_states: jax.Array,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecN_states: jax.Array,  # (B, NH, (NC+1) * DHQK)
    scaMinter_states: jax.Array,  # (B, NH, (NC+1))
    vecI: jax.Array,  # (B, NH, NC, L)
    vecB: jax.Array,  # (B, NH, NC, L)
    qk_scale: float | None = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:  # matH_out (B, NH, S, DHHV), vecN_out (B, NH, S)
    """
    Execute the parallel forward kernel for the H computation in the mLSTM chunkwise formulation.

    This function defines the grid and block sizes for the kernel launch and calls the kernel. See
    the fwbw backend implementation and the triton kernels for more information.

    Args:
        matQ: Tensor containing the queries. Shape (B, NH, S, DHQK).
        matK: Tensor containing the keys. Shape (B, NH, S, DHQK).
        matV: Tensor containing the values. Shape (B, NH, S, DHHV).
        matC_states: States of the C matrix. Shape (B, NH, NC * DHQK, DHHV).
            This state and following states must be all states up to the last chunk, i.e. :-1.
        vecN_states: States of the N vector. Shape (B, NH, NC * DHQK).
        scaMinter_states: States of the M scalar. Shape (B, NH, NC).
        vecI: Tensor containing the input gate. Shape (B, NH, NC, L).
        vecB: Tensor containing the summed log forget gate activations. Shape (B, NH, NC, L).
        qk_scale: Scaling factor for the QK matrix. Defaults to None and will be inferred.
        CHUNK_SIZE: Chunk size for the kernel. Defaults to 64.
        NUM_CHUNKS: Number of chunks. Defaults to 1.
        EPS: Small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        Tuple containing the output matrix H (shape (B, NH, S, DHHV)) and the N vector (shape (B, NH, S)).
    """
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]
    NC = NUM_CHUNKS
    L = CHUNK_SIZE

    assert NC == vecB.shape[2], "Number of chunks must match the number of chunks in vecB."
    assert L == vecB.shape[3], "Chunk size must match the chunk size in vecB."
    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    # Prepare the output shapes.
    matH_out = jax.ShapeDtypeStruct((B, NH, S, DHHV), matQ.dtype)
    vecN_out = jax.ShapeDtypeStruct((B, NH, S), jnp.float32)
    vecM_out = jax.ShapeDtypeStruct((B, NH, S), jnp.float32)

    # Define the grid and call the triton kernel.
    grid = (num_b_DHHV, NC, B * NH)
    matH_out, vecN_out, vecM_out = jt.triton_call(
        matQ,  # (B, NH, S, DHQK)
        matK,  # (B, NH, S, DHQK)
        matV,  # (B, NH, S, DHHV)
        matC_states,  # (B, NH, NC * DHQK, DHHV)
        vecN_states,  # (B, NH, NC * DHQK)
        scaMinter_states,  # (B, NH, NC)
        vecI,  # (B, NH, NC, L)
        vecB,  # (B, NH, NC, L)
        out_shape=(matH_out, vecN_out, vecM_out),
        qk_scale=qk_scale,
        str_matQK_B_NH=get_stride(matQ, axis=1),
        str_matQK_S=get_stride(matQ, axis=2),
        str_matQK_DHQK=get_stride(matQ, axis=3),
        str_matHV_B_NH=get_stride(matV, axis=1),
        str_matHV_S=get_stride(matV, axis=2),
        str_matHV_DHHV=get_stride(matV, axis=3),
        str_matCstates_B_NH=get_stride(matC_states, axis=1),
        str_matCstates_NCDHQK=get_stride(matC_states, axis=2),
        str_matCstates_DHHV=get_stride(matC_states, axis=3),
        str_vecNstates_B_NH=get_stride(vecN_states, axis=1),
        str_vecNstates_NCDHQK=get_stride(vecN_states, axis=2),
        str_scaMinterstates_B_NH=get_stride(scaMinter_states, axis=1),
        str_vecBI_B_NH=get_stride(vecB, axis=1),
        str_vecBI_NC=get_stride(vecB, axis=2),
        str_vecBI_L=get_stride(vecB, axis=3),
        str_vecMN_B_NH=get_stride(vecN_out, axis=1),
        str_vecMN_S=get_stride(vecN_out, axis=2),
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        DTYPE=jax2triton_dtype(matQ.dtype),
        EPS=EPS,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=grid,
        kernel=mlstm_chunkwise__parallel_fw_H_kernel,
    )

    return matH_out, vecN_out, vecM_out