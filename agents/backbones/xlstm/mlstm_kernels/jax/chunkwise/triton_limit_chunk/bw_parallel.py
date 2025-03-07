#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from ....triton.chunkwise.limit_chunk.bw_kernel_parallel import (
    mlstm_chunkwise__parallel_bw_dQKV_kernel,
)
from ....utils.kernels import is_power_of_2
from ...stride_utils import get_stride
from ...utils import jax2triton_dtype


def mlstm_chunkwise__parallel_bw_dQKV(
    matQ: jax.Array,  # (B, NH, S, DHQK)
    matK: jax.Array,  # (B, NH, S, DHQK)
    matV: jax.Array,  # (B, NH, S, DHHV)
    vecB: jax.Array,  # (B, NH, NC, L)
    vecI: jax.Array,  # (B, NH, NC, L)
    vecM_combine: jax.Array,  # (B, NH, S) = (B, NH, NC * L)
    scaM_inter: jax.Array,  # (B, NH, NC+1)
    matC_states: jax.Array,  # (B, NH, (NC+1) * DHQK, DHHV)
    matDeltaH: jax.Array,  # (B, NH, S, DHHV)
    vecN_out: jax.Array,  # (B, NH, S)
    matDeltaC_states: jax.Array,  # (B, NH, (NC+1) * DHQK, DHHV)
    qk_scale: float | None = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> tuple[
    jax.Array, jax.Array, jax.Array
]:  # matDeltaQ (B,NH,S,DHQK), matDeltaK (B,NH,S,DHQK), matDeltaV (B,NH,S,DHHV)
    """
    Computes the gradients for the query, key and value matrices.

    Args:
        matQ: Tensor containing the query vectors. Shape (B, NH, S, DHQK).
        matK: Tensor containing the key vectors. Shape (B, NH, S, DHQK).
        matV: Tensor containing the value vectors. Shape (B, NH, S, DHHV).
        vecB: Tensor containing the summed log forget gate activations. Shape (B, NH, NC, L).
        vecI: Tensor containing the input gate pre-activations. Shape (B, NH, NC, L).
        vecM_combine: Combined M states. Shape (B, NH, S).
        scaM_inter: States of the M scalar. Shape (B, NH, NC+1).
        matC_states: States of the C matrix. Shape (B, NH, NC * DHQK, DHHV).
        matDeltaH: Tensor containing the H gradients. Shape (B, NH, S, DHHV).
        vecN_out: States of the N vector. Shape (B, NH, S).
        matDeltaC_states: Tensor containing the C gradients. Shape (B, NH, (NC+1) * DHQK, DHHV).
        qk_scale: Scale factor for the QK matrix. Defaults to None.
        CHUNK_SIZE (int, optional): Chunk size. Defaults to 64.
        NUM_CHUNKS (int, optional): Number of chunks. Defaults to 1.
        EPS: Epsilon value. Defaults to 1e-6.

    Returns:
        Gradients for the query, key and value matrices. Shapes (B, NH, S, DHQK), (B, NH, S, DHQK), (B, NH, S, DHHV).
    """
    B, NH, S, DHQK, DHHV = *matQ.shape, matV.shape[-1]
    NC = NUM_CHUNKS
    L = CHUNK_SIZE
    _dtype = matQ.dtype

    assert NC == vecB.shape[2], "Number of chunks must match the number in vecB."
    assert L == vecB.shape[3], "Chunk size must match the chunk size in vecB."
    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = min(32 if str(_dtype) == "float32" else 64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(32 if str(_dtype) == "float32" else 64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    # Specify output shapes.
    matDeltaQ = jax.ShapeDtypeStruct(shape=(B, NH, S, DHQK), dtype=_dtype)
    matDeltaK = jax.ShapeDtypeStruct(shape=(B, NH, S, DHQK), dtype=_dtype)
    # each b_DHQK thread block computes the contribution of its siz_b_DHQK block of matDeltaC
    # we need to sum them up to get the final result (we do this outside the kernel)
    matDeltaV = jax.ShapeDtypeStruct(shape=(num_b_DHQK, B, NH, S, DHHV), dtype=_dtype)

    # Define the grid and call the triton kernel.
    grid = (num_b_DHQK, NC, B * NH)
    matDeltaQ, matDeltaK, matDeltaV = jt.triton_call(
        matQ,  # (B, NH, S, DHQK)
        matK,  # (B, NH, S, DHQK)
        matV,  # (B, NH, S, DHHV)
        vecB,  # (B, NH, NC, L)
        vecI,  # (B, NH, NC, L)
        vecM_combine,  # (B, NH, S)
        scaM_inter,  # (B, NH, NC+1)
        matC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
        matDeltaH,  # (B, NH, S, DHHV)
        vecN_out,  # (B, NH, S)
        matDeltaC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
        out_shape=(matDeltaQ, matDeltaK, matDeltaV),
        qk_scale=qk_scale,
        str_matQK_B_NH=get_stride(matQ, axis=1),
        str_matQK_S=get_stride(matQ, axis=2),
        str_matQK_DHQK=get_stride(matQ, axis=3),
        str_matDV_num_b_DHQK=get_stride(matDeltaV, axis=0),
        str_matHV_B_NH=get_stride(matV, axis=1),
        str_matHV_S=get_stride(matV, axis=2),
        str_matHV_DHHV=get_stride(matV, axis=3),
        str_vecBI_B_NH=get_stride(vecI, axis=1),
        str_vecBI_NC=get_stride(vecI, axis=2),
        str_vecBI_L=get_stride(vecI, axis=3),
        str_vecM_combine_B_NH=get_stride(vecM_combine, axis=1),
        str_vecM_combine_S=get_stride(vecM_combine, axis=2),
        str_scaM_inter_B_NH=get_stride(scaM_inter, axis=1),
        str_scaM_inter_NC=get_stride(scaM_inter, axis=2),
        str_matC_states_B_NH=get_stride(matC_states, axis=1),
        str_matC_states_NCDHQK=get_stride(matC_states, axis=2),
        str_matC_states_DHHV=get_stride(matC_states, axis=3),
        str_vecN_out_B_NH=get_stride(vecN_out, axis=1),
        str_vecN_out_S=get_stride(vecN_out, axis=2),
        str_matDeltaC_states_B_NH=get_stride(matDeltaC_states, axis=1),
        str_matDeltaC_states_NCDHQK=get_stride(matDeltaC_states, axis=2),
        str_matDeltaC_states_DHHV=get_stride(matDeltaC_states, axis=3),
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        DTYPE=jax2triton_dtype(_dtype),
        EPS=EPS,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=grid,
        kernel=mlstm_chunkwise__parallel_bw_dQKV_kernel,
    )
    # sum up the contributions of each siz_b_DHQK block
    matDeltaV = matDeltaV.sum(axis=0)  # (B, NH, S, DHHV)

    return matDeltaQ, matDeltaK, matDeltaV