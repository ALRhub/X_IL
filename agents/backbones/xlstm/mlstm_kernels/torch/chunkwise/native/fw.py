#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""PyTorch.

Forward pass of the mLSTM chunkwise formulation.

Notation:
Dimensions:
    B: batch size
    NH: number of heads
    S: sequence length
    DH: hidden dimension
    NC: number of chunks
    L: chunk size

Variables:
    vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to current timestep t
    vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk state C_k (form current timestep t)
    scaG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.
"""

import torch
import torch.nn.functional as F
from einops import rearrange


def mlstm_chunkwise__recurrent_fw_C(
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecB: torch.Tensor,  # (B, NH, NC, L) # cumsum(logsigmoid(f))
    vecI: torch.Tensor,  # (B, NH, NC, L)
    matC_states: torch.Tensor = None,  # (B, NH, (NC + 1) * DHQK, DHHV)
    vecN_states: torch.Tensor = None,  # (B, NH, (NC + 1) * DHQK)
    scaMinter_states: torch.Tensor = None,  # (B, NH, (NC + 1)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaMinter_initial: torch.Tensor = None,  # (B, NH, 1)
    qk_scale: float = None,
    chunk_size: int = 64,
    num_chunks: int = 1,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matC_states (B, NH, (NC+1) * DHQK, DHHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1))
    B, NH, S, DHQK, DHHV = *matK.shape, matV.shape[-1]
    NC = num_chunks
    _dtype, _device = matK.dtype, matK.device

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    # initialize the states tensors
    if matC_states is None:
        matC_states = torch.zeros(
            (B, NH, (NC + 1) * DHQK, DHHV), dtype=_dtype, device=_device
        )
    if vecN_states is None:
        vecN_states = torch.zeros(
            (B, NH, (NC + 1) * DHQK), dtype=_dtype, device=_device
        )
    if scaMinter_states is None:
        scaMinter_states = torch.zeros((B, NH, (NC + 1)), dtype=_dtype, device=_device)

    # assign the initial states to the running states
    matC_k = (
        torch.zeros((B, NH, DHQK, DHHV), dtype=_dtype, device=_device)
        if matC_initial is None
        else matC_initial
    )
    vecN_k = (
        torch.zeros((B, NH, DHQK), dtype=_dtype, device=_device)
        if vecN_initial is None
        else vecN_initial
    )
    scaM_inter_k = (
        torch.zeros((B, NH, 1), dtype=_dtype, device=_device)
        if scaMinter_initial is None
        else scaMinter_initial
    )
    vecA = vecB[..., -1, None] - vecB + vecI
    scaG = vecB[..., -1]
    scaA_max = vecA.max(-1).values

    # for this implementation we actually want to have shape (B, NH) for scaM_inter_k
    scaM_inter_k = scaM_inter_k.squeeze(-1)

    for k in range(0, num_chunks):
        # store the states from the previous iteration before updating them
        # in the first iteration, these are the initial states
        matC_states[:, :, k * DHQK : (k + 1) * DHQK, :] = matC_k
        vecN_states[:, :, k * DHQK : (k + 1) * DHQK] = vecN_k
        scaMinter_states[:, :, k] = scaM_inter_k

        # m_k update
        scaA_max_k = scaA_max[:, :, k]
        scaG_k = scaG[:, :, k]
        scaM_inter_k_next = torch.max(scaG_k + scaM_inter_k, scaA_max_k)
        # C_k update
        matK_chunk = matK[:, :, k * chunk_size : (k + 1) * chunk_size, :]  # * qk_scale
        matV_chunk = matV[:, :, k * chunk_size : (k + 1) * chunk_size, :]
        vecA_k = vecA[:, :, k, :]

        vecAbar_k = torch.exp(vecA_k - scaM_inter_k_next[..., None])[:, :, :, None]

        matK_chunk_gated = matK_chunk * vecAbar_k

        scaGbar_k = torch.exp(scaG_k + scaM_inter_k - scaM_inter_k_next)[:, :, None]

        # NOTE: no update in-place (i.e. +=) as this gives error for autograd backward
        matC_k_next = scaGbar_k[..., None] * matC_k + matK_chunk_gated.transpose(
            -2, -1
        ) @ (matV_chunk)

        # n_k update
        vecN_k_next = scaGbar_k * vecN_k + matK_chunk_gated.transpose(-2, -1).sum(-1)

        # move to the next iteration
        scaM_inter_k = scaM_inter_k_next
        matC_k = matC_k_next
        vecN_k = vecN_k_next

    # store the states from the last iteration
    matC_states[:, :, -DHQK:, :] = matC_k
    vecN_states[:, :, -DHQK:] = vecN_k
    scaMinter_states[:, :, -1] = scaM_inter_k

    return matC_states, vecN_states, scaMinter_states


def mlstm_chunkwise__parallel_fw_H(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    # these states must be all states up to the last chunk, i.e. :-1
    matC_states: torch.Tensor,  # (B, NH, NC * DHQK, DHHV)
    vecN_states: torch.Tensor,  # (B, NH, NC * DHQK)
    scaMinter_states: torch.Tensor,  # (B, NH, NC)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    qk_scale: float,
    chunk_size: int = 64,
    num_chunks: int = 1,
    eps: float = 1e-6,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matH_out (B, NH, S, DHHV), vecN_out (B, NH, S), vecM_out (B, NH, S)
    _device = matQ.device
    NC, L = num_chunks, chunk_size
    matC_k_states = rearrange(
        matC_states, "b nh (nc dhqk) dhv -> b nh nc dhqk dhv", nc=NC
    )
    vecN_k_states = rearrange(vecN_states, "b nh (nc dhqk) -> b nh nc dhqk", nc=NC)
    scaMinter_k_states = scaMinter_states

    matQ = rearrange(matQ, "b nh (nc l) dh -> b nh nc l dh", l=L)
    matK = rearrange(matK, "b nh (nc l) dh -> b nh nc l dh", l=L)
    matV = rearrange(matV, "b nh (nc l) dh -> b nh nc l dh", l=L)

    ltr = torch.tril(
        torch.ones(
            (L, L),
            dtype=torch.bool,
            device=_device,
        )
    )

    # compute the H_states in parallel

    # Compute intra chunk contribution: H_intra
    matF_logsig_chunk = vecB[:, :, :, :, None] - vecB[:, :, :, None, :]

    matF_logsig_mask_chunk = torch.where(ltr, matF_logsig_chunk, -float("inf"))

    matLogD_chunk = matF_logsig_mask_chunk + vecI[:, :, :, None, :]

    # max_state intra
    vecMintra_k = torch.max(
        matLogD_chunk, dim=-1, keepdim=False
    ).values  # (B, NH, NC, L)

    # max_state combined
    vecM_b_inter = vecB + scaMinter_k_states[:, :, :, None]  # (B, NH, NC, L)
    vecM_k_combine = torch.maximum(vecM_b_inter, vecMintra_k)  # (B, NH, NC, L)

    vecM_k_combine = vecM_k_combine[:, :, :, :, None]  # (B, NH, NC, L, 1)
    vecM_b_inter = vecM_b_inter[:, :, :, :, None]  # (B, NH, NC, L, 1)

    matLogD_stabilized_chunk = matLogD_chunk - vecM_k_combine
    matD_chunk = torch.exp(matLogD_stabilized_chunk)

    matS_chunk = (matQ @ matK.transpose(-2, -1)) * qk_scale

    matM_chunk = matS_chunk * matD_chunk

    # ? Combine H_intra with H_inter
    vecBbar = torch.exp(vecM_b_inter - vecM_k_combine)
    matQ_chunk_gated = matQ * vecBbar * qk_scale

    matNumerator_common = (
        matQ_chunk_gated @ matC_k_states + matM_chunk @ matV
    )  # (B, NH, NC, L, DHHV)

    vecDenom_l_common = matQ_chunk_gated @ vecN_k_states.unsqueeze(-1) + matM_chunk.sum(
        dim=-1, keepdim=True
    )  # (B, NH, NC, L, 1)

    vecDenom_max_common = torch.maximum(
        torch.abs(vecDenom_l_common), torch.exp(-vecM_k_combine)
    )

    matH_k_chunk = matNumerator_common / (vecDenom_max_common + eps)

    matH_out = rearrange(matH_k_chunk, "b nh nc l dh -> b nh (nc l) dh")

    # we need the denominator and the overall max state for the backward pass
    vecN_out = rearrange(
        vecDenom_max_common, "b nh nc l 1 -> b nh (nc l)"
    )  # (B, NH, S)
    vecM_out = rearrange(vecM_k_combine, "b nh nc l 1 -> b nh (nc l)")  # (B, NH, S)
    return matH_out, vecN_out, vecM_out


def mlstm_chunkwise_fw(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH, 1)
    qk_scale: float = None,
    return_last_states: bool = False,
    return_all_states: bool = False,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> tuple[
    torch.Tensor,  # matH_out (B, NH, S, DHHV)
    torch.Tensor,  # vecN_out (B, NH, S)
    torch.Tensor,  # vecM_out (B, NH, S)
    None
    | (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ),  # last_states (matC_states (B, NH, DHQK, DHHV), vecN_states (B, NH, DHQK), scaMinter_states (B, NH, 1))
    None
    | (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ),  # all_states (matC_states (B, NH, (NC+1) * DHQK, DHHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1)))
]:
    B, NH, S, DHQK = matQ.shape
    assert (
        S % chunk_size == 0
    ), f"Sequence length {S} is not divisible by chunk size {chunk_size}."
    NC = S // chunk_size

    vecI = rearrange(vecI, "b nh (nc l) -> b nh nc l", l=chunk_size)
    vecF = rearrange(vecF, "b nh (nc l) -> b nh nc l", l=chunk_size)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = F.logsigmoid(vecF)
    vecB = vecF_logsig.cumsum(-1)

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    #! materialize the  C_k, n_k, m_k states for each chunk
    matC_k_states, vecN_k_states, scaMinter_k_states = mlstm_chunkwise__recurrent_fw_C(
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaMinter_initial=scaM_initial,
        qk_scale=qk_scale,
        chunk_size=chunk_size,
        num_chunks=NC,
    )

    #! compute the outputs within each chunk
    matH_out, vecN_out, vecM_out = mlstm_chunkwise__parallel_fw_H(
        matQ=matQ,
        matK=matK,
        matV=matV,
        matC_states=matC_k_states[:, :, :-DHQK, :],
        vecN_states=vecN_k_states[:, :, :-DHQK],
        scaMinter_states=scaMinter_k_states[:, :, :-1],
        vecI=vecI,
        vecB=vecB,
        qk_scale=qk_scale,
        chunk_size=chunk_size,
        num_chunks=NC,
        eps=eps,
    )

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
                scaMinter_k_states[:, :, -1:],
            ),
        )
    else:
        ret_tuple += (None,)

    if return_all_states:
        ret_tuple += ((matC_k_states, vecN_k_states, scaMinter_k_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple  # (matH_out, vecN_out, vecM_out, optional(last_states), optional(all_states))


def mlstm_chunkwise(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH, 1)
    qk_scale: float = None,
    return_last_states: bool = False,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> (
    torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):  # matH_out (B, NH, S, DHHV), optional(last_states (matC_states (B, NH, DHQK, DHHV), vecN_states (B, NH, DHQK), scaMinter_states (B, NH, 1)))
    matH_out, _, _, last_states, _ = mlstm_chunkwise_fw(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecF=vecF,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaM_initial=scaM_initial,
        qk_scale=qk_scale,
        return_last_states=return_last_states,
        return_all_states=False,
        eps=eps,
        chunk_size=chunk_size,
    )
    if return_last_states:
        return matH_out, last_states
    else:
        return matH_out
