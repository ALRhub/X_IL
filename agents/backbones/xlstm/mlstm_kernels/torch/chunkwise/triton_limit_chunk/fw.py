#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
from torch.nn.functional import logsigmoid

from ...utils import contiguous_noctx
from .fw_parallel import mlstm_chunkwise__parallel_fw_H
from .fw_recurrent import mlstm_chunkwise__recurrent_fw_C


@contiguous_noctx
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
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
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
    DHHV = matV.shape[-1]
    assert (
        S % CHUNK_SIZE == 0
    ), f"Sequence length {S} is not divisible by chunk size {CHUNK_SIZE}."
    NC = S // CHUNK_SIZE

    # vecI = rearrange(vecI, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE)
    # vecF = rearrange(vecF, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE).to(torch.float32)
    vecI = vecI.reshape(B, NH, NC, CHUNK_SIZE)
    vecF = vecF.reshape(B, NH, NC, CHUNK_SIZE).to(torch.float32)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = logsigmoid(vecF)
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
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
    )
    # print("matC_k_states - fw_C", matC_k_states.shape, matC_k_states.dtype)

    #! compute the outputs within each chunk
    # we pass NC+1 states into this kernel but load only the first NC states
    # the NC+1th state is the next state after the last chunk
    matH_out, vecN_out, vecM_out = mlstm_chunkwise__parallel_fw_H(
        matQ=matQ,
        matK=matK,
        matV=matV,
        matC_states=matC_k_states,  # (B, NH, (NC+1) * DHQK, DHHV)
        vecN_states=vecN_k_states,  # (B, NH, (NC+1) * DHQK)
        scaMinter_states=scaMinter_k_states,  # (B, NH, (NC+1))
        vecI=vecI,
        vecB=vecB,
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )

    ret_tuple = (
        matH_out,
        vecN_out,
        vecM_out,
    )
    if return_last_states:
        # Note: we need to make the states contiguous here, because the last states are not contiguous
        # if we return a slice of the larger tensor.
        # For generation afterwards we will use these state tensors and update them in place.
        # For this in place operation the tensor needs to be contiguous.
        # In this case the contigous should result in a copy operation.
        ret_tuple += (
            (
                matC_k_states[:, :, -DHQK:, :].contiguous(),
                vecN_k_states[:, :, -DHQK:].contiguous(),
                scaMinter_k_states[:, :, -1:].contiguous(),
            ),
        )
    else:
        ret_tuple += (None,)

    if return_all_states:
        ret_tuple += ((matC_k_states, vecN_k_states, scaMinter_k_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple  # (matH_out, vecN_out, vecM_out, optional(last_states), optional(all_states))
