#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
from torch.nn.functional import logsigmoid

from ...utils import contiguous_noctx
from .bw_parallel import mlstm_chunkwise__parallel_bw_dQKV
from .bw_recurrent import mlstm_chunkwise__recurrent_bw_dC
from .fw_recurrent import mlstm_chunkwise__recurrent_fw_C
from .chunkwise_gates import compute_gate_grads_vecDeltaI_vecDeltaF


@contiguous_noctx
def mlstm_chunkwise_bw(
    ## Forward arguments
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH, 1)
    qk_scale: float = None,
    ## Backward arguments
    matC_all: torch.Tensor = None,  # (B, NH, NC * DHQK, DHV)
    vecN_all: torch.Tensor = None,  # (B, NH, NC * DHQK)
    scaM_all: torch.Tensor = None,  # (B, NH, NC)
    vecN_out: torch.Tensor = None,  # (B, NH, NC * L) = (B, NH, S)
    vecM_out: torch.Tensor = None,  # (B, NH, NC * L) = (B, NH, S)
    matDeltaH: torch.Tensor = None,  # (B, NH, S, DHV)
    matDeltaC_last: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecDeltaN_last: torch.Tensor = None,  # (B, NH, DHQK) # TODO not used, maybe leave out
    scaDeltaM_last: torch.Tensor = None,  # (B, NH) # TODO not used, maybe leave out
    ## Common arguments
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
):
    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]

    assert S % CHUNK_SIZE == 0, f"Sequence length {S} is not divisible by chunk size {CHUNK_SIZE}."

    NC = S // CHUNK_SIZE

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    # vecI = rearrange(vecI, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE)
    # vecF = rearrange(vecF, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE).to(torch.float32)
    vecI = vecI.reshape(B, NH, NC, CHUNK_SIZE)
    vecF_reshaped = vecF.reshape(B, NH, NC, CHUNK_SIZE).to(torch.float32)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = logsigmoid(vecF_reshaped)
    vecB = vecF_logsig.cumsum(-1)

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

    matC_k_states = matC_all  # [:, :, :-DHQK, :]  # take the first NC states
    matDeltaC_k_states = matDeltaC_states  # [:, :, DHQK:, :]  # take the last NC states

    matDeltaQ, matDeltaK, matDeltaV = mlstm_chunkwise__parallel_bw_dQKV(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        vecM_combine=vecM_out,
        scaM_inter=scaM_all,  # (B, NH, NC)
        matC_states=matC_k_states,  # (B, NH, (NC+1) * DHQK, DHV) # we only need the first NC states
        matDeltaH=matDeltaH,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_states=matDeltaC_k_states,  # (B, NH, (NC+1) * DHQK, DHV)
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )

    vecDeltaI, vecDeltaF = compute_gate_grads_vecDeltaI_vecDeltaF(
        matQ=matQ, matK=matK, matDeltaQ=matDeltaQ, matDeltaK=matDeltaK, vecF=vecF
    )

    matDeltaC_initial = matDeltaC_states[:, :, :DHQK, :] if matC_initial is not None else None
    vecDeltaN_initial = torch.zeros_like(vecN_initial) if vecN_initial is not None else None
    scaDeltaM_initial = torch.zeros_like(scaM_initial) if scaM_initial is not None else None

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
