#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import math

import torch
import torch.nn.functional as F

"""
PyTorch

mLSTM forward and backward pass. Parallel formulation.
Implemented in a tiled fashion as template for kernels.

Experimental code. Not meant for real usage.
"""


def construct_log_gate_matrix_paper(fgs: torch.Tensor, igs: torch.Tensor) -> torch.Tensor:
    _device = fgs.device
    _dtype = fgs.dtype
    B, NH, S, _ = fgs.shape
    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )
    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(fgs, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igs.transpose(-2, -1)  # (B, NH, S, S)
    return log_D_matrix


def construct_log_gate_matrix_tiled(
    fgs: torch.Tensor,
    igs: torch.Tensor,
    BQ: int,
    BKV: int,
    idx_BQ: int,
    idx_BKV: int,
    fgs_cs: torch.Tensor = None,
) -> torch.Tensor:
    B, NH, S = fgs.shape
    if fgs_cs is None:
        fgs_cs = torch.cumsum(fgs, dim=-1)
    fgs_cs_chunk_Q = fgs_cs[:, :, idx_BQ * BQ : (idx_BQ + 1) * BQ]
    fgs_cs_chunk_KV = fgs_cs[:, :, idx_BKV * BKV : (idx_BKV + 1) * BKV]

    fgate_tile = fgs_cs_chunk_Q[:, :, :, None] - fgs_cs_chunk_KV[:, :, None, :]

    igs_chunk = igs[:, :, idx_BKV * BKV : (idx_BKV + 1) * BKV]
    log_D_matrix = fgate_tile + igs_chunk

    # causal masking
    if idx_BKV * BKV >= idx_BQ * BQ:
        bq_idxes = torch.arange(idx_BQ * BQ, (idx_BQ + 1) * BQ)
        kv_idxes = torch.arange(idx_BKV * BKV, (idx_BKV + 1) * BKV)
        idx_mask = bq_idxes[:, None] - kv_idxes[None, :]  # or bq_idxes[:, None] >= kv_idxes[None, :]
        log_D_matrix = torch.where(idx_mask < 0, -float("inf"), log_D_matrix)
    return log_D_matrix


def _mlstm_fw(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    bq_tile_size: int = -1,
    bkv_tile_size: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """This is the core mLSTM operation in parallel form computed in tiles.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, 1)
        fgate_preact (torch.Tensor): (B, NH, S, 1)
        bq_tile_size (int, optional): Tile size along sequence dim for queries. Defaults to -1.
                                      If -1, no tiling is performed.
        bkv_tile_size (int, optional): Tile size along sequence dim for keys and values. Defaults to -1.
                                        If -1, no tiling is performed.

    Returns:
        torch.Tensor: (B, NH, S, DH), retrieved values

    # TODO adapt notation
    # TODO do not precompute the gate matrix. Use the tiled versions from above. (See also backward pass).
    """
    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device
    if bq_tile_size == -1:
        bq_tile_size = S
    else:
        assert S % bq_tile_size == 0, "S must be divisible by bq_tile_size"
    if bkv_tile_size == -1:
        bkv_tile_size = S
    else:
        assert S % bkv_tile_size == 0, "S must be divisible by bkv_tile_size"

    #! We compute the gate matrix D in non tiled way:
    # forget gate matrix
    log_fgates = F.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )
    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)

    #! From here begin tiling:
    q_tiles = torch.split(queries, bq_tile_size, dim=2)
    k_tiles = torch.split(keys, bkv_tile_size, dim=2)
    v_tiles = torch.split(values, bkv_tile_size, dim=2)
    print(f"q_tiles: {len(q_tiles)}, {q_tiles[0].shape}")
    print(f"kv_tiles: {len(k_tiles)}, {k_tiles[0].shape}")

    # we do not break causality since the log_fg_matrix is already causal

    h_matrix = torch.zeros_like(queries)  # the output matrix
    for q_idx, q_tile in enumerate(q_tiles):
        m_prev = torch.zeros((B, NH, bq_tile_size, 1), dtype=_dtype, device=_device)
        l_prev = torch.zeros((B, NH, bq_tile_size, 1), dtype=_dtype, device=_device)
        n_prev = torch.zeros((B, NH, bq_tile_size, 1), dtype=_dtype, device=_device)
        h_tile = torch.zeros_like(q_tile)
        for kv_idx, (k_tile, v_tile) in enumerate(zip(k_tiles, v_tiles)):
            # print(f"q_idx: {q_idx*bq_tile_size}, kv_idx: {kv_idx*bkv_tile_size}")
            d_tile = log_D_matrix[
                :,
                :,
                q_idx * bq_tile_size : (q_idx + 1) * bq_tile_size,
                kv_idx * bkv_tile_size : (kv_idx + 1) * bkv_tile_size,
            ]
            s_tile = q_tile @ (k_tile.transpose(-2, -1) / math.sqrt(DH))

            #! this bounding of m_temp by -10 stabilizes the forward pass
            m_temp = torch.maximum(
                torch.tensor([[[-10.0]]], dtype=_dtype, device=_device),
                torch.max(d_tile, dim=-1, keepdim=True)[0],
            )

            m = torch.maximum(m_prev, m_temp)
            l = torch.exp(m_prev - m) * l_prev + (s_tile * torch.exp(d_tile - m)).sum(dim=-1, keepdim=True)

            n = torch.maximum(torch.abs(l), torch.exp(-m))
            c_tile = (s_tile * torch.exp(d_tile - m)) / (n + eps)

            h_tile = torch.exp(m_prev - m) * (n_prev / n) * h_tile + c_tile @ v_tile

            if q_idx == 10:
                print(
                    f"q_idx: {q_idx}, kv_idx: {kv_idx}, m_prev: {m_prev}, m: {m}, l_prev: {l_prev}, l: {l}, n_prev: {n_prev}, n: {n}"
                )
            m_prev = m
            l_prev = l
            n_prev = n
        h_matrix[:, :, q_idx * bq_tile_size : (q_idx + 1) * bq_tile_size, :] = h_tile

    return h_matrix, m, n


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _mlstm_bw(
    matDeltaHtilde: torch.Tensor,
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    vecM: torch.Tensor,
    vecN: torch.Tensor,
    BLOCK_Q: int = 32,
    BLOCK_KV: int = 32,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, ...]:
    B, NH, S, DH = matQ.shape
    _dtype, _device = matQ.dtype, matQ.device
    assert BLOCK_KV <= BLOCK_Q

    assert vecF.shape == (B, NH, S)
    assert vecI.shape == (B, NH, S)

    ## ? preprocessing
    # precompute gate cumsums
    vecF_act = F.logsigmoid(vecF)
    vecF_cs = torch.cumsum(vecF_act, dim=-1)
    ## ? end preprocessing

    ## ? setup
    # ? tile the input tensors
    # we keep the batch and num_head dimensions
    # in a kernel we would embarrassingly parallelize over these dimensions

    # split along BLOCK_Q dimension:
    matQ_tiles = torch.split(matQ, BLOCK_Q, dim=2)
    matDeltaHtilde_tiles = torch.split(matDeltaHtilde, BLOCK_Q, dim=2)
    vecM_chunks = torch.split(vecM, BLOCK_Q, dim=2)
    vecN_chunks = torch.split(vecN, BLOCK_Q, dim=2)
    vecF_cs_chunks = torch.split(vecF_cs, BLOCK_Q, dim=2)

    # split along BLOCK_KV dimension:
    matK_tiles = torch.split(matK, BLOCK_KV, dim=2)
    matV_tiles = torch.split(matV, BLOCK_KV, dim=2)
    vecI_chunks = torch.split(vecI, BLOCK_KV, dim=2)

    # ? define the kernel output tensors
    matDeltaQ = torch.zeros_like(matQ)
    matDeltaK = torch.zeros_like(matK)
    matDeltaV = torch.zeros_like(matV)
    vecDeltaI = torch.zeros_like(vecI)
    vecDeltaF = torch.zeros_like(vecF)

    print(
        f"matQ_tiles: {len(matQ_tiles)}, {matQ_tiles[0].shape} | matK_tiles: {len(matK_tiles)}, {matK_tiles[0].shape}"
    )
    ## ? end setup

    ## ? begin the backward pass kernel
    #! KV dim loop
    # we will parallelize over this loop later
    # we start at the leftmost block of the KV dimension and work our way right
    for kvIdx, (matK_tile, matV_tile, vecI_chunk) in enumerate(zip(matK_tiles, matV_tiles, vecI_chunks)):
        # init matDeltaK_tile, matDeltaV_tile to zero
        matDeltaK_tile = torch.zeros_like(matK_tile)
        matDeltaV_tile = torch.zeros_like(matV_tile)

        # init vecDeltaF_cs_chunk_KV, vecDeltaI_chunk_KV
        vecDeltaI_sum_chunk_KV = torch.zeros_like(vecI_chunk)

        vecF_cs_chunk_KV = vecF_cs[:, :, kvIdx * BLOCK_KV : (kvIdx + 1) * BLOCK_KV]

        #! Q dim loop
        # we start at the diagonal of the S & D matrices and work our way down
        qStartIdx = (kvIdx * BLOCK_KV) // BLOCK_Q
        qEndIdx = ceildiv(S, BLOCK_Q)
        for qIdx in range(qStartIdx, qEndIdx):
            matQ_tile = matQ_tiles[qIdx]
            matDeltaHtilde_tile = matDeltaHtilde_tiles[qIdx]
            vecM_chunk = vecM_chunks[qIdx]
            vecN_chunk = vecN_chunks[qIdx]
            vecF_cs_chunk_Q = vecF_cs_chunks[qIdx]

            matDeltaC = matDeltaHtilde_tile @ matV_tile.transpose(-2, -1) / (vecN_chunk + eps)

            # ? recomputation of S & D matrices
            matS = (matQ_tile @ matK_tile.transpose(-2, -1)) / math.sqrt(DH)

            # construct D matrix
            vecF_cs_tile = vecF_cs_chunk_Q[:, :, :, None] - vecF_cs_chunk_KV[:, :, None, :]
            matLogD_tile = vecF_cs_tile + vecI_chunk

            # causal masking of matLogD_tile
            if kvIdx * BLOCK_KV >= qIdx * BLOCK_Q:
                bq_idxes = torch.arange(qIdx * BLOCK_Q, (qIdx + 1) * BLOCK_Q, device=vecI.device)
                kv_idxes = torch.arange(kvIdx * BLOCK_KV, (kvIdx + 1) * BLOCK_KV, device=vecI.device)
                idx_mask = bq_idxes[:, None] - kv_idxes[None, :]
                matLogD_tile = torch.where(idx_mask < 0, -float("inf"), matLogD_tile)

            matDprime = torch.exp(matLogD_tile - vecM_chunk)
            # ? end recomputation of S & D matrices

            matDeltaCtilde = matDeltaC * matS * matDprime

            # ? compute sum for vecDeltaI
            vecDeltaI_sum_chunk_KV += matDeltaCtilde.sum(dim=-2)

            # matDeltaCtilde_cumsum = matDeltaCtilde.cumsum(-1)
            # # causal masking of matDeltaCtilde_cumsum
            # if kvIdx * BLOCK_KV >= qIdx * BLOCK_Q:
            #     bq_idxes = torch.arange(
            #         qIdx * BLOCK_Q, (qIdx + 1) * BLOCK_Q, device=vecI.device
            #     )
            #     kv_idxes = (
            #         torch.arange(
            #             kvIdx * BLOCK_KV, (kvIdx + 1) * BLOCK_KV, device=vecI.device
            #         )
            #         + 1.0  #! we need to add 1 here to get the correct mask (e.g. .tril(-1))
            #     )
            #     idx_mask = bq_idxes[:, None] - kv_idxes[None, :]

            #     matDeltaCtilde_cumsum = torch.where(
            #         idx_mask < 0, 0.0, matDeltaCtilde_cumsum
            #     )

            matP = matDeltaC * matDprime
            matR = matS * matDprime

            matDeltaQ_tile = matP @ (matK_tile / math.sqrt(DH))
            # * store matDeltaQ in HBM (this access is in parallel at the same HBM location, e.g. must be atomic)
            matDeltaQ[:, :, qIdx * BLOCK_Q : (qIdx + 1) * BLOCK_Q] += matDeltaQ_tile

            matDeltaK_tile += (matP.transpose(-2, -1) @ matQ_tile) / math.sqrt(DH)

            matDeltaV_tile += matR.transpose(-2, -1) @ (matDeltaHtilde_tile / (vecN_chunk + eps))
            #! end Q dim loop

        # * store matDeltaK_tile & matDeltaV_tile in HBM (every thread block writes to a different HBM location)
        matDeltaK[:, :, kvIdx * BLOCK_KV : (kvIdx + 1) * BLOCK_KV] = matDeltaK_tile
        matDeltaV[:, :, kvIdx * BLOCK_KV : (kvIdx + 1) * BLOCK_KV] = matDeltaV_tile

        # * store vecDeltaIF_sum_chunk_KV in HBM (every thread block writes to a different HBM location)
        vecDeltaI[:, :, kvIdx * BLOCK_KV : (kvIdx + 1) * BLOCK_KV] = vecDeltaI_sum_chunk_KV
        #! end KV dim loop

    ## ? end the backward pass kernel

    ## ? postprocessing
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    vecDeltaFbar_acc = (matQ * matDeltaQ - matK * matDeltaK).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)
    ## ? end postprocessing

    return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF


def mlstm_fwbw(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    BLOCK_Q: int = 32,
    BLOCK_KV: int = 16,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matH, _, _ = _mlstm_fwbw.apply(matQ, matK, matV, vecI, vecF, BLOCK_Q, BLOCK_KV, eps)
    return matH


class _mlstm_fwbw(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        matQ: torch.Tensor,
        matK: torch.Tensor,
        matV: torch.Tensor,
        vecI: torch.Tensor,
        vecF: torch.Tensor,
        BLOCK_Q: int = 32,
        BLOCK_KV: int = 16,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matH, vecM, vecN = _mlstm_fw(
            matQ,
            matK,
            matV,
            vecI,
            vecF,
            eps=eps,
        )
        ctx.save_for_backward(matQ, matK, matV, vecI, vecF, vecM, vecN, BLOCK_Q, BLOCK_KV, eps)
        return matH, vecM, vecN

    @staticmethod
    def backward(
        ctx,
        matDeltaHtilde: torch.Tensor,
        vecDeltaM_unused: torch.Tensor,
        vecDeltaN_unused: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        (matQ, matK, matV, vecI, vecF, vecM, vecN, BLOCK_Q, BLOCK_KV, eps) = ctx.saved_tensors
        matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF = _mlstm_bw(
            matDeltaHtilde=matDeltaHtilde,
            matQ=matQ,
            matK=matK,
            matV=matV,
            vecI=vecI,
            vecF=vecF,
            vecM=vecM,
            vecN=vecN,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
            eps=eps,
        )
        return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF, None, None, None
