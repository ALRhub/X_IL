#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
Triton

mLSTM backward pass. Parallel formulation.

The backward pass is implemented in two kernels in order to avoid the the necessity of synchronisation between two blocks.

The first loop computes the delta errors for the keys, values and the input gates.
The second loop computes the delta errors for the queries.

After these two loops we compute the delta errors for the forget gates.
"""

import triton
import triton.language as tl

from .fw_kernel import configs, keep


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def mlstm_parallel_bw_dQ_kernel(
    matDeltaHtilde,
    matQ,
    matK,
    matV,
    vecI,
    vecF_cs,
    vecM,
    vecN,
    qk_scale,
    matDeltaQ,
    matDeltaK,
    matDeltaV,
    vecDeltaI,
    stride_dhtz,
    stride_dhth,
    stride_dhts,
    stride_dhtd,
    stride_qz,
    stride_qh,
    stride_qs,
    stride_qd,  #
    stride_kz,
    stride_kh,
    stride_ks,
    stride_kd,  #
    stride_vz,
    stride_vh,
    stride_vs,
    stride_vd,  #
    stride_ifmn_z,
    stride_ifmn_h,
    stride_ifmn_s,
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_Q: tl.constexpr,  #
    BLOCK_KV: tl.constexpr,  #
    EPS: tl.constexpr = 1e-6,
):
    ## Notation
    # z: batch size
    # h: number of heads
    # s: sequence length
    # d: head dimension

    tl.static_assert(BLOCK_KV <= HEAD_DIM)
    qIdx = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qkvh_batchhead_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    ifmn_batchhead_offset = off_z.to(tl.int64) * stride_ifmn_z + off_h.to(tl.int64) * stride_ifmn_h

    # input block pointers
    # Note: the order argument specifies the memory traversal order within a block
    matDeltaHtilde_block_ptr = tl.make_block_ptr(
        base=matDeltaHtilde + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dhts, stride_dhtd),
        offsets=(qIdx * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )
    matQ_block_ptr = tl.make_block_ptr(
        base=matQ + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qs, stride_qd),
        offsets=(qIdx * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )
    matK_block_ptr = tl.make_block_ptr(
        base=matK + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_ks, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0),
    )
    # directly transpose matV while loading
    matV_block_ptr = tl.make_block_ptr(
        base=matV + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vd, stride_vs),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_KV),  # this is the transposed shape in SRAM
        order=(
            0,
            1,
        ),  # adapt the order to the underlying layout (which is not transposed), we load HEAD_DIM first
    )

    # output block pointers
    matDeltaQ_block_ptr = tl.make_block_ptr(
        base=matDeltaQ + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dhts, stride_dhtd),
        offsets=(qIdx * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )

    # ? LOADING AND INITIALIZATION
    # define q_block_idxes for causal masking
    q_offset = qIdx * BLOCK_Q
    q_offset = tl.multiple_of(q_offset, BLOCK_Q)
    q_block_idxes = q_offset + tl.arange(0, BLOCK_Q)

    # load matQ_tile & matDeltaHtilde_tile
    matQ_tile = tl.load(matQ_block_ptr)  # (BLOCK_Q, HEAD_DIM)
    matDeltaHtilde_tile = tl.load(matDeltaHtilde_block_ptr)  # (BLOCK_Q, HEAD_DIM)

    # load vecM_chunk_Q, vecN_chunk_Q
    vecMN_offsets = ifmn_batchhead_offset + q_offset + tl.arange(0, BLOCK_Q)
    vecM_chunk_Q_ptr = vecM + vecMN_offsets
    vecN_chunk_Q_ptr = vecN + vecMN_offsets

    vecM_chunk_Q = tl.load(vecM_chunk_Q_ptr)  # (BLOCK_Q,)
    vecN_chunk_Q = tl.load(vecN_chunk_Q_ptr)  # (BLOCK_Q,)

    # load vecF_cs_chunk_Q
    vecF_cs_chunk_Q_ptr = vecF_cs + ifmn_batchhead_offset + q_offset + tl.arange(0, BLOCK_Q)
    vecF_cs_chunk_Q = tl.load(vecF_cs_chunk_Q_ptr)
    vecF_cs_chunk_Q = vecF_cs_chunk_Q.to(tl.float32)

    # init matDeltaQ_tile accumulator
    matDeltaQ_tile = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # ? LOOP1: compute matDeltaK, matDeltaV, vecDeltaI_sum
    kvEndIdx = tl.cdiv((qIdx + 1) * BLOCK_Q, BLOCK_KV)

    # loop over BLOCK_Q dimension and update matDeltK, matDeltaV, vecDeltaI_sum accumulators
    for kvIdx in range(0, kvEndIdx):
        # tl.device_print("kvIdx: %d\n", kvIdx)
        kv_offset = kvIdx * BLOCK_KV
        kv_offset = tl.multiple_of(kv_offset, BLOCK_Q)

        # load vecF_cs_chunk_KV
        vecF_cs_chunk_KV_ptr = vecF_cs + ifmn_batchhead_offset + kv_offset + tl.arange(0, BLOCK_KV)
        vecF_cs_chunk_KV = tl.load(vecF_cs_chunk_KV_ptr)
        vecF_cs_chunk_KV = vecF_cs_chunk_KV.to(tl.float32)

        # load vecI_chunk
        vecI_chunk_KV_ptr = vecI + ifmn_batchhead_offset + kv_offset + tl.arange(0, BLOCK_KV)
        vecI_chunk_KV = tl.load(vecI_chunk_KV_ptr)  # (BLOCK_KV,)

        # load matK_tile & matV_tile
        matK_tile = tl.load(matK_block_ptr)  # (BLOCK_KV, HEAD_DIM)
        matV_tile = tl.load(matV_block_ptr)  # (HEAD_DIM, BLOCK_KV)

        # compute matDeltaC_tile
        # tl.static_print("matDeltaHtilde_tile", matDeltaHtilde_tile)
        # tl.static_print("matV_tile", matV_tile)
        matDeltaC_tile = tl.dot(matDeltaHtilde_tile, matV_tile)  # (BLOCK_Q, BLOCK_KV)
        matDeltaC_tile = matDeltaC_tile / (vecN_chunk_Q[:, None] + EPS)

        # ? recomputation of S & D matrices
        # compute matS_tile
        matK_tile_transposed = tl.trans(matK_tile)  # (HEAD_DIM, BLOCK_KV)
        # tl.static_print("matK_tile_transposed", matK_tile_transposed)
        # tl.static_print("matQ_tile", matQ_tile)
        matS_tile = tl.dot(matQ_tile, matK_tile_transposed)  # (BLOCK_Q, BLOCK_KV)
        matS_tile = matS_tile / qk_scale

        # compute matLogD_tile
        matLogD_Fgates_tile = vecF_cs_chunk_Q[:, None] - vecF_cs_chunk_KV[None, :]
        matLogD_tile = matLogD_Fgates_tile + vecI_chunk_KV[None, :]

        # causal masking
        if kv_offset >= q_offset:
            # we are on diagonal
            kv_block_idxes = kv_offset + tl.arange(0, BLOCK_KV)
            mask = q_block_idxes[:, None] - kv_block_idxes[None, :]
            # we set all values above the main diagonal to -inf
            matLogD_tile = tl.where(mask >= 0, matLogD_tile, -float("inf"))

        # else: below main diagonal

        matDprime_tile = tl.exp(matLogD_tile - vecM_chunk_Q[:, None])  # (BLOCK_Q, BLOCK_KV)
        # ? end recomputation of S & D matrices

        matP_tile = matDeltaC_tile * matDprime_tile  # (BLOCK_Q, BLOCK_KV)

        # update matDeltaQ_tile in SRAM
        matP_tile = matP_tile.to(matK_tile.type.element_ty)
        # tl.static_print("matP_tile", matP_tile)
        # tl.static_print("matK_tile", matK_tile)
        matDeltaQ_tile_iter = tl.dot(matP_tile, matK_tile)  # (BLOCK_Q, HEAD_DIM)
        matDeltaQ_tile_iter = matDeltaQ_tile_iter / qk_scale
        matDeltaQ_tile += matDeltaQ_tile_iter

        # advance pointers matK, matV
        matK_block_ptr = tl.advance(matK_block_ptr, (BLOCK_KV, 0))
        matV_block_ptr = tl.advance(matV_block_ptr, (0, BLOCK_KV))
        # ? END LOOP1

    # epilogue
    tl.store(matDeltaQ_block_ptr, matDeltaQ_tile.to(matDeltaQ.type.element_ty))


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def mlstm_parallel_bw_dKdV_kernel(
    matDeltaHtilde,
    matQ,
    matK,
    matV,
    vecI,
    vecF_cs,
    vecM,
    vecN,
    qk_scale,
    matDeltaQ,
    matDeltaK,
    matDeltaV,
    vecDeltaI,
    stride_dhtz,
    stride_dhth,
    stride_dhts,
    stride_dhtd,
    stride_qz,
    stride_qh,
    stride_qs,
    stride_qd,  #
    stride_kz,
    stride_kh,
    stride_ks,
    stride_kd,  #
    stride_vz,
    stride_vh,
    stride_vs,
    stride_vd,  #
    stride_ifmn_z,
    stride_ifmn_h,
    stride_ifmn_s,
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_Q: tl.constexpr,  #
    BLOCK_KV: tl.constexpr,  #
    EPS: tl.constexpr = 1e-6,
):
    ## Notation
    # z: batch size
    # h: number of heads
    # s: sequence length
    # d: head dimension

    tl.static_assert(BLOCK_KV <= HEAD_DIM)
    kvIdx = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qkvh_batchhead_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    ifmn_batchhead_offset = off_z.to(tl.int64) * stride_ifmn_z + off_h.to(tl.int64) * stride_ifmn_h

    # input block pointers
    # Note: the order argument specifies the memory traversal order within a block
    matDeltaHtilde_block_ptr = tl.make_block_ptr(
        base=matDeltaHtilde + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dhts, stride_dhtd),
        offsets=(0, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )
    matQ_block_ptr = tl.make_block_ptr(
        base=matQ + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qs, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )
    matK_block_ptr = tl.make_block_ptr(
        base=matK + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_ks, stride_kd),
        offsets=(kvIdx * BLOCK_KV, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0),
    )
    # directly transpose matV while loading
    matV_block_ptr = tl.make_block_ptr(
        base=matV + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vd, stride_vs),
        offsets=(0, kvIdx * BLOCK_KV),
        block_shape=(HEAD_DIM, BLOCK_KV),  # this is the transposed shape in SRAM
        order=(
            0,
            1,
        ),  # adapt the order to the underlying layout (which is not transposed), we load HEAD_DIM first
    )

    # output block pointers
    matDeltaK_block_ptr = tl.make_block_ptr(
        base=matDeltaK + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dhts, stride_dhtd),
        offsets=(kvIdx * BLOCK_KV, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0),
    )
    matDeltaV_block_ptr = tl.make_block_ptr(
        base=matDeltaV + qkvh_batchhead_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dhts, stride_dhtd),
        offsets=(kvIdx * BLOCK_KV, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0),
    )

    # ? LOADING AND INITIALIZATION
    # define kv_block_idxes for causal masking
    kv_offset = kvIdx * BLOCK_KV
    kv_offset = tl.multiple_of(kv_offset, BLOCK_KV)
    kv_block_idxes = kv_offset + tl.arange(0, BLOCK_KV)

    # load matK_tile, matV_tile
    matK_tile = tl.load(matK_block_ptr)  # (BLOCK_KV, HEAD_DIM)
    matV_tile = tl.load(matV_block_ptr)  # (HEAD_DIM, BLOCK_KV)
    # init matDeltaK_tile, matDeltaV_tile accumulators
    matDeltaK_tile = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    matDeltaV_tile = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # load vecI_chunk
    vecI_chunk_KV_ptr = vecI + ifmn_batchhead_offset + kv_offset + tl.arange(0, BLOCK_KV)
    vecI_chunk_KV = tl.load(vecI_chunk_KV_ptr)  # (BLOCK_KV,)

    # init vecDeltaI_sum accumulator
    vecDeltaI_sum_chunk_KV = tl.zeros([BLOCK_KV], dtype=tl.float32)

    # load vecF_cs_chunk_KV
    vecF_cs_chunk_KV_ptr = vecF_cs + ifmn_batchhead_offset + kv_offset + tl.arange(0, BLOCK_KV)
    vecF_cs_chunk_KV = tl.load(vecF_cs_chunk_KV_ptr)
    vecF_cs_chunk_KV = vecF_cs_chunk_KV.to(tl.float32)

    # ? LOOP1: compute matDeltaK, matDeltaV, vecDeltaI_sum
    qStartIdx = (kvIdx * BLOCK_KV) // BLOCK_Q
    qEndIdx = tl.cdiv(N_CTX, BLOCK_Q)
    qEndIdx = tl.multiple_of(qEndIdx, BLOCK_Q)

    # move mat(Delta)Q_block_ptr & matDeltaHtilde_block_ptr to the position for the current thread block
    # input pointers:
    matQ_block_ptr = tl.advance(matQ_block_ptr, (qStartIdx * BLOCK_Q, 0))
    matDeltaHtilde_block_ptr = tl.advance(matDeltaHtilde_block_ptr, (qStartIdx * BLOCK_Q, 0))
    # output pointers:
    # matDeltaQ_block_ptr = tl.advance(matDeltaQ_block_ptr, (qStartIdx * BLOCK_Q, 0))

    # loop over BLOCK_Q dimension and update matDeltK, matDeltaV, vecDeltaI_sum accumulators
    for qIdx in range(qStartIdx, qEndIdx):
        q_offset = qIdx * BLOCK_Q
        q_offset = tl.multiple_of(q_offset, BLOCK_Q)

        # load matQ_tile & matDeltaHtilde_tile
        matQ_tile = tl.load(matQ_block_ptr)  # (BLOCK_Q, HEAD_DIM)
        matDeltaHtilde_tile = tl.load(matDeltaHtilde_block_ptr)  # (BLOCK_Q, HEAD_DIM)

        # load vecM_chunk_Q, vecN_chunk_Q
        vecMN_offsets = ifmn_batchhead_offset + q_offset + tl.arange(0, BLOCK_Q)
        vecM_chunk_Q_ptr = vecM + vecMN_offsets
        vecN_chunk_Q_ptr = vecN + vecMN_offsets

        vecM_chunk_Q = tl.load(vecM_chunk_Q_ptr)  # (BLOCK_Q,)
        vecN_chunk_Q = tl.load(vecN_chunk_Q_ptr)  # (BLOCK_Q,)

        # load vecF_cs_chunk_Q
        vecF_cs_chunk_Q_ptr = vecF_cs + ifmn_batchhead_offset + q_offset + tl.arange(0, BLOCK_Q)
        vecF_cs_chunk_Q = tl.load(vecF_cs_chunk_Q_ptr)
        vecF_cs_chunk_Q = vecF_cs_chunk_Q.to(tl.float32)

        # compute matDeltaC_tile
        # tl.static_print("matDeltaHtilde_tile", matDeltaHtilde_tile)
        # tl.static_print("matV_tile", matV_tile)
        matDeltaC_tile = tl.dot(matDeltaHtilde_tile, matV_tile)  # (BLOCK_Q, BLOCK_KV)
        matDeltaC_tile = matDeltaC_tile / (vecN_chunk_Q[:, None] + EPS)

        # ? recomputation of S & D matrices
        # compute matS_tile
        matK_tile_transposed = tl.trans(matK_tile)  # (HEAD_DIM, BLOCK_KV)
        # tl.static_print("matK_tile_transposed", matK_tile_transposed)
        # tl.static_print("matQ_tile", matQ_tile)
        matS_tile = tl.dot(matQ_tile, matK_tile_transposed)  # (BLOCK_Q, BLOCK_KV)
        matS_tile = matS_tile / qk_scale

        # compute matLogD_tile
        matLogD_Fgates_tile = vecF_cs_chunk_Q[:, None] - vecF_cs_chunk_KV[None, :]
        matLogD_tile = matLogD_Fgates_tile + vecI_chunk_KV[None, :]

        # causal masking
        if kv_offset >= q_offset:
            # we are on diagonal
            q_block_idxes = q_offset + tl.arange(0, BLOCK_Q)
            mask = q_block_idxes[:, None] - kv_block_idxes[None, :]
            # we set all values above the main diagonal to -inf
            matLogD_tile = tl.where(mask >= 0, matLogD_tile, -float("inf"))

        # else: below main diagonal

        matDprime_tile = tl.exp(matLogD_tile - vecM_chunk_Q[:, None])  # (BLOCK_Q, BLOCK_KV)
        # ? end recomputation of S & D matrices

        matDeltaCTilde_tile = matDeltaC_tile * matS_tile * matDprime_tile

        # compute sum for vecDeltaI
        # sum up the columns of matDeltaCTilde_tile
        vecDeltaI_sum_chunk_KV += tl.sum(matDeltaCTilde_tile, axis=0)  # (BLOCK_KV,)

        matP_tile = matDeltaC_tile * matDprime_tile  # (BLOCK_Q, BLOCK_KV)
        matR_tile = matS_tile * matDprime_tile  # (BLOCK_Q, BLOCK_KV)
        matR_tile = matR_tile.to(matQ_tile.type.element_ty)

        # update matDeltaQ_tile in HBM
        matP_tile = matP_tile.to(matQ_tile.type.element_ty)

        # update matDeltaK_tile, matDeltaV_tile in SRAM
        matP_tile_transposed = tl.trans(matP_tile)  # (BLOCK_KV, BLOCK_Q)
        # tl.static_print("matP_tile_transposed", matP_tile_transposed)
        # tl.static_print("matQ_tile", matQ_tile)
        matDeltaK_tile_temp = tl.dot(matP_tile_transposed, matQ_tile)  # (BLOCK_KV, HEAD_DIM)
        matDeltaK_tile += matDeltaK_tile_temp / qk_scale

        matR_tile_transposed = tl.trans(matR_tile)  # (BLOCK_KV, BLOCK_Q)
        matDeltaHtilde_tile_normalized = matDeltaHtilde_tile / (vecN_chunk_Q[:, None] + EPS)  # (BLOCK_Q, HEAD_DIM)
        matDeltaHtilde_tile_normalized = matDeltaHtilde_tile_normalized.to(matQ_tile.type.element_ty)
        # tl.static_print("matR_tile_transposed", matR_tile_transposed)
        # tl.static_print(
        #     "matDeltaHtilde_tile_normalized", matDeltaHtilde_tile_normalized
        # )
        matDeltaV_tile += tl.dot(matR_tile_transposed, matDeltaHtilde_tile_normalized)  # (BLOCK_KV, HEAD_DIM)

        # advance pointers (delta_)Q + deltaHtilde
        matQ_block_ptr = tl.advance(matQ_block_ptr, (BLOCK_Q, 0))
        matDeltaHtilde_block_ptr = tl.advance(matDeltaHtilde_block_ptr, (BLOCK_Q, 0))
        # ? END LOOP1

    # epilogue
    # store matDeltaK_tile, matDeltaV_tile
    tl.store(matDeltaK_block_ptr, matDeltaK_tile.to(matDeltaK.type.element_ty))
    tl.store(matDeltaV_block_ptr, matDeltaV_tile.to(matDeltaV.type.element_ty))
    # store vecDeltaI_sum
    vecDeltaI_chunk_KV_ptr = vecDeltaI + ifmn_batchhead_offset + kv_offset + tl.arange(0, BLOCK_KV)
    tl.store(vecDeltaI_chunk_KV_ptr, vecDeltaI_sum_chunk_KV.to(vecDeltaI.type.element_ty))
