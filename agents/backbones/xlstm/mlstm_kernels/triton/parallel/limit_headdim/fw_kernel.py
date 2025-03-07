#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
Triton

mLSTM forward pass. Parallel formulation.

Similar work partitioning as Flash-Attention2.
The head dimension is limited to 256 and it only supports the
same head dimension for Q, K, and V.

# TODO adapt to notation of backward pass.
"""

import triton
import triton.language as tl

ENABLE_AUTOTUNING = True

if ENABLE_AUTOTUNING:
    configs = [
        triton.Config({"BLOCK_Q": BQ, "BLOCK_KV": BKV}, num_stages=s, num_warps=w)
        for BQ, BKV in [
            # (128, 128),
            # (128, 64),
            # (128, 32),
            # (128, 16),
            (64, 64),
            # (64, 32),
            # (64, 16),
            (32, 32),
            # (32, 16),
            # (16, 16),
        ]
        for s in [1, 4]
        for w in [2, 4]
    ]
else:
    configs = [
        triton.Config({"BLOCK_Q": BQ, "BLOCK_KV": BKV}, num_stages=s, num_warps=w)
        for BQ, BKV in [
            # (128, 128),
            # (128, 64),
            # (128, 32),
            # (128, 16),
            # (64, 64),
            # (64, 32),
            # (64, 16),
            # (32, 32),
            # (32, 16),
            (16, 16),
        ]
        for s in [1]
        for w in [2]
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_Q"]
    BLOCK_N = conf.kwargs["BLOCK_KV"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# BLOCK_Q = 16
# BLOCK_KV = 16

MINIMUM_MAX_VAL = -10  # -float("inf")  # -10.0


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def mlstm_parallel_fw_kernel(
    matQ,
    matK,
    matV,
    vecI,
    vecF_cs,
    qk_scale,
    matH,  #
    vecM,
    vecN,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_hz,
    stride_hh,
    stride_hm,
    stride_hn,  #
    stride_ifmn_z,
    stride_ifmn_h,
    stride_ifmn_m,
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_Q: tl.constexpr,  #
    BLOCK_KV: tl.constexpr,  #
    MINIMUM_MAX_VAL: tl.constexpr,
    EPS: tl.constexpr = 1e-6,
):
    tl.static_assert(BLOCK_KV <= HEAD_DIM)
    start_m_idx = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    ifmn_offset = off_z.to(tl.int64) * stride_ifmn_z + off_h.to(tl.int64) * stride_ifmn_h

    # block pointers
    # Note on order argument:
    # For the K_block_ptr, the order=(0, 1) indicates that within each block, the kernel will access elements by
    # first traversing along the HEAD_DIM dimension, and then along the BLOCK_KV dimension.
    # This row-major access pattern aligns with typical matrix operations and can be beneficial
    # for memory access efficiency, especially if HEAD_DIM elements are stored contiguously in memory.
    # The order must match the underlying memory layout of the tensor.
    Q_block_ptr = tl.make_block_ptr(
        base=matQ + qkv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m_idx * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=matV + qkv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=matK + qkv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_KV),
        order=(0, 1),
    )
    H_block_ptr = tl.make_block_ptr(
        base=matH + qkv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_hm, stride_hn),
        offsets=(start_m_idx * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )

    # ? LOADING AND INITIALIZATION
    # initialize accumulator
    h_out = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)

    # define q_block_idxes for causal masking
    q_offset = start_m_idx * BLOCK_Q
    q_block_idxes = q_offset + tl.arange(0, BLOCK_Q)

    # load vecF_cs: ifmn_offset defines the proper batch-head, q_offset defines the location
    # in the sequence for the current thread block
    vecF_cs_chunkQ_ptr = vecF_cs + ifmn_offset + q_offset + tl.arange(0, BLOCK_Q)
    vecF_cs_chunkQ = tl.load(vecF_cs_chunkQ_ptr)
    vecF_cs_chunkQ = vecF_cs_chunkQ.to(tl.float32)

    # init l, m, n vectors
    m_new = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    m_old = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    n_new = tl.zeros([BLOCK_Q], dtype=tl.float32)
    n_old = tl.zeros([BLOCK_Q], dtype=tl.float32)
    l_new = tl.zeros([BLOCK_Q], dtype=tl.float32)
    l_old = tl.zeros([BLOCK_Q], dtype=tl.float32)

    # ? MAIN LOOP
    lo = 0
    hi = (start_m_idx + 1) * BLOCK_Q
    hi = tl.multiple_of(hi, BLOCK_Q)

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_KV):
        start_n = tl.multiple_of(start_n, BLOCK_KV)
        # -- compute matS --
        k = tl.load(K_block_ptr)
        matS = tl.dot(q, k)
        matS = matS / qk_scale

        # ? -- create gate matrix tile D --
        # load vecF_cs_chunkKV
        vecF_cs_vecI_chunkKV_offset = ifmn_offset + start_n + tl.arange(0, BLOCK_KV)
        vecF_cs_chunkKV_ptr = vecF_cs + vecF_cs_vecI_chunkKV_offset
        vecF_cs_chunkKV = tl.load(vecF_cs_chunkKV_ptr)
        vecF_cs_chunkKV = vecF_cs_chunkKV.to(tl.float32)

        # load vecI_chunkKV
        vecI_ptr = vecI + vecF_cs_vecI_chunkKV_offset
        vecI_chunkKV = tl.load(vecI_ptr)
        vecI_chunkKV = vecI_chunkKV.to(tl.float32)

        # compute D matrix
        matD_log_fgates = vecF_cs_chunkQ[:, None] - vecF_cs_chunkKV[None, :]
        matD = matD_log_fgates + vecI_chunkKV[None, :]

        # ? -- causal masking --
        # TODO with this if I get a weird error: operation scheduled before its operands
        if start_n >= q_offset:
            # we are on diagonal
            kv_block_idxes = start_n + tl.arange(0, BLOCK_KV)
            mask = q_block_idxes[:, None] - kv_block_idxes[None, :]
            matD = tl.where(mask >= 0, matD, -float("inf"))

        # else: below diagonal

        # ? -- compute m_state --
        m_temp = tl.max(matD, axis=1)  # rowwise max
        m_temp = tl.maximum(MINIMUM_MAX_VAL, m_temp)  # elementwise max
        m_new = tl.maximum(m_old, m_temp)
        m_ratio = tl.exp(m_old - m_new)

        # ? -- compute matC --
        matD_prime = tl.exp(matD - m_new[:, None])
        matC = matS * matD_prime

        # ? -- compute l_state --
        # tl.fma did not bring performance improvement
        l_temp = m_ratio * l_old
        l_new = l_temp + tl.sum(matC, axis=1)

        # ? -- compute n_state --
        n_new = tl.maximum(tl.abs(l_new), tl.exp(-m_new))

        # ? -- compute h_out -- update h_out --
        # compute weighting factor
        # tl.fdiv did not bring any performance improvement
        h_out_old_weight = (m_ratio * n_old) / (n_new + EPS)
        h_out = h_out * h_out_old_weight[:, None]

        v = tl.load(V_block_ptr)

        matC = matC / (n_new[:, None] + EPS)
        matC = matC.to(q.type.element_ty)
        h_out = tl.dot(matC, v, h_out)

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_KV))

        l_old = l_new
        m_old = m_new
        n_old = n_new

    # epilogue
    tl.store(H_block_ptr, h_out.to(matH.type.element_ty))
    vecM_ptr = vecM + ifmn_offset + q_offset + tl.arange(0, BLOCK_Q)
    vecN_ptr = vecN + ifmn_offset + q_offset + tl.arange(0, BLOCK_Q)
    tl.store(vecM_ptr, m_old.to(vecM.type.element_ty))
    tl.store(vecN_ptr, n_old.to(vecN.type.element_ty))
