#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
Triton.

This module contains the recurrent step of the mLSTM in triton.

This is a fused forward decoding step kernel for the mLSTM. Factor of 2 speedup compared to torch.compile.
Ca. 30% faster than non-fused triton step kernel.
"""

import triton
import triton.language as tl


@triton.jit
def recurrent_step_fw_kernel(
    matC_old,  # (B, NH, DHQK, DHHV)
    vecN_old,  # (B, NH, DHQK)
    scaM_old,  # (B, NH, 1)
    vecQ,  # (B, NH, DHQK)
    vecK,  # (B, NH, DHQK)
    vecV,  # (B, NH, DHHV)
    scaI,  # (B, NH, 1)
    scaF,  # (B, NH, 1)
    vecH,  # (B, NH, DHHV)
    matC_new,  # (B, NH, DHQK, DHHV)
    vecN_new,  # (B, NH, DHQK)
    scaM_new,  # (B, NH, 1)
    qk_scale: tl.constexpr,
    str_matC_B_NH: tl.constexpr,
    str_matC_DHQK: tl.constexpr,
    str_matC_DHHV: tl.constexpr,
    str_vecN_B_NH: tl.constexpr,
    str_vecN_DHQK: tl.constexpr,
    str_scaM_B_NH: tl.constexpr,
    str_vecQK_NH: tl.constexpr,
    str_vecQK_DHQK: tl.constexpr,
    str_vecVH_B_NH: tl.constexpr,
    str_vecVH_DHHV: tl.constexpr,
    str_scaIF_B_NH: tl.constexpr,
    B: tl.constexpr,
    NH: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    EPS: tl.constexpr = 1e-6,
    DTYPE: tl.constexpr = tl.float32,
    DTYPE_STATE: tl.constexpr = tl.float32,
):
    idx_b_DHHV, idx_b_BNH = tl.program_id(1), tl.program_id(2)

    # ? Define pointers
    matC_old_bptr = tl.make_block_ptr(
        base=matC_old + idx_b_BNH * str_matC_B_NH,
        shape=(DHQK, DHHV),
        strides=(str_matC_DHQK, str_matC_DHHV),
        offsets=(0, idx_b_DHHV * siz_b_DHHV),
        block_shape=(siz_b_DHQK, siz_b_DHHV),
        order=(0, 1),
    )
    matC_new_bptr = tl.make_block_ptr(
        base=matC_new + idx_b_BNH * str_matC_B_NH,
        shape=(DHQK, DHHV),
        strides=(str_matC_DHQK, str_matC_DHHV),
        offsets=(0, idx_b_DHHV * siz_b_DHHV),
        block_shape=(siz_b_DHQK, siz_b_DHHV),
        order=(0, 1),
    )
    vecH_ptr = (
        vecH
        + idx_b_BNH * str_vecVH_B_NH
        + idx_b_DHHV * siz_b_DHHV * str_vecVH_DHHV
        + tl.arange(0, siz_b_DHHV)
    )

    scaI_ptr = scaI + idx_b_BNH * str_scaIF_B_NH
    scaF_ptr = scaF + idx_b_BNH * str_scaIF_B_NH

    scaM_old_ptr = scaM_old + idx_b_BNH * str_scaM_B_NH
    scaM_new_ptr = scaM_new + idx_b_BNH * str_scaM_B_NH

    # ? Load data
    # gates
    # the numbers are the conversion factors from log -> log2 and exp -> exp2
    # math.log2(math.e) = 1.4426950408889634
    # (1/math.log2(math.e)) = 0.6931471805599453
    # tl.exp and tl.sigmoid only work with float32
    scaF_val = tl.load(scaF_ptr).to(tl.float32)
    scaI_val = tl.load(scaI_ptr).to(tl.float32)
    scaFlog_val = tl.log(tl.sigmoid(scaF_val))

    scaM_old_val = tl.load(scaM_old_ptr)
    scaM_new_val = tl.maximum(scaFlog_val + scaM_old_val, scaI_val)
    if idx_b_DHHV == 0:  # only one thread block writes the scaM_new
        tl.store(scaM_new_ptr, scaM_new_val.to(DTYPE_STATE))

    max_val = tl.exp(-scaM_new_val.to(tl.float32)).to(DTYPE)

    # gate computation for all dimensions
    scaF_act = tl.exp(scaFlog_val + scaM_old_val - scaM_new_val).to(DTYPE)
    scaI_act = tl.exp(scaI_val - scaM_new_val).to(DTYPE)
    # tl.static_print("scaF_act", scaF_act)
    # ? init accumulators
    h_num = tl.zeros((siz_b_DHHV,), dtype=tl.float32)
    qn_dotproduct = tl.zeros((1,), dtype=tl.float32)

    NUM_BLOCKS_DQK = triton.cdiv(DHQK, siz_b_DHQK)

    for i_dhqk in range(NUM_BLOCKS_DQK):
        vecN_old_ptr = (
            vecN_old
            + idx_b_BNH * str_vecN_B_NH
            + i_dhqk * siz_b_DHQK * str_vecN_DHQK
            + tl.arange(0, siz_b_DHQK)
        )
        vecN_new_ptr = (
            vecN_new
            + idx_b_BNH * str_vecN_B_NH
            + i_dhqk * siz_b_DHQK * str_vecN_DHQK
            + tl.arange(0, siz_b_DHQK)
        )

        vecQ_ptr = (
            vecQ
            + idx_b_BNH * str_vecQK_NH
            + i_dhqk * siz_b_DHQK * str_vecQK_DHQK
            + tl.arange(0, siz_b_DHQK)
        )
        vecK_ptr = (
            vecK
            + idx_b_BNH * str_vecQK_NH
            + i_dhqk * siz_b_DHQK * str_vecQK_DHQK
            + tl.arange(0, siz_b_DHQK)
        )
        vecV_ptr = (
            vecV
            + idx_b_BNH * str_vecVH_B_NH
            + idx_b_DHHV * siz_b_DHHV * str_vecVH_DHHV
            + tl.arange(0, siz_b_DHHV)
        )

        # update rule
        vecK_val = tl.load(vecK_ptr)
        vecV_val = tl.load(vecV_ptr)
        matC_old_val = tl.load(
            matC_old_bptr, boundary_check=(0, 1), padding_option="zero"
        ).to(dtype=DTYPE_STATE)
        matC_old_val = tl.load(
            matC_old_bptr, boundary_check=(0, 1), padding_option="zero"
        ).to(dtype=DTYPE_STATE)

        matC_new_val = scaF_act * matC_old_val + scaI_act * (
            vecK_val[:, None] * vecV_val[None, :]
        )
        matC_new_val = scaF_act * matC_old_val + scaI_act * (
            vecK_val[:, None] * vecV_val[None, :]
        )

        vecN_new_val = (
            scaF_act * tl.load(vecN_old_ptr).to(dtype=DTYPE_STATE) + scaI_act * vecK_val
        )
        vecN_new_val = (
            scaF_act * tl.load(vecN_old_ptr).to(dtype=DTYPE_STATE) + scaI_act * vecK_val
        )
        # ? Store data
        tl.store(
            matC_new_bptr,
            matC_new_val.to(matC_new.type.element_ty),
            boundary_check=(0, 1),
        )
        if idx_b_DHHV == 0:  # only one thread block writes the vecN_new
            tl.store(vecN_new_ptr, vecN_new_val.to(vecN_new.type.element_ty))

        # ? advance pointers
        matC_old_bptr = tl.advance(matC_old_bptr, (siz_b_DHQK, 0))
        matC_new_bptr = tl.advance(matC_new_bptr, (siz_b_DHQK, 0))

        # ? accumulate h_num & qn_dotproduct
        vecQ_val = tl.load(vecQ_ptr) * qk_scale
        # outputs
        h_num_temp = vecQ_val[:, None] * matC_new_val.to(dtype=DTYPE)
        # we keep h_num and qn_dotproduct in float32 as they are accumulated
        h_num += tl.sum(h_num_temp, axis=0)
        qn_dotproduct += tl.sum(vecQ_val * vecN_new_val.to(dtype=DTYPE))

    # we compute h in float32 and then cast to DTYPE
    h_denom = tl.maximum(tl.abs(qn_dotproduct), max_val) + EPS
    h = tl.fdiv(h_num, h_denom)

    # ? Store data
    tl.store(vecH_ptr, h.to(DTYPE))
