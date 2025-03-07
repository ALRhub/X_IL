#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
import triton
import triton.language as tl

"""
Triton.

This module contains the recurrent step of the mLSTM in triton.

This is a non-fused forward decoding step triton kernel for the mLSTM.
Ca. 30% faster than the torch.compile version.

First kernel computes the next C, n, m states. Second kernel computes the output H.
"""

ENABLE_AUTOTUNING = True

# TODO find better heuristic
# need to adapt the block size if DHQK != DHV, then we need rectangular blocks (instead of quadratic)

if ENABLE_AUTOTUNING:
    configs = [
        triton.Config({"BLOCK_DQK": BQ, "BLOCK_DV": BV}, num_stages=s, num_warps=w)
        for BQ, BV, w in [
            (256, 256, 8),
            (256, 256, 16),
            (128, 128, 4),
            (128, 128, 8),
            (128, 128, 16),
            (64, 64, 2),
            (64, 64, 4),
            (64, 64, 8),
            (32, 32, 1),
            (32, 32, 2),
            (32, 32, 4),
            (16, 16, 1),
        ]
        for s in [1]
    ]
else:
    configs = [
        triton.Config({"BLOCK_DQK": BQ, "BLOCK_DV": BV}, num_stages=s, num_warps=w)
        for BQ, BV, w in [
            (128, 128, 8),
        ]
        for s in [1]
    ]


def keep(conf):
    BQ = conf.kwargs["BLOCK_DQK"]
    BV = conf.kwargs["BLOCK_DV"]
    if BQ * BV < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["DHQK", "DHV"])
@triton.jit
def recurrent_step_fw_kernel_C(
    matC_old,  # (B, NH, DHQK, DHV)
    vecN_old,  # (B, NH, DHQK)
    scaM_old,  # (B, NH, 1)
    vecK,  # (B, NH, DHQK)
    vecV,  # (B, NH, DHV)
    scaI,  # (B, NH, 1)
    scaF,  # (B, NH, 1)
    matC_new,  # (B, NH, DHQK, DHV)
    vecN_new,  # (B, NH, DHQK)
    scaM_new,  # (B, NH, 1)
    qk_scale,
    s_matC_b,
    s_matC_nh,
    s_matC_dhqk,
    s_matC_dhv,
    s_vecN_b,
    s_vecN_nh,
    s_vecN_dhqk,
    s_scaM_b,
    s_scaM_nh,
    s_vecQK_b,
    s_vecQK_nh,
    s_vecQK_dhqk,
    s_vecVH_b,
    s_vecVH_nh,
    s_vecVH_dhv,
    s_scaIF_b,
    s_scaIF_nh,
    B,
    NH,
    DHQK: tl.constexpr,
    DHV: tl.constexpr,
    BLOCK_DQK: tl.constexpr,  # DHQK = BLOCK_DQK * NUM_BLOCKS_DQK
    BLOCK_DV: tl.constexpr,  # DHV = BLOCK_DV * NUM_BLOCKS_DV
    EPS: tl.constexpr = 1e-6,
):
    i_dhqk, i_dhv, i_bnh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # ? Define pointers
    matC_old_bptr = tl.make_block_ptr(
        base=matC_old + i_bnh * s_matC_nh,
        shape=(DHQK, DHV),
        strides=(s_matC_dhqk, s_matC_dhv),
        offsets=(i_dhqk * BLOCK_DQK, i_dhv * BLOCK_DV),
        block_shape=(BLOCK_DQK, BLOCK_DV),
        order=(0, 1),
    )
    matC_new_bptr = tl.make_block_ptr(
        base=matC_new + i_bnh * s_matC_nh,
        shape=(DHQK, DHV),
        strides=(s_matC_dhqk, s_matC_dhv),
        offsets=(i_dhqk * BLOCK_DQK, i_dhv * BLOCK_DV),
        block_shape=(BLOCK_DQK, BLOCK_DV),
        order=(0, 1),
    )

    vecN_old_ptr = vecN_old + i_bnh * s_vecN_nh + i_dhqk * BLOCK_DQK * s_vecN_dhqk + tl.arange(0, BLOCK_DQK)
    vecN_new_ptr = vecN_new + i_bnh * s_vecN_nh + i_dhqk * BLOCK_DQK * s_vecN_dhqk + tl.arange(0, BLOCK_DQK)

    scaM_old_ptr = scaM_old + i_bnh * s_scaM_nh
    scaM_new_ptr = scaM_new + i_bnh * s_scaM_nh

    vecK_ptr = vecK + i_bnh * s_vecQK_nh + i_dhqk * BLOCK_DQK * s_vecQK_dhqk + tl.arange(0, BLOCK_DQK)
    vecV_ptr = vecV + i_bnh * s_vecVH_nh + i_dhv * BLOCK_DV * s_vecVH_dhv + tl.arange(0, BLOCK_DV)

    scaI_ptr = scaI + i_bnh * s_scaIF_nh
    scaF_ptr = scaF + i_bnh * s_scaIF_nh

    # ? Load data
    # gates
    # tl.exp and tl.sigmoid only work with float32
    scaF_val = tl.load(scaF_ptr).to(tl.float32)
    scaI_val = tl.load(scaI_ptr).to(tl.float32)

    scaFlog_val = tl.log(tl.sigmoid(scaF_val)).to(scaM_old.type.element_ty)

    scaM_old_val = tl.load(scaM_old_ptr)

    # update rule
    # cast back to state type
    scaM_new_val = tl.maximum(scaFlog_val + scaM_old_val, scaI_val)  # .to(scaM_old.type.element_ty)

    scaF_act = tl.exp(scaFlog_val + scaM_old_val - scaM_new_val).to(scaM_old.type.element_ty)
    scaI_act = tl.exp(scaI_val - scaM_new_val).to(scaM_old.type.element_ty)

    vecK_val = tl.load(vecK_ptr)
    vecV_val = tl.load(vecV_ptr)

    matC_old_val = tl.load(matC_old_bptr, boundary_check=(0, 1), padding_option="zero")

    matC_new_val = scaF_act * matC_old_val + scaI_act * (vecK_val[:, None] * vecV_val[None, :])

    vecN_new_val = scaF_act * tl.load(vecN_old_ptr) + scaI_act * vecK_val

    # ? Store data
    tl.store(matC_new_bptr, matC_new_val.to(matC_new.type.element_ty), boundary_check=(0, 1))
    tl.store(vecN_new_ptr, vecN_new_val.to(vecN_new.type.element_ty))
    tl.store(scaM_new_ptr, scaM_new_val.to(scaM_new.type.element_ty))


@triton.autotune(list(filter(keep, configs)), key=["DHQK", "DHV"])
@triton.jit
def recurrent_step_fw_kernel_H(
    vecQ,  # (B, NH, DHQK)
    vecH,  # (B, NH, DHV)
    matC_new,  # (B, NH, DHQK, DHV)
    vecN_new,  # (B, NH, DHQK)
    scaM_new,  # (B, NH, 1)
    qk_scale,
    s_matC_b,
    s_matC_nh,
    s_matC_dhqk,
    s_matC_dhv,
    s_vecN_b,
    s_vecN_nh,
    s_vecN_dhqk,
    s_scaM_b,
    s_scaM_nh,
    s_vecQK_b,
    s_vecQK_nh,
    s_vecQK_dhqk,
    s_vecVH_b,
    s_vecVH_nh,
    s_vecVH_dhv,
    s_scaIF_b,
    s_scaIF_nh,
    B,
    NH,
    DHQK: tl.constexpr,
    DHV: tl.constexpr,
    BLOCK_DQK: tl.constexpr,  # DHQK = BLOCK_DQK * NUM_BLOCKS_DQK
    BLOCK_DV: tl.constexpr,  # DHV = BLOCK_DV * NUM_BLOCKS_DV
    EPS: tl.constexpr = 1e-6,
):
    i_dhv, i_bnh = tl.program_id(1), tl.program_id(2)

    # ? Define pointers
    matC_new_bptr = tl.make_block_ptr(
        base=matC_new + i_bnh * s_matC_nh,
        shape=(DHQK, DHV),
        strides=(s_matC_dhqk, s_matC_dhv),
        offsets=(0, i_dhv * BLOCK_DV),
        block_shape=(BLOCK_DQK, BLOCK_DV),
        order=(0, 1),
    )
    scaM_new_ptr = scaM_new + i_bnh * s_scaM_nh
    scaM_new_val = tl.load(scaM_new_ptr)
    vecH_ptr = vecH + i_bnh * s_vecVH_nh + i_dhv * BLOCK_DV * s_vecVH_dhv + tl.arange(0, BLOCK_DV)

    h_num = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    qn_dotproduct = tl.zeros((1,), dtype=tl.float32)

    NUM_BLOCKS_DQK = triton.cdiv(DHQK, BLOCK_DQK)

    for i_dhqk in range(NUM_BLOCKS_DQK):
        vecN_new_ptr = vecN_new + i_bnh * s_vecN_nh + i_dhqk * BLOCK_DQK * s_vecN_dhqk + tl.arange(0, BLOCK_DQK)

        vecQ_ptr = vecQ + i_bnh * s_vecQK_nh + i_dhqk * BLOCK_DQK * s_vecQK_dhqk + tl.arange(0, BLOCK_DQK)

        # ? Load data
        matC_new_val = tl.load(matC_new_bptr, boundary_check=(0, 1), padding_option="zero")
        vecN_new_val = tl.load(vecN_new_ptr)

        vecQ_val = tl.load(vecQ_ptr) * qk_scale

        # outputs
        h_num_temp = vecQ_val[:, None] * matC_new_val
        h_num += tl.sum(h_num_temp, axis=0)

        qn_dotproduct += tl.sum(vecQ_val * vecN_new_val)
        matC_new_bptr = tl.advance(matC_new_bptr, (BLOCK_DQK, 0))

    max_val = tl.exp(-scaM_new_val.to(tl.float32)).to(scaM_new.type.element_ty)
    h_denom = tl.maximum(tl.abs(qn_dotproduct), max_val) + EPS
    h = tl.fdiv(h_num, h_denom)

    # ? Store data
    tl.store(vecH_ptr, h.to(vecH.type.element_ty))
