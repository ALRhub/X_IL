#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
import torch


def mlstm_recurrent_step__native_fw(
    matC_old: torch.Tensor,  # (B, NH, DHQK, DHV)
    vecN_old: torch.Tensor,  # (B, NH, DHQK)
    scaM_old: torch.Tensor,  # (B, NH, 1)
    vecQ: torch.Tensor,  # (B, NH, DHQK)
    vecK: torch.Tensor,  # (B, NH, DHQK)
    vecV: torch.Tensor,  # (B, NH, DHV)
    scaI: torch.Tensor,  # (B, NH, 1)
    scaF: torch.Tensor,  # (B, NH, 1)
    eps: float = 1e-6,
    dtype_state: torch.dtype = torch.float32,
    **kwargs,
) -> tuple[
    torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:  # vecH, (matC_state_new (B, NH, DHQK, DHV), vecN_state_new (B, NH, DHQK), vecM_state_new (B, NH, 1))
    """This is a single step of the mLSTM operation in recurrent form.

    Args:
        matC_old: (B, NH, DHQK, DHV)
        vecN_old: (B, NH, DHQK)
        scaM_old: (B, NH, 1)
        vecQ: (B, NH, DHQK)
        vecK: (B, NH, DHQK)
        vecV: (B, NH, DHV)
        scaI: (B, NH, 1)
        scaF: (B, NH, 1)
        eps: Used for building the forgetgate matrix. Defaults to 1e-6.

    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            (hidden_state [B, NH, DHV], (c_state_new [B, NH, DHQK, DHV], n_state_new [B, NH, DHQK]], m_state_new [B, NH, 1]))
    """

    dtype_qkv = vecQ.dtype
    matC_old = matC_old.to(dtype=dtype_state)
    vecN_old = vecN_old.to(dtype=dtype_state)
    scaM_old = scaM_old.to(dtype=dtype_state)

    B, NH, DHQK = vecQ.shape
    _, _, DHHV = vecV.shape
    assert vecQ.shape == vecK.shape, "q and k must have the same shape"
    assert matC_old.shape == (
        B,
        NH,
        DHQK,
        DHHV,
    ), f"matC_old has wrong shape, got {matC_old.shape}"
    assert vecN_old.shape == (
        B,
        NH,
        DHQK,
    ), f"vecN_old has wrong shape, got {vecN_old.shape}"
    assert scaM_old.shape == (
        B,
        NH,
        1,
    ), f"scaM_old has wrong shape, got {scaM_old.shape}"
    assert scaI.shape == (B, NH, 1), f"scaI has wrong shape, got {scaI.shape}"
    assert scaF.shape == (B, NH, 1), f"scaF has wrong shape, got {scaF.shape}"

    # gates
    scaF_log = torch.nn.functional.logsigmoid(scaF)

    # update rule
    scaM_state_new = torch.max(scaF_log + scaM_old, scaI)  # (B, NH, 1)

    scaF_act = torch.exp(scaF_log + scaM_old - scaM_state_new)  # (B, NH, 1)
    scaI_act = torch.exp(scaI - scaM_state_new)  # (B, NH, 1)

    vecQ_scaled = vecQ * (DHQK ** (-0.5))  # (B, NH, DHQK)
    matC_state_new = scaF_act[:, :, :, None] * matC_old + scaI_act[:, :, :, None] * (
        vecK[:, :, :, None] @ vecV[:, :, None, :]
    )  # (B, NH, DHQK, DHV)
    vecN_state_new = scaF_act * vecN_old + scaI_act * vecK  # (B, NH, DHQK)
    h_num = vecQ_scaled[:, :, None, :] @ matC_state_new.to(
        dtype=dtype_qkv
    )  # (B, NH, 1, DHV)
    h_num = h_num.squeeze(2).to(dtype=dtype_state)  # (B, NH, DHV)

    qn_dotproduct = vecQ_scaled[:, :, None, :] @ vecN_state_new[:, :, :, None].to(
        dtype=dtype_qkv
    )  # (B, NH, 1, 1)
    qn_dotproduct = qn_dotproduct.squeeze(2)  # (B, NH, 1)
    max_val = torch.exp(-scaM_state_new)  # (B, NH, 1)
    h_denom = (torch.maximum(qn_dotproduct.abs(), max_val) + eps).to(
        dtype=dtype_state
    )  # (B, NH, 1)
    h = h_num / h_denom  # (B, NH, DHV) / (B, NH, 1) = (B, NH, DHV)

    h = h.to(dtype=dtype_qkv)
    matC_state_new = matC_state_new.to(dtype=dtype_state)
    vecN_state_new = vecN_state_new.to(dtype=dtype_state)
    scaM_state_new = scaM_state_new.to(dtype=dtype_state)
    return h, (matC_state_new, vecN_state_new, scaM_state_new)


def mlstm_recurrent_step__native(
    q: torch.Tensor,  # (B, NH, DHQK)
    k: torch.Tensor,  # (B, NH, DHQK)
    v: torch.Tensor,  # (B, NH, DHV)
    i: torch.Tensor,  # (B, NH, 1)
    f: torch.Tensor,  # (B, NH, 1)
    c: torch.Tensor,  # (B, NH, DHQK, DHV)
    n: torch.Tensor,  # (B, NH, DHQK)
    m: torch.Tensor,  # (B, NH, 1)
    eps: float = 1e-6,
    dtype_state: torch.dtype = torch.float32,
    **kwargs,
) -> tuple[
    torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:  # vecH, (matC_state_new (B, NH, DHQK, DHV), vecN_state_new (B, NH, DHQK), vecM_state_new (B, NH, 1))
    """This is a single step of the mLSTM operation in recurrent form."""
    return mlstm_recurrent_step__native_fw(
        matC_old=c,
        vecN_old=n,
        scaM_old=m,
        vecQ=q,
        vecK=k,
        vecV=v,
        scaI=i,
        scaF=f,
        eps=eps,
        dtype_state=dtype_state,
        **kwargs,
    )
