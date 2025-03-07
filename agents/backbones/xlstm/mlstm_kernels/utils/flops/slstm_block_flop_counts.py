#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""Contains the sLSTM block flop counts."""

from collections.abc import Callable


def count_flops_slstm_cell_fw(S, d, Nh, factor_exp=1, factor_div=1):
    """Counts the number of flops in the forward pass of an sLSTM cell."""
    assert d % Nh == 0, f"d must be divisible by Nh, got d={d} and Nh={Nh}"
    dh = d // Nh

    return S * Nh * dh * (8 * dh + 5 * factor_exp + 4 + 5 + 1 * factor_div)


def _count_ln_flops(d):
    # not sure about this
    # d for mean, d to subtract mean, d for variance, d for division
    # return 4 * d

    # do not count ln flops
    return 0


def count_flops_slstm_block_fw(
    S, d, Nh, conv1d_kernel_size=4, pf_ffn=1.3, factor_exp=1, count_ln_flops: Callable[[int], int] = _count_ln_flops
):
    slstm_cell_flops = count_flops_slstm_cell_fw(S=S, d=d, Nh=Nh)
    dh = d // Nh
    conv1d_flops = 2 * conv1d_kernel_size * (S + conv1d_kernel_size - 1) * dh * Nh + S * dh * Nh
    ffn_flops = 4 * S * d * d * pf_ffn + S * d * factor_exp

    skip_ln_flops = 2 * S * d + (2 + 1) * S * count_ln_flops(d)  # 2 block pre-norm, 1 group norm

    total_flops = int(slstm_cell_flops + conv1d_flops + ffn_flops + skip_ln_flops)
    linear_layer_flops = int(ffn_flops)
    return total_flops, linear_layer_flops, int(slstm_cell_flops)


def get_slstm_fw_flop_dict(sequence_length: int) -> dict[str, tuple[int, int, int]]:
    slstm_size_dict = {
        "125M": dict(d=768, Nh=4),
        "350M": dict(d=1024, Nh=4),
        "760M": dict(d=1536, Nh=4),
        "1.3B": dict(d=2048, Nh=4),
    }

    flops_dict = {k: count_flops_slstm_block_fw(sequence_length, **v) for k, v in slstm_size_dict.items()}
    return flops_dict
