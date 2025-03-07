#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

def count_flops_transformer_block_fw(S: int, d: int, Nh: int, ff_ratio: float = 4.0) -> tuple[int, int, int]:
    """DeepMind method for forwad pass FLOPs counting of decoder-only Transformer
    See Chinchilla paper or this blog post:
    https://www.adamcasson.com/posts/transformer-flops
    """
    d_attn = d // Nh
    d_ff = d * ff_ratio

    attn_qkv = 2 * S * 3 * d * (d_attn * Nh)
    attn_logits = 2 * S * S * (d_attn * Nh)
    attn_softmax = 3 * Nh * S * S
    attn_reduce = 2 * S * S * (d_attn * Nh)
    attn_project = 2 * S * (d_attn * Nh) * d
    total_attn = attn_qkv + attn_logits + attn_softmax + attn_reduce + attn_project

    ff = 2 * S * (d * d_ff + d * d_ff)

    total_flops = total_attn + ff

    return int(total_flops), int(ff), int(total_attn)


def get_transformer_fw_flop_dict(sequence_length: int) -> dict[str, tuple[int, int, int]]:
    """Returns a dictionary with the FLOPs of the transformer block for different hidden sizes."""
    transformer_size_dict = {
        "125M": dict(d=768, Nh=12),
        "350M": dict(d=1024, Nh=16),
        "760M": dict(d=1536, Nh=16),
        "1.3B": dict(d=2048, Nh=16),
        "2.7B": dict(d=2560, Nh=32),
        "7B": dict(d=4096, Nh=32),
    }

    flops_dict = {k: count_flops_transformer_block_fw(sequence_length, **v) for k, v in transformer_size_dict.items()}
    return flops_dict
