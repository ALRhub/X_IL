#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

def compute_total_model_flops(
    total_fw_block_flops: int = None,
    batch_size: int = None,
    num_blocks: int = None,
    vocab_size: int = None,
    embedding_dim: int = None,
    sequence_length: int = None,
    num_train_steps: int = 1,
    include_embedding_flops: bool = True,
    include_logits_flops: bool = True,
    backward_flop_factor: float = 2.0,
    total_fw_block2_flops: int = None,
    num_blocks2: int = None,
) -> int:
    total_flops = 0

    total_block_flops = total_fw_block_flops * num_blocks
    if total_fw_block2_flops is not None:
        assert num_blocks2 is not None, "num_blocks2 must be provided if total_fw_block2_flops is provided"
        total_block_flops += total_fw_block2_flops * num_blocks2

    total_flops += total_block_flops
    if include_embedding_flops:
        embedding_flops = 2 * sequence_length * vocab_size * embedding_dim
        total_flops += embedding_flops
    if include_logits_flops:
        logit_flops = 2 * sequence_length * vocab_size * embedding_dim
        total_flops += logit_flops

    total_flops = total_flops * batch_size * backward_flop_factor * num_train_steps

    return total_flops


def compute_total_model_flops_for_block_flops_dict(
    block_flops_dict: dict[str, tuple[int, ...]],
    batch_size: int = None,
    num_blocks: int = None,
    vocab_size: int = None,
    embedding_dim: int = None,
    sequence_length: int = None,
    num_train_steps: int = None,
    sizes_dict: dict[str, dict[str, int | float]] = {},
    include_embedding_flops: bool = True,
    include_logits_flops: bool = True,
    backward_flop_factor: float = 2.0,
    block2_flops_dict: dict[str, tuple[int, ...]] = None,
    num_blocks2: int = None,
) -> dict[str, tuple[int]]:
    total_flops_dict = {}
    for model_size in sizes_dict.keys():
        block_flops = block_flops_dict[model_size]
        total_fw_block_flops = block_flops[0]
        size_dict = sizes_dict.get(model_size, {})
        if batch_size is not None:
            size_dict["batch_size"] = batch_size
        if num_blocks is not None:
            size_dict["num_blocks"] = num_blocks
        if vocab_size is not None:
            size_dict["vocab_size"] = vocab_size
        if embedding_dim is not None:
            size_dict["embedding_dim"] = embedding_dim
        if sequence_length is not None:
            size_dict["sequence_length"] = sequence_length
        if num_train_steps is not None:
            size_dict["num_train_steps"] = num_train_steps

        if block2_flops_dict is not None:
            block_flops2 = block2_flops_dict[model_size]
            total_fw_block2_flops = block_flops2[0]
        else:
            total_fw_block2_flops = None

        if num_blocks2 is not None:
            size_dict["num_blocks2"] = num_blocks2

        total_flops = compute_total_model_flops(
            total_fw_block_flops=total_fw_block_flops,
            total_fw_block2_flops=total_fw_block2_flops,
            include_embedding_flops=include_embedding_flops,
            include_logits_flops=include_logits_flops,
            backward_flop_factor=backward_flop_factor,
            **size_dict,
        )
        total_flops_dict[model_size] = (total_flops,)

    return total_flops_dict
