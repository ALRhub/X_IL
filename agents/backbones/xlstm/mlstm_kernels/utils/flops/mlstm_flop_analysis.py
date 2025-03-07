#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""Contains the final counts of the mlstm blocks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mlstm_block_flop_counts import (
    FLOPsComputation,
    count_flops_mlstm_chunkwise_fw,
    count_fw_flops,
    embedding_dims,
    flop_factors,
    get_mlstm_v1_fw_flops,
    get_mlstm_v2_fw_flops,
)


def get_mlstm_v1_fw_7B_flops(sequence_length: int, chunk_size: int, batch_size: int = 1):
    num_heads_mlstm_v1 = {
        "7B_nh1": 1,
        "7B_nh2": 2,
        "7B_nh4": 4,
        "7B_nh8": 8,
        "7B_nh16": 16,
    }
    pf_ffn = 4
    mlstm_v1_flop_computations = {}
    for model_tag in num_heads_mlstm_v1.keys():
        flopcomp = FLOPsComputation(
            batch_size=batch_size,
            model_type="mlstm_v1",
            model_tag=model_tag,
            model_config={
                "S": sequence_length,
                "d": embedding_dims["7B"],
                "Nh": num_heads_mlstm_v1[model_tag],
                "dqk": embedding_dims["7B"] // num_heads_mlstm_v1[model_tag],
                "dv": embedding_dims["7B"] // num_heads_mlstm_v1[model_tag],
                "chunk_size": chunk_size,
                "pf": pf_ffn,
                **flop_factors,
            },
        )
        mlstm_v1_flop_computations[model_tag] = count_fw_flops(flopcomp)
    return mlstm_v1_flop_computations


def get_mlstm_v2_fw_7B_flops(sequence_length: int, chunk_size: int, batch_size: int = 1):
    num_heads_mlstm_v2 = {
        "7B_nh1": 1,
        "7B_nh2": 2,
        "7B_nh4": 4,
        "7B_nh8": 8,
        "7B_nh16": 16,
    }
    qk_block_size = 4
    qk_pf = 1
    v_block_size = 4
    v_pf = 1
    conv1d_kernel_size = 4
    pf = 2
    mlstm_v2_flop_computations = {}
    for model_tag in num_heads_mlstm_v2.keys():
        flopcomp = FLOPsComputation(
            batch_size=batch_size,
            model_type="mlstm_v2",
            model_tag=model_tag,
            model_config={
                "S": sequence_length,
                "d": embedding_dims["7B"],
                "Nh": num_heads_mlstm_v2[model_tag],
                "dqk": (embedding_dims["7B"] * pf) // num_heads_mlstm_v2[model_tag],
                "dv": (embedding_dims["7B"] * pf) // num_heads_mlstm_v2[model_tag],
                "qk_block_size": qk_block_size,
                "qk_pf": qk_pf,
                "v_block_size": v_block_size,
                "v_pf": v_pf,
                "conv1d_kernel_size": conv1d_kernel_size,
                "chunk_size": chunk_size,
                "pf": pf,
                **flop_factors,
            },
        )
        mlstm_v2_flop_computations[model_tag] = count_fw_flops(flopcomp)
    return mlstm_v2_flop_computations


## plotting
def get_flops_array_for_sizes(
    model_size_keys: list[str],
    flop_computation_dict: dict[str, FLOPsComputation],
    flop_type: str = "total_other_flops",
) -> list[int]:
    flop_vals = []
    for msk in model_size_keys:
        if flop_type == "total_flops":
            flop_vals.append(flop_computation_dict[msk].total_flops)
        elif flop_type == "mlstm_other_flops":
            flop_vals.append(flop_computation_dict[msk].mlstm_other_flops)
        elif flop_type == "linear_layer_flops":
            flop_vals.append(flop_computation_dict[msk].linear_layer_flops)
    return flop_vals


def plot_mlstm_v1_v2_flop_comparison(
    sequence_length: int,
    batch_size: int,
    chunk_size: int,
    model_size_keys: list[str],
    plot_only_total_flops: bool = True,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    mlstm_flops_v1 = get_mlstm_v1_fw_flops(
        sequence_length=sequence_length, batch_size=batch_size, chunk_size=chunk_size
    )
    mlstm_flops_v2 = get_mlstm_v2_fw_flops(
        sequence_length=sequence_length, batch_size=batch_size, chunk_size=chunk_size
    )

    total_flops_v1 = get_flops_array_for_sizes(
        model_size_keys, flop_computation_dict=mlstm_flops_v1, flop_type="total_flops"
    )
    mlstm_other_flops_v1 = get_flops_array_for_sizes(
        model_size_keys,
        flop_computation_dict=mlstm_flops_v1,
        flop_type="mlstm_other_flops",
    )
    linear_layer_flops_v1 = get_flops_array_for_sizes(
        model_size_keys,
        flop_computation_dict=mlstm_flops_v1,
        flop_type="linear_layer_flops",
    )

    total_flops_v2 = get_flops_array_for_sizes(
        model_size_keys, flop_computation_dict=mlstm_flops_v2, flop_type="total_flops"
    )
    mlstm_other_flops_v2 = get_flops_array_for_sizes(
        model_size_keys,
        flop_computation_dict=mlstm_flops_v2,
        flop_type="mlstm_other_flops",
    )
    linear_layer_flops_v2 = get_flops_array_for_sizes(
        model_size_keys,
        flop_computation_dict=mlstm_flops_v2,
        flop_type="linear_layer_flops",
    )

    x_vals = np.arange(len(model_size_keys))
    x_tick_labels = model_size_keys

    ax.grid(alpha=0.5)
    v1_color = "tab:blue"
    v2_color = "tab:red"
    ax.plot(x_vals, total_flops_v1, label="Total FLOPs mLSTMv1", marker="o", color=v1_color)
    ax.plot(x_vals, total_flops_v2, label="Total FLOPs mLSTMv2", marker="o", color=v2_color)
    if not plot_only_total_flops:
        ax.plot(
            x_vals,
            mlstm_other_flops_v1,
            label="mLSTM FLOPs mLSTMv1",
            linestyle="--",
            color=v1_color,
        )
        ax.plot(
            x_vals,
            mlstm_other_flops_v2,
            label="mLSTM FLOPs mLSTMv2",
            linestyle="--",
            color=v2_color,
        )
        ax.plot(
            x_vals,
            linear_layer_flops_v1,
            label="Linear Layer FLOPs mLSTMv1",
            linestyle="-.",
            color=v1_color,
        )

        ax.plot(
            x_vals,
            linear_layer_flops_v2,
            label="Linear Layer FLOPs mLSTMv2",
            linestyle="-.",
            color=v2_color,
        )

    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yscale("log")

    ax.set_xlabel("Model Size")
    ax.set_ylabel("FLOPs")
    ax.set_title("mLSTM v1 vs mLSTM v2 Block FLOPs Comparison")
    ax.legend()
    return fig


def make_flop_table(
    model_size_keys: list[str],
    mlstm_flops_v1: dict[str, FLOPsComputation],
    mlstm_flops_v2: dict[str, FLOPsComputation],
):
    total_flops_v1 = get_flops_array_for_sizes(
        model_size_keys, flop_computation_dict=mlstm_flops_v1, flop_type="total_flops"
    )
    mlstm_other_flops_v1 = get_flops_array_for_sizes(
        model_size_keys,
        flop_computation_dict=mlstm_flops_v1,
        flop_type="mlstm_other_flops",
    )
    linear_layer_flops_v1 = get_flops_array_for_sizes(
        model_size_keys,
        flop_computation_dict=mlstm_flops_v1,
        flop_type="linear_layer_flops",
    )
    total_flops_v2 = get_flops_array_for_sizes(
        model_size_keys, flop_computation_dict=mlstm_flops_v2, flop_type="total_flops"
    )
    mlstm_other_flops_v2 = get_flops_array_for_sizes(
        model_size_keys,
        flop_computation_dict=mlstm_flops_v2,
        flop_type="mlstm_other_flops",
    )
    linear_layer_flops_v2 = get_flops_array_for_sizes(
        model_size_keys,
        flop_computation_dict=mlstm_flops_v2,
        flop_type="linear_layer_flops",
    )

    v2_v1_ratio = np.array(total_flops_v2) / np.array(total_flops_v1)

    v1_linear_layer_ratio = np.array(linear_layer_flops_v1) / np.array(total_flops_v1)
    v2_linear_layer_ratio = np.array(linear_layer_flops_v2) / np.array(total_flops_v2)

    v1_mlstm_other_ratio = np.array(mlstm_other_flops_v1) / np.array(total_flops_v1)
    v2_mlstm_other_ratio = np.array(mlstm_other_flops_v2) / np.array(total_flops_v2)

    dat_dict = {
        "Model Size": model_size_keys,
        "Total FLOPs v1": total_flops_v1,
        "Total FLOPs v2": total_flops_v2,
        "v2/v1 Ratio": v2_v1_ratio,
        "v1 Linear Layer FLOPs Ratio": v1_linear_layer_ratio,
        "v2 Linear Layer FLOPs Ratio": v2_linear_layer_ratio,
        "v1 MLSTM Other FLOPs Ratio": v1_mlstm_other_ratio,
        "v2 MLSTM Other FLOPs Ratio": v2_mlstm_other_ratio,
    }
    return pd.DataFrame(dat_dict)


def make_chunkwise_flop_chunksize_sweep(
    seq_len, chunk_sizes: list[int], dqk, dv, Nh, factor_exp, factor_max, factor_mask
):
    num_chunks = [seq_len // chunk_size for chunk_size in chunk_sizes]

    num_chunks = np.array(num_chunks)
    chunk_sizes = np.array(chunk_sizes)

    flop_results = []
    for chunk_size, num_chunk in zip(chunk_sizes, num_chunks):
        total_flops, fw_C_flops, fw_h_flops = count_flops_mlstm_chunkwise_fw(
            L=chunk_size,
            Nc=num_chunk,
            dqk=dqk,
            dv=dv,
            Nh=Nh,
            factor_exp=factor_exp,
            factor_mask=factor_mask,
            factor_max=factor_max,
        )
        flop_results.append([total_flops, fw_C_flops, fw_h_flops])
        print(
            f"Chunk Size: {chunk_size}, Num Chunk: {num_chunk}, Total FLOPs: {total_flops / 1e6:.2f}M, Parallel FLOPs: {fw_h_flops / 1e6:.2f}M, Recurrent FLOPs: {fw_C_flops / 1e6:.2f}M"
        )

    flop_results = np.array(flop_results)
    fig, ax = plt.subplots()

    ax.plot(chunk_sizes, flop_results[:, 0], label="Total FLOPs", marker="o")
    ax.plot(chunk_sizes, flop_results[:, 1], label="Recurrent FLOPs", marker="o")
    ax.plot(chunk_sizes, flop_results[:, 2], label="Parallel FLOPs", marker="o")

    ax.set_xlabel("Chunk Size")
    ax.set_ylabel("FLOPs")
    ax.set_title(f"FLOPs vs Chunk Size - Seq_len={seq_len}, dqk={dqk}, dv={dv}, Nh={Nh}")
    ax.grid(alpha=0.5)
    ax.legend()
    return fig


def make_chunkwise_flop_sequence_length_sweep(
    seq_lengths: list[int], chunk_size, dqk, dv, Nh, factor_exp, factor_max, factor_mask
):
    sequence_lengths = np.array(seq_lengths)

    flops_results_parallel = []
    flop_results = []
    for seq_len in seq_lengths:
        num_chunk = seq_len // chunk_size
        total_flops, fw_C_flops, fw_h_flops = count_flops_mlstm_chunkwise_fw(
            L=chunk_size,
            Nc=num_chunk,
            dqk=dqk,
            dv=dv,
            Nh=Nh,
            factor_exp=factor_exp,
            factor_mask=factor_mask,
            factor_max=factor_max,
        )
        flop_results.append([total_flops, fw_C_flops, fw_h_flops])

        # parallel baseline with only one chunk of chunk_size=seq_len
        total_flops_p, fw_C_flops_p, fw_h_flops_p = count_flops_mlstm_chunkwise_fw(
            L=seq_len,
            Nc=1,
            dqk=dqk,
            dv=dv,
            Nh=Nh,
            factor_exp=factor_exp,
            factor_mask=factor_mask,
            factor_max=factor_max,
        )
        flops_results_parallel.append([total_flops_p, fw_C_flops_p, fw_h_flops_p])
        print(f"seq_len={seq_len}, chunkwise FLOPs={total_flops / 1e6:.2f}M, parallel FLOPs={total_flops_p / 1e6:.2f}M")

    flop_results = np.array(flop_results)
    flops_results_parallel = np.array(flops_results_parallel)
    fig, ax = plt.subplots()

    ax.plot(sequence_lengths, flop_results[:, 0], label="Total FLOPs")
    ax.plot(sequence_lengths, flop_results[:, 1], label="Recurrent FLOPs")
    ax.plot(sequence_lengths, flop_results[:, 2], label="Parallel FLOPs")
    ax.plot(
        sequence_lengths,
        flops_results_parallel[:, 0],
        label="Total FLOPs - Parallel",
        color="red",
    )

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("FLOPs")
    ax.set_title(f"FLOPs vs Sequence Length - chunk_size={chunk_size}, dqk={dqk}, dv={dv}, Nh={Nh}")
    ax.grid(alpha=0.5)
    ax.legend()
    return fig
