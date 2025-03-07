#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import numpy as np
from matplotlib import pyplot as plt


def convert_to_diff_imarray_torch(target, baseline=None):
    """Converts the difference of two result torch.Tensors
    to a numpy array for plotting."""
    import torch

    assert isinstance(target, torch.Tensor)
    if baseline is not None:
        assert isinstance(baseline, torch.Tensor)

    if baseline is None:
        imarr = target.detach().float().abs().squeeze().cpu().numpy()
    else:
        assert baseline.ndim == target.ndim
        if baseline.ndim > 2:
            baseline = baseline.clone()
            target = target.clone()
            # take only first element of each dimension
            while baseline.ndim > 2:
                baseline = baseline[0, ...]
            while target.ndim > 2:
                target = target[0, ...]
        imarr = (target.detach().float() - baseline.detach().float()).abs().cpu().numpy()
    if imarr.ndim < 2:
        imarr = imarr[:, None]
    return imarr


def convert_to_diff_imarray_numpy(target, baseline=None):
    """Converts the difference of two result numpy.ndarrays
    to a single numpy array for plotting."""
    assert isinstance(target, np.ndarray)
    if baseline is not None:
        assert isinstance(baseline, np.ndarray)

    if baseline is None:
        imarr = target
    else:
        assert baseline.ndim == target.ndim
        if baseline.ndim > 2:
            baseline = baseline.copy()
            target = target.copy()
            # take only first element of each dimension
            while baseline.ndim > 2:
                baseline = baseline[0, ...]
            while target.ndim > 2:
                target = target[0, ...]
        imarr = np.abs(target - baseline)
    if imarr.ndim < 2:
        imarr = imarr[:, None]
    return imarr


def plot_numerical_diffs(
    pt_fp32_baseline,
    cu_fp32,
    cu_bf16,
    cu_half,
    title,
    vmin=0.0,
    vmax=1e-2,
    figsize=(10, 6),
    convert_to_diff_imarray_fn: callable = convert_to_diff_imarray_numpy,
):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=figsize, ncols=3)

    pos1 = ax1.imshow(
        convert_to_diff_imarray_fn(cu_fp32, pt_fp32_baseline),
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("float32")
    fig.colorbar(pos1, ax=ax1)
    pos2 = ax2.imshow(
        convert_to_diff_imarray_fn(cu_bf16, pt_fp32_baseline),
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title("bfloat16")
    fig.colorbar(pos2, ax=ax2)
    pos3 = ax3.imshow(
        convert_to_diff_imarray_fn(cu_half, pt_fp32_baseline),
        vmin=vmin,
        vmax=vmax,
    )
    ax3.set_title("float16")
    fig.colorbar(pos3, ax=ax3)
    fig.suptitle(title)
    return fig


def plot_numerical_diffs_single(
    baseline,
    target=None,
    title="",
    vmin=0.0,
    vmax=1e-2,
    figsize=(10, 6),
    convert_to_diff_imarray_fn: callable = convert_to_diff_imarray_numpy,
):
    fig, ax1 = plt.subplots(figsize=figsize)
    pos1 = ax1.imshow(
        convert_to_diff_imarray_fn(baseline=baseline, target=target),
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title(title)
    fig.colorbar(pos1, ax=ax1)
    return fig


def plot_numerical_diffs_per_batchhead(
    baseline,
    target=None,
    title="",
    vmin=0.0,
    vmax=1e-2,
    figsize=(10, 6),
    rtol: float = None,
    atol: float = None,
    max_num_batchhead_plots: int = -1,  # -1 means all
    convert_to_diff_imarray_fn: callable = convert_to_diff_imarray_numpy,
):
    baseline = baseline.reshape(-1, baseline.shape[-2], baseline.shape[-1])
    if target is not None:
        target = target.reshape(-1, target.shape[-2], target.shape[-1])

    if max_num_batchhead_plots > 0:
        num_batchheads = min(max_num_batchhead_plots, baseline.shape[0])
    else:
        num_batchheads = baseline.shape[0]

    figs = []
    for i in range(num_batchheads):
        max_diff = np.max(np.abs(baseline[i, ...] - target[i, ...]))
        title_i = f"BH({i}):{title}|max_diff:{max_diff}"
        if rtol is not None and atol is not None:
            allclose = np.allclose(baseline[i, ...], target[i, ...], rtol=rtol, atol=atol)
            title_i += f"|allclose(atol={atol},rtol={rtol}):{allclose}"
        fig = plot_numerical_diffs_single(
            baseline=baseline[i, ...],
            target=target[i, ...],
            title=title_i,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            convert_to_diff_imarray_fn=convert_to_diff_imarray_fn,
        )
        figs.append(fig)
    return figs
