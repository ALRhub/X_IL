#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..plot import (
    compute_errors_per_batchhead,
    plot_error_statistics_over_time_per_batchhead,
    plot_numerical_diffs_per_batchhead,
)
from ..plot.diff_imshow import convert_to_diff_imarray_numpy

LOGGER = logging.getLogger(__name__)


def check_correctness(
    test_specifier: str,
    baseline: np.ndarray,
    target: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    vmax: float = None,
    max_num_batchhead_plots: int = -1,
    percentiles: list = [50, 90],
    savepath: str = None,
    dtype_str: str = "NAtype",
) -> bool:
    assert isinstance(baseline, np.ndarray)
    assert isinstance(target, np.ndarray)

    assert baseline.shape == target.shape

    # closeness in highest precision
    baseline = baseline.astype(np.float64)
    target = target.astype(np.float64)

    result = np.allclose(baseline, target, atol=atol, rtol=rtol)

    errors = np.abs(baseline - target)
    rel_errors = np.abs(baseline - target) / (np.abs(baseline) + 1e-6)

    error_percentiles = np.percentile(errors, percentiles)

    def make_percentile_str(error_percentiles: np.ndarray, percentiles: int) -> str:
        percentile_str = ""
        for i, percentile in enumerate(percentiles):
            percentile_str += f"p{percentile:<3}: {error_percentiles[i]:>5.5e}"
            if i < len(percentiles) - 1:
                percentile_str += "|"
        return percentile_str

    # title = f"{test_specifier:>20}|{dtype_str:>6}| max diff: {(baseline - target).abs().max():>25}| mean diff: {(baseline - target).abs().mean():25} | allclose(atol={atol},rtol={rtol}): {result}"
    title = f"{test_specifier:>20}|{dtype_str:>6}| diff: {make_percentile_str(error_percentiles, percentiles):>35} | maxreldiff: {np.max(rel_errors):25} | maxdiff: {np.max(errors):25} | meandiff: {np.mean(errors):25} | allclose(atol={atol},rtol={rtol}): {result} | max abs bl: {np.max(np.abs(baseline)):7.6f} | max abs tg: {np.max(np.abs(target)):.5}"

    print(title)
    LOGGER.info(title)
    if savepath is not None:
        if vmax is None:
            vmax = atol
        savepath = Path(savepath) / f"{test_specifier}--{dtype_str}"
        savepath.mkdir(parents=True, exist_ok=True)

        figs = plot_numerical_diffs_per_batchhead(
            baseline=baseline,
            target=target,
            title=f"{test_specifier}|{dtype_str}",
            vmin=0,
            vmax=vmax,
            rtol=rtol,
            atol=atol,
            max_num_batchhead_plots=max_num_batchhead_plots,
            convert_to_diff_imarray_fn=convert_to_diff_imarray_numpy,
        )
        for i, fig in enumerate(figs):
            fig.savefig(savepath / f"diff_imshow--batchhead_{i}.pdf")
            plt.close()

        figs = plot_error_statistics_over_time_per_batchhead(
            errors=compute_errors_per_batchhead(baseline=baseline, target=target),
            percentiles=[50, 90, 100],
            add_mean=True,
            title=f"{test_specifier}|{dtype_str}",
            ema_alpha=0.02,
            max_num_batchhead_plots=max_num_batchhead_plots,
        )

        for i, fig in enumerate(figs):
            fig.savefig(savepath / f"diff_lineplot--batchhead_{i}.pdf")
            plt.close()

    return result


def verify_output(
    name: str,
    baseline: np.ndarray,
    target: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    vmax: float = 1e-2,
):
    check_correctness(name, baseline=baseline, target=target, rtol=rtol, atol=atol)
    fig = plot_numerical_diffs_per_batchhead(
        baseline=baseline,
        target=target,
        title=name,
        rtol=rtol,
        atol=atol,
        vmax=vmax,
        convert_to_diff_imarray_fn=convert_to_diff_imarray_numpy,
    )
    return fig
