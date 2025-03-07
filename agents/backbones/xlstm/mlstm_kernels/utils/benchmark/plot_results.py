#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .plot_config import (
    FIGSIZE,
    FIGSIZE_2COL,
    FONTSIZE,
    FONTSIZE_SMALL,
    FONTSIZE_TICKS,
    GRIDSPEC_KWARGS,
    LINEWIDTH,
    MARKERSIZE,
)

fontsize_delta = 0


def rc_context_wrapper(func: Callable, **kwargs):
    with mpl.rc_context(
        rc={
            "text.usetex": False,
            "font.size": FONTSIZE + fontsize_delta,
            "axes.labelsize": FONTSIZE + fontsize_delta,
            "legend.fontsize": FONTSIZE_SMALL + fontsize_delta,
            "xtick.labelsize": FONTSIZE_TICKS + fontsize_delta,
            "ytick.labelsize": FONTSIZE_TICKS + fontsize_delta,
            "axes.titlesize": FONTSIZE + fontsize_delta,
            "lines.markersize": MARKERSIZE,
            "lines.linewidth": LINEWIDTH,
        }
    ):
        return func(**kwargs)


def savefig(fig, filename: str):
    dir = Path("./plots/")
    dir.mkdir(parents=True, exist_ok=True)

    if filename is not None:
        for file_ending in ["png", "pdf", "svg"]:
            file = Path(f"./plots/plot_{filename}.{file_ending}")
            fig.savefig(
                file,
                dpi=300,
                bbox_inches="tight",
            )


def plot_benchmark_result_table(
    result_df: pd.DataFrame,
    x_axis_param: str,
    title: str = None,
    legend_args: dict[str, Any] = dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)),
    legend_order: list[str] = None,
    figsize=FIGSIZE,
    grid_alpha: float = 0.2,
    plot_kwargs: dict[str, Any] = {"marker": "o", "linestyle": "-"},
    style_dict: dict[str, Any] = None,
    style_dict_colname_mapping_exact: bool = True,
    linestyle_mapping: dict[str, Any] = None,
    additional_exclude_col_regex: str = None,
    filename: str = None,
    y_label: str = "Time [ms]",
    x_label: str = None,
    add_legend: bool = True,
    ax: Axes = None,
) -> Figure:
    """Plot benchmark results from a DataFrame as line plot.
    This function is used in the benchmarks for plotting the result raw data.
    It produces similar line plots to `create_runtime_line_plot`. It does not support
    column grouping and also includes saving options for the plot.

    Example usage with rc_context_wrapper:
    ```python
    fig = rc_context_wrapper(
        func=plot_benchmark_result_table,
        result_df=token_per_sec_plot_df,
        x_axis_param="prefill_length",
        # linestyle_mapping=linestyle_mapping,
        style_dict=style_dict,
        style_dict_colname_mapping_exact=False,
        y_label="Tokens per Second",
        title="",
        x_label="Prefill Length",
        figsize=(1.6 * 12 * 1 / 2.54, 1.5 * 8 * 1 / 2.54),
        filename=f"timetofirsttoken_tokens_per_sec{filename_suffix}",
        add_legend=add_legend,
        legend_args={
            "loc": "lower center",
            "ncol": 3,
            "bbox_to_anchor": (0.0, 1.02, 1.0, 0.502),
            "frameon": False,
            "facecolor": "white",
        },
    )
    ```

    Args:
        result_df: DataFrame with benchmark results.
        x_axis_param: Name of the column to use for the x-axis.
        title: Title of the plot. Defaults to None.
        legend_args: Legend arguments. Defaults to dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)).
        legend_order: Order of the legend entries. Defaults to None.
        figsize: Figure size. Defaults to (2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54).
        grid_alpha: Alpha value for the grid. Defaults to 0.2.
        plot_kwargs: Plot arguments. Defaults to {"marker": "o", "linestyle": "-"}.
        style_dict: Style dictionary for the plot. Defaults to None.
        style_dict_colname_mapping_exact: If True, the style_dict is used for exact column names. Defaults to True.
        linestyle_mapping: Mapping for the linestyle. Defaults to None.
        additional_exclude_col_regex: Additional regex to exclude columns. Defaults to None.
        filename: Filename for saving the plot. Defaults to None.
        y_label: Label for the y-axis. Defaults to "Time [ms]".
        x_label: Label for the x-axis. Defaults to None.
        add_legend: If True, the legend is added. Defaults to True.
        ax: Axis for the plot. Defaults to None.

    Returns:
        The figure object.
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    else:
        f = ax.get_figure()
        ax.set_title(title)

    x_axis_vals = result_df[f"P--{x_axis_param}"]

    exclude_regex = "P--.*|Unnamed.*"
    if additional_exclude_col_regex is not None:
        exclude_regex += f"|{additional_exclude_col_regex}"
    y_axis_val_df = result_df.drop(
        result_df.filter(regex=exclude_regex, axis=1).columns, axis=1
    )

    for col in y_axis_val_df.columns:
        plot_kwargs_col = copy.deepcopy(plot_kwargs)
        if style_dict is not None:
            if style_dict_colname_mapping_exact:
                plot_kwargs_col.update(style_dict.get(col, {}))
            else:
                for col_key in style_dict.keys():
                    if col_key in col:
                        plot_kwargs_col.update(style_dict.get(col_key, {}))
        if linestyle_mapping is not None:
            for col_key in linestyle_mapping.keys():
                if col_key in col:
                    plot_kwargs_col.update(linestyle_mapping.get(col_key, {}))
        if "label" in plot_kwargs_col:
            ax.plot(x_axis_vals, y_axis_val_df[col].values, **plot_kwargs_col)
        else:
            ax.plot(
                x_axis_vals, y_axis_val_df[col].values, label=col, **plot_kwargs_col
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_label = x_axis_param if x_label is None else x_label
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if add_legend:
        ax.legend(**legend_args)
    ax.grid(alpha=grid_alpha)

    def savefig(file_ending):
        dir = Path("./plots/")
        dir.mkdir(parents=True, exist_ok=True)
        file = Path(f"./plots/plot_{filename}.{file_ending}")
        f.savefig(
            file,
            dpi=300,
            bbox_inches="tight",
        )

    if filename is not None:
        for file_ending in ["png", "pdf", "svg"]:
            savefig(file_ending)

    return f


def select_columns(df, selected_columns, keep_col_regex: str):
    """Select columns from a DataFrame and keep columns matching a regex."""
    keep_col = df.filter(regex=keep_col_regex)
    selected_df = df[selected_columns.values()]
    selected_df.columns = selected_columns.keys()
    selected_df = pd.concat([keep_col, selected_df], axis=1)
    return selected_df


def create_group_names_from_cols(
    data_df: pd.DataFrame, colnames: str, add_colname: bool = False
) -> list[str]:
    """Create group names from columns in a DataFrame."""
    group_names = []
    group_cols = data_df[colnames].astype(int)
    for i, row in group_cols.iterrows():
        group_str = ""
        for i, colname in enumerate(colnames):
            if add_colname:
                group_str += f"{colname}={row[colname]}"
            else:
                group_str += f"{row[colname]}"
            if i < len(colnames) - 1:
                group_str += "\n"
        group_names.append(group_str)
    return group_names


def create_bar_length_df(
    data_df: pd.DataFrame, colnames: list[str], offset: float = 0.0
) -> pd.DataFrame:
    """Create a DataFrame with bar lengths for a bar plot."""
    # subtract the min from selected columns respectively and add an arbitrary offset
    bar_length_df = data_df[colnames].sub(data_df[colnames].min()).add(offset)
    bar_length_data_df = data_df.copy()
    bar_length_data_df[colnames] = bar_length_df
    return bar_length_data_df


def create_runtime_bar_plot(
    data_df: pd.DataFrame,
    group_col_names: list[str],
    title: str = None,
    bar_label_font_size: int = 10,
    bar_length_df: pd.DataFrame | None = None,
    plot_column_order: list[str] = None,
    style_dict: dict[str, Any] = None,
    fillna_val: float = -0.2,
    fillna_exclude_cols: list[str] = None,
    fillna_str: str = "NA",
    legend_args: dict[str, Any] = dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)),
    legend_order: list[str] = None,
    figsize=FIGSIZE,
    grid_alpha: float = 0.2,
    yticks: list[float] = None,
    ylim: tuple[float, float] | None = None,
    y_label: str | None = None,
    x_label: str = "Sequence Length",
    ax=None,
    add_colname: bool = True,
) -> Figure:
    """Create a bar plot for runtime results.

    Example usage with rc_context_wrapper:
    ```python
    fig = rc_context_wrapper(
        func=create_runtime_bar_plot,
        data_df=plot_throughput_df,
        group_col_names=["BS", "CTX"],
        style_dict=style_dict,
        figsize=(2 * 12 * 1 / 2.54, 1.5 * 8 * 1 / 2.54),
        y_label="Tokens per Second",
        legend_args={
            "loc": "lower center",
            "ncol": 3,
            "bbox_to_anchor": (0.0, 1.02, 1.0, 0.502),
            "frameon": False,
            "facecolor": "white",
        },
    )
    ```

    Args:
        data_df: DataFrame with the data to plot.
        group_col_names: List of column names to group the bars by.
                         The group names must be columns in the dataframe and are added as x-axis labels.
        title: Title of the plot. Defaults to None.
        bar_label_font_size: Font size for the bar labels. Defaults to 10.
        bar_length_df: DataFrame with the bar lengths. Defaults to None.
        plot_column_order: Order of the columns to plot. Defaults to None.
        style_dict: Style dictionary for the plot. Defaults to None.
        fillna_val: Value to fill NaN values with. Defaults to -0.2.
        fillna_exclude_cols: Columns to exclude from fillna. Defaults to None.
        fillna_str: String to fill NaN values with. Defaults to "NA".
        legend_args: Legend arguments. Defaults to dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)).
        legend_order: Order of the legend entries. Defaults to None.
        figsize: Figure size. Defaults to FIGSIZE.
        grid_alpha: Alpha value for the grid. Defaults to 0.2.
        yticks: Y-ticks. Defaults to None.
        ylim: Y-limits. Defaults to None.
        y_label: Label for the y-axis. Defaults to None.
        x_label: Label for the x-axis. Defaults to "Sequence Length".

    Returns:
        The figure object.
    """

    group_names = create_group_names_from_cols(
        data_df=data_df,
        colnames=group_col_names,
        add_colname=add_colname,
    )
    raw_data_df = data_df.drop(columns=group_col_names)
    # data df contains only the columns to plot
    if fillna_exclude_cols is not None:
        raw_data_nan_cols_df = raw_data_df[fillna_exclude_cols].round(2)
    else:
        fillna_exclude_cols = []
        raw_data_nan_cols_df = pd.DataFrame()
    raw_data_nonan_cols_df = raw_data_df.drop(columns=fillna_exclude_cols)
    if bar_length_df is None:
        bar_length_df = raw_data_nonan_cols_df.fillna(fillna_val).round(2)
    else:
        bar_length_df = bar_length_df.drop(columns=fillna_exclude_cols)
        bar_length_df = (
            bar_length_df.drop(columns=group_col_names).fillna(fillna_val).round(2)
        )
    raw_data_nonan_cols_df = raw_data_nonan_cols_df.round(2).fillna(fillna_str)

    if fillna_exclude_cols is not None:
        raw_data_df = pd.concat([raw_data_nonan_cols_df, raw_data_nan_cols_df], axis=1)
        bar_length_df = pd.concat([bar_length_df, raw_data_nan_cols_df], axis=1)
    else:
        raw_data_df = raw_data_nonan_cols_df

    if plot_column_order is not None:
        columns = plot_column_order
    else:
        columns = bar_length_df.columns

    # x-axis locations
    x = np.arange(len(raw_data_df))
    width = 0.9 / (len(columns) + 1)
    # align labels to bars
    multiplier = -len(raw_data_df) / 2 if x_label == "Sequence Length" else 0

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    else:
        f = ax.get_figure()
        ax.set_title(title)

    for col in columns:
        offset = width * multiplier
        if style_dict is None:
            rects = ax.bar(x + offset, bar_length_df[col], width, label=col)
        else:
            rects = ax.bar(x + offset, bar_length_df[col], width, **style_dict[col])
        ax.bar_label(rects, labels=raw_data_df[col], fontsize=bar_label_font_size)
        multiplier += 1

    if ylim:
        ax.set_ylim(ylim)

    ax.set_ylabel("Time [ms]" if y_label is None else y_label)
    ax.set_xlabel(x_label)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xticks(x + width, group_names)
    if legend_args and legend_order is None:
        ax.legend(**legend_args)
    elif legend_args and legend_order is not None:
        handles, labels = ax.get_legend_handles_labels()
        label_handle_dict = dict(zip(labels, handles))
        handles = [label_handle_dict[label] for label in legend_order]
        ax.legend(handles=handles, **legend_args)
    ax.grid(alpha=grid_alpha, which="both")

    if yticks is not None:
        ax.set_yticks(yticks)
        y_formatter = plt.ScalarFormatter()
        # y_formatter.set_scientific(False)
        # y_formatter.set_useOffset(10.0)
        ax.get_yaxis().set_major_formatter(y_formatter)

    return f


def create_runtime_line_plot(
    data_df: pd.DataFrame,
    group_col_names: list[str],
    title: str = None,
    plot_column_order: list[str] = None,
    style_dict: dict[str, Any] = None,
    legend_args: dict[str, Any] = dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)),
    legend_order: list[str] = None,
    figsize=FIGSIZE,
    grid_alpha: float = 0.2,
    yticks: list[float] = None,
    ylim: tuple[float, float] | None = None,
    x_label: str = "Sequence Length",
    ax: Axes = None,
    add_colname: bool = False,
):
    """Create a line plot for runtime results.
    Simliar to `create_runtime_bar_plot`, but creates a line plot instead of a bar plot.

    Args:
        data_df: DataFrame with the data to plot.
        group_col_names: List of column names to group the bars by.
                         The group names must be columns in the dataframe and are added as x-axis labels.
        title: Title of the plot. Defaults to None.
        plot_column_order: Order of the columns to plot. Defaults to None.
        style_dict: Style dictionary for the plot. Defaults to None.
        legend_args: Legend arguments. Defaults to dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)).
        legend_order: Order of the legend entries. Defaults to None.
        figsize: Figure size. Defaults to FIGSIZE.
        grid_alpha: Alpha value for the grid. Defaults to 0.2.
        yticks: Y-ticks. Defaults to None.
        ylim: Y-limits. Defaults to None.
        x_label: Label for the x-axis. Defaults to "Sequence Length".
        ax: Axis for the plot. Defaults to None.
        add_colname: If True, the column name is added to the group names. Defaults to False.

    Returns:
        The figure object.

    """

    group_names = create_group_names_from_cols(
        data_df=data_df, colnames=group_col_names, add_colname=add_colname
    )
    raw_data_df = data_df.drop(columns=group_col_names)

    # x-axis locations
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    else:
        f = ax.get_figure()
        ax.set_title(title)

    if plot_column_order is not None:
        columns = plot_column_order
    else:
        columns = raw_data_df.columns

    for col in columns:
        if style_dict is None:
            ax.plot(range(len(raw_data_df)), raw_data_df[col], label=col, marker="s")
        else:
            ax.plot(
                range(len(raw_data_df)), raw_data_df[col], marker="s", **style_dict[col]
            )

    ax.set_ylabel("Time [ms]")
    ax.set_xlabel(x_label)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xticks(range(len(raw_data_df)), group_names)
    if legend_args and legend_order is None:
        ax.legend(**legend_args)
    elif legend_args and legend_order is not None:
        handles, labels = ax.get_legend_handles_labels()
        label_handle_dict = dict(zip(labels, handles))
        handles = [label_handle_dict[label] for label in legend_order]
        ax.legend(handles=handles, **legend_args)
    ax.grid(alpha=grid_alpha, which="both")

    if yticks is not None:
        ax.set_yticks(yticks)
        # y_formatter.set_scientific(False)
        # y_formatter.set_useOffset(10.0)

    return f


def plot_runtime_results(
    data_df: pd.DataFrame,
    plot_column_order: list[str],
    group_cols: list[str],
    yticks: list[float],
    bar_label_fontsize: int = 9,
    filename: str = None,
    fillna_exclude_cols: list[str] = None,
    style_dict: dict[str, Any] = None,
    legend_args: dict[str, Any] = dict(
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.0, 0.97, 1.0, 0.102),
        frameon=False,
        facecolor="white",
    ),
    plot_type: Literal["line", "bar"] = "bar",
    ylim: tuple[float, float] | None = None,
    x_label: str = "Sequence Length",
    legend_order: list[str] = None,
    figsize: tuple[float, float] = None,
    ax: Axes = None,
    add_colname: bool = True,
) -> Figure:
    """Plot runtime results from a DataFrame.
    Convenience function to create a bar or line plot for runtime results.
    Also sets the matplotlib rc parameters for the plot.

    Args:
        data_df: DataFrame with the data to plot.
        plot_column_order: Order of the columns to plot.
        group_cols: List of column names to group the bars by.
                    The group names must be columns in the dataframe and are added as x-axis labels.
        yticks: Y-ticks.
        bar_label_fontsize: Font size for the bar labels. Defaults to 9.
        filename: Filename for saving the plot. Defaults to None.
        fillna_exclude_cols: Columns to exclude from fillna. Defaults to None.
        style_dict: Style dictionary for the plot. Defaults to None.
        legend_args: Legend arguments. Defaults to dict(loc="lower center", ncol=4, bbox_to_anchor=(0.0, 0.97, 1.0, 0.102), frameon=False, facecolor="white").
        plot_type: Type of the plot. Defaults to "bar".
        ylim: Y-limits. Defaults to None.
        x_label: Label for the x-axis. Defaults to "Sequence Length".
        legend_order: Order of the legend entries. Defaults to None.
        figsize: Figure size. Defaults to None.
        ax: Axis for the plot. Defaults to None.
        add_colname: If True, the column name is added to the group names. Defaults to True.

    Returns:
        The figure object.
    """

    with mpl.rc_context(
        rc={
            "text.usetex": False,
            "font.size": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "legend.fontsize": FONTSIZE_SMALL,
            "xtick.labelsize": FONTSIZE_TICKS,
            "ytick.labelsize": FONTSIZE_TICKS,
            "axes.titlesize": FONTSIZE,
            "lines.markersize": MARKERSIZE,
            "lines.linewidth": LINEWIDTH,
        }
    ):
        if plot_type == "bar":
            f = create_runtime_bar_plot(
                data_df=data_df,
                bar_length_df=None,
                bar_label_font_size=bar_label_fontsize,
                group_col_names=group_cols,
                plot_column_order=plot_column_order,
                style_dict=style_dict,
                legend_args=legend_args,  # {"loc": "upper right", "bbox_to_anchor": (1.1, 1.0)},
                legend_order=legend_order,
                yticks=yticks,
                figsize=FIGSIZE if figsize is None else figsize,
                fillna_val=-10,
                fillna_exclude_cols=fillna_exclude_cols,
                ylim=ylim,
                x_label=x_label,
                ax=ax,
                add_colname=add_colname,
            )
        else:
            f = create_runtime_line_plot(
                data_df=data_df,
                group_col_names=group_cols,
                plot_column_order=plot_column_order,
                style_dict=style_dict,
                legend_args=legend_args,  # {"loc": "upper right", "bbox_to_anchor": (1.1, 1.0)},
                legend_order=legend_order,
                yticks=yticks,
                figsize=FIGSIZE if figsize is None else figsize,
                ylim=ylim,
                x_label=x_label,
                ax=ax,
                add_colname=add_colname,
            )
        if filename is not None:
            savefig(f, filename)

        return f


def plot_runtime_results_fwbw(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    col_order_left: list[str] = None,
    col_order_right: list[str] = None,
    yticks_left: list[float] = [],
    yticks_right: list[float] = [],
    group_cols: list[str] = [],
    filename_wo_ending: str = "",
    style_dict: dict[str, Any] = None,
    legend_args: dict[str, Any] = {
        "loc": "lower center",
        "ncol": 3,
        "bbox_to_anchor": (0.0, 0.97, 1.0, 0.102),
        "frameon": False,
        "facecolor": "white",
    },
    modify_df_func=None,
    plot_type: Literal["line", "bar"] = "bar",
    ylim_left: tuple[float, float] | None = None,
    ylim_right: tuple[float, float] | None = None,
    fillna_exclude_cols_left: list[str] = None,
    fillna_exclude_cols_right: list[str] = None,
    x_label: str = "Sequence Length",
    add_colname: bool = True,
) -> Figure:
    """Similar to `plot_runtime_results`, but plots two figures side by side.
    
    Args:
        df_left: DataFrame with the data to plot on the left side.
        df_right: DataFrame with the data to plot on the right side.
        col_order_left: Order of the columns to plot on the left side. Defaults to None.
        col_order_right: Order of the columns to plot on the right side. Defaults to None.
        yticks_left: Y-ticks for the left side. Defaults to [].
        yticks_right: Y-ticks for the right side. Defaults to [].
        group_cols: List of column names to group the bars by. Defaults to [].
        filename_wo_ending: Filename for saving the plot. Defaults to "".
        style_dict: Style dictionary for the plot. Defaults to None.
        legend_args: Legend arguments. Defaults to dict(loc="lower center", ncol=3, bbox_to_anchor=(0.0, 0.97, 1.0, 0.102), frameon=False, facecolor="white").
        modify_df_func: Function to modify the dataframes before plotting. Defaults to None.
        plot_type: Type of the plot. Defaults to "bar".
        ylim_left: Y-limits for the left side. Defaults to None.
        ylim_right: Y-limits for the right side. Defaults to None.
        fillna_exclude_cols_left: Columns to exclude from fillna on the left side. Defaults to None.
        fillna_exclude_cols_right: Columns to exclude from fillna on the right side. Defaults to None.
        x_label: Label for the x-axis. Defaults to "Sequence Length".
        add_colname: If True, the column name is added to the group names. Defaults to True.

    Returns:
        The figure object.    
    """

    f, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=FIGSIZE_2COL, gridspec_kw=GRIDSPEC_KWARGS
    )

    if modify_df_func is not None:
        df_left = modify_df_func(df_left)
        df_right = modify_df_func(df_right)

    f = plot_runtime_results(
        data_df=df_left,
        group_cols=group_cols,
        yticks=yticks_left,
        plot_column_order=col_order_left,
        legend_args=legend_args,
        style_dict=style_dict,
        fillna_exclude_cols=fillna_exclude_cols_left,
        ax=ax_left,
        plot_type=plot_type,
        ylim=ylim_left,
        x_label=x_label,
        add_colname=add_colname,
    )
    f = plot_runtime_results(
        data_df=df_right,
        group_cols=group_cols,
        yticks=yticks_right,
        plot_column_order=col_order_right,
        legend_args=legend_args,
        style_dict=style_dict,
        fillna_exclude_cols=fillna_exclude_cols_right,
        plot_type=plot_type,
        ylim=ylim_right,
        ax=ax_right,
        x_label=x_label,
        add_colname=add_colname,
    )
    savefig(f, filename=filename_wo_ending)
    return f
