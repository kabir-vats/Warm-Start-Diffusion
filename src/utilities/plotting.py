from __future__ import annotations

import itertools
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import xarray as xr

from src.optimization.losses import crps_ensemble
from src.utilities.naming import (
    clean_metric_name,
    get_label_names_for_wandb_group_names,
    normalize_run_name,
)
from src.utilities.wandb_api import (
    get_runs_for_group_with_any_metric,
    has_summary_metric,
    metrics_of_runs_to_arrays,
)


FIXED_CMAPS = {
    "UnetResNet": "Greys",
    "DDPM": "Blues",
    "MixUp": "Oranges",
    "MixUpPretrainedInterpolator": "Oranges",
    "Deblurring": "Greens",
}
FIXED_LINE_STYLES = {
    "UnetResNet": "-",
    "SimMultiHorizon7-DDPM": "-",
    "DDPM": "--",
    "MixUp": "-",
    "MixUpPretrainedInterpolator": "--",
    "DY2s-v1": "-",
    "MCVD": "-.",
    "Deblurring": "-",
    "h7": "--",
}
FIXED_LINE_WIDTHS = {
    "MCVD": 2,
}
for key in list(FIXED_CMAPS.keys()):
    FIXED_CMAPS["SimMultiHorizon7-" + key] = FIXED_CMAPS["MultiHorizon7-" + key] = FIXED_CMAPS[key]
FIXED_CMAPS["SimMultiHorizon7"] = FIXED_CMAPS["MultiHorizon7"] = FIXED_CMAPS["UnetResNet"]

log = logging.getLogger(__name__)


def get_marker_kwargs(wandb_group: str):
    if "MCVD" in wandb_group:
        marker = "o"
    elif "DDPM" in wandb_group:
        marker = "v"
    elif "MixUp" in wandb_group or "DY2S" in wandb_group:
        marker = "x"
    else:
        marker = None
    return {"marker": marker, "markersize": 8}


def get_fig_and_axes(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[float, float] = (10, 10),
    to_2d_array: bool = False,
    hspace: float = None,
    wspace: float = None,
    **kwargs,
) -> (plt.Figure, np.ndarray):
    if hspace is not None or wspace is not None:
        assert "gridspec_kw" not in kwargs, "Cannot use gridspec_kw with hspace or wspace"
        kwargs["gridspec_kw"] = {"hspace": hspace, "wspace": wspace}
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [axs] if not isinstance(axs, np.ndarray) else axs
    if to_2d_array:
        axs = np.array(axs).reshape(nrows, ncols)
    return fig, axs


def set_wb2_style() -> None:
    """Changes MPL defaults to WB2 style."""
    plt.rcParams["axes.grid"] = True
    # plt.rcParams['lines.linewidth'] = 2
    plt.rcParams["figure.facecolor"] = "None"
    plt.rcParams["axes.facecolor"] = "0.95"
    plt.rcParams["grid.color"] = "white"
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams.update({"axes.labelsize": 16})
    plt.rcParams["hatch.linewidth"] = 0.5  # previous pdf hatch linewidth
    plt.rcParams["lines.markersize"] = 10


def plot_shaded_std_for_xarray_mean(da, stats_dim: str = "batch", alpha=0.3, **plot_kwargs):
    """Plots the mean of a xarray.DataArray with the standard deviation shaded.
    Args:
        da: xarray.DataArray with dimensions (batch, time, ...)
        stats_dim: dimension to compute the mean and std over
        alpha: alpha value for the shaded area
        **plot_kwargs: passed to plt.plot
    """
    ax_plt = plot_kwargs.get("ax", plt)
    mean, std = da.mean(dim=stats_dim), da.std(dim=stats_dim)
    mean.plot.line(**plot_kwargs)
    xaxis = mean.coords[mean.dims[0]].values
    mean, std = mean.values, std.values
    ax_plt.fill_between(xaxis, mean - std, mean + std, alpha=alpha)


def plot_avg_and_std(
    data_arrays: Sequence[np.ndarray],
    x_axis: Sequence[float],
    axes: Sequence[plt.Axes],
    label: str = None,
    plot_shaded_std: bool = True,
    shaded_std_alpha: float = 0.2,
    plot_runs_separately: bool = False,
    label_suffix_format: str | None = None,
    plot_kwargs: Dict[str, Any] = None,
) -> (List[float], List[float]):
    """Plots the mean and standard deviation of a sequence of numpy arrays

    Args:
        data_arrays (Sequence[np.ndarray]): The data arrays to plot, with shape (num_samples, num_xticks).
        x_axis (Sequence[float]): The x-axis values, with shape (num_xticks,).
        axes (Sequence[plt.Axes]): The axes to plot on.
        label (str): The label for the lines to plot.
        plot_shaded_std (bool): Whether to plot the standard deviation as a shaded area.
        shaded_std_alpha (float): The alpha value for the shaded standard deviation area.
        plot_runs_separately (bool): Whether to plot each run separately (no standard deviation).
        label_suffix_format (str | None): The format string for the label suffix, e.g. 'mean+std'
        plot_kwargs (Dict[str, Any]): The plot kwargs.

    Returns:
        (List[float], List[float]): The scalar means and standard deviations, i.e. one (mean, std) for each data array.
    """
    assert len(data_arrays) == len(axes), f"len(data_arrays) = {len(data_arrays)}, len(axes) = {len(axes)}"
    if plot_runs_separately:
        n_runs = data_arrays[0].shape[0]
        for i in range(n_runs):
            only_run_i = [data_array[i : i + 1] for data_array in data_arrays]
            plot_avg_and_std(
                only_run_i,
                x_axis,
                axes,
                label,
                plot_shaded_std,
                shaded_std_alpha,
                False,
                label_suffix_format,
                plot_kwargs,
            )
        return
    plot_kwargs = plot_kwargs or {}
    need_separate_labels = len(axes) > 1 and label_suffix_format is not None
    scalar_means, scalar_stds = [], []
    for i, (data_array, ax) in enumerate(zip(data_arrays, axes)):
        mean = np.nanmean(data_array, axis=0)  # use nanmean?
        std = np.nanstd(data_array, axis=0)  # use nanstd?

        label_ax = label if (i == 0 or need_separate_labels) else None
        scalar_mean = np.mean(mean)
        scalar_std = np.mean(std)
        scalar_means.append(float(scalar_mean))
        scalar_stds.append(float(scalar_std))

        if label_ax is not None and label_suffix_format is not None:
            if label_suffix_format == "mean+std" and scalar_std > 0:
                label_ax = f"{label_ax} ({scalar_mean:.3f} ± {scalar_std:.2f})"
            elif "mean" in label_suffix_format:
                label_ax = f"{label_ax} (Avg={scalar_mean:.4f})"
            elif label_suffix_format == "min":
                # get min value and corresponding x-axis value
                min_idx = np.nanargmin(mean)
                min_value, x_for_min_value = mean[min_idx], x_axis[min_idx]
                label_ax = f"{label_ax} (Min={min_value:.3f} @ {x_for_min_value})"
            elif label_suffix_format == "max":
                # get max value and corresponding x-axis value
                max_idx = np.nanargmax(mean)
                max_value, x_for_max_value = mean[max_idx], x_axis[max_idx]
                label_ax = f"{label_ax} (Max={max_value:.3f} @ {x_for_max_value})"

        # ax.plot(x_axis, mean, label=label, **plot_kwargs)
        ax.errorbar(x_axis, mean, yerr=std, label=label_ax, **plot_kwargs)

        if plot_shaded_std:
            shaded_std_kwargs = {k: v for k, v in plot_kwargs.items() if k not in ("label", "marker", "markersize")}
            ax.fill_between(
                x_axis,
                mean - std,
                mean + std,
                alpha=shaded_std_alpha,
                **shaded_std_kwargs,
            )
    return scalar_means, scalar_stds


def plot_ensemble_performance_vs_noise(
    wandb_groups: Sequence[str],
    labels: Sequence[str] = None,
    ensemble_size: int = 50,
    split: str = "test/2020",
    noise_levels: List[float] = None,
    skip_noise_levels: List[float] = None,
    metrics: List[str] = ("mse", "crps", "corr"),
    metrics_prefix: str = "avg/",
    figsize: Tuple[int, int] = (20, 5),
    n_labels_per_row: int = None,
    ylims: Dict[str, Tuple[float, float]] = None,
    plot_shaded_std: bool = True,
    add_num_runs_to_label: bool = True,
    remove_lr_from_label: bool = False,
    wandb_api=None,
    wandb_kwargs: Dict[str, Any] = None,
    format_kwargs: Dict[str, Any] = None,
):
    """
    Args:
        wandb_groups (Sequence[str]): A list of wandb group names.
        labels (Sequence[str]): A list of labels for the wandb groups. If None, the wandb group names are used.
        ensemble_size (int): The ensemble size (how many samples to draw from the model).
        split (str): The split to plot. One of 'val' or 'test/2020/88'.
        noise_levels (List[float]): The noise levels to plot. If None, all noise levels are plotted.
        skip_noise_levels (List[float]): The noise levels to skip. If None, no noise levels are skipped.
        metrics (List[str]): The metrics to plot. Any combination of 'mse', 'crps', 'corr'.
        metrics_prefix (str): The prefix to use for the metrics. E.g. '' or 't7/'.
        figsize (Tuple[int, int]): The figure size.
        n_labels_per_row (int): The number of labels per row in the legend.
        ylims (Dict[str, Tuple[float, float]]): The y-limits for each metric, e.g. {'mse': (0, 0.1)}.
        plot_shaded_std (bool): Whether to plot the standard deviation as a shaded area.
        add_num_runs_to_label (bool): Whether to add the number of runs to the label.
        wandb_api (wandb.Api): The wandb API.
        wandb_kwargs (Dict[str, Any]): The kwargs to pass to wandb.Api.runs().

    """
    wandb_api = wandb_api or wandb.Api(timeout=60)
    noise_levels = noise_levels or [
        1e-4,
        1e-3,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.75,
        0.9,
        1.0,
    ]
    if skip_noise_levels is not None:
        noise_levels = [nl for nl in noise_levels if nl not in skip_noise_levels]
    noise_levels = sorted(noise_levels)

    wandb_kwargs = wandb_kwargs or {}
    format_kwargs = format_kwargs or {}
    metrics_prefix = f"{metrics_prefix}/".replace("//", "/") if metrics_prefix else ""

    n_labels_per_row = n_labels_per_row or (5 if labels is None else 8)
    wandb_groups, labels = get_label_names_for_wandb_group_names(
        wandb_groups, labels, remove_lr_from_label=remove_lr_from_label
    )
    fig, axs = get_fig_and_axes(nrows=1, ncols=len(metrics), figsize=figsize)
    formats = RollingPlotFormats(all_keys=wandb_groups, **format_kwargs)  # base_keys_sep='_',

    any_noise_key = f"{split}/{noise_levels[0]}eps/{ensemble_size}ens_mems/{metrics_prefix}crps"
    if "filter_functions" not in wandb_kwargs:
        wandb_kwargs["filter_functions"] = has_summary_metric(any_noise_key)

    for g_i, (group, label) in enumerate(zip(wandb_groups, labels)):
        group_runs = get_runs_for_group_with_any_metric(group, wandb_api=wandb_api, **wandb_kwargs)

        if len(group_runs) == 0:
            logging.warning(f"No runs found for group {group} and key {any_noise_key}! Skipping...")
            continue

        plt_kwargs = formats[group] if len(formats.unique_base_keys) > 1 else dict()  # marker='x',
        if add_num_runs_to_label:
            label = f"{label} (#{len(group_runs)})"

        group_metrics = metrics_of_runs_to_arrays(
            group_runs,
            metrics,
            columns=noise_levels,
            column_to_wandb_key=lambda e: f"{split}/{e}eps/{ensemble_size}ens_mems/{metrics_prefix}",
        )
        plot_avg_and_std(
            list(group_metrics.values()),
            x_axis=noise_levels,
            label=label,
            axes=axs,
            plot_shaded_std=plot_shaded_std,
            label_suffix_format=None,
            plot_kwargs=plt_kwargs,
        )
    beautify_plots_with_metrics(
        axs,
        metrics,
        xlabel="Noise level",
        ylabel_prefix=metrics_prefix,
        ylabel_prefix_left_ax=split,
        ylims=ylims,
        n_labels_per_row=n_labels_per_row,
    )
    return fig, axs


def plot_ensemble_performance_vs_horizon(
    wandb_groups: Sequence[str] | Dict[str, str],
    labels: Sequence[str] = None,
    ensemble_size: int = 50,
    split: str = "test/2020/88",
    noise_level: float = 0.0,
    metrics: List[str] = ("mse", "crps"),
    horizon_ticks: List[int] = None,
    figsize: Tuple[int, int] = (20, 5),
    n_labels_per_row: int = None,
    xlabel: str = "Horizon",
    ylims: Dict[str, Tuple[float, float]] = None,
    metrics_in_cols: bool = True,
    plot_shaded_std: bool = True,
    label_suffix_format: str | None = "mean+std",
    add_num_runs_to_label: bool = True,
    remove_lr_from_label: bool = False,
    plot_runs_separately: bool = False,
    wandb_api=None,
    wandb_kwargs: Dict[str, Any] = None,
    format_kwargs: Dict[str, Any] = None,
    legend_kwargs: Dict[str, Any] = None,
):
    """
    Args:
        wandb_groups (Sequence[str]): A list of wandb group names.
        labels (Sequence[str]): A list of labels for the wandb groups. If None, the wandb group names are used.
        ensemble_size (int): The ensemble size (how many samples to draw from the model).
        split (str): The split to plot. One of 'val' or 'test/2020/88'.
        noise_level (float): The noise level to plot (usually 0 or 1e-4).
        metrics (List[str]): The metrics to plot. Any combination of 'mse', 'crps', 'corr'.
        horizon_ticks (List[int]): The horizon ticks to plot. If None, all horizons are plotted.
        figsize (Tuple[int, int]): The figure size.
        n_labels_per_row (int): The number of labels per row in the legend.
        ylims (Dict[str, Tuple[float, float]]): The y-limits for each metric, e.g. {'mse': (0, 0.1)}.
        metrics_in_cols (bool): Whether to plot the metrics in columns (True) or rows (False).
        plot_shaded_std (bool): Whether to plot the standard deviation as a shaded area.
        label_suffix_format (str): The format for the label suffix. One of 'mean+std', 'mean', 'min'.
        add_num_runs_to_label (bool): Whether to add the number of runs to the label.
        wandb_api (wandb.Api): The wandb API.
        wandb_kwargs (Dict[str, Any]): The kwargs to pass to wandb.Api.runs().

    """
    wandb_api = wandb_api or wandb.Api(timeout=60)
    wandb_kwargs = wandb_kwargs or {}
    format_kwargs = format_kwargs or {}
    legend_kwargs = legend_kwargs or {}

    n_labels_per_row = n_labels_per_row or (5 if labels is None else 8)
    if isinstance(wandb_groups, dict):
        wandb_groups, labels = wandb_groups.values(), wandb_groups.keys()
    wandb_groups, labels = get_label_names_for_wandb_group_names(
        wandb_groups,
        labels,
        remove_lr_from_label=remove_lr_from_label,
        remove_weight_decay_from_label=True,
    )
    nrows, ncols = (1, len(metrics)) if metrics_in_cols else (len(metrics), 1)
    fig, axs = get_fig_and_axes(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=False)
    formats = RollingPlotFormats(all_keys=wandb_groups, **format_kwargs)  # base_keys_sep='_',

    any_group = get_runs_for_group(wandb_groups[0], wandb_api=wandb_api, **wandb_kwargs)
    if len(any_group) == 0:
        raise ValueError(f"No runs found for group {wandb_groups[0]}, wandb_kwargs={wandb_kwargs}")
    if noise_level == 0.0 and any(f"{split}/avg/{ensemble_size}ens_mems/crps" in r.summary.keys() for r in any_group):

        def split_to_any_metric_stem(spl):
            return f"{spl}/avg/{ensemble_size}ens_mems"

    else:

        def split_to_any_metric_stem(spl):
            return f"{spl}/avg/{noise_level}eps/{ensemble_size}ens_mems"

    for g_i, (group, label) in enumerate(zip(wandb_groups, labels)):
        group_runs, any_metric_stem = get_runs_for_group_with_any_metric(
            group, split, split_to_any_metric_stem, wandb_api, **wandb_kwargs
        )
        if group_runs is None:
            continue

        plt_kwargs = formats[group] if len(formats.unique_base_keys) > 1 else dict()  # {}
        if add_num_runs_to_label:
            label = f"{label} (#{len(group_runs)})" if len(group_runs) > 1 else f"{group_runs[0].id}: {label}"

        group_horizon = group_runs[0].config.get("datamodule/horizon")
        if horizon_ticks is not None:
            horizons = horizon_ticks
        elif "InterpolationExperiment".lower() in group_runs[0].config.get("exp/_target_", "").lower():
            horizons = list(range(1, group_horizon))
        else:
            horizons = list(range(1, group_horizon + 1))

        group_metrics = metrics_of_runs_to_arrays(
            group_runs,
            metrics,
            columns=horizons,
            column_to_wandb_key=lambda h: (
                any_metric_stem.replace("avg", f"t{h}") if h is not None else any_metric_stem.replace("/avg", "")
            ),
        )
        plot_avg_and_std(
            list(group_metrics.values()),
            x_axis=horizons,
            label=label,
            axes=axs,
            plot_shaded_std=plot_shaded_std,
            label_suffix_format=label_suffix_format,
            plot_kwargs=plt_kwargs,
            plot_runs_separately=plot_runs_separately,
        )
    beautify_plots_with_metrics(
        axs,
        metrics,
        xlabel=xlabel,
        ylabel_prefix_left_ax=split if isinstance(split, str) else split[0],
        ylims=ylims,
        n_labels_per_row=n_labels_per_row,
        **legend_kwargs,
    )
    return fig, axs


def plot_sampling_ablation_vs_horizon(
    wandb_group: str,
    title: str = "",
    sampling_keys: Sequence[str] = None,
    ensemble_size: int = 50,
    split: str = "test/2020/88",
    noise_level: float = 0.0,
    metrics: List[str] = ("mse", "crps"),
    horizon_ticks: Sequence[int] = None,  # None -> all horizons
    figsize: Tuple[int, int] = (20, 5),
    n_labels_per_row: int = None,
    ylims: Dict[str, Tuple[float, float]] = None,
    plot_shaded_std: bool = True,
    remove_lr_from_label: bool = False,
    wandb_api=None,
    wandb_kwargs: Dict[str, Any] = None,
    format_kwargs: Dict[str, Any] = None,
):
    """
    Args:
        wandb_group (str): The wandb group name.
        title (str): The title for the wandb group. If None, the (normalized) wandb group name is used.
        ensemble_size (int): The ensemble size (how many samples to draw from the model).
        split (str): The split to plot. One of 'val' or 'test/2020/88'.
        noise_level (float): The noise level to plot (usually 0 or 1e-4).
        metrics (List[str]): The metrics to plot. Any combination of 'mse', 'crps', 'corr'.
        horizon_ticks (Sequence[int]): The horizons to plot. If None, all horizons are plotted.
        figsize (Tuple[int, int]): The figure size.
        n_labels_per_row (int): The number of labels per row in the legend.
        ylims (Dict[str, Tuple[float, float]]): The y-limits for each metric, e.g. {'mse': (0, 0.1)}.
        plot_shaded_std (bool): Whether to plot the standard deviation as a shaded area.
        remove_lr_from_label (bool): Whether to remove the learning rate from the label.
        wandb_api (wandb.Api): The wandb API.
        wandb_kwargs (Dict[str, Any]): The kwargs to pass to wandb.Api.runs().
        format_kwargs (Dict[str, Any]): The kwargs to pass to RollingPlotFormats.
    """
    wandb_api = wandb_api or wandb.Api(timeout=60)
    wandb_kwargs = wandb_kwargs or {}
    format_kwargs = format_kwargs or {}

    if title == "":
        title = normalize_run_name(
            wandb_group,
            remove_lr_from_label=remove_lr_from_label,
            remove_weight_decay_from_label=True,
        )
    if sampling_keys is None:
        sampling_keys = []
        for st in ["", "naive/"]:
            for sched in ["", "only_dynamics", "first50", "every2nd"]:
                sampling_keys += [f"{st}{sched}"]
    # make labels by adding _ before numbers (e.g first50 -> first_50, odp1 -> odp_1)
    sampling_keys_labels = [
        re.sub(r"(\d+)", r"_\1", re.sub(r"([a-z])([A-Z])", r"\1_\2", key)) for key in sampling_keys  # .capitalize()
    ]

    if n_labels_per_row is None:
        # basically, we don't want more than 10 labels per row
        n_labels_per_row = min(10, (len(sampling_keys) // 2)) if len(sampling_keys) >= 8 else len(sampling_keys)

    any_metric_key = f"{split}/{sampling_keys[-1]}/{noise_level}eps/{ensemble_size}ens_mems/avg/crps"
    if "filter_functions" not in wandb_kwargs:
        wandb_kwargs["filter_functions"] = has_summary_metric(any_metric_key)
    group_runs = get_runs_for_group_with_any_metric(wandb_group, wandb_api=wandb_api, **wandb_kwargs)
    if len(group_runs) == 0:
        raise ValueError(f"No runs found for group {wandb_group} and key {any_metric_key}!")

    group_horizon = group_runs[0].config.get("datamodule/horizon")
    horizons = list(range(1, group_horizon + 1)) if horizon_ticks is None else horizon_ticks

    fig, axs = get_fig_and_axes(nrows=1, ncols=len(metrics), figsize=figsize)
    formats = RollingPlotFormats(all_keys=sampling_keys_labels, **format_kwargs)
    for i, (sampling_key, label) in enumerate(zip(sampling_keys, sampling_keys_labels)):
        plt_kwargs = formats[label] if len(formats.unique_base_keys) > 1 else dict()

        group_metrics = metrics_of_runs_to_arrays(
            group_runs,
            metrics,
            columns=horizons,
            column_to_wandb_key=lambda h: f"{split}/{sampling_key}/{noise_level}eps/{ensemble_size}ens_mems/t{h}",
        )
        label = (
            "default"
            if label == ""
            else sampling_key.replace("odp", "+").replace("_discrete", "D").replace("_random", "Rand")
        )
        plot_avg_and_std(
            list(group_metrics.values()),
            x_axis=horizons,
            label=label,
            axes=axs,
            plot_shaded_std=plot_shaded_std,
            label_suffix_format="mean+std",
            plot_kwargs=plt_kwargs,
        )
    beautify_plots_with_metrics(
        axs,
        metrics,
        title=title,
        xlabel="Horizon (days)",
        ylabel_prefix_left_ax=split,
        ylims=ylims,
        n_labels_per_row=n_labels_per_row,
        plot_legend_on_top=len(metrics) == 1,
    )
    return fig, axs


def plot_sampling_ablation_vs_number_sampling_steps(
    wandb_groups: List[str] | Dict[str, str],
    sampling_key_stems: Sequence[str] = ("odp", "odp_discrete"),
    sampling_key_ranges: Dict[str, Union[List[int], Tuple[int, int]]] | List[int] | Tuple[int, int] = (1, 15),
    ensemble_size: int = 50,
    split: str = "test/2020",
    noise_level: float = 0.0,
    metrics: List[str] = ("mse", "crps"),
    relative_to_artificial_diffusion_steps: bool = False,
    add_only_dynamics: bool = True,
    add_full_sampling: bool = True,
    figsize: Tuple[int, int] = (20, 5),
    n_labels_per_row: int = None,
    ylims: Dict[str, Tuple[float, float]] = None,
    plot_shaded_std: bool = True,
    remove_lr_from_label: bool = False,
    wandb_api=None,
    wandb_kwargs: Dict[str, Any] = None,
    format_kwargs: Dict[str, Any] = None,
):
    """
    Args:
        wandb_groups (List[str] | Dict[str, str]): The wandb groups to plot. If a list, the group names are used as labels, if a dict, the keys are used as labels.
        sampling_key_stems (Sequence[str]): The sampling key stems to plot in ascending order.
        sampling_key_ranges (Dict[str, Tuple[int, int]] | List[Tuple[int, int]]): The sampling key ranges to plot.
        ensemble_size (int): The ensemble size (how many samples to draw from the model).
        split (str): The split to plot. One of 'val' or 'test/2020/88'.
        noise_level (float): The noise level to plot (usually 0 or 1e-4).
        metrics (List[str]): The metrics to plot. Any combination of 'mse', 'crps', 'corr'.
        relative_to_artificial_diffusion_steps (bool): Whether to plot the x-axis relative to the total number of artificial diffusion steps.
        add_only_dynamics (bool): Whether to add the only_dynamic sampling key to the front (i.e. 0 artificial diffusion steps).
        add_full_sampling (bool): Whether to add/plot the full sampling key (i.e. all artificial diffusion steps).
        figsize (Tuple[int, int]): The figure size.
        n_labels_per_row (int): The number of labels per row in the legend.
        ylims (Dict[str, Tuple[float, float]]): The y-limits for each metric, e.g. {'mse': (0, 0.1)}.
        plot_shaded_std (bool): Whether to plot the standard deviation as a shaded area.
        remove_lr_from_label (bool): Whether to remove the learning rate from the label.
        wandb_api (wandb.Api): The wandb API.
        wandb_kwargs (Dict[str, Any]): The kwargs to pass to wandb.Api.runs().
        format_kwargs (Dict[str, Any]): The kwargs to pass to RollingPlotFormats.
    """
    wandb_groups = [wandb_groups] if isinstance(wandb_groups, str) else wandb_groups
    wandb_api = wandb_api or wandb.Api(timeout=60)
    wandb_kwargs = wandb_kwargs or {}
    format_kwargs = format_kwargs or {}

    if not isinstance(wandb_groups, dict):
        nkwargs = dict(
            remove_lr_from_label=remove_lr_from_label,
            remove_weight_decay_from_label=True,
        )
        wandb_groups = {normalize_run_name(wandb_group, **nkwargs): wandb_group for wandb_group in wandb_groups}

    num_lines = len(wandb_groups) * len(sampling_key_stems)
    n_labels_per_row = get_num_labels_per_legend_row(num_lines, n_labels_per_row)

    fig, axs = get_fig_and_axes(nrows=1, ncols=len(metrics), figsize=figsize)
    formats = RollingPlotFormats(
        all_keys=[g + "$" for g in wandb_groups.values()],
        base_keys_sep="$",
        **format_kwargs,
    )
    for i, (label, wandb_group) in enumerate(wandb_groups.items()):
        any_metric_key = f"{split}/only_dynamics/avg/{noise_level}eps/{ensemble_size}ens_mems/crps"
        if "filter_functions" not in wandb_kwargs:
            wandb_kwargs["filter_functions"] = has_summary_metric(any_metric_key)
        group_runs = get_runs_for_group(wandb_group, wandb_api=wandb_api, **wandb_kwargs)
        if len(group_runs) == 0:
            raise ValueError(f"No runs found for group {wandb_group} and key {any_metric_key}!")

        if i == 0:
            horizon = group_runs[0].config.get("datamodule/horizon")
        else:
            assert horizon == group_runs[0].config.get(
                "datamodule/horizon"
            ), "Horizon must be the same for all groups!"

        n_art_diffusion_steps = get_number_diffusion_steps(group_runs[0], wandb_api=wandb_api, only_artificial=True)
        horizons = list(range(1, horizon + 1))
        eps_lvl_ens_size = f"{noise_level}eps/{ensemble_size}ens_mems"

        # Get metrics for default sampling schedule
        group_metrics_fs = metrics_of_runs_to_arrays(
            group_runs,
            metrics,
            columns=horizons,
            column_to_wandb_key=lambda h: [
                f"{split}/{eps_lvl_ens_size}/t{h}",
                f"{split}/odp{n_art_diffusion_steps}/{eps_lvl_ens_size}/t{h}",
            ],
        )

        for j, sampling_key_stem in enumerate(sampling_key_stems):
            plt_kwargs = formats[wandb_group + "$"]
            if isinstance(sampling_key_ranges, dict):
                sampling_key_range = sampling_key_ranges[sampling_key_stem]
            if isinstance(sampling_key_ranges, tuple):
                sampling_key_range = list(range(*sampling_key_ranges))

            group_metrics = metrics_of_runs_to_arrays(
                group_runs,
                metrics,
                columns=sampling_key_range,
                column_to_wandb_key=lambda s: f"{split}/{sampling_key_stem}{s}/{eps_lvl_ens_size}/avg",
            )
            # group_metrics2 = {k: np.nan_to_num(v) for k, v in group_metrics2.items()}

            if add_only_dynamics and sampling_key_range[0] != 0:
                group_metrics_od = metrics_of_runs_to_arrays(
                    group_runs,
                    metrics,
                    columns=horizons,
                    column_to_wandb_key=lambda h: f"{split}/only_dynamics/{eps_lvl_ens_size}/t{h}",
                )
                for k, v in group_metrics_od.items():
                    group_metrics[k] = np.concatenate([v.mean(axis=-1, keepdims=True), group_metrics[k]], axis=1)
                sampling_key_range = [0] + sampling_key_range
            if relative_to_artificial_diffusion_steps:
                sampling_key_range = [n / n_art_diffusion_steps for n in sampling_key_range]

            label = f"{sampling_key_stem} {label}" if len(sampling_key_stems) > 1 else label
            plot_avg_and_std(
                list(group_metrics.values()),
                x_axis=sampling_key_range,
                label=label,
                axes=axs,
                plot_shaded_std=plot_shaded_std,
                label_suffix_format="min",
                plot_kwargs=plt_kwargs,
            )

            # Plot the full schedule dot as a errorbar
            if add_full_sampling:
                plot_avg_and_std(
                    [v.mean(axis=-1, keepdims=True) for v in group_metrics_fs.values()],
                    x_axis=[1 if relative_to_artificial_diffusion_steps else n_art_diffusion_steps],
                    axes=axs,
                    plot_shaded_std=False,
                    plot_kwargs={
                        **plt_kwargs,
                        "marker": "x",
                        "markersize": 10,
                        "markeredgewidth": 3,
                    },
                )
                # sampling_key_range = list(sorted(sampling_key_range + [n_total_diffusion_steps]))
    xlabel = (
        "Diffusion steps" if not relative_to_artificial_diffusion_steps else "Rel. diffusion steps from default (in %)"
    )
    beautify_plots_with_metrics(
        axs,
        metrics,
        title="",
        xlabel=xlabel,
        ylabel_prefix_left_ax=split,
        ylabel_prefix=f"t=1:{horizon} avg. ",
        ylims=ylims,
        n_labels_per_row=n_labels_per_row,
        plot_legend_on_top=len(metrics) == 1,
    )
    return fig, axs


def plot_sampling_ablation_vs_time(
    wandb_groups: List[str] | Dict[str, str],
    sampling_keys: List[str],
    ensemble_size: int = 50,
    split: str = "test/2020",
    box: int = 84,
    noise_level: float = 0.0,
    sampling_type: str = "cold",
    metrics: List[str] = ("crps", "mse"),
    time_in: str = "seconds",
    figsize: Tuple[int, int] = (20, 5),
    n_labels_per_row: int = None,
    ylims: Dict[str, Tuple[float, float]] = None,
    scatter: bool = False,
    min_keys_per_run: int = 1,
    remove_lr_from_label: bool = False,
    wandb_api=None,
    axes: Optional[plt.Axes] = None,
    wandb_kwargs: Dict[str, Any] = None,
    format_kwargs: Dict[str, Any] = None,
):
    """
    Args:
        wandb_groups (List[str] | Dict[str, str]): The wandb groups to plot. If a list, the group names are used as labels, if a dict, the keys are used as labels.
        sampling_keys (List[str]): The sampling keys to plot.
        ensemble_size (int): The ensemble size (how many samples to draw from the model).
        split (str): The split to plot. One of 'val' or 'test/2020/88'.
        noise_level (float): The noise level to plot (usually 0 or 1e-4).
        metrics (List[str]): The metrics to plot. Any combination of 'mse', 'crps', 'corr'.
        figsize (Tuple[int, int]): The figure size.
        n_labels_per_row (int): The number of labels per row in the legend.
        ylims (Dict[str, Tuple[float, float]]): The y-limits for each metric, e.g. {'mse': (0, 0.1)}.
        remove_lr_from_label (bool): Whether to remove the learning rate from the label.
        wandb_api (wandb.Api): The wandb API.
        wandb_kwargs (Dict[str, Any]): The kwargs to pass to wandb.Api.runs().
        format_kwargs (Dict[str, Any]): The kwargs to pass to RollingPlotFormats.
    """
    wandb_groups = [wandb_groups] if isinstance(wandb_groups, str) else wandb_groups
    wandb_api = wandb_api or wandb.Api(timeout=60)
    wandb_kwargs = wandb_kwargs or {}
    format_kwargs = format_kwargs or {}

    if not isinstance(wandb_groups, dict):
        nkwargs = dict(
            remove_lr_from_label=remove_lr_from_label,
            remove_weight_decay_from_label=True,
        )
        wandb_groups = {normalize_run_name(wandb_group, **nkwargs): wandb_group for wandb_group in wandb_groups}

    num_lines = len(wandb_groups)  # Number of colors for the different scatter plots
    n_labels_per_row = get_num_labels_per_legend_row(num_lines, n_labels_per_row)

    # Each subplot is a scatter plot of the different sampling keys metric vs. time
    fig, axs = get_fig_and_axes(nrows=1, ncols=len(metrics), figsize=figsize) if axes is None else (None, axes)
    formats = RollingPlotFormats(
        all_keys=[g + "$" for g in wandb_groups.values()],
        base_keys_sep="$",
        **format_kwargs,
    )
    for i, (label, wandb_group) in enumerate(wandb_groups.items()):
        any_metric_key = f"{split}/{box}/{sampling_keys[0]}/{noise_level}eps/{ensemble_size}ens_mems/avg/crps".replace(
            "//", "/"
        )
        if "filter_functions" not in wandb_kwargs:
            wandb_kwargs["filter_functions"] = has_summary_metric(any_metric_key)
        group_runs = get_runs_for_group(wandb_group, wandb_api=wandb_api, **wandb_kwargs)
        if len(group_runs) == 0:
            log.warning(f"No runs found for group {wandb_group} and key {any_metric_key}!")
            continue

        if i == 0:
            horizon = group_runs[0].config.get("datamodule/horizon")
        else:
            assert horizon == group_runs[0].config.get(
                "datamodule/horizon"
            ), "Horizon must be the same for all groups!"

        horizons = list(range(1, horizon + 1))
        eps_lvl_ens_size = f"{noise_level}eps/{ensemble_size}ens_mems"

        # Get metrics for default sampling schedule
        metrics_of_runs_to_arrays(
            group_runs,
            metrics,
            columns=horizons,
            column_to_wandb_key=lambda h: f"{split}/{eps_lvl_ens_size}/t{h}",
        )
        plt_kwargs = formats[wandb_group + "$"]
        shades = np.linspace(0.5, 1, len(sampling_keys))
        for r_i, run in enumerate(group_runs):
            stem = f"{split}/{box}/$$$/{noise_level}eps/{ensemble_size}ens_mems/avg".replace("//", "/")
            if run.summary.get(f'{stem.replace("$$$", sampling_keys[0])}/crps') is None:
                print(f"Run {run.name} does not have metrics for sampling key {sampling_keys[0]}!")
                print("tried to get", f'{stem.replace("$$$", sampling_keys[0])}/crps')
                continue
            m_to_vals = defaultdict(list)
            times = []
            for j, sampling_key in enumerate(sampling_keys):
                time_key = f"time/{box}/{split}/{sampling_type}/{sampling_key}"
                time = run.summary.get(time_key)
                time2 = run.summary.get(f"inference_speed_ms/{sampling_key}")
                if time2 is not None:
                    time = time2 / 1000  # convert to seconds
                if time is None and run.summary.get(time_key.replace("odp", "only_dynamics_plus")) is not None:
                    time = run.summary.get(time_key.replace("odp", "only_dynamics_plus"))
                if time is None and run.summary.get(time_key.replace("cold", sampling_key)) is not None:
                    time = run.summary.get(time_key.replace("cold", sampling_key))
                if time is None:
                    print(f"Run {run.id} does not have time key``{time_key}``!")
                    continue
                if time < 0.1 and sampling_key == "every4th":
                    if run.summary.get(time_key.replace(str(box), "85")):
                        time = run.summary.get(time_key.replace(str(box), "85"))
                    elif run.summary.get(time_key.replace(str(box), "85").replace("cold", sampling_key)):
                        time = run.summary.get(time_key.replace(str(box), "85").replace("cold", sampling_key))
                    else:
                        print(f"Run {run.id} does not have time key``{time_key}``: {time}!")
                        continue
                if time_in in ["minutes", "min", "m"]:
                    time = time / 60
                elif time_in not in ["seconds", "s"]:
                    raise ValueError(f"Unknown time unit {time_in}!")
                for m_i, m in enumerate(metrics):
                    m_val = run.summary.get(f"{stem.replace('$$$', sampling_key)}/{m}")
                    if m_i == 0:
                        label_sc = f"{run.id} ({sampling_key})" if len(sampling_keys) > 1 else label
                    else:
                        label_sc = None
                    if scatter:
                        axs[m_i].scatter(
                            time,
                            m_val,
                            color=plt_kwargs["color"],
                            alpha=shades[j],
                            s=20,
                            label=label_sc,
                        )
                    else:
                        m_to_vals[m].append(m_val)
                times.append(time / 55)
            if not scatter and len(times) > min_keys_per_run:
                for m_i, m in enumerate(metrics):
                    label_line = f"{label}" if m_i == 0 else None
                    plt_kwargs["linestyle"] = "-"
                    plt_kwargs["marker"] = plt_kwargs.get("marker", "o")
                    axs[m_i].plot(times, m_to_vals[m], **plt_kwargs, alpha=0.9, label=label_line)

    xlabel = "Time [s]" if time_in in ["seconds", "s"] else "Time [min]"
    beautify_plots_with_metrics(
        axs,
        metrics,
        title="",
        xlabel=xlabel,
        ylabel_prefix_left_ax=split,
        ylabel_prefix=f"t=1:{horizon} avg. ",
        ylims=ylims,
        n_labels_per_row=n_labels_per_row,
        plot_legend_on_top=True,
    )
    return fig, axs


def plot_sampling_ablation_vs_ensemble_size(
    wandb_groups: List[str] | Dict[str, str],
    ensemble_sizes: List[int] = (
        2,
        3,
        4,
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
    ),
    sampling_key: str = "",
    split: str = "test/2020",
    noise_level: float = 0.0,
    target_key: str = "avg",
    metrics: List[str] = ("mse", "crps"),
    figsize: Tuple[int, int] = (20, 5),
    n_labels_per_row: int = None,
    ylims: Dict[str, Tuple[float, float]] = None,
    plot_shaded_std: bool = True,
    remove_lr_from_label: bool = False,
    print_values: bool = False,
    wandb_api=None,
    wandb_kwargs: Dict[str, Any] = None,
    format_kwargs: Dict[str, Any] = None,
):
    """
    Args:
        wandb_groups (List[str] | Dict[str, str]): The wandb groups to plot. If a list, the group names are used as labels, if a dict, the keys are used as labels.
        ensemble_sizes (List[int]): The ensemble sizes to plot in ascending order.
        sampling_key (str): The sampling key to plot, e.g. 'only_dynamics', 'odp10', or '' for the default schedule.
        split (str): The split to plot. One of 'val' or 'test/2020/88'.
        noise_level (float): The noise level to plot (usually 0 or 1e-4).
        target_key (str): The target key to plot, e.g. 'avg' or 't1' for the first time step, 't4' for the fourth time step etc.
        metrics (List[str]): The metrics to plot. Any combination of 'mse', 'crps', 'corr'.
        figsize (Tuple[int, int]): The figure size.
        n_labels_per_row (int): The number of labels per row in the legend.
        ylims (Dict[str, Tuple[float, float]]): The y-limits for each metric, e.g. {'mse': (0, 0.1)}.
        plot_shaded_std (bool): Whether to plot the standard deviation as a shaded area.
        remove_lr_from_label (bool): Whether to remove the learning rate from the label.
        wandb_api (wandb.Api): The wandb API.
        wandb_kwargs (Dict[str, Any]): The kwargs to pass to wandb.Api.runs().
        format_kwargs (Dict[str, Any]): The kwargs to pass to RollingPlotFormats.
    """
    wandb_groups = [wandb_groups] if isinstance(wandb_groups, str) else wandb_groups
    wandb_api = wandb_api or wandb.Api(timeout=60)
    wandb_kwargs = wandb_kwargs or {}
    format_kwargs = format_kwargs or {}

    num_lines = len(wandb_groups) * len(ensemble_sizes)
    n_labels_per_row = get_num_labels_per_legend_row(num_lines, n_labels_per_row)

    if not isinstance(wandb_groups, dict):
        nkwargs = dict(
            remove_lr_from_label=remove_lr_from_label,
            remove_weight_decay_from_label=True,
        )
        wandb_groups = {normalize_run_name(wandb_group, **nkwargs): wandb_group for wandb_group in wandb_groups}

    key_prefix = f"{split}/{sampling_key}/{noise_level}eps".replace("//", "/")
    any_metric_key = f"{key_prefix}/{ensemble_sizes[0]}ens_mems/{target_key}/crps"
    if "filter_functions" not in wandb_kwargs:
        wandb_kwargs["filter_functions"] = has_summary_metric(any_metric_key)

    fig, axs = get_fig_and_axes(nrows=1, ncols=len(metrics), figsize=figsize)
    formats = RollingPlotFormats(
        all_keys=[g + "$" for g in wandb_groups.values()],
        base_keys_sep="$",
        **format_kwargs,
    )
    for i, (label, wandb_group) in enumerate(wandb_groups.items()):
        group_runs = get_runs_for_group(wandb_group, wandb_api=wandb_api, **wandb_kwargs)
        if len(group_runs) == 0:
            log.warning(f"No runs found for group {wandb_group} and key {any_metric_key}!")
            continue

        if i == 0:
            horizon = group_runs[0].config.get("datamodule/horizon")
        else:
            h2 = group_runs[0].config.get("datamodule/horizon", horizon)
            assert horizon == h2, f"Horizon must be the same for all groups!, but got {horizon} and {h2}"

        plt_kwargs = formats[wandb_group + "$"]
        plt_kwargs["linewidth"] = 1
        group_metrics = metrics_of_runs_to_arrays(
            group_runs,
            metrics,
            columns=ensemble_sizes,
            column_to_wandb_key=lambda e: f"{key_prefix}/{e}ens_mems/{target_key}",
        )
        if print_values:
            print(f"Plotting {label[:50]} ", group_metrics["crps"].squeeze())

        plot_avg_and_std(
            list(group_metrics.values()),
            x_axis=ensemble_sizes,
            label=label,
            axes=axs,
            plot_shaded_std=plot_shaded_std,
            label_suffix_format="min",
            plot_kwargs={
                **plt_kwargs,
                **get_marker_kwargs(wandb_group),
            },  # , 'markersize': 10, 'markeredgewidth': 1},
        )

    xlabel = "Ensemble size"
    ylabel_prefix = f"t=1:{horizon} avg. " if target_key == "avg" else f"{target_key} "
    beautify_plots_with_metrics(
        axs,
        metrics,
        title="",
        xlabel=xlabel,
        ylabel_prefix_left_ax=split,
        ylabel_prefix=ylabel_prefix,
        ylims=ylims,
        n_labels_per_row=n_labels_per_row,
        plot_legend_on_top=len(metrics) == 1,
    )
    return fig, axs


def plot_metrics_vs_box(
    wandb_groups: List[str] | Dict[str, str],
    boxes: List[int] = (84, 85, 86, 87, 88, 89, 108, 109, 110, 111, 112, 165),
    split: str = "test/2020",
    noise_level: float = 0.0,
    ensemble_size: int = 50,
    target_key: str = "avg",
    metric: str = "crps",
    only_show_average: bool = False,
    figsize: Tuple[int, int] = (20, 5),
    n_labels_per_row: int = None,
    remove_lr_from_label: bool = False,
    remove_weight_decay_from_label: bool = False,
    add_num_runs_to_label: bool = False,
    wandb_api=None,
    wandb_kwargs: Dict[str, Any] = None,
    format_kwargs: Dict[str, Any] = None,
    legend_kwargs: Dict[str, Any] = None,
):
    """
    Plot a bar plot of the given metric for each of the given boxes.
    Args:
        wandb_groups (List[str] | Dict[str, str]): The wandb groups to plot. If a list, the group names are used as labels, if a dict, the keys are used as labels.
        boxes (List[int]): The boxes to plot.
        split (str): The split to plot. One of 'val' or 'test/2020/{BOX-ID}'. If only_show_average is True, multiple splits can be given.
        noise_level (float): The noise level to plot (usually 0 or 1e-4).
        ensemble_size (int): The ensemble size to use for the plot.
        target_key (str): The target key to plot, e.g. 'avg' or 't1' for the first time step, 't4' for the fourth time step etc.
        metric (str): The metric to plot. Any of 'mse', 'crps', 'corr'.
        only_show_average (bool): Whether to only show the average performance across all boxes (i.e. one bar plot only).
        figsize (Tuple[int, int]): The figure size.
        n_labels_per_row (int): The number of labels per row in the legend.
        remove_lr_from_label (bool): Whether to remove the learning rate from the label.
        add_num_runs_to_label (bool): Whether to add the number of runs to the label.
        wandb_api (wandb.Api): The wandb API.
        wandb_kwargs (Dict[str, Any]): The kwargs to pass to wandb.Api.runs().
        format_kwargs (Dict[str, Any]): The kwargs to pass to RollingPlotFormats.
    """
    wandb_groups = [wandb_groups] if isinstance(wandb_groups, str) else wandb_groups
    wandb_api = wandb_api or wandb.Api(timeout=60)
    wandb_kwargs = wandb_kwargs or {}
    format_kwargs = format_kwargs or {}
    legend_kwargs = legend_kwargs or {}

    num_bars_per_plot = len(wandb_groups)
    n_labels_per_row = get_num_labels_per_legend_row(num_bars_per_plot, n_labels_per_row)

    if not isinstance(metric, str) and not only_show_average:
        raise ValueError(
            f"Can only plot multiple metrics if only_show_average is True, but got {metric} and {only_show_average}."
        )
    metric_names = [metric] if isinstance(metric, str) else metric

    if only_show_average:
        nrows, ncols = 1, len(metric_names)
    elif len(boxes) <= 5:
        nrows, ncols = 1, len(boxes)
    elif len(boxes) <= 11:
        nrows, ncols = 2, len(boxes) // 2
    else:
        nrows, ncols = 3, len(boxes) // 3

    if not isinstance(wandb_groups, dict):
        was_dict = False
        nkwargs = dict(
            remove_lr_from_label=remove_lr_from_label,
            remove_weight_decay_from_label=remove_weight_decay_from_label,
        )
        wandb_groups = {normalize_run_name(wandb_group, **nkwargs): wandb_group for wandb_group in wandb_groups}
    else:
        was_dict = True

    def split_to_any_metric_stem(spl):
        bid = 84 if 84 in boxes else boxes[0]
        spl = spl.replace("$$", str(bid)) if "$$" in spl else f"{spl}/{bid}"
        return f"{spl}/{noise_level}eps/{ensemble_size}ens_mems/{target_key}"

    metrics, labels, horizon = dict(), dict(), None
    for i, (name, wandb_group) in enumerate(wandb_groups.items()):
        group_runs, any_metric_stem = get_runs_for_group_with_any_metric(
            wandb_group,
            split,
            split_to_any_metric_stem,
            wandb_api,
            **wandb_kwargs,
            metric=metric_names[0],
        )
        if group_runs is None:
            continue
        any_metric_stem = any_metric_stem.replace("/84/", "/$$/")

        name = group_runs[0].summary.get("short_group_name", name) if not was_dict else name

        if add_num_runs_to_label:
            labels[name] = f"{name} (#{len(group_runs)})" if len(group_runs) > 1 else f"{group_runs[0].id}: {name}"
        else:
            labels[name] = name

        if horizon is None:
            horizon = group_runs[0].config.get("datamodule/horizon")
        else:
            group_runs[0].config.get("datamodule/horizon", horizon)
            # assert horizon == h2, f'Horizon must be the same for all groups!, but got {horizon} and {h2}'

        metrics[name] = metrics_of_runs_to_arrays(
            group_runs,
            metric_names,
            columns=boxes,
            column_to_wandb_key=lambda b: any_metric_stem.replace("$$", str(b)),
        )  # [metric]
        if not only_show_average:
            metrics[name] = metrics[name][metric]
        # ml2 = metrics_of_runs_to_arrays(
        #     group_runs, metric_names, columns=boxes,
        #     column_to_wandb_key=lambda b: any_metric_stem.replace('$$', str(b)).replace("0.0", str(1e-4)),
        # )[metric]
        # set metrics[label] to ml2 for all nan values
        # metrics[name][np.isnan(metrics[name])] = ml2[np.isnan(metrics[name])]

    formats = RollingPlotFormats(
        all_keys=list(metrics.keys()),
        base_keys_sep="_",
        use_hatches=True,
        **format_kwargs,
    )
    fig, axs = get_fig_and_axes(nrows=nrows, ncols=ncols, figsize=figsize)
    if only_show_average:
        # Plot the average performance across all boxes for each metric and group
        # Sort the performance by the first metric for better visualisation
        for i, (metric, ax) in enumerate(zip(metric_names, axs)):
            label_to_mean_std = dict()
            for name in metrics:
                group_mean = np.nanmean(metrics[name][metric], axis=0)
                group_std = np.nanstd(metrics[name][metric], axis=0)
                # if there's still nan values, don't plot this group
                if np.isnan(group_mean).any():
                    continue
                # Store the average mean/std across all boxes for this group
                # print(f'{label}:', group_mean, f"±{np.std(group_mean):.4f}, {np.mean(group_mean):.4f}")
                group_box_std = np.mean(group_std)  # np.std(group_mean)
                label_to_mean_std[name] = (np.mean(group_mean), group_box_std)
            means = [label_to_mean_std[label][0] for label in label_to_mean_std]
            # Sort the groups by the average mean across all boxes  (only for the first metric)
            if i == 0:
                sorted_labels = sorted(label_to_mean_std, key=lambda label: label_to_mean_std[label][0])
            # Plot the bars
            for i_l, name in enumerate(sorted_labels):
                mean, std = label_to_mean_std[name]
                ax.bar(
                    x=i_l,
                    height=mean,
                    yerr=std,
                    **formats[name],
                    label=labels[name] if i == 0 else None,
                )

            ax.set_xlabel(f"Avg. of boxes {boxes}")
            ax.set_xticks([])  # disable xticks
            # set ylimits slightly around the min, max value
            ax.set_ylim(0.98 * min(means), 1.02 * max(means))
            ax.grid(True)
            if i > 0:
                ax.set_ylabel(clean_metric_name(metric))
    else:
        ax_flat = axs.flatten() if nrows > 1 or ncols > 1 else axs
        label_to_plt_kwargs = dict()
        for b, (box, ax) in enumerate(zip(boxes, ax_flat)):
            # Plot a bar for each group, but first sort the groups by the metric value
            metrics_for_box = {
                label: metrics[label][:, b] for label in metrics if not np.isnan(np.nanmean(metrics[label][:, b]))
            }
            label_to_mean = {label: np.nanmean(metrics_for_box[label]) for label in metrics_for_box}
            sorted_labels = sorted(metrics_for_box, key=lambda label: label_to_mean[label])  # sort from low to high
            means = label_to_mean.values()
            for i_l, name in enumerate(sorted_labels):
                if name not in label_to_plt_kwargs:
                    label_to_plt_kwargs[name] = formats[name]

                mean = np.nanmean(metrics_for_box[name])
                std = np.nanstd(metrics_for_box[name])
                # Plot error bars, one next to the other
                # To differentiate bars with similar shades, we use a different hatch for each bar
                ax.bar(
                    x=i_l * 0.1,
                    height=mean,
                    yerr=std,
                    width=0.1,
                    label=name if b == 0 else None,
                    **label_to_plt_kwargs[name],
                    # capsize=3, ecolor='black'  # Plot error bars with vertical lines to make them more visible
                )
            ax.set_xlabel(f"Box {box}")
            ax.set_xticks([])  # disable xticks
            # set ylimits slightly around the min, max value
            ax.set_ylim(0.98 * min(means), 1.02 * max(means))

    # Set ylabels on the leftmost axes
    ylabel_prefix = f"t=1:{horizon} avg. " if target_key == "avg" else f"{target_key} "
    split_name = (split if isinstance(split, str) else split[0]).replace("/$$", "")
    ylabel = ylabel_prefix + clean_metric_name(metric_names[0]) + "\n" + split_name
    for ax in axs[:, 0] if nrows > 1 else [axs[0]]:
        ax.set_ylabel(ylabel)
    show_legend_on_top(axs, n_labels_per_row=n_labels_per_row, **legend_kwargs)
    return fig, axs


# ------------------------------------------ Auxiliary functions ------------------------------------------ #
def get_num_labels_per_legend_row(total_lines: int, n_labels_per_row: int = None, max_lines_per_row: int = 10) -> int:
    if n_labels_per_row is None:
        # basically, we don't want more than ``max_lines_per_row`` labels per row
        n_labels_per_row = min(max_lines_per_row, (total_lines // 2)) if total_lines >= 8 else total_lines
    return n_labels_per_row


def show_legend_on_top(
    axes: List[plt.Axes],
    n_labels_per_row: int,
    fontsize: int = None,
    anchor_y: float = 0.96,
    loc="upper center",
    for_all_axes: bool = True,
):
    """Make legend show on top of plot"""
    # first, get number of elements in legend
    if isinstance(axes, np.ndarray) and axes.ndim == 2:
        any_ax = axes[0, 0]
    elif isinstance(axes, list) or (isinstance(axes, np.ndarray) and axes.ndim == 1):
        any_ax = axes[0]
    else:
        any_ax = axes
    handles, labels = any_ax.get_legend_handles_labels()
    n_labels = len(labels)
    n_label_rows = int(np.ceil(n_labels / n_labels_per_row))
    # Get optimal position for legend (anchor_y), right above the plot
    anchor_y = anchor_y + 0.04 * (n_label_rows - 1)
    legend_handle = any_ax.figure if for_all_axes else any_ax
    legend_handle.legend(
        loc=loc,
        bbox_to_anchor=(0.5, anchor_y),
        ncol=n_labels_per_row,
        fontsize=fontsize,
    )
    return anchor_y


def beautify_plots_with_metrics(
    axes: List[plt.Axes],
    metric_names: List[str],
    title: str = None,
    xtick_labels: List[str] = None,
    xlabel: str = "",
    ylabel_prefix: str = "",
    ylabel_prefix_left_ax: str = "",
    ylims: Dict[str, Tuple[float, float]] = None,
    n_labels_per_row: int = 3,
    plot_legend_on_top: bool = True,
    anchor_y: float = 0.96,
    grid: bool = True,
    fontsize: int = 10,
):
    """
    Args:
        axes: list of axes to beautify
        metric_names: list of metric names to use as y-labels
        title (str): title to use
        xtick_labels (List[str]): x-tick labels to override default x-axis labels
        xlabel (str): x-label to use
        ylabel_prefix: prefix to use for y-labels
        ylabel_prefix_left_ax: prefix to use for y-labels of leftmost axis
        ylims: dict of metric name to (min, max) ylims
        n_labels_per_row: number of labels per row in legend
        plot_legend_on_top (bool): whether to plot legend on top of plot (instead of inside plot)
        grid (bool): whether to show grid
    """
    for i, (ax, m_name) in enumerate(zip(axes, metric_names)):
        if xtick_labels is not None:
            ax.set_xticklabels(xtick_labels)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ylabel = ylabel_prefix + clean_metric_name(m_name)
        if i == 0 and ylabel_prefix_left_ax:
            # Bolden first y-label
            ylabel = r"$\bf{" + ylabel_prefix_left_ax.replace("_", r"\_") + "}$\n" + clean_metric_name(m_name)

        ax.set_ylabel(ylabel, fontsize=fontsize)
        if ylims is not None and m_name in ylims:
            ax.set_ylim(*ylims[m_name])

        # if grid: ax.grid(visible=True)
        ax.grid(visible=grid)

    if plot_legend_on_top:
        anchor_y = show_legend_on_top(axes, n_labels_per_row, fontsize=fontsize, anchor_y=anchor_y)
    else:
        for ax in axes:
            ax.legend(loc="best", ncol=min(3, n_labels_per_row))

    if title is not None:
        # Make title show on top of legend
        y = anchor_y + 0.035 if plot_legend_on_top else 0.93
        axes[0].figure.suptitle(title, y=y)


def create_wandb_figures(target, gen, var_name, fig_shared_label, coords, show_log_precip=True):
    import cartopy.crs as ccrs

    # Some plotting parameters
    map_transform = ccrs.PlateCarree()
    _plot_params = {
        "pr": {"cmap": "BrBG", "add_colorbar": False, "transform": map_transform},  # , "cbar_kwargs": {"shrink": 0.9}
        "tas": {"cmap": "inferno", "add_colorbar": False, "transform": map_transform},
        "crps": {"cmap": "viridis", "transform": map_transform, "cbar_kwargs": {"shrink": 0.8}},
        "error": {
            "cmap": "coolwarm",
            "add_colorbar": False,
            "transform": map_transform,
        },  # , "cbar_kwargs": {"shrink": 0.5}
    }
    cbar_kwargs = {"fraction": 0.046, "pad": 0.04}

    def to_xr_dataarray(data, is_ensemble=False):
        if isinstance(data, dict):
            return {k: to_xr_dataarray(v) for k, v in data.items()}
        data = data.detach().cpu().numpy() if torch.is_tensor(data) else data
        if is_ensemble:
            dims = ["member", "batch"] if len(data.shape) == 4 else ["member"]
        else:
            dims = ["batch"] if len(data.shape) == 3 else []
        dims += list(coords.keys()) if coords is not None else ["dim2", "dim3"]
        return xr.DataArray(data, coords=coords, dims=dims)

    def get_random_ensembles(data):
        # get 2 random ensemble members
        idxs = np.random.choice(np.arange(ensemble_size), 2, replace=False)
        # get ens member and transform to xr dataarray
        ens_1 = to_xr_dataarray(data[idxs[0]], is_ensemble=False)
        ens_2 = to_xr_dataarray(data[idxs[1]], is_ensemble=False)
        return ens_1, ens_2, idxs

    gen = gen.cpu() if torch.is_tensor(gen) else gen[var_name].cpu()
    target = target.cpu() if torch.is_tensor(target) else target[var_name].cpu()
    any_gen_shape = gen.shape
    any_target_shape = target.shape
    is_ensemble = len(any_gen_shape) == len(any_target_shape) + 1
    if is_ensemble:
        crps = crps_ensemble(predicted=gen, truth=target, reduction="none")
    else:
        crps = None
    ensemble_size = any_gen_shape[0] if is_ensemble else 1
    gen_ens_1, gen_ens_2, ens_idxs = get_random_ensembles(gen)
    gen_ens_mean = to_xr_dataarray(gen.mean(dim=0), is_ensemble=False)
    target = to_xr_dataarray(target, is_ensemble=False)
    snapshots = dict()
    y = 0.85
    # Handle Precipitation unique case
    if var_name == "pr" and show_log_precip:
        # get log version to make more visible
        target_log = np.log(target + 1)
        gen_log = np.log(gen_ens_mean + 1)
        gen_ens_1_log = np.log(gen_ens_1 + 1)
        gen_ens_2_log = np.log(gen_ens_2 + 1)

        fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={"projection": map_transform})
        fig.suptitle(f"log({var_name}) - {fig_shared_label}", y=y + 0.1)

        # First row - target and generated mean
        vmin = min(np.min(target_log), np.min(gen_log), np.min(gen_ens_1_log), np.min(gen_ens_2_log))
        vmax = max(np.max(target_log), np.max(gen_log), np.max(gen_ens_1_log), np.max(gen_ens_2_log))
        im1 = target_log.plot(ax=axs[0, 0], vmin=vmin, vmax=vmax, **_plot_params[var_name])
        axs[0, 0].set_title("Target")
        _ = gen_log.plot(ax=axs[0, 1], vmin=vmin, vmax=vmax, **_plot_params[var_name])
        axs[0, 1].set_title("Generated: Mean")

        # Second row - sampled ensembles
        _ = gen_ens_1_log.plot(ax=axs[1, 0], vmin=vmin, vmax=vmax, **_plot_params[var_name])
        axs[1, 0].set_title(f"Generated: Ensemble {ens_idxs[0]}")
        _ = gen_ens_2_log.plot(ax=axs[1, 1], vmin=vmin, vmax=vmax, **_plot_params[var_name])
        axs[1, 1].set_title(f"Generated: Ensemble {ens_idxs[1]}")

        # create cbar
        _ = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation="vertical", **cbar_kwargs)

        # Add coastlines
        for ax in axs.flat:
            ax.coastlines()

        snapshots[f"image-full-field-log/{fig_shared_label}/{var_name}"] = wandb.Image(fig)

        # Handle Error plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 6), subplot_kw={"projection": map_transform})
        fig.suptitle(f"Bias log({var_name}) - {fig_shared_label}", y=y)

        # Calculate error in log scale # todo: Better/same to log after calculating error in general case?
        error = gen_log - target_log
        error_gen_ens_1 = gen_ens_1_log - target_log
        error_gen_ens_2 = gen_ens_2_log - target_log
        vmin = min(np.min(error), np.min(error_gen_ens_1), np.min(error_gen_ens_2))
        vmax = max(np.max(error), np.max(error_gen_ens_1), np.max(error_gen_ens_2))
        im1 = error.plot(ax=axs[0], vmin=vmin, vmax=vmax, **_plot_params["error"])
        axs[0].set_title("Error - Generated Mean")
        _ = error_gen_ens_1.plot(ax=axs[1], vmin=vmin, vmax=vmax, **_plot_params["error"])  # im2
        axs[1].set_title(f"Error - Generated Ensemble {ens_idxs[0]}")
        _ = error_gen_ens_2.plot(ax=axs[2], vmin=vmin, vmax=vmax, **_plot_params["error"])  # im3
        axs[2].set_title(f"Error - Generated Ensemble {ens_idxs[1]}")

        # create cbar
        _ = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation="vertical", **cbar_kwargs)  # cbar

        # Add coastlines
        for ax in axs.flat:
            ax.coastlines()

        snapshots[f"image-error-log/{fig_shared_label}/{var_name}"] = wandb.Image(fig)

    # Handle General plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={"projection": map_transform})
    fig.suptitle(f"{var_name}Full Field {var_name} - {fig_shared_label}", y=y + 0.1)

    # First row Target and Generated Mean
    vmin = min(np.min(target), np.min(gen_ens_mean), np.min(gen_ens_1), np.min(gen_ens_2))
    vmax = max(np.max(target), np.max(gen_ens_mean), np.max(gen_ens_1), np.max(gen_ens_2))
    im1 = target.plot(ax=axs[0, 0], vmin=vmin, vmax=vmax, **_plot_params[var_name])
    axs[0, 0].set_title("Target")
    _ = gen_ens_mean.plot(ax=axs[0, 1], vmin=vmin, vmax=vmax, **_plot_params[var_name])  # im2
    axs[0, 1].set_title("Generated: Mean")

    # Second row - Sampled Ensembles
    _ = gen_ens_1.plot(ax=axs[1, 0], vmin=vmin, vmax=vmax, **_plot_params[var_name])  # im3
    axs[1, 0].set_title(f"Generated: Ensemble {ens_idxs[0]}")
    _ = gen_ens_2.plot(ax=axs[1, 1], vmin=vmin, vmax=vmax, **_plot_params[var_name])  # im4
    axs[1, 1].set_title(f"Generated: Ensemble {ens_idxs[1]}")

    # create cbar
    _ = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation="vertical", **cbar_kwargs)  # cbar

    # Add coastlines
    for ax in axs.flat:
        ax.coastlines()

    snapshots[f"image-full-field/{fig_shared_label}/{var_name}"] = wandb.Image(fig)

    # Handle Error plot
    fig, axs = plt.subplots(1, 3, figsize=(16, 4), subplot_kw={"projection": map_transform})
    fig.suptitle(f"{var_name}Error {var_name} - {fig_shared_label}", y=y)

    # Calculate error
    error = gen_ens_mean - target
    error_gen_ens_1 = gen_ens_1 - target
    error_gen_ens_2 = gen_ens_2 - target
    vmin = min(np.min(error), np.min(error_gen_ens_1), np.min(error_gen_ens_2))
    vmax = max(np.max(error), np.max(error_gen_ens_1), np.max(error_gen_ens_2))
    im1 = error.plot(ax=axs[0], vmin=vmin, vmax=vmax, **_plot_params["error"])
    axs[0].set_title("Error - Generated Mean")
    _ = error_gen_ens_1.plot(ax=axs[1], vmin=vmin, vmax=vmax, **_plot_params["error"])  # im2
    axs[1].set_title(f"Error - Generated Ensemble {ens_idxs[0]}")
    _ = error_gen_ens_2.plot(ax=axs[2], vmin=vmin, vmax=vmax, **_plot_params["error"])  # im3
    axs[2].set_title(f"Error - Generated Ensemble {ens_idxs[1]}")

    # create cbar
    _ = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation="vertical", **cbar_kwargs)  # cbar

    # Add coastlines
    for ax in axs.flat:
        ax.coastlines()

    snapshots[f"image-error/{fig_shared_label}/{var_name}"] = wandb.Image(fig)
    if is_ensemble:
        # Handle CRPS plot
        fig, axs = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={"projection": map_transform})

        # convert to xr dataarray
        crps = to_xr_dataarray(crps)

        im1 = crps.plot(ax=axs, **_plot_params["crps"])
        axs.set_title(f"{fig_shared_label} CRPS {var_name}")
        axs.coastlines()

        snapshots[f"image-crps-fair/{fig_shared_label}/{var_name}"] = wandb.Image(fig)
    #
    snapshots = {k.strip("/").replace("//", "/"): v for k, v in snapshots.items()}
    return snapshots


class RollingPlotFormats:
    def __init__(
        self,
        all_keys: list,
        # unique_base_keys: list = None,
        base_keys_sep: str = "_",
        ignore_common_prefix: bool = True,
        cmaps: list = None,
        cmap_strength_start: float = 0.75,
        linestyles: list = None,
        linestyle_only_for_single_key: bool = True,
        extra_formats: Dict[str, Dict[str, Any]] = None,
        use_hatches: bool = False,
        always_return_same_format_for_same_key: bool = True,
        shift_cmaps_by: int = 0,
    ):
        """
        Args:
            all_keys: list of all unique keys to be potentially plotted
            base_keys_sep: separator to map from full key to base key (stem), e.g. 'loss' for 'loss_train' and 'loss_val'
            ignore_common_prefix: if True, ignore common prefix/suffix for labels
            cmaps: list of cmaps to use
            cmap_strength_start: start strength of cmap (so that we don't use too bright colors)
            linestyles: list of linestyles to use
            linestyle_only_for_single_key: if True, only use linestyles for single-key plots
            extra_formats: dict of extra formats to use for specific keys
            use_hatches: if True, use hatches (use for bar plots)
            always_return_same_format_for_same_key: if True, always return the same format for the same key
        """
        all_keys = list(all_keys)
        self.sep = base_keys_sep
        self.use_hatches = use_hatches

        # extra formats
        extra_formats = extra_formats or {}
        self.extra_formats = {k.lower(): v for k, v in extra_formats.items()}

        self.ignore_common_prefix = ignore_common_prefix
        self.common_prefix = os.path.commonprefix(all_keys)

        # if unique_base_keys is None:
        base_keys = [self.get_base_key(k) for k in all_keys]
        unique_base_keys = sorted(list(set(base_keys)))
        self.unique_base_keys = unique_base_keys

        cmaps = cmaps or []
        cmaps_pair = cmaps or []
        try:
            import seaborn as sns
            from matplotlib.colors import ListedColormap

            # Use colorblind seaborn palette
            sns_cmaps = sns.color_palette("colorblind", len(unique_base_keys), as_cmap=True)
            sns_cmaps_plot_pair = [
                "Blues",
                "Oranges",
                "Greens",
                "Purples",
                "Greys",
                "Reds",
                "YlOrBr",
                "YlOrRd",
                "OrRd",
                "PuRd",
                "RdPu",
                "BuPu",
                "GnBu",
                "PuBu",
                "YlGnBu",
                "PuBuGn",
                "BuGn",
                "YlGn",
            ]
            if len(unique_base_keys) > len(sns_cmaps):
                print(f"Not enough colormaps for {unique_base_keys}, sns_cmaps: {sns_cmaps}")
            for i in range(len(unique_base_keys)):
                i = i + shift_cmaps_by
                index = i % len(sns_cmaps) if i >= len(sns_cmaps) else i
                cmaps.append(ListedColormap(sns_cmaps[index]))
                cmaps_pair.append(sns_cmaps_plot_pair[index])
            # cmaps += ListedColormap(sns_cmaps)
            # cmaps += [ListedColormap(sns_cmaps(i)) for i in range(len(unique_base_keys))]
        except ImportError:
            cmaps += [
                c
                for c in [
                    "Greens",
                    "Oranges",
                    "Blues",
                    "Purples",
                    "Greys",
                    "Reds",
                    "YlOrBr",
                    "YlOrRd",
                    "OrRd",
                    "PuRd",
                    "RdPu",
                    "BuPu",
                    "GnBu",
                    "PuBu",
                    "YlGnBu",
                    "PuBuGn",
                    "BuGn",
                    "YlGn",
                ]
                if c not in cmaps
            ]

        HATCHES = ["/", "o", "+", "|", "-", "///", "x", "O", ".", "*"]
        self.key_to_hatch = {k: h for k, h in zip(unique_base_keys, itertools.cycle(HATCHES))}

        # linestyles
        fixed_linestyles = {k.lower(): v for k, v in FIXED_LINE_STYLES.items()}
        fixed_linewidths = {k.lower(): v for k, v in FIXED_LINE_WIDTHS.items()}
        linestyles = linestyles or ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
        linestyles = [ls for ls in linestyles if ls not in fixed_linestyles.values()]
        linestyles += list(fixed_linestyles.values())  # add to the end (only use if really necessary)
        if linestyle_only_for_single_key:  # remove '-' if multiple keys
            linestyles = [ls for ls in linestyles if ls != "-"]
        # make iterator
        linestyles = itertools.cycle(linestyles)

        self.max_key_occurences, self.linestyles, self.linewidths = (
            dict(),
            dict(),
            dict(),
        )
        for i, key in enumerate(unique_base_keys):
            n_occurences = len([1 for k in base_keys if key == k])
            self.max_key_occurences[key] = n_occurences

            if key.lower() in fixed_linestyles.keys():
                self.linestyles[key] = fixed_linestyles[key.lower()]
            elif key.split("-")[-1].lower() in fixed_linestyles.keys():
                self.linestyles[key] = fixed_linestyles[key.split("-")[-1].lower()]
            elif linestyle_only_for_single_key and n_occurences > 1:
                self.linestyles[key] = "-"
            else:
                self.linestyles[key] = next(linestyles)
            # print(f"key: {key}, n_occurences: {n_occurences}, linestyle: {self.linestyles[key]}")

            if key.lower() in fixed_linewidths.keys():
                self.linewidths[key] = fixed_linewidths[key.lower()]
            elif n_occurences == 1:
                self.linewidths[key] = 2.5
            else:
                self.linewidths[key] = 1.4

        # assert len(unique_base_keys) <= len(cmaps), f'Not enough colormaps for {unique_base_keys}'
        fixed_cmaps = {k.lower(): v for k, v in FIXED_CMAPS.items()}
        key_to_cmap = {k: fixed_cmaps[k.lower()] for k in unique_base_keys if k.lower() in fixed_cmaps.keys()}
        cmaps_remaining = itertools.cycle([c for c in cmaps if c not in key_to_cmap.values()])
        for i, k in enumerate(unique_base_keys):
            if k in key_to_cmap.keys():
                continue
            key_to_cmap[k] = next(cmaps_remaining) if self.max_key_occurences[k] == 1 else cmaps_pair[i]

        self.key_to_cmap = {k: plt.get_cmap(v) for k, v in key_to_cmap.items()}
        # print(f"key_to_cmap: {self.key_to_cmap}")

        self.cmap_strength_start = dict()
        for key in unique_base_keys:
            if self.max_key_occurences[key] == 1:
                self.cmap_strength_start[key] = 0.66
            elif self.max_key_occurences[key] == 2:
                self.cmap_strength_start[key] = 0.75
            else:
                self.cmap_strength_start[key] = cmap_strength_start

        self.pos_per_cmap = {k: self.cmap_strength_start[k] for k in unique_base_keys}
        if always_return_same_format_for_same_key:
            self._format_cache = dict()
        else:
            self._format_cache = None

    def get_base_key(self, key):
        if self.ignore_common_prefix:
            key = key.replace(self.common_prefix, "").lstrip("-_")
        return key.split(self.sep)[0] if self.sep is not None else key

    def __getitem__(self, key: str) -> Dict[str, Any]:
        if self._format_cache is not None and key in self._format_cache.keys():
            return self._format_cache[key]
        base_key = self.get_base_key(key)
        cur_cmap_pos = self.pos_per_cmap[base_key]
        max_cmap_pos = self.max_key_occurences[base_key]
        color = self.key_to_cmap[base_key](cur_cmap_pos / max_cmap_pos)
        format_dict = {
            "color": color,
            "linestyle": self.linestyles[base_key],
            "linewidth": self.linewidths[base_key],
        }
        if base_key.lower() in self.extra_formats:
            format_dict.update(self.extra_formats[base_key.lower()])
        # print(f"key: {key}, base_key: {base_key}, format_dict: {format_dict}")

        if self.use_hatches:
            # use hatches for every second line (so that we can see the lines better)
            if (cur_cmap_pos - self.cmap_strength_start[base_key]) % 2 == 0 and max_cmap_pos > 1:
                format_dict["hatch"] = self.key_to_hatch[base_key]
                # make hatch lines thinner
                format_dict["linewidth"] = 1.0

        # update pos counter
        if cur_cmap_pos + 1 > max_cmap_pos:
            self.pos_per_cmap[base_key] = self.cmap_strength_start[base_key]  # reset
        else:
            self.pos_per_cmap[base_key] += 1
        if self._format_cache is not None:
            self._format_cache[key] = format_dict
        return format_dict

    def get(self, key: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            return self[key]
        except KeyError:
            return default


# Placeholder for plotting.RollingPlotFormats by Gemini
class MockRollingPlotFormats:
    def __init__(
        self,
        keys: List[str],
        base_keys_sep: str,
        extra_formats: Dict,
        linestyle_only_for_single_key: bool,
        shift_cmaps_by: int,
    ):
        import seaborn as sns

        self.keys = keys
        self.base_keys_sep = base_keys_sep
        self.extra_formats = extra_formats
        self.linestyle_only_for_single_key = linestyle_only_for_single_key
        self.shift_cmaps_by = shift_cmaps_by
        self.num_base_colors = 10  # Number of distinct colors in a typical seaborn palette
        # self.palette = sns.color_palette("deep", self.num_base_colors)  # Example palette
        self.palette = sns.color_palette("colorblind", self.num_base_colors)  # Example palette
        self.linestyles = ["-", "--", "-.", ":"]
        self.key_to_format = {}
        self._assign_formats()

    def _assign_formats(self):
        base_key_to_color_idx = {}
        base_key_counter = 0
        current_linestyle_idx = {}

        for key_idx, full_key in enumerate(self.keys):
            if full_key in self.extra_formats:
                self.key_to_format[full_key] = self.extra_formats[full_key]
                continue

            base_key = full_key.split(self.base_keys_sep)[0]

            if base_key not in base_key_to_color_idx:
                base_key_to_color_idx[base_key] = (base_key_counter + self.shift_cmaps_by) % self.num_base_colors
                base_key_counter += 1
                current_linestyle_idx[base_key] = 0

            color = self.palette[base_key_to_color_idx[base_key]]
            linestyle = self.linestyles[current_linestyle_idx[base_key] % len(self.linestyles)]

            # Determine if this key represents a single variant or one of many for its base
            is_single_variant = True
            if self.base_keys_sep in full_key:  # means it's like "model@$@0"
                # Count how many variants exist for this base_key
                count = sum(1 for k_check in self.keys if k_check.startswith(base_key + self.base_keys_sep))
                if count > 1:
                    is_single_variant = False

            fmt = {"color": color}
            if not (self.linestyle_only_for_single_key and not is_single_variant):
                fmt["linestyle"] = linestyle

            self.key_to_format[full_key] = fmt
            current_linestyle_idx[base_key] += 1

    def get(self, key: str, default: Optional[Dict] = None) -> Dict:
        return self.key_to_format.get(key, default if default is not None else {})


if __name__ == "__main__":
    api = wandb.Api()
    run = api.run("convex-diffusion/ConvexDiffusion/jxpmf2t7")
    print(type(run.group), run.group)
    # pritn other run ids with same group
    group: str = run.group
    project = api.project("ConvexDiffusion", entity="convex-diffusion")
    print(project.__dict__)

    arr = np.zeros((5, 10))
    for i, run in enumerate([1, 2, 3, 4, 5]):
        arr[i, :] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(arr)
