"""Plotting helpers for IntelliCage place-learning analyses.

The plotting functions aim to keep the new Python figures close to the visual
logic of the legacy MATLAB outputs while still using a cleaner, reusable
implementation. The main design choices are:

- step plots for binned trajectories
- optional SEM or standard deviation shading around the mean
- phase and day annotations
- awake/sleep background shading
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


GROUP_COLORS = {
    "WT": "#264653",
    "tdTomato": "#6c757d",
    "Tau 66-421": "#2a9d8f",
    "Tau 1-421": "#e9a820",
    "Tau 1-441": "#4ade80",
}
PHASE_COLORS = {
    1: "#bfe4f7",
    2: "#6fb2e5",
    3: "#3d80b8",
    4: "#1f2a78",
}
DEFAULT_SLEEP_SHADE_COLOR = "#e6e6e6"


def _group_color(group_name: str) -> str:
    """Return a stable display color for one pathology group."""

    return GROUP_COLORS.get(group_name, "#457b9d")


def _prepare_output_path(output_path: Path) -> None:
    """Create the parent directory for one output file if necessary."""

    output_path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_filename_part(value: str) -> str:
    """Convert a label into a filesystem-friendly filename part."""

    return (
        value.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
    )


def _spread_column(spread_metric: str) -> str:
    """Return the summary column name for the selected spread metric."""

    return "std_value" if spread_metric == "std" else "sem_value"


def _add_awake_sleep_background(
    ax: plt.Axes,
    *,
    x_end: float,
    x_start: float = 0.0,
    origin_clock_hour: float,
    awake_start_clock_hour: float,
    awake_end_clock_hour: float,
    label_y: float | None = None,
) -> None:
    """Shade sleep intervals based on a wall-clock cycle."""

    if awake_end_clock_hour <= awake_start_clock_hour:
        raise ValueError("Only same-day awake windows are supported.")

    if label_y is None:
        y_low, y_high = ax.get_ylim()
        label_y = y_high - (y_high - y_low) * 0.07

    cycle_start = x_start - 24.0
    while cycle_start <= x_end + 24.0:
        awake_start = cycle_start + ((awake_start_clock_hour - origin_clock_hour) % 24.0)
        awake_end = awake_start + (awake_end_clock_hour - awake_start_clock_hour)
        sleep1_start = cycle_start
        sleep1_end = awake_start
        sleep2_start = awake_end
        sleep2_end = cycle_start + 24.0

        for left, right in (
            (sleep1_start, sleep1_end),
            (sleep2_start, sleep2_end),
        ):
            if right <= x_start or left >= x_end:
                continue
            draw_left = max(left, x_start)
            draw_right = min(right, x_end)
            ax.axvspan(
                draw_left,
                draw_right,
                color=DEFAULT_SLEEP_SHADE_COLOR,
                alpha=0.55,
                linewidth=0,
                zorder=0,
            )
            if draw_right - draw_left >= 5:
                ax.text(
                    draw_left + (draw_right - draw_left) / 2.0,
                    label_y,
                    "sleep",
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="#4d4d4d",
                )
        cycle_start += 24.0


def _add_day_annotations(
    ax: plt.Axes,
    *,
    x_end: float,
    x_start: float = 0.0,
    label_every_days: int = 1,
    starting_day: int = 1,
) -> None:
    """Add day labels across the top of the plot."""

    day_index = 0
    day_start = x_start
    while day_start < x_end:
        day_end = min(day_start + 24.0, x_end)
        if day_index % label_every_days == 0:
            color = "#cfe7f8" if (day_index // label_every_days) % 2 == 0 else "#8fc5ea"
            ax.axvspan(day_start, day_end, ymin=0.93, ymax=0.965, color=color, alpha=0.9, linewidth=0, zorder=1)
            ax.text(
                day_start + (day_end - day_start) / 2.0,
                0.9475,
                f"Day {starting_day + day_index}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#1f1f1f",
                transform=ax.get_xaxis_transform(),
            )
        day_index += 1
        day_start += 24.0


def _add_phase_band(
    ax: plt.Axes,
    phase_window_table: pd.DataFrame,
    *,
    phase_display_names: dict[int, str],
) -> None:
    """Add a colored top band that indicates the active phase."""

    for _, row in phase_window_table.iterrows():
        phase_number = int(row["PhaseNumber"])
        start_hours = float(row["start_hours"])
        end_hours = float(row["end_hours"])
        if end_hours <= start_hours:
            continue
        ax.axvspan(
            start_hours,
            end_hours,
            ymin=0.965,
            ymax=1.0,
            color=PHASE_COLORS.get(phase_number, "#cccccc"),
            alpha=0.75,
            linewidth=0,
            zorder=2,
        )
        ax.text(
            start_hours + (end_hours - start_hours) / 2.0,
            0.982,
            phase_display_names.get(phase_number, f"Phase {phase_number}"),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#1f1f1f",
            transform=ax.get_xaxis_transform(),
        )


def _draw_step_with_band(
    ax: plt.Axes,
    x_starts: pd.Series,
    x_ends: pd.Series,
    y_mean: pd.Series,
    y_spread: pd.Series,
    *,
    color: str,
    label: str | None,
    linewidth: float = 2.8,
) -> None:
    """Draw a step mean trace with matching shaded spread."""

    x = x_starts.to_numpy(dtype=float)
    x_end = x_ends.to_numpy(dtype=float)
    y = y_mean.to_numpy(dtype=float)
    spread = y_spread.to_numpy(dtype=float)
    x_step = np.append(x, x_end[-1])
    y_step = np.append(y, y[-1])
    lower = np.append(y - spread, (y - spread)[-1])
    upper = np.append(y + spread, (y + spread)[-1])

    ax.fill_between(x_step, lower, upper, step="post", color=color, alpha=0.18, linewidth=0, zorder=3)
    ax.step(x_step, y_step, where="post", color=color, linewidth=linewidth, label=label, zorder=4)


def _draw_line_with_band(
    ax: plt.Axes,
    x_centers: pd.Series,
    y_mean: pd.Series,
    y_spread: pd.Series,
    *,
    color: str,
    label: str | None,
    linewidth: float = 2.8,
) -> None:
    """Draw a line mean trace with matching shaded spread."""

    x = x_centers.to_numpy(dtype=float)
    y = y_mean.to_numpy(dtype=float)
    spread = y_spread.to_numpy(dtype=float)
    ax.fill_between(x, y - spread, y + spread, color=color, alpha=0.18, linewidth=0, zorder=3)
    ax.plot(x, y, color=color, linewidth=linewidth, label=label, zorder=4)


def _draw_trace_with_band(
    ax: plt.Axes,
    summary_frame: pd.DataFrame,
    *,
    y_col: str,
    spread_col: str,
    color: str,
    label: str | None,
    plot_style: str = "step",
    linewidth: float = 2.8,
) -> None:
    """Draw either a step or line mean trace with shaded spread."""

    if plot_style == "line":
        _draw_line_with_band(
            ax,
            summary_frame["bin_center_hours"],
            summary_frame[y_col],
            summary_frame[spread_col],
            color=color,
            label=label,
            linewidth=linewidth,
        )
    else:
        _draw_step_with_band(
            ax,
            summary_frame["bin_start_hours"],
            summary_frame["bin_end_hours"],
            summary_frame[y_col],
            summary_frame[spread_col],
            color=color,
            label=label,
            linewidth=linewidth,
        )


def _draw_individual_trace(
    ax: plt.Axes,
    trace_frame: pd.DataFrame,
    *,
    y_col: str,
    color: str,
    plot_style: str = "step",
    linewidth: float = 1.0,
    alpha: float = 0.35,
    label: str | None = None,
) -> None:
    """Draw an individual mouse trace as either a step or line plot."""

    if plot_style == "line":
        ax.plot(
            trace_frame["bin_center_hours"],
            trace_frame[y_col],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=2,
        )
    else:
        x_step = np.append(trace_frame["bin_start_hours"].to_numpy(), trace_frame["bin_end_hours"].iloc[-1])
        y_step = np.append(trace_frame[y_col].to_numpy(), trace_frame[y_col].iloc[-1])
        ax.step(
            x_step,
            y_step,
            where="post",
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=2,
        )


def _format_rate_axis(ax: plt.Axes) -> None:
    """Format a rate axis in percent."""

    ax.set_ylim(0, 105)
    ax.set_ylabel("Correct visit rate [%]")
    ax.set_yticks(np.arange(0, 110, 10))


def _p_to_star_label(p_value: float) -> str:
    """Convert a p-value to the legacy star notation."""

    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def plot_experiment_overview(
    mouse_bins: pd.DataFrame,
    summary_bins: pd.DataFrame,
    *,
    group_name: str,
    bin_hours: int,
    output_path: Path,
    phase_window_table: pd.DataFrame,
    phase_display_names: dict[int, str],
    spread_metric: str,
    x_end_hours: float | None = None,
    plot_style: str = "step",
    show_individual_labels: bool = True,
) -> None:
    """Plot full-experiment visit activity for one pathology group."""

    group_mouse = mouse_bins.loc[mouse_bins["Group"].astype(str).eq(group_name)].copy()
    group_summary = summary_bins.loc[summary_bins["Group"].astype(str).eq(group_name)].copy()
    if group_mouse.empty or group_summary.empty:
        return

    _prepare_output_path(output_path)
    color = _group_color(group_name)
    spread_col = _spread_column(spread_metric)

    fig, ax = plt.subplots(figsize=(16, 6.5))
    for et_label, mouse_data in group_mouse.groupby("ETLabel", observed=True):
        _draw_individual_trace(
            ax,
            mouse_data,
            y_col="value",
            color="#666666",
            plot_style=plot_style,
            linewidth=1.0,
            alpha=0.35,
            label=str(et_label) if show_individual_labels else None,
        )

    _draw_trace_with_band(
        ax,
        group_summary,
        y_col="mean_value",
        spread_col=spread_col,
        color=color,
        label=f"Group mean ± {spread_metric.upper()}",
        plot_style=plot_style,
    )

    max_hour = float(group_summary["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    ax.set_xlim(0, max_hour)
    ax.set_ylim(0, max(5.0, float(group_summary["mean_value"].max() + group_summary[spread_col].max()) * 1.18))
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=0.0,
        origin_clock_hour=12.0,
        awake_start_clock_hour=8.0,
        awake_end_clock_hour=20.0,
    )
    _add_day_annotations(ax, x_end=max_hour, x_start=0.0, label_every_days=2, starting_day=0)
    _add_phase_band(ax, phase_window_table, phase_display_names=phase_display_names)

    ax.set_title(f"{group_name}: visits across all phases ({bin_hours} h bins)")
    ax.set_xlabel("Elapsed experimental time [hours]")
    ax.set_ylabel("Visits per mouse and bin")
    ax.grid(axis="y", alpha=0.25)
    legend = ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper right")
    if not show_individual_labels:
        for text in legend.get_texts():
            if text.get_text().startswith("ET") or text.get_text().startswith("Lo"):
                text.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_experiment_overview_groups(
    summary_bins: pd.DataFrame,
    *,
    output_path: Path,
    phase_window_table: pd.DataFrame,
    phase_display_names: dict[int, str],
    spread_metric: str,
    x_end_hours: float | None = None,
    plot_style: str = "step",
) -> None:
    """Plot all pathology-group visit means across the full experiment."""

    if summary_bins.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    fig, ax = plt.subplots(figsize=(16, 6.5))
    max_hour = float(summary_bins["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    ax.set_xlim(0, max_hour)
    y_max = float((summary_bins["mean_value"] + summary_bins[spread_col]).max())
    ax.set_ylim(0, max(5.0, y_max * 1.18))

    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=0.0,
        origin_clock_hour=12.0,
        awake_start_clock_hour=8.0,
        awake_end_clock_hour=20.0,
    )
    _add_day_annotations(ax, x_end=max_hour, x_start=0.0, label_every_days=2, starting_day=0)
    _add_phase_band(ax, phase_window_table, phase_display_names=phase_display_names)

    for group_name, group_summary in summary_bins.groupby("Group", observed=True):
        _draw_trace_with_band(
            ax,
            group_summary,
            y_col="mean_value",
            spread_col=spread_col,
            color=_group_color(str(group_name)),
            label=str(group_name),
            plot_style=plot_style,
            linewidth=2.4,
        )

    ax.set_title("Visits across all phases by group")
    ax.set_xlabel("Elapsed experimental time [hours]")
    ax.set_ylabel("Visits per mouse and bin")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_phase2_adaptation(
    primary_summary: pd.DataFrame,
    secondary_summary: pd.DataFrame,
    *,
    group_name: str,
    bin_hours: int,
    output_path: Path,
    secondary_label: str,
    phase_display_name: str,
) -> None:
    """Plot the phase-2 adaptation metric as paired bar charts."""

    visits_group = primary_summary.loc[primary_summary["Group"].astype(str).eq(group_name)].copy()
    secondary_group = secondary_summary.loc[secondary_summary["Group"].astype(str).eq(group_name)].copy()
    if visits_group.empty or secondary_group.empty:
        return

    _prepare_output_path(output_path)
    color = _group_color(group_name)

    fig, ax = plt.subplots(figsize=(14, 6))
    width = float(bin_hours) * 0.42
    x = visits_group["bin_center_hours"]
    ax.bar(
        x - width / 2.0,
        visits_group["mean_value"],
        width=width,
        color="#cfcfcf",
        edgecolor="#7f7f7f",
        yerr=visits_group["sem_value"],
        capsize=2,
        label="Visits",
    )
    ax.bar(
        x + width / 2.0,
        secondary_group["mean_value"],
        width=width,
        color=color,
        edgecolor=color,
        yerr=secondary_group["sem_value"],
        capsize=2,
        label=secondary_label,
        alpha=0.88,
    )

    ax.set_title(f"{group_name}: {phase_display_name} adaptation ({bin_hours} h bins)")
    ax.set_xlabel("Hours since start of phase 2")
    ax.set_ylabel("Mean count per mouse and bin")
    ax.set_xlim(0, float(visits_group["bin_end_hours"].max()))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_phase_learning_counts(
    mouse_bins: pd.DataFrame,
    summary_bins: pd.DataFrame,
    *,
    group_name: str,
    phase_number: int,
    phase_display_name: str,
    bin_hours: int,
    output_path: Path,
    spread_metric: str,
    x_end_hours: float | None = None,
    plot_style: str = "step",
) -> None:
    """Plot correct-visit counts for phase 3 or phase 4."""

    group_mouse = mouse_bins.loc[mouse_bins["Group"].astype(str).eq(group_name)].copy()
    group_summary = summary_bins.loc[summary_bins["Group"].astype(str).eq(group_name)].copy()
    if group_mouse.empty or group_summary.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    color = _group_color(group_name)

    fig, ax = plt.subplots(figsize=(14, 6))
    for _, mouse_data in group_mouse.groupby("ETLabel", observed=True):
        _draw_individual_trace(
            ax,
            mouse_data,
            y_col="value",
            color="#9aa0a6",
            plot_style=plot_style,
            linewidth=1.0,
            alpha=0.35,
        )

    max_hour = float(group_summary["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    y_max = max(5.0, float(group_summary["mean_value"].max() + group_summary[spread_col].max()) * 1.18)
    ax.set_xlim(0, max_hour)
    ax.set_ylim(0, y_max)
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=0.0,
        origin_clock_hour=8.0,
        awake_start_clock_hour=8.0,
        awake_end_clock_hour=20.0,
    )
    _add_day_annotations(ax, x_end=max_hour, x_start=0.0, label_every_days=1, starting_day=1)
    _draw_trace_with_band(
        ax,
        group_summary,
        y_col="mean_value",
        spread_col=spread_col,
        color=color,
        label=f"Group mean ± {spread_metric.upper()}",
        plot_style=plot_style,
    )

    ax.set_title(f"{group_name}: rewarded correct visits ({phase_display_name}, {bin_hours} h bins)")
    ax.set_xlabel(f"Hours since start of {phase_display_name}")
    ax.set_ylabel("Rewarded correct visits per mouse and bin")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_phase_learning_rate(
    mouse_bins: pd.DataFrame,
    summary_bins: pd.DataFrame,
    *,
    group_name: str,
    phase_number: int,
    phase_display_name: str,
    bin_hours: int,
    output_path: Path,
    spread_metric: str,
    x_end_hours: float | None = None,
    plot_style: str = "step",
) -> None:
    """Plot correct-visit rates for phase 3 or phase 4."""

    group_mouse = mouse_bins.loc[mouse_bins["Group"].astype(str).eq(group_name)].copy()
    group_summary = summary_bins.loc[summary_bins["Group"].astype(str).eq(group_name)].copy()
    if group_mouse.empty or group_summary.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    color = _group_color(group_name)

    fig, ax = plt.subplots(figsize=(14, 6))
    for _, mouse_data in group_mouse.groupby("ETLabel", observed=True):
        clean = mouse_data.dropna(subset=["value"])
        if clean.empty:
            continue
        clean = clean.copy()
        clean["value_pct"] = clean["value"] * 100.0
        _draw_individual_trace(
            ax,
            clean,
            y_col="value_pct",
            color="#b8bdc4",
            plot_style=plot_style,
            linewidth=1.0,
            alpha=0.35,
        )

    max_hour = float(group_summary["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    ax.set_xlim(0, max_hour)
    _format_rate_axis(ax)
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=0.0,
        origin_clock_hour=8.0,
        awake_start_clock_hour=8.0,
        awake_end_clock_hour=20.0,
    )
    _add_day_annotations(ax, x_end=max_hour, x_start=0.0, label_every_days=1, starting_day=1)
    group_summary = group_summary.copy()
    group_summary["mean_value_pct"] = group_summary["mean_value"] * 100.0
    group_summary["spread_pct"] = group_summary[spread_col] * 100.0
    _draw_trace_with_band(
        ax,
        group_summary,
        y_col="mean_value_pct",
        spread_col="spread_pct",
        color=color,
        label=f"Group mean ± {spread_metric.upper()}",
        plot_style=plot_style,
    )
    ax.axhline(25.0, color="#4f4f4f", linestyle="--", linewidth=1.4, label="Chance level (25%)")

    ax.set_title(f"{group_name}: correct visit rate ({phase_display_name}, {bin_hours} h bins)")
    ax.set_xlabel(f"Hours since start of {phase_display_name}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_phase_learning_rate_groups(
    summary_bins: pd.DataFrame,
    *,
    phase_number: int,
    phase_display_name: str,
    bin_hours: int,
    output_path: Path,
    spread_metric: str,
    x_end_hours: float | None = None,
    plot_style: str = "step",
) -> None:
    """Plot all pathology-group rate means in one figure."""

    if summary_bins.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    fig, ax = plt.subplots(figsize=(15, 6.5))
    max_hour = float(summary_bins["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    ax.set_xlim(0, max_hour)
    _format_rate_axis(ax)
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=0.0,
        origin_clock_hour=8.0,
        awake_start_clock_hour=8.0,
        awake_end_clock_hour=20.0,
    )
    _add_day_annotations(ax, x_end=max_hour, x_start=0.0, label_every_days=1, starting_day=1)

    for group_name, group_summary in summary_bins.groupby("Group", observed=True):
        color = _group_color(str(group_name))
        group_summary = group_summary.copy()
        group_summary["mean_value_pct"] = group_summary["mean_value"] * 100.0
        group_summary["spread_pct"] = group_summary[spread_col] * 100.0
        _draw_trace_with_band(
            ax,
            group_summary,
            y_col="mean_value_pct",
            spread_col="spread_pct",
            color=color,
            label=str(group_name),
            plot_style=plot_style,
            linewidth=2.4,
        )

    ax.axhline(25.0, color="#4f4f4f", linestyle="--", linewidth=1.4, label="Chance level (25%)")
    ax.set_title(f"Correct visit rate across groups ({phase_display_name}, {bin_hours} h bins)")
    ax.set_xlabel(f"Hours since start of {phase_display_name}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_phase_learning_counts_groups(
    summary_bins: pd.DataFrame,
    *,
    phase_display_name: str,
    bin_hours: int,
    output_path: Path,
    spread_metric: str,
    x_end_hours: float | None = None,
    plot_style: str = "step",
    title_prefix: str = "Rewarded correct visit counts across groups",
    ylabel: str = "Rewarded correct visits per mouse and bin",
) -> None:
    """Plot all pathology-group correct-visit counts in one figure."""

    if summary_bins.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    fig, ax = plt.subplots(figsize=(15, 6.5))
    max_hour = float(summary_bins["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    ax.set_xlim(0, max_hour)
    y_max = float((summary_bins["mean_value"] + summary_bins[spread_col]).max())
    ax.set_ylim(0, max(5.0, y_max * 1.18))
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=0.0,
        origin_clock_hour=8.0,
        awake_start_clock_hour=8.0,
        awake_end_clock_hour=20.0,
    )
    _add_day_annotations(ax, x_end=max_hour, x_start=0.0, label_every_days=1, starting_day=1)

    for group_name, group_summary in summary_bins.groupby("Group", observed=True):
        _draw_trace_with_band(
            ax,
            group_summary,
            y_col="mean_value",
            spread_col=spread_col,
            color=_group_color(str(group_name)),
            label=str(group_name),
            plot_style=plot_style,
            linewidth=2.4,
        )

    ax.set_title(f"{title_prefix} ({phase_display_name}, {bin_hours} h bins)")
    ax.set_xlabel(f"Hours since start of {phase_display_name}")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_experiment_dual_metric_bars(
    primary_summary: pd.DataFrame,
    secondary_summary: pd.DataFrame,
    *,
    group_name: str,
    bin_hours: int,
    output_path: Path,
    secondary_label: str,
    phase_window_table: pd.DataFrame,
    phase_display_names: dict[int, str],
) -> None:
    """Plot paired bar summaries across the full experiment timeline."""

    visits_group = primary_summary.loc[primary_summary["Group"].astype(str).eq(group_name)].copy()
    secondary_group = secondary_summary.loc[secondary_summary["Group"].astype(str).eq(group_name)].copy()
    if visits_group.empty or secondary_group.empty:
        return

    _prepare_output_path(output_path)
    color = _group_color(group_name)
    fig, ax = plt.subplots(figsize=(16, 6.5))
    max_hour = float(visits_group["bin_end_hours"].max())
    ax.set_xlim(0, max_hour)
    y_max = max(
        float((visits_group["mean_value"] + visits_group["sem_value"]).max()),
        float((secondary_group["mean_value"] + secondary_group["sem_value"]).max()),
    )
    ax.set_ylim(0, max(5.0, y_max * 1.18))

    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=0.0,
        origin_clock_hour=12.0,
        awake_start_clock_hour=8.0,
        awake_end_clock_hour=20.0,
    )
    _add_day_annotations(ax, x_end=max_hour, x_start=0.0, label_every_days=2, starting_day=0)
    _add_phase_band(ax, phase_window_table, phase_display_names=phase_display_names)

    width = float(bin_hours) * 0.42
    x = visits_group["bin_center_hours"]
    ax.bar(
        x - width / 2.0,
        visits_group["mean_value"],
        width=width,
        color="#cfcfcf",
        edgecolor="#7f7f7f",
        yerr=visits_group["sem_value"],
        capsize=2,
        label="Visits",
        zorder=3,
    )
    ax.bar(
        x + width / 2.0,
        secondary_group["mean_value"],
        width=width,
        color=color,
        edgecolor=color,
        yerr=secondary_group["sem_value"],
        capsize=2,
        label=secondary_label,
        alpha=0.85,
        zorder=3,
    )

    ax.set_title(f"{group_name}: visits vs {secondary_label.lower()} across all phases ({bin_hours} h bins)")
    ax.set_xlabel("Elapsed experimental time [hours]")
    ax.set_ylabel("Mean count per mouse and bin")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_phase_activity_boxplot(
    mouse_phase_activity: pd.DataFrame,
    stats_table: pd.DataFrame,
    *,
    phase_display_names: dict[int, str],
    output_path: Path,
) -> None:
    """Plot median hourly activity per phase and pathology group."""

    if mouse_phase_activity.empty:
        return

    _prepare_output_path(output_path)
    phase_numbers = sorted(mouse_phase_activity["PhaseNumber"].unique())
    group_order = [str(group) for group in mouse_phase_activity["Group"].dropna().unique()]
    fig, ax = plt.subplots(figsize=(12, 9))

    positions: dict[tuple[str, int], float] = {}
    group_centers: dict[str, float] = {}
    box_width = 0.18
    offsets = np.linspace(-0.30, 0.30, num=len(phase_numbers))
    legend_handles: list[Patch] = []

    for phase_number in phase_numbers:
        legend_handles.append(
            Patch(
                facecolor=PHASE_COLORS.get(int(phase_number), "#cccccc"),
                edgecolor=PHASE_COLORS.get(int(phase_number), "#cccccc"),
                label=phase_display_names.get(int(phase_number), f"Phase {phase_number}"),
            )
        )

    for group_index, group_name in enumerate(group_order, start=1):
        group_center = float(group_index)
        group_centers[group_name] = group_center
        medians_x: list[float] = []
        medians_y: list[float] = []

        for offset, phase_number in zip(offsets, phase_numbers):
            position = group_center + float(offset)
            positions[(group_name, int(phase_number))] = position
            values = mouse_phase_activity.loc[
                mouse_phase_activity["Group"].astype(str).eq(group_name)
                & mouse_phase_activity["PhaseNumber"].eq(int(phase_number)),
                "median_visits_per_hour",
            ].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            box = ax.boxplot(
                values,
                positions=[position],
                widths=box_width,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "#ff5a36", "linewidth": 2.3},
                whiskerprops={"color": "#555555", "linewidth": 1.0, "linestyle": "--"},
                capprops={"color": "#555555", "linewidth": 1.0},
                boxprops={"linewidth": 1.1},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(PHASE_COLORS.get(int(phase_number), "#cccccc"))
                patch.set_edgecolor(PHASE_COLORS.get(int(phase_number), "#cccccc"))
                patch.set_alpha(0.95)
            for flier in box["fliers"]:
                flier.set_marker("o")
                flier.set_markersize(6.5)
                flier.set_markerfacecolor("white")
                flier.set_markeredgecolor("black")
                flier.set_markeredgewidth(1.2)
            medians_x.append(position)
            medians_y.append(float(np.median(values)))

        if medians_x:
            ax.plot(medians_x, medians_y, color="red", marker="o", linewidth=2.1, label="Median" if group_index == 1 else None)

    y_min = max(0.0, float(mouse_phase_activity["median_visits_per_hour"].min()) - 4.0)
    y_max = float(mouse_phase_activity["median_visits_per_hour"].max()) + 8.0
    ax.set_ylim(y_min, y_max)

    for _, row in stats_table.iterrows():
        group_name = str(row["Group"])
        phase_number = int(row["PhaseNumber"])
        label = _p_to_star_label(float(row["pairwise_p_value_vs_phase1"]))
        if not label:
            continue
        position = positions.get((group_name, phase_number))
        if position is None:
            continue
        ax.text(position, y_min + 0.6, label, color="red", fontsize=13, ha="center", va="bottom", fontweight="bold")

    xticks: list[float] = []
    xlabels: list[str] = []
    for group_name in group_order:
        xticks.append(group_centers[group_name])
        n_value = mouse_phase_activity.loc[mouse_phase_activity["Group"].astype(str).eq(group_name), "ET"].nunique()
        xlabels.append(f"{group_name}\nn={n_value}")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_title("Mice activity per group and phase")
    ax.set_ylabel("Median number of corner visits per hour")
    ax.grid(axis="y", alpha=0.22)

    median_handle = plt.Line2D([0], [0], color="red", marker="o", linewidth=2.1, label="Median")
    outlier_handle = plt.Line2D(
        [0],
        [0],
        color="black",
        marker="o",
        linestyle="None",
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Outlier",
    )
    ax.legend(handles=[*legend_handles, median_handle, outlier_handle], frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
