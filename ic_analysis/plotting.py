"""Reusable figure functions for IntelliCage analysis outputs.

The plotting functions aim to keep the new Python figures close to the visual
logic of the legacy MATLAB outputs while still using a cleaner, reusable
implementation. The main design choices are:

- step plots for binned trajectories
- optional SEM or standard deviation shading around the mean
- phase and day annotations
- awake/sleep background shading

author: Fabrizio Musacchio
date: May 2026
"""
# %% IMPORTS
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import textwrap
from typing import Any, Literal

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
# %% CONSTANTS
GROUP_COLORS = {
    "Group 1": "#264653",
    "Group 2": "#6c757d",
    "Group 3": "#2a9d8f",
    "Group 4": "#e9a820",
    "Group 5": "#4ade80",
    "Group 6": "#457b9d",
    "Group 7": "#c7523f",
    "Group 8": "#8a5cf6",
    "Group 9": "#2f9e44",
    "Group 10": "#8d6e63"}
PHASE_COLORS = {
    1: "#bfe4f7",
    2: "#6fb2e5",
    3: "#3d80b8",
    4: "#1f2a78"}
DEFAULT_SLEEP_SHADE_COLOR = "#e6e6e6"
ROLE_COLORS = {
    "PL target corner": "#2a9d8f",
    "PR target corner": "#e76f51",
    "Neutral corner 1": "#7f7f7f",
    "Neutral corner 2": "#b0b0b0"}
DRINKING_VISIT_COLOR = "#f28e2b"
CM_TO_INCH = 2.54
DEFAULT_FIGURE_SIZE_CM = {
    "LONG_FIGSIZE_CM": (18.2, 7.4),
    "LONG_FIGSIZE_2_CM": (15.2, 7.4),
    "PHASE2_FIGSIZE_CM": (10.4, 7.0),
    "MEDIUM_FIGSIZE_CM": (11.8, 7.6),
    "MEDIUM_WIDE_FIGSIZE_CM": (12.8, 8.0),
    "SEGMENT_FIGSIZE_CM": (12.6, 7.9),
    "VIOLIN_FIGSIZE_CM": (5.8, 7.2),
    "ONSET_FIGSIZE_CM": (5.8, 7.0),
    "ACTIVITY_FIGSIZE_CM": (8.8, 8.1),
    "WIDE_GROUP_FIGSIZE_CM": (18.2, 7.4)}
LONG_FIGSIZE_CM = DEFAULT_FIGURE_SIZE_CM["LONG_FIGSIZE_CM"]
LONG_FIGSIZE_2_CM = DEFAULT_FIGURE_SIZE_CM["LONG_FIGSIZE_2_CM"]
PHASE2_FIGSIZE_CM = DEFAULT_FIGURE_SIZE_CM["PHASE2_FIGSIZE_CM"]
MEDIUM_FIGSIZE_CM = DEFAULT_FIGURE_SIZE_CM["MEDIUM_FIGSIZE_CM"]
MEDIUM_WIDE_FIGSIZE_CM = DEFAULT_FIGURE_SIZE_CM["MEDIUM_WIDE_FIGSIZE_CM"]
SEGMENT_FIGSIZE_CM = DEFAULT_FIGURE_SIZE_CM["SEGMENT_FIGSIZE_CM"]
VIOLIN_FIGSIZE_CM = DEFAULT_FIGURE_SIZE_CM["VIOLIN_FIGSIZE_CM"]
ONSET_FIGSIZE_CM = DEFAULT_FIGURE_SIZE_CM["ONSET_FIGSIZE_CM"]
ACTIVITY_FIGSIZE_CM = DEFAULT_FIGURE_SIZE_CM["ACTIVITY_FIGSIZE_CM"]
WIDE_GROUP_FIGSIZE_CM = DEFAULT_FIGURE_SIZE_CM["WIDE_GROUP_FIGSIZE_CM"]
# %% FUNCTIONS
def set_group_colors(color_mapping: dict[str, str] | None) -> None:
    """Update the global experimental-group color mapping used across all plots."""

    if not color_mapping:
        return
    GROUP_COLORS.update({str(key): str(value) for key, value in color_mapping.items()})

def set_figure_size_presets(size_mapping: dict[str, tuple[float, float]] | None) -> None:
    """Update the global centimeter-based figure-size presets used by all plots."""

    if not size_mapping:
        return
    globals_dict = globals()
    for key, value in size_mapping.items():
        if key not in DEFAULT_FIGURE_SIZE_CM:
            continue
        width_cm, height_cm = value
        globals_dict[key] = (float(width_cm), float(height_cm))

def configure_plot_style(*, font_size: float = 10.0, font_family: str = "Arial") -> None:
    """Apply global matplotlib defaults for IntelliCage publication figures."""

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [font_family, "Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["font.size"] = float(font_size)
    mpl.rcParams["axes.titlesize"] = float(font_size)
    mpl.rcParams["axes.labelsize"] = float(font_size)
    mpl.rcParams["xtick.labelsize"] = max(7.0, float(font_size) - 1.0)
    mpl.rcParams["ytick.labelsize"] = max(7.0, float(font_size) - 1.0)
    mpl.rcParams["legend.fontsize"] = max(6.0, float(font_size) - 2.0)
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False

def _font_size(offset: float = 0.0) -> float:
    """Return a plot text size relative to the configured global base size."""

    return float(mpl.rcParams["font.size"]) + float(offset)

def _figsize_cm(width_cm: float, height_cm: float) -> tuple[float, float]:
    """Convert a figure size from centimeters to matplotlib inches."""

    return (float(width_cm) / CM_TO_INCH, float(height_cm) / CM_TO_INCH)

def _resolve_figsize_cm(default_size: tuple[float, float], override: tuple[float, float] | None = None) -> tuple[float, float]:
    """Return a matplotlib figure size from an optional centimeter override."""

    size = override if override is not None else default_size
    return _figsize_cm(*size)

def _title_start(value: str) -> str:
    """Upper-case the first character of a title fragment when possible."""

    if not value:
        return value
    return value[0].upper() + value[1:]

def _wrap_axis_label(value: str, *, width: int = 28) -> str:
    """Insert simple line breaks into long axis labels for narrow figures."""

    if not value or len(value) <= width:
        return value
    if " [" in value:
        stem, unit = value.split(" [", 1)
        return f"{stem}\n[{unit}"
    wrapped = textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False)
    return "\n".join(wrapped) if wrapped else value

def _wrap_title(value: str, *, width: int = 38) -> str:
    """Wrap long titles across multiple lines without breaking words."""

    if not value or len(value) <= width:
        return value
    protected = value.replace(" h bins)", " h_bins)")
    wrapped = textwrap.wrap(protected, width=width, break_long_words=False, break_on_hyphens=False)
    wrapped = [line.replace("h_bins", "h bins") for line in wrapped]
    return "\n".join(wrapped) if wrapped else value

def _x_axis_unit_factor(x_unit: Literal["hours", "days", "weeks"]) -> tuple[float, str, str]:
    """Return the divisor, axis unit, and short bin unit for one x-axis unit."""

    if x_unit == "hours":
        return 1.0, "h", "h"
    if x_unit == "days":
        return 24.0, "days", "d"
    if x_unit == "weeks":
        return 168.0, "weeks", "wk"
    raise ValueError("x_unit must be one of 'hours', 'days', or 'weeks'.")

def _count_axis_upper(y_max: float) -> float:
    """Reserve headroom so data do not intrude into annotation bands."""

    return max(5.0, float(y_max) * 1.35 + 0.5)

def _group_color(group_name: str) -> str:
    """Return a stable display color for one experimental group."""

    return GROUP_COLORS.get(group_name, "#457b9d")

def _ordered_group_names_from_series(series: pd.Series) -> list[str]:
    """Return experimental groups in categorical order when available."""

    categories = getattr(series.dtype, "categories", None)
    if categories is not None:
        return [str(category) for category in categories if str(category) != "nan"]
    return [str(group) for group in series.dropna().astype(str).unique()]

def _mouse_trace_colors(mouse_labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Return stable distinct colors for individual mouse traces within one panel."""

    if not mouse_labels:
        return {}
    cmap = plt.get_cmap("tab20")
    label_count = max(1, len(mouse_labels))
    return {
        str(label): cmap(index / max(1, label_count - 1)) if label_count > 1 else cmap(0.0)
        for index, label in enumerate(mouse_labels)}

def _prepare_output_path(output_path: Path) -> None:
    """Create the parent directory for one output file if necessary."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save each figure as both PNG and PDF with tight bounding boxes."""

    _prepare_output_path(output_path)
    pdf_output_path = output_path.parent / "pdf" / f"{output_path.stem}.pdf"
    _prepare_output_path(pdf_output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(pdf_output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

def sanitize_filename_part(value: str) -> str:
    """Convert a label into a filesystem-friendly filename part."""

    return (
        value.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct"))

def plot_bottle_preference_groups(
    summary_bins: pd.DataFrame,
    *,
    output_path: Path,
    left_bottle: str,
    right_bottle: str,
    calc: str,
    bin_h: int | float,
    spread_metric: str = "sem",
    plot_style: str = "line",
    x_unit: Literal["hours", "days", "weeks"] = "hours",
    indicate_dots: bool = False,
    figsize_cm: tuple[float, float] | None = None,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    phase_window_table: pd.DataFrame | None = None,
    phase_display_names: dict[int, str] | None = None,
    title_label: str | None = None,
    ylabel: str | None = None,
    origin_clock_hour: float = 6.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep")) -> None:
    """Plot binned bottle consumption or preference across groups."""

    if summary_bins.empty:
        return
    prepared = summary_bins.copy()
    is_preference = calc in {"left_bottle/right_bottle", "right_bottle/left_bottle"}
    if is_preference:
        prepared["mean_value"] = prepared["mean_value"] * 100.0
        prepared["sem_value"] = prepared["sem_value"] * 100.0
        prepared["std_value"] = prepared["std_value"] * 100.0
    spread_col = _spread_column(spread_metric)
    group_names = _ordered_group_names_from_series(prepared["Group"])
    unit_factor, unit_label, short_unit = _x_axis_unit_factor(x_unit)
    fig, ax = plt.subplots(figsize=_resolve_figsize_cm(LONG_FIGSIZE_CM, figsize_cm))
    x_end = float(prepared["bin_end_hours"].max()) / unit_factor
    if x_unit == "hours":
        background_phase_window_table = phase_window_table
    elif phase_window_table is None or phase_window_table.empty:
        background_phase_window_table = phase_window_table
    else:
        background_phase_window_table = phase_window_table.copy()
        background_phase_window_table["start_hours"] = background_phase_window_table["start_hours"].astype(float) / unit_factor
        background_phase_window_table["end_hours"] = background_phase_window_table["end_hours"].astype(float) / unit_factor
    if x_unit == "hours":
        _add_awake_sleep_background(
            ax,
            x_end=x_end,
            origin_clock_hour=origin_clock_hour,
            awake_start_clock_hour=awake_start_clock_hour,
            awake_end_clock_hour=awake_end_clock_hour,
            day_night_indicator=day_night_indicator)
    if background_phase_window_table is not None and not background_phase_window_table.empty:
        _add_phase_band(ax, background_phase_window_table, phase_display_names=phase_display_names or {})
    for group_name in group_names:
        group_data = prepared.loc[prepared["Group"].astype(str).eq(group_name)].sort_values("bin_start_hours")
        x_values = group_data["bin_center_hours"].astype(float) / unit_factor
        y_values = group_data["mean_value"].astype(float)
        spread = group_data[spread_col].fillna(0.0).astype(float)
        color = _group_color(group_name)
        if plot_style == "step":
            ax.step(x_values, y_values, where="mid", color=color, linewidth=2.0, label=group_name)
        else:
            ax.plot(x_values, y_values, color=color, linewidth=2.0, label=group_name)
        if indicate_dots:
            ax.scatter(x_values, y_values, color=color, edgecolor="white", linewidth=0.4, s=18, zorder=5)
        ax.fill_between(x_values, y_values - spread, y_values + spread, color=color, alpha=0.18, linewidth=0)
    if is_preference:
        ax.axhline(50.0, color="#6c757d", linewidth=1.0, linestyle="--")
        ax.set_ylim(0.0, 112.0)
        ax.set_yticks(np.arange(0.0, 101.0, 10.0))
    if ylim is not None:
        current_ylim = ax.get_ylim()
        bottom = current_ylim[0] if ylim[0] is None else float(ylim[0])
        top = current_ylim[1] if ylim[1] is None else float(ylim[1])
        ax.set_ylim(bottom, top)
    if xlim is not None:
        current_xlim = ax.get_xlim()
        left = current_xlim[0] if xlim[0] is None else float(xlim[0])
        right = current_xlim[1] if xlim[1] is None else float(xlim[1])
        ax.set_xlim(left, right)
    else:
        ax.set_xlim(0.0, x_end)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    if yticks is not None:
        ax.set_yticks(list(yticks))
    bin_label = float(bin_h) / unit_factor
    bin_text = f"{bin_label:g} {short_unit}"
    ax.set_xlabel(xlabel or f"Experiment time [{unit_label}], bin={bin_text}")
    ax.set_ylabel(ylabel or _bottle_preference_ylabel(calc, left_bottle=left_bottle, right_bottle=right_bottle))
    ax.set_title(_wrap_title(title_label or _bottle_preference_title(calc, left_bottle=left_bottle, right_bottle=right_bottle)))
    if show_legend:
        _draw_legend(
            ax,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, Path(output_path))

def _bottle_preference_ylabel(calc: str, *, left_bottle: str, right_bottle: str) -> str:
    """Return a readable y-axis label for bottle-preference plots."""

    if calc == "left_bottle":
        return f"{left_bottle} consumption [licks]"
    if calc == "right_bottle":
        return f"{right_bottle} consumption [licks]"
    if calc == "left_bottle/right_bottle":
        return f"{left_bottle} preference [%]"
    return f"{right_bottle} preference [%]"

def _bottle_preference_title(calc: str, *, left_bottle: str, right_bottle: str) -> str:
    """Return a readable title for bottle-preference plots."""

    if calc == "left_bottle":
        return f"{left_bottle} consumption"
    if calc == "right_bottle":
        return f"{right_bottle} consumption"
    if calc == "left_bottle/right_bottle":
        return f"{left_bottle} / total bottle consumption"
    return f"{right_bottle} / total bottle consumption"

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
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Shade sleep intervals and label both sleep and awake periods."""

    if awake_end_clock_hour <= awake_start_clock_hour:
        raise ValueError("Only same-day awake windows are supported.")

    if label_y is None:
        label_y = 0.835

    awake_duration = awake_end_clock_hour - awake_start_clock_hour
    awake_shift = (awake_start_clock_hour - origin_clock_hour) % 24.0
    interval_indices = range(-2, int(np.ceil((x_end - x_start) / 24.0)) + 3)
    awake_intervals: list[tuple[float, float]] = []
    for index in interval_indices:
        awake_left = awake_shift + 24.0 * index
        awake_right = awake_left + awake_duration
        if awake_right <= x_start or awake_left >= x_end:
            continue
        awake_intervals.append((awake_left, awake_right))
    awake_intervals.sort()

    total_range = x_end - x_start
    label_font_size = _font_size(-2.0)
    min_label_width = 5.5
    awake_label = ""
    sleep_label = ""
    if day_night_indicator is not None:
        awake_label = str(day_night_indicator[0])
        sleep_label = str(day_night_indicator[1])
    for awake_left, awake_right in awake_intervals:
        draw_awake_left = max(awake_left, x_start)
        draw_awake_right = min(awake_right, x_end)
        if awake_label and draw_awake_right - draw_awake_left >= min_label_width:
            ax.text(
                draw_awake_left + (draw_awake_right - draw_awake_left) / 2.0,
                label_y,
                awake_label,
                ha="center",
                va="top",
                fontsize=label_font_size,
                color="#4d4d4d",
                transform=ax.get_xaxis_transform())

    current_left = x_start
    for awake_left, awake_right in awake_intervals:
        draw_sleep_left = current_left
        draw_sleep_right = min(max(awake_left, x_start), x_end)
        if draw_sleep_right > draw_sleep_left:
            ax.axvspan(
                draw_sleep_left,
                draw_sleep_right,
                color=DEFAULT_SLEEP_SHADE_COLOR,
                alpha=0.55,
                linewidth=0,
                zorder=0)
            if sleep_label and draw_sleep_right - draw_sleep_left >= min_label_width:
                ax.text(
                    draw_sleep_left + (draw_sleep_right - draw_sleep_left) / 2.0,
                    label_y,
                    sleep_label,
                    ha="center",
                    va="top",
                    fontsize=label_font_size,
                    color="#4d4d4d",
                    transform=ax.get_xaxis_transform())
        current_left = max(current_left, min(awake_right, x_end))

    if current_left < x_end:
        ax.axvspan(
            current_left,
            x_end,
            color=DEFAULT_SLEEP_SHADE_COLOR,
            alpha=0.55,
            linewidth=0,
            zorder=0)
        if sleep_label and x_end - current_left >= min_label_width:
            ax.text(
                current_left + (x_end - current_left) / 2.0,
                label_y,
                sleep_label,
                ha="center",
                va="top",
                fontsize=label_font_size,
                color="#4d4d4d",
                transform=ax.get_xaxis_transform())

def _apply_axis_overrides(
    ax: plt.Axes,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    """Apply explicit user axis overrides after plot defaults."""

    if xlim is not None:
        current_xlim = ax.get_xlim()
        left = current_xlim[0] if xlim[0] is None else float(xlim[0])
        right = current_xlim[1] if xlim[1] is None else float(xlim[1])
        ax.set_xlim(left, right)
    if ylim is not None:
        current_ylim = ax.get_ylim()
        bottom = current_ylim[0] if ylim[0] is None else float(ylim[0])
        top = current_ylim[1] if ylim[1] is None else float(ylim[1])
        ax.set_ylim(bottom, top)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    if yticks is not None:
        ax.set_yticks(list(yticks))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(_wrap_axis_label(ylabel))
    if title is not None:
        ax.set_title(_wrap_title(title))

def _add_day_annotations(
    ax: plt.Axes,
    *,
    x_end: float,
    x_start: float = 0.0,
    label_every_days: int = 1,
    starting_day: int = 1,
    min_label_width_hours: float = 10.0,
) -> None:
    """Add day labels across the top of the plot."""

    day_index = 0
    day_start = x_start
    while day_start < x_end:
        day_end = min(day_start + 24.0, x_end)
        if day_index % label_every_days == 0:
            color = "#cfe7f8" if (day_index // label_every_days) % 2 == 0 else "#8fc5ea"
            ax.axvspan(day_start, day_end, ymin=0.84, ymax=0.92, color=color, alpha=0.9, linewidth=0, zorder=1)
            if (day_end - day_start) >= min_label_width_hours:
                ax.text(
                    day_start + (day_end - day_start) / 2.0,
                    0.88,
                    f"Day {starting_day + day_index}",
                    ha="center",
                    va="center",
                    fontsize=_font_size(-1.2),
                    fontweight="bold",
                    color="#1f1f1f",
                    transform=ax.get_xaxis_transform())
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
            ymin=0.92,
            ymax=1.0,
            color=PHASE_COLORS.get(phase_number, "#cccccc"),
            alpha=0.75,
            linewidth=0,
            zorder=2)
        ax.text(
            start_hours + (end_hours - start_hours) / 2.0,
            0.96,
            phase_display_names.get(phase_number, f"Phase {phase_number}"),
            ha="center",
            va="center",
            fontsize=_font_size(-1.2),
            fontweight="bold",
            color="#1f1f1f",
            transform=ax.get_xaxis_transform())

def _add_single_phase_band(
    ax: plt.Axes,
    *,
    phase_number: int,
    label: str,
    start_hours: float,
    end_hours: float,
) -> None:
    """Add one phase band for phase-specific plots with relative x-axes."""

    if end_hours <= start_hours:
        return
    ax.axvspan(
        start_hours,
        end_hours,
        ymin=0.92,
        ymax=1.0,
        color=PHASE_COLORS.get(phase_number, "#cccccc"),
        alpha=0.75,
        linewidth=0,
        zorder=2)
    ax.text(
        start_hours + (end_hours - start_hours) / 2.0,
        0.96,
        label,
        ha="center",
        va="center",
        fontsize=_font_size(-1.2),
        fontweight="bold",
        color="#1f1f1f",
        transform=ax.get_xaxis_transform())

def _add_segment_annotations(
    ax: plt.Axes,
    *,
    max_segment: int,
    phase_number: int,
    phase_display_name: str,
    starting_day: int = 1,
) -> None:
    """Add awake/sleep, day, and phase annotations to segment-based plots."""

    for segment_index in range(max_segment):
        left = float(segment_index)
        right = float(segment_index + 1)
        is_sleep = segment_index % 2 == 1
        if is_sleep:
            ax.axvspan(
                left,
                right,
                color=DEFAULT_SLEEP_SHADE_COLOR,
                alpha=0.55,
                linewidth=0,
                zorder=0)
        ax.text(
            left + 0.5,
            0.835,
            "sleep" if is_sleep else "awake",
            ha="center",
            va="top",
            fontsize=_font_size(-1.8),
            color="#4d4d4d",
            transform=ax.get_xaxis_transform())

    day_count = int(np.ceil(max_segment / 2.0))
    for day_index in range(day_count):
        left = float(day_index * 2)
        right = min(float((day_index + 1) * 2), float(max_segment))
        color = "#cfe7f8" if day_index % 2 == 0 else "#8fc5ea"
        ax.axvspan(left, right, ymin=0.84, ymax=0.92, color=color, alpha=0.9, linewidth=0, zorder=1)
        ax.text(
            left + (right - left) / 2.0,
            0.88,
            f"Day {starting_day + day_index}",
            ha="center",
            va="center",
            fontsize=_font_size(-1.2),
            fontweight="bold",
            color="#1f1f1f",
            transform=ax.get_xaxis_transform())

    _add_single_phase_band(
        ax,
        phase_number=phase_number,
        label=phase_display_name,
        start_hours=0.0,
        end_hours=float(max_segment))

def _draw_step_with_band(
    ax: plt.Axes,
    x_starts: pd.Series,
    x_ends: pd.Series,
    y_mean: pd.Series,
    y_spread: pd.Series,
    *,
    color: str,
    label: str | None,
    linewidth: float = 1.6,
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
    linewidth: float = 1.6,
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
    linewidth: float = 1.6,
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
            linewidth=linewidth)
    else:
        _draw_step_with_band(
            ax,
            summary_frame["bin_start_hours"],
            summary_frame["bin_end_hours"],
            summary_frame[y_col],
            summary_frame[spread_col],
            color=color,
            label=label,
            linewidth=linewidth)

def _draw_individual_trace(
    ax: plt.Axes,
    trace_frame: pd.DataFrame,
    *,
    y_col: str,
    color: str,
    plot_style: str = "step",
    linewidth: float = 0.9,
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
            zorder=2)
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

def _format_rate_axis(ax: plt.Axes, *, ylabel: str = "Visit rate [%]") -> None:
    """Format a rate axis in percent."""

    ax.set_ylim(0, 130)
    ax.set_ylabel(_wrap_axis_label(ylabel))
    ax.set_yticks(np.arange(0, 110, 10))

def _phase_plot_x_start(origin_clock_hour: float, awake_start_clock_hour: float) -> float:
    """Return the relative x-axis start so day boundaries remain mouse-day aligned."""

    raw_offset = awake_start_clock_hour - origin_clock_hour
    if raw_offset > 0:
        raw_offset -= 24.0
    return float(raw_offset)

def _place_legend(
    ax: plt.Axes,
    *,
    ncol: int = 1,
    show: bool = True,
    loc: str = "upper right",
    y_anchor: float = 0.80,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
) -> plt.Legend | None:
    """Place a legend slightly below the top annotations to avoid overlap."""

    if not show:
        return None
    kwargs: dict[str, Any] = {
        "frameon": False,
        "ncol": ncol,
        "loc": legend_loc or loc,
        "borderaxespad": 0.0,
        "fontsize": _font_size(0.0) if legend_font_size is None else float(legend_font_size)}
    if legend_loc is None:
        kwargs["bbox_to_anchor"] = (1.0, y_anchor)
    return ax.legend(**kwargs)

def _place_external_legend(
    ax: plt.Axes,
    *,
    ncol: int = 1,
    loc: str = "upper left",
    x_anchor: float = 1.01,
    y_anchor: float = 1.0,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
) -> plt.Legend:
    """Place a legend outside the plotting area in a single clean column."""

    kwargs: dict[str, Any] = {
        "frameon": False,
        "ncol": ncol,
        "loc": legend_loc or loc,
        "borderaxespad": 0.0,
        "fontsize": _font_size(0.0) if legend_font_size is None else float(legend_font_size)}
    if legend_loc is None:
        kwargs["bbox_to_anchor"] = (x_anchor, y_anchor)
    return ax.legend(**kwargs)

def _draw_legend(
    ax: plt.Axes,
    *,
    handles: list[Any] | None = None,
    ncol: int = 1,
    loc: str = "upper left",
    bbox_to_anchor: tuple[float, float] | None = None,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
) -> plt.Legend:
    """Draw a legend with consistent default font-size behavior."""

    kwargs: dict[str, Any] = {
        "frameon": False,
        "ncol": ncol,
        "loc": legend_loc or loc,
        "fontsize": _font_size(0.0) if legend_font_size is None else float(legend_font_size)}
    if handles is not None:
        kwargs["handles"] = handles
    if legend_loc is None and bbox_to_anchor is not None:
        kwargs["bbox_to_anchor"] = bbox_to_anchor
        kwargs["borderaxespad"] = 0.0
    return ax.legend(**kwargs)

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
    legend_spread_label: bool = True,
    x_end_hours: float | None = None,
    plot_style: str = "step",
    show_individual_labels: bool = True,
    title_label: str = "visits across all phases",
    ylabel: str = "Visits per mouse and bin",
    origin_clock_hour: float = 6.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    show_legend: bool = True,
    legend_inside: bool = False,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Plot full-experiment visit activity for one experimental group."""

    group_mouse = mouse_bins.loc[mouse_bins["Group"].astype(str).eq(group_name)].copy()
    group_summary = summary_bins.loc[summary_bins["Group"].astype(str).eq(group_name)].copy()
    if group_mouse.empty or group_summary.empty:
        return

    _prepare_output_path(output_path)
    color = _group_color(group_name)
    spread_col = _spread_column(spread_metric)

    fig, ax = plt.subplots(figsize=_figsize_cm(*LONG_FIGSIZE_CM))
    mouse_color_map = _mouse_trace_colors([str(label) for label in group_mouse["ETLabel"].dropna().unique()])
    for et_label, mouse_data in group_mouse.groupby("ETLabel", observed=True):
        _draw_individual_trace(
            ax,
            mouse_data,
            y_col="value",
            color=mouse_color_map.get(str(et_label), "#666666"),
            plot_style=plot_style,
            linewidth=0.75,
            alpha=0.45,
            label=str(et_label) if show_individual_labels else None)

    _draw_trace_with_band(
        ax,
        group_summary,
        y_col="mean_value",
        spread_col=spread_col,
        color=color,
        label=f"Group\n$\\mu$ ± {spread_metric.upper()}" if legend_spread_label else "Group mean",
        plot_style=plot_style,
        linewidth=1.0)

    max_hour = float(group_summary["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    x_start = _phase_plot_x_start(origin_clock_hour, awake_start_clock_hour)
    ax.set_xlim(x_start, max_hour)
    y_max = max(
        float(group_mouse["value"].max()) if not group_mouse.empty else 0.0,
        float((group_summary["mean_value"] + group_summary[spread_col]).max()))
    ax.set_ylim(0, _count_axis_upper(y_max))
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=0.0,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        day_night_indicator=day_night_indicator)
    _add_day_annotations(ax, x_end=max_hour, x_start=0.0, label_every_days=1, starting_day=0, min_label_width_hours=12.0)
    _add_phase_band(ax, phase_window_table, phase_display_names=phase_display_names)

    ax.set_title(_wrap_title(f"{group_name}: {_title_start(title_label)}; bin={bin_hours} h"))
    ax.set_xlabel(xlabel or "Elapsed experimental time [hours]")
    ax.set_ylabel(_wrap_axis_label(ylabel))
    _apply_axis_overrides(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.grid(False)
    if show_legend and legend_inside:
        legend = _place_legend(
            ax,
            ncol=1,
            y_anchor=0.76,
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    elif show_legend:
        legend = _place_external_legend(
            ax,
            ncol=1,
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    else:
        legend = None
    if legend is not None and not show_individual_labels:
        for text in legend.get_texts():
            if text.get_text().startswith("ET") or text.get_text().startswith("Lo"):
                text.set_visible(False)
    _save_figure(fig, output_path)

def plot_experiment_overview_groups(
    summary_bins: pd.DataFrame,
    *,
    output_path: Path,
    phase_window_table: pd.DataFrame,
    phase_display_names: dict[int, str],
    spread_metric: str,
    legend_spread_label: bool = True,
    x_end_hours: float | None = None,
    plot_style: str = "step",
    title_label: str = "Visits across all phases by group",
    ylabel: str = "Visits per mouse and bin",
    origin_clock_hour: float = 6.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    show_legend: bool = True,
    legend_inside: bool = False,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Plot all experimental-group visit means across the full experiment."""

    if summary_bins.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    fig, ax = plt.subplots(figsize=_figsize_cm(*WIDE_GROUP_FIGSIZE_CM))
    max_hour = float(summary_bins["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    x_start = _phase_plot_x_start(origin_clock_hour, awake_start_clock_hour)
    ax.set_xlim(x_start, max_hour)
    y_max = float((summary_bins["mean_value"] + summary_bins[spread_col]).max())
    ax.set_ylim(0, _count_axis_upper(y_max))

    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=0.0,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        day_night_indicator=day_night_indicator)
    _add_day_annotations(ax, x_end=max_hour, x_start=0.0, label_every_days=1, starting_day=0, min_label_width_hours=12.0)
    _add_phase_band(ax, phase_window_table, phase_display_names=phase_display_names)

    for group_name, group_summary in summary_bins.groupby("Group", observed=True):
        _draw_trace_with_band(
            ax,
            group_summary,
            y_col="mean_value",
            spread_col=spread_col,
            color=_group_color(str(group_name)),
            label=f"{group_name}\n$\\mu$ ± {spread_metric.upper()}" if legend_spread_label else str(group_name),
            plot_style=plot_style,
            linewidth=1.0)

    ax.set_title(_wrap_title(_title_start(title_label)))
    ax.set_xlabel(xlabel or "Elapsed experimental time [hours]")
    ax.set_ylabel(_wrap_axis_label(ylabel))
    _apply_axis_overrides(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        if legend_inside:
            _place_legend(
                ax,
                ncol=1,
                y_anchor=0.76,
                legend_loc=legend_loc,
                legend_font_size=legend_font_size)
        else:
            _place_external_legend(
                ax,
                ncol=1,
                legend_loc=legend_loc,
                legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

def plot_phase2_adaptation(
    primary_summary: pd.DataFrame,
    secondary_summary: pd.DataFrame,
    *,
    group_name: str,
    bin_hours: int,
    output_path: Path,
    secondary_label: str,
    phase_display_name: str,
    plot_style: str = "bar",
    spread_metric: str = "sem",
    x_end_hours: float | None = None,
    origin_clock_hour: float = 8.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    starting_day: int = 1,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Plot the phase-2 adaptation metric as bars or mean traces with spread."""

    visits_group = primary_summary.loc[primary_summary["Group"].astype(str).eq(group_name)].copy()
    secondary_group = secondary_summary.loc[secondary_summary["Group"].astype(str).eq(group_name)].copy()
    if visits_group.empty or secondary_group.empty:
        return

    _prepare_output_path(output_path)
    color = _group_color(group_name)

    fig, ax = plt.subplots(figsize=_figsize_cm(*PHASE2_FIGSIZE_CM))
    spread_col = _spread_column(spread_metric)
    max_hour = float(visits_group["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    y_max = max(
        float((visits_group["mean_value"] + visits_group[spread_col]).max()),
        float((secondary_group["mean_value"] + secondary_group[spread_col]).max()),
    )
    x_start = _phase_plot_x_start(origin_clock_hour, awake_start_clock_hour)
    ax.set_xlim(x_start, max_hour)
    ax.set_ylim(0, _count_axis_upper(y_max))
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=x_start,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        day_night_indicator=day_night_indicator)
    _add_day_annotations(ax, x_end=max_hour, x_start=x_start, label_every_days=1, starting_day=starting_day)
    _add_single_phase_band(ax, phase_number=2, label=phase_display_name, start_hours=0.0, end_hours=max_hour)

    if plot_style == "line":
        _draw_trace_with_band(
            ax,
            visits_group,
            y_col="mean_value",
            spread_col=spread_col,
            color="#7f7f7f",
            label="Visits",
            plot_style="line",
            linewidth=1.0)
        _draw_trace_with_band(
            ax,
            secondary_group,
            y_col="mean_value",
            spread_col=spread_col,
            color=DRINKING_VISIT_COLOR,
            label=secondary_label,
            plot_style="line",
            linewidth=1.0)
    else:
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
            zorder=3)
        ax.bar(
            x + width / 2.0,
            secondary_group["mean_value"],
            width=width,
            color=DRINKING_VISIT_COLOR,
            edgecolor=DRINKING_VISIT_COLOR,
            yerr=secondary_group["sem_value"],
            capsize=2,
            label=secondary_label,
            alpha=0.88,
            zorder=3)

    ax.set_title(_wrap_title(f"{group_name}: {phase_display_name} adaptation; bin={bin_hours} h"))
    ax.set_xlabel("Hours since start of phase 2")
    ax.set_ylabel(_wrap_axis_label("Mean count per mouse and bin"))
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        _place_legend(
            ax,
            ncol=1,
            y_anchor=0.76,
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

def plot_phase2_adaptation_groups(
    primary_summary: pd.DataFrame,
    secondary_summary: pd.DataFrame,
    *,
    group_names: list[str],
    bin_hours: int,
    output_path: Path,
    secondary_label: str,
    phase_display_name: str,
    x_end_hours: float | None = None,
    origin_clock_hour: float = 8.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    starting_day: int = 1,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title_label: str | None = None,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Plot the phase-2 adaptation metric for all groups in one combined panel."""

    available_groups = [
        group_name
        for group_name in group_names
        if group_name in set(primary_summary["Group"].astype(str))
        and group_name in set(secondary_summary["Group"].astype(str))]
    if not available_groups:
        return

    _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=_figsize_cm(*PHASE2_FIGSIZE_CM))
    spread_col = "sem_value"

    max_hour = float(
        max(primary_summary["bin_end_hours"].max(), secondary_summary["bin_end_hours"].max())
        if x_end_hours is None
        else float(x_end_hours))
    y_max = 0.0
    for group_name in available_groups:
        primary_group = primary_summary.loc[primary_summary["Group"].astype(str).eq(group_name)].copy()
        secondary_group = secondary_summary.loc[secondary_summary["Group"].astype(str).eq(group_name)].copy()
        if primary_group.empty or secondary_group.empty:
            continue
        group_color = _group_color(group_name)
        y_max = max(
            y_max,
            float((primary_group["mean_value"] + primary_group[spread_col]).max()),
            float((secondary_group["mean_value"] + secondary_group[spread_col]).max()))
        ax.plot(
            primary_group["bin_center_hours"],
            primary_group["mean_value"],
            color=group_color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            zorder=3)
        ax.plot(
            secondary_group["bin_center_hours"],
            secondary_group["mean_value"],
            color=group_color,
            linestyle="-",
            linewidth=1.0,
            alpha=0.95,
            zorder=4)

    x_start = _phase_plot_x_start(origin_clock_hour, awake_start_clock_hour)
    ax.set_xlim(x_start, max_hour)
    ax.set_ylim(0, _count_axis_upper(y_max))
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=x_start,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        day_night_indicator=day_night_indicator)
    _add_day_annotations(ax, x_end=max_hour, x_start=x_start, label_every_days=1, starting_day=starting_day)
    _add_single_phase_band(ax, phase_number=2, label=phase_display_name, start_hours=0.0, end_hours=max_hour)

    ax.set_title(_wrap_title(f"Visits vs {secondary_label.lower()}; {phase_display_name}; bin={bin_hours} h"))
    ax.set_xlabel("Hours since start of phase 2")
    ax.set_ylabel(_wrap_axis_label("Mean count per mouse and bin"))
    ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color=_group_color(group_name), linestyle="-", linewidth=1.2, label=group_name)
        for group_name in available_groups]
    legend_handles.extend(
        [
            Line2D([0], [0], color="#000000", linestyle="--", linewidth=1.2, label="All visits"),
            Line2D([0], [0], color="#000000", linestyle="-", linewidth=1.2, label=secondary_label),
        ])
    if show_legend:
        _draw_legend(
            ax,
            handles=legend_handles,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

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
    title_label: str,
    ylabel: str,
    x_end_hours: float | None = None,
    plot_style: str = "step",
    origin_clock_hour: float = 8.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    starting_day: int = 1,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Plot one selected place-learning count metric for phase 3 or phase 4."""

    group_mouse = mouse_bins.loc[mouse_bins["Group"].astype(str).eq(group_name)].copy()
    group_summary = summary_bins.loc[summary_bins["Group"].astype(str).eq(group_name)].copy()
    if group_mouse.empty or group_summary.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    color = _group_color(group_name)

    fig, ax = plt.subplots(figsize=_figsize_cm(*MEDIUM_FIGSIZE_CM))
    mouse_color_map = _mouse_trace_colors([str(label) for label in group_mouse["ETLabel"].dropna().unique()])
    for et_label, mouse_data in group_mouse.groupby("ETLabel", observed=True):
        _draw_individual_trace(
            ax,
            mouse_data,
            y_col="value",
            color=mouse_color_map.get(str(et_label), "#9aa0a6"),
            plot_style=plot_style,
            linewidth=0.75,
            alpha=0.45)

    max_hour = float(group_summary["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    x_start = _phase_plot_x_start(origin_clock_hour, awake_start_clock_hour)
    y_max = _count_axis_upper(
        max(
            float(group_mouse["value"].max()) if not group_mouse.empty else 0.0,
            float((group_summary["mean_value"] + group_summary[spread_col]).max()),
        ))
    ax.set_xlim(x_start, max_hour)
    ax.set_ylim(0, y_max)
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=x_start,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour)
    _add_day_annotations(ax, x_end=max_hour, x_start=x_start, label_every_days=1, starting_day=starting_day)
    _add_single_phase_band(
        ax,
        phase_number=phase_number,
        label=phase_display_name,
        start_hours=0.0,
        end_hours=max_hour)
    _draw_trace_with_band(
        ax,
        group_summary,
        y_col="mean_value",
        spread_col=spread_col,
        color=color,
        label=f"Group\n$\\mu$ ± {spread_metric.upper()}",
        plot_style=plot_style,
        linewidth=1.0)

    ax.set_title(_wrap_title(f"{group_name}: {_title_start(title_label)}; {phase_display_name}; bin={bin_hours} h"))
    ax.set_xlabel(xlabel or f"Hours since start of {phase_display_name}")
    ax.set_ylabel(_wrap_axis_label(ylabel))
    _apply_axis_overrides(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        _place_external_legend(
            ax,
            ncol=1,
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

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
    title_label: str,
    ylabel: str,
    chance_level: float | None = None,
    x_end_hours: float | None = None,
    plot_style: str = "step",
    origin_clock_hour: float = 8.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    starting_day: int = 1,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Plot one selected place-learning rate metric for phase 3 or phase 4."""

    group_mouse = mouse_bins.loc[mouse_bins["Group"].astype(str).eq(group_name)].copy()
    group_summary = summary_bins.loc[summary_bins["Group"].astype(str).eq(group_name)].copy()
    if group_mouse.empty or group_summary.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    color = _group_color(group_name)

    fig, ax = plt.subplots(figsize=_figsize_cm(*MEDIUM_FIGSIZE_CM))
    mouse_color_map = _mouse_trace_colors([str(label) for label in group_mouse["ETLabel"].dropna().unique()])
    for et_label, mouse_data in group_mouse.groupby("ETLabel", observed=True):
        clean = mouse_data.dropna(subset=["value"])
        if clean.empty:
            continue
        clean = clean.copy()
        clean["value_pct"] = clean["value"] * 100.0
        _draw_individual_trace(
            ax,
            clean,
            y_col="value_pct",
            color=mouse_color_map.get(str(et_label), "#b8bdc4"),
            plot_style=plot_style,
            linewidth=0.75,
            alpha=0.45)

    max_hour = float(group_summary["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    x_start = _phase_plot_x_start(origin_clock_hour, awake_start_clock_hour)
    ax.set_xlim(x_start, max_hour)
    _format_rate_axis(ax, ylabel=ylabel)
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=x_start,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        day_night_indicator=day_night_indicator)
    _add_day_annotations(ax, x_end=max_hour, x_start=x_start, label_every_days=1, starting_day=starting_day)
    _add_single_phase_band(
        ax,
        phase_number=phase_number,
        label=phase_display_name,
        start_hours=0.0,
        end_hours=max_hour)
    group_summary = group_summary.copy()
    group_summary["mean_value_pct"] = group_summary["mean_value"] * 100.0
    group_summary["spread_pct"] = group_summary[spread_col] * 100.0
    _draw_trace_with_band(
        ax,
        group_summary,
        y_col="mean_value_pct",
        spread_col="spread_pct",
        color=color,
        label=f"Group\n$\\mu$ ± {spread_metric.upper()}",
        plot_style=plot_style,
        linewidth=1.0)
    if chance_level is not None:
        ax.axhline(
            chance_level,
            color="#4f4f4f",
            linestyle="--",
            linewidth=1.4,
            label="Chance level")

    ax.set_title(_wrap_title(f"{group_name}: {_title_start(title_label)}; {phase_display_name}; bin={bin_hours} h"))
    ax.set_xlabel(xlabel or f"Hours since start of {phase_display_name}")
    _apply_axis_overrides(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        _place_legend(
            ax,
            ncol=1,
            y_anchor=0.76,
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

def plot_phase_learning_rate_groups(
    summary_bins: pd.DataFrame,
    *,
    phase_number: int,
    phase_display_name: str,
    bin_hours: int,
    output_path: Path,
    spread_metric: str,
    title_label: str,
    ylabel: str,
    chance_level: float | None = None,
    x_end_hours: float | None = None,
    plot_style: str = "step",
    origin_clock_hour: float = 8.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    starting_day: int = 1,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Plot one selected place-learning rate metric across all experimental groups."""

    if summary_bins.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    fig, ax = plt.subplots(figsize=_figsize_cm(*WIDE_GROUP_FIGSIZE_CM))
    max_hour = float(summary_bins["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    x_start = _phase_plot_x_start(origin_clock_hour, awake_start_clock_hour)
    ax.set_xlim(x_start, max_hour)
    _format_rate_axis(ax, ylabel=ylabel)
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=x_start,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        day_night_indicator=day_night_indicator)
    _add_day_annotations(ax, x_end=max_hour, x_start=x_start, label_every_days=1, starting_day=starting_day)
    _add_single_phase_band(
        ax,
        phase_number=phase_number,
        label=phase_display_name,
        start_hours=0.0,
        end_hours=max_hour)

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
            linewidth=1.0,
        )

    if chance_level is not None:
        ax.axhline(
            chance_level,
            color="#4f4f4f",
            linestyle="--",
            linewidth=1.4,
            label="Chance level")
    ax.set_title(_wrap_title(f"{_title_start(title_label)} across groups; {phase_display_name}; bin={bin_hours} h"))
    ax.set_xlabel(xlabel or f"Hours since start of {phase_display_name}")
    _apply_axis_overrides(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        _place_legend(
            ax,
            ncol=1,
            y_anchor=0.76,
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

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
    origin_clock_hour: float = 8.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    starting_day: int = 1,
    phase_number: int | None = None,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Plot all experimental-group correct-visit counts in one figure."""

    if summary_bins.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    fig, ax = plt.subplots(figsize=_figsize_cm(*WIDE_GROUP_FIGSIZE_CM))
    max_hour = float(summary_bins["bin_end_hours"].max()) if x_end_hours is None else float(x_end_hours)
    x_start = _phase_plot_x_start(origin_clock_hour, awake_start_clock_hour)
    ax.set_xlim(x_start, max_hour)
    y_max = float((summary_bins["mean_value"] + summary_bins[spread_col]).max())
    ax.set_ylim(0, _count_axis_upper(y_max))
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=x_start,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        day_night_indicator=day_night_indicator)
    _add_day_annotations(ax, x_end=max_hour, x_start=x_start, label_every_days=1, starting_day=starting_day)
    phase_number = phase_number or (3 if phase_display_name == "PL" else 4 if phase_display_name == "PR" else 3)
    _add_single_phase_band(
        ax,
        phase_number=phase_number,
        label=phase_display_name,
        start_hours=0.0,
        end_hours=max_hour)

    for group_name, group_summary in summary_bins.groupby("Group", observed=True):
        _draw_trace_with_band(
            ax,
            group_summary,
            y_col="mean_value",
            spread_col=spread_col,
            color=_group_color(str(group_name)),
            label=str(group_name),
            plot_style=plot_style,
            linewidth=1.0)

    ax.set_title(_wrap_title(f"{_title_start(title_prefix)}; {phase_display_name}; bin={bin_hours} h"))
    ax.set_xlabel(xlabel or f"Hours since start of {phase_display_name}")
    ax.set_ylabel(_wrap_axis_label(ylabel))
    _apply_axis_overrides(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        _place_legend(
            ax,
            ncol=1,
            y_anchor=0.76,
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

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
    plot_style: str = "bar",
    spread_metric: str = "sem",
    origin_clock_hour: float = 6.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title_label: str | None = None,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Plot paired dual-metric summaries across the full experiment timeline."""

    visits_group = primary_summary.loc[primary_summary["Group"].astype(str).eq(group_name)].copy()
    secondary_group = secondary_summary.loc[secondary_summary["Group"].astype(str).eq(group_name)].copy()
    if visits_group.empty or secondary_group.empty:
        return

    _prepare_output_path(output_path)
    color = _group_color(group_name)
    fig, ax = plt.subplots(figsize=_figsize_cm(*LONG_FIGSIZE_2_CM))
    spread_col = _spread_column(spread_metric)
    max_hour = float(visits_group["bin_end_hours"].max())
    ax.set_xlim(0, max_hour)
    y_max = max(
        float((visits_group["mean_value"] + visits_group[spread_col]).max()),
        float((secondary_group["mean_value"] + secondary_group[spread_col]).max()))
    ax.set_ylim(0, _count_axis_upper(y_max))

    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=0.0,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        day_night_indicator=day_night_indicator)
    _add_day_annotations(ax, x_end=max_hour, x_start=0.0, label_every_days=1, starting_day=0, min_label_width_hours=12.0)
    _add_phase_band(ax, phase_window_table, phase_display_names=phase_display_names)

    if plot_style == "line":
        _draw_trace_with_band(
            ax,
            visits_group,
            y_col="mean_value",
            spread_col=spread_col,
            color="#7f7f7f",
            label="Visits",
            plot_style="line",
            linewidth=1.0)
        _draw_trace_with_band(
            ax,
            secondary_group,
            y_col="mean_value",
            spread_col=spread_col,
            color=DRINKING_VISIT_COLOR,
            label=secondary_label,
            plot_style="line",
            linewidth=1.0)
    else:
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
            zorder=3)
        ax.bar(
            x + width / 2.0,
            secondary_group["mean_value"],
            width=width,
            color=DRINKING_VISIT_COLOR,
            edgecolor=DRINKING_VISIT_COLOR,
            yerr=secondary_group["sem_value"],
            capsize=2,
            label=secondary_label,
            alpha=0.85,
            zorder=3)

    ax.set_title(_wrap_title(title_label or f"{group_name}: Visits vs {secondary_label.lower()} across selected phases ({bin_hours} h bins)"))
    ax.set_xlabel(xlabel or "Elapsed experimental time [hours]")
    ax.set_ylabel(_wrap_axis_label(ylabel or "Mean count per mouse and bin"))
    _apply_axis_overrides(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.grid(axis="y", alpha=0.25)
    _place_legend(
        ax,
        y_anchor=0.76,
        show=show_legend,
        legend_loc=legend_loc,
        legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

def plot_experiment_dual_metric_groups(
    primary_summary: pd.DataFrame,
    secondary_summary: pd.DataFrame,
    *,
    group_names: list[str],
    bin_hours: int,
    output_path: Path,
    secondary_label: str,
    phase_window_table: pd.DataFrame,
    phase_display_names: dict[int, str],
    origin_clock_hour: float = 6.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title_label: str | None = None,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Plot all-groups full-experiment dual metrics with reduced legend semantics."""

    if primary_summary.empty or secondary_summary.empty:
        return

    available_groups = [
        group_name
        for group_name in group_names
        if group_name in set(primary_summary["Group"].astype(str))
        and group_name in set(secondary_summary["Group"].astype(str))]
    if not available_groups:
        return

    _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=_figsize_cm(*LONG_FIGSIZE_2_CM))
    max_hour = float(max(primary_summary["bin_end_hours"].max(), secondary_summary["bin_end_hours"].max()))
    spread_col = "sem_value"
    y_max = 0.0
    for group_name in available_groups:
        primary_group = primary_summary.loc[primary_summary["Group"].astype(str).eq(group_name)].copy()
        secondary_group = secondary_summary.loc[secondary_summary["Group"].astype(str).eq(group_name)].copy()
        if primary_group.empty or secondary_group.empty:
            continue
        y_max = max(
            y_max,
            float((primary_group["mean_value"] + primary_group[spread_col]).max()),
            float((secondary_group["mean_value"] + secondary_group[spread_col]).max()))
        group_color = _group_color(group_name)
        ax.plot(
            primary_group["bin_center_hours"],
            primary_group["mean_value"],
            color=group_color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=3)
        ax.plot(
            secondary_group["bin_center_hours"],
            secondary_group["mean_value"],
            color=group_color,
            linestyle="-",
            linewidth=1.0,
            alpha=0.95,
            zorder=4)

    ax.set_xlim(0, max_hour)
    ax.set_ylim(0, _count_axis_upper(y_max))
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=0.0,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        day_night_indicator=day_night_indicator)
    _add_day_annotations(ax, x_end=max_hour, x_start=0.0, label_every_days=1, starting_day=0, min_label_width_hours=12.0)
    _add_phase_band(ax, phase_window_table, phase_display_names=phase_display_names)

    ax.set_title(_wrap_title(title_label or f"Visits vs {secondary_label.lower()} across selected phases by group ({bin_hours} h bins)"))
    ax.set_xlabel(xlabel or "Elapsed experimental time [hours]")
    ax.set_ylabel(_wrap_axis_label(ylabel or "Mean count per mouse and bin"))
    _apply_axis_overrides(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color=_group_color(group_name), linestyle="-", linewidth=1.2, label=group_name)
        for group_name in available_groups]
    legend_handles.extend(
        [
            Line2D([0], [0], color="#000000", linestyle="--", linewidth=1.2, label="All visits"),
            Line2D([0], [0], color="#000000", linestyle="-", linewidth=1.2, label=secondary_label),
        ])
    if show_legend:
        _draw_legend(
            ax,
            handles=legend_handles,
            ncol=1,
            loc="upper right",
            bbox_to_anchor=(1.0, 0.76),
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

def plot_phase4_reversal_components(
    summary_by_component: dict[str, pd.DataFrame],
    *,
    group_name: str,
    phase_display_name: str,
    bin_hours: int,
    output_path: Path,
    spread_metric: str,
    x_end_hours: float | None = None,
    plot_style: str = "step",
    origin_clock_hour: float = 8.0,
    awake_start_clock_hour: float = 6.0,
    awake_end_clock_hour: float = 18.0,
    starting_day: int = 1,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title_label: str | None = None,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
) -> None:
    """Plot new-correct, previous-correct, and neutral-corner visit rates for reversal."""

    component_labels = {
        "new_correct_corner": ("New correct corner", "#2a9d8f"),
        "previous_correct_corner": ("Previous correct corner", "#e76f51"),
        "neutral_incorrect_corner_1": ("Neutral incorrect corner 1", "#7f7f7f"),
        "neutral_incorrect_corner_2": ("Neutral incorrect corner 2", "#b0b0b0")}
    spread_col = _spread_column(spread_metric)
    prepared: dict[str, pd.DataFrame] = {}
    for component_name, summary_frame in summary_by_component.items():
        group_summary = summary_frame.loc[summary_frame["Group"].astype(str).eq(group_name)].copy()
        if group_summary.empty:
            continue
        group_summary["mean_value_pct"] = group_summary["mean_value"] * 100.0
        group_summary["spread_pct"] = group_summary[spread_col] * 100.0
        prepared[component_name] = group_summary
    if not prepared:
        return

    _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=_figsize_cm(*MEDIUM_WIDE_FIGSIZE_CM))
    max_hour = (max(float(frame["bin_end_hours"].max()) for frame in prepared.values()) if x_end_hours is None else float(x_end_hours))
    x_start = _phase_plot_x_start(origin_clock_hour, awake_start_clock_hour)
    ax.set_xlim(x_start, max_hour)
    _format_rate_axis(ax, ylabel="Corner visit rate [%]")
    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=x_start,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        day_night_indicator=day_night_indicator)
    _add_day_annotations(ax, x_end=max_hour, x_start=x_start, label_every_days=1, starting_day=starting_day)
    _add_single_phase_band(ax, phase_number=4, label=phase_display_name, start_hours=0.0, end_hours=max_hour)

    for component_name, group_summary in prepared.items():
        label, color = component_labels[component_name]
        _draw_trace_with_band(
            ax,
            group_summary,
            y_col="mean_value_pct",
            spread_col="spread_pct",
            color=color,
            label=label,
            plot_style=plot_style,
            linewidth=1.0)

    ax.set_title(_wrap_title(title_label or f"{group_name}: Reversal corner visit components ({phase_display_name}, {bin_hours} h bins)"))
    ax.set_xlabel(xlabel or f"Hours since start of {phase_display_name}")
    if ylabel is not None:
        ax.set_ylabel(_wrap_axis_label(ylabel))
    _apply_axis_overrides(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        _place_legend(
            ax,
            ncol=1,
            y_anchor=0.76,
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

def plot_phase_segment_rate_groups(
    summary_bins: pd.DataFrame,
    *,
    phase_number: int,
    phase_display_name: str,
    title_label: str,
    ylabel: str,
    output_path: Path,
    spread_metric: str,
    add_zero_start: bool = True,
    starting_day: int = 1,
    chance_level: float | None = None,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
) -> None:
    """Plot group mean rates across awake/sleep segments within one phase."""

    if summary_bins.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    fig, ax = plt.subplots(figsize=_figsize_cm(*SEGMENT_FIGSIZE_CM))
    max_segment = int(summary_bins["segment_order"].max())

    for group_name, group_summary in summary_bins.groupby("Group", observed=True):
        group_summary = group_summary.sort_values("segment_order").copy()
        x_values = group_summary["segment_order"].to_numpy(dtype=float) - 0.5
        y_values = group_summary["mean_value"].to_numpy(dtype=float) * 100.0
        spread_values = group_summary[spread_col].to_numpy(dtype=float) * 100.0
        if add_zero_start:
            x_values = np.insert(x_values, 0, 0.0)
            y_values = np.insert(y_values, 0, 0.0)
            spread_values = np.insert(spread_values, 0, 0.0)
        ax.fill_between(
            x_values,
            y_values - spread_values,
            y_values + spread_values,
            color=_group_color(str(group_name)),
            alpha=0.18,
            linewidth=0)
        ax.plot(
            x_values,
            y_values,
            color=_group_color(str(group_name)),
            linewidth=1.0,
            marker="o",
            markersize=4.5,
            label=str(group_name))

    ax.set_xlim(0, max_segment)
    _format_rate_axis(ax, ylabel=ylabel)
    _add_segment_annotations(
        ax,
        max_segment=max_segment,
        phase_number=phase_number,
        phase_display_name=phase_display_name,
        starting_day=starting_day)
    if chance_level is not None:
        ax.axhline(
            float(chance_level),
            color="#4f4f4f",
            linestyle="--",
            linewidth=1.0,
            zorder=1)
    tick_positions = [0.0] + [position - 0.5 for position in range(1, max_segment + 1)]
    tick_labels = ["start"] + [
        f"D{starting_day + ((position - 1) // 2)} {'awake' if position % 2 == 1 else 'sleep'}"
        for position in range(1, max_segment + 1)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_title(f"{_title_start(title_label)}\nAcross awake/sleep segments ({phase_display_name})")
    ax.set_xlabel("Mouse-day segment")
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        _place_external_legend(
            ax,
            ncol=1,
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

def plot_group_day_violin(
    mouse_day_rates: pd.DataFrame,
    *,
    phase_number: int,
    phase_display_name: str,
    phase_day: int,
    metric_title: str,
    ylabel: str,
    pairwise_stats: pd.DataFrame,
    chance_stats: pd.DataFrame | None,
    output_path: Path,
    dayphase_label: str = "awake",
    outlier_col: str = "is_outlier",
    reference_line: float | None = 25.0,
    value_scale: float = 100.0,
    format_as_percent: bool = True,
    y_limits: tuple[float | None, float | None] | None = None,
) -> None:
    """Plot one day-wise group violin panel with significance annotations."""

    panel = mouse_day_rates.loc[
        mouse_day_rates["PhaseNumber"].eq(phase_number) & mouse_day_rates["phase_day"].eq(phase_day)
    ].copy()
    if panel.empty:
        return
    if outlier_col in panel.columns:
        panel["__is_outlier"] = panel[outlier_col].fillna(False).astype(bool)
    else:
        panel["__is_outlier"] = False
    panel_inliers = panel.loc[~panel["__is_outlier"]].copy()

    _prepare_output_path(output_path)
    group_order = _ordered_group_names_from_series(panel["Group"])
    positions = np.arange(1, len(group_order) + 1)
    fig, ax = plt.subplots(figsize=_figsize_cm(*VIOLIN_FIGSIZE_CM))

    violin_data: list[np.ndarray] = []
    for group_name in group_order:
        violin_data.append(
            panel_inliers.loc[panel_inliers["Group"].astype(str).eq(group_name), "value"].dropna().to_numpy(dtype=float) * float(value_scale))
    violins = ax.violinplot(violin_data, positions=positions, widths=0.8, showmeans=False, showmedians=True)
    for body, group_name in zip(violins["bodies"], group_order):
        color = _group_color(group_name)
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_linewidth(0.0)
        body.set_alpha(0.25)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in violins:
            violins[key].set_color("#555555")
            violins[key].set_linewidth(1.0)

    for position, group_name, values in zip(positions, group_order, violin_data):
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=22,
            color=_group_color(group_name),
            edgecolor="none",
            linewidth=0.0,
            alpha=0.85,
            zorder=3)
        outlier_values = (
            panel.loc[
                panel["Group"].astype(str).eq(group_name) & panel["__is_outlier"],
                "value"]
            .dropna()
            .to_numpy(dtype=float)
            * float(value_scale))
        if len(outlier_values) > 0:
            outlier_jitter = np.linspace(-0.08, 0.08, len(outlier_values)) if len(outlier_values) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(outlier_values), position) + outlier_jitter,
                outlier_values,
                s=34,
                color="#c1121f",
                marker="x",
                linewidth=1.0,
                alpha=0.95,
                zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(group_order, rotation=25, ha="right")
    if format_as_percent:
        _format_rate_axis(ax, ylabel=ylabel)
    else:
        y_max = float(panel["value"].dropna().max() * float(value_scale)) if panel["value"].notna().any() else 1.0
        ax.set_ylim(0, max(1.2, y_max * 1.20))
        ax.set_ylabel(_wrap_axis_label(ylabel))
    ax.set_title(f"{_title_start(metric_title)}\n{phase_display_name} day {phase_day} {dayphase_label}")
    if reference_line is not None:
        ax.axhline(float(reference_line), color="#4f4f4f", linestyle="--", linewidth=1.0, zorder=1)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    significant_pairs = pairwise_stats.loc[pairwise_stats["phase_day"].eq(phase_day) & pairwise_stats["p_value"].lt(0.05)].copy()
    default_min = 100.0 if format_as_percent else 1.0
    y_data_max = float(max(default_min, panel["value"].dropna().max() * float(value_scale) if panel["value"].notna().any() else default_min))
    y_base = max((104.0 if format_as_percent else y_data_max * 1.05), y_data_max + (5.0 if format_as_percent else max(0.1, y_data_max * 0.08)))
    y_step = 10.0 if format_as_percent else max(0.12, y_data_max * 0.10)
    y_limit = 124.0 if format_as_percent else max(1.25, y_base + 0.4)
    for pair_index, (_, row) in enumerate(significant_pairs.iterrows()):
        left = group_order.index(str(row["group1"])) + 1
        right = group_order.index(str(row["group2"])) + 1
        line_y = y_base + pair_index * y_step
        y_limit = max(y_limit, line_y + 7.0)
        ax.plot([left, left, right, right], [line_y - 1.1, line_y, line_y, line_y - 1.1], color="#444444", linewidth=1.0)
        ax.text(
            (left + right) / 2.0,
            line_y + (1.8 if format_as_percent else max(0.04, y_step * 0.18)),
            f"p={float(row['p_value']):.3g}",
            ha="center",
            va="bottom",
            fontsize=_font_size(-2.0),
            color="#444444")

    chance_star_y = max((116.0 if format_as_percent else y_base + len(significant_pairs) * y_step + max(0.08, y_step * 0.3)), y_base + len(significant_pairs) * y_step + (3.5 if format_as_percent else max(0.08, y_step * 0.3)))
    if chance_stats is not None and not chance_stats.empty:
        for position, group_name in zip(positions, group_order):
            chance_row = chance_stats.loc[chance_stats["phase_day"].eq(phase_day) & chance_stats["Group"].astype(str).eq(group_name)]
            if not chance_row.empty and float(chance_row["p_value"].iloc[0]) < 0.05:
                ax.text(position, chance_star_y, "*", ha="center", va="center", fontsize=_font_size(6.0), color="#222222")

    auto_top = max(y_limit, chance_star_y + (6.0 if format_as_percent else max(0.12, y_step * 0.45)))
    if y_limits is None:
        ax.set_ylim(0, auto_top)
    else:
        current_bottom, _ = ax.get_ylim()
        bottom = current_bottom if y_limits[0] is None else float(y_limits[0])
        top = auto_top if y_limits[1] is None else float(y_limits[1])
        ax.set_ylim(bottom, top)
    _save_figure(fig, output_path)

def plot_cumulative_role_curves(
    summary_bins: pd.DataFrame,
    *,
    group_name: str,
    output_path: Path,
    title_label: str,
    ylabel: str,
    value_col: str,
    spread_col: str,
    plot_style: str,
    phase_window_table: pd.DataFrame,
    phase_display_names: dict[int, str],
    origin_clock_hour: float,
    awake_start_clock_hour: float,
    awake_end_clock_hour: float,
    x_start_hours: float | None = None,
    onset_points: list[dict[str, float | str]] | None = None,
    starting_day: int = 1,
    day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
) -> None:
    """Plot cumulative role-based corner trajectories for one experimental group."""

    group_summary = summary_bins.loc[summary_bins["Group"].astype(str).eq(group_name)].copy()
    if group_summary.empty:
        return

    _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=_figsize_cm(*MEDIUM_WIDE_FIGSIZE_CM))
    x_start = _phase_plot_x_start(origin_clock_hour, awake_start_clock_hour) if x_start_hours is None else float(x_start_hours)
    max_hour = float(group_summary["bin_end_hours"].max())
    ax.set_xlim(x_start, max_hour)

    y_max = float((group_summary[value_col] + group_summary[spread_col]).max())
    if "rate" in value_col or "relative" in value_col:
        _format_rate_axis(ax, ylabel=ylabel)
    else:
        ax.set_ylim(0, _count_axis_upper(y_max))
        ax.set_ylabel(_wrap_axis_label(ylabel))

    _add_awake_sleep_background(
        ax,
        x_end=max_hour,
        x_start=x_start,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        day_night_indicator=day_night_indicator)
    _add_day_annotations(ax, x_end=max_hour, x_start=x_start, label_every_days=1, starting_day=starting_day)
    _add_phase_band(ax, phase_window_table, phase_display_names=phase_display_names)

    for role_name, role_summary in group_summary.groupby("corner_role", observed=True):
        _draw_trace_with_band(
            ax,
            role_summary,
            y_col=value_col,
            spread_col=spread_col,
            color=ROLE_COLORS.get(str(role_name), "#555555"),
            label=str(role_name),
            plot_style=plot_style,
            linewidth=1.0)

    if onset_points:
        onset_label_drawn = False
        for point in onset_points:
            role_name = str(point["corner_role"])
            x_value = float(point["x_hours"])
            role_summary = group_summary.loc[group_summary["corner_role"].astype(str).eq(role_name)].copy()
            if role_summary.empty:
                continue
            nearest_index = (role_summary["bin_center_hours"] - x_value).abs().idxmin()
            y_value = float(role_summary.loc[nearest_index, value_col])
            ax.scatter(
                [x_value],
                [y_value],
                color="#c1121f",
                edgecolor="white",
                linewidth=0.0,
                s=12,
                alpha=0.75,
                label="Learning onset" if not onset_label_drawn else None,
                zorder=6)
            onset_label_drawn = True

    ax.set_title(_wrap_title(f"{group_name}: {_title_start(title_label)}"))
    ax.set_xlabel("Hours since start of PL")
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        _place_external_legend(
            ax,
            ncol=1,
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

def plot_visit_learning_curve_groups(
    summary_bins: pd.DataFrame,
    *,
    phase_display_name: str,
    title_label: str,
    ylabel: str,
    output_path: Path,
    spread_metric: str,
    chance_level: float | None = 25.0,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
    xlabel: str | None = None,
    figsize_cm: tuple[float, float] | None = None,
) -> None:
    """Plot experience-dependent learning curves across groups."""

    if summary_bins.empty:
        return

    _prepare_output_path(output_path)
    spread_col = _spread_column(spread_metric)
    fig, ax = plt.subplots(figsize=_resolve_figsize_cm(WIDE_GROUP_FIGSIZE_CM, figsize_cm))
    max_visit = float(summary_bins["window_end_visit"].max())
    ax.set_xlim(0, max_visit)
    _format_rate_axis(ax, ylabel=ylabel)

    for group_name, group_summary in summary_bins.groupby("Group", observed=True):
        prepared = group_summary.copy()
        prepared["mean_value_pct"] = prepared["mean_value"] * 100.0
        prepared["spread_pct"] = prepared[spread_col] * 100.0
        ax.fill_between(
            prepared["window_center_visit"],
            prepared["mean_value_pct"] - prepared["spread_pct"],
            prepared["mean_value_pct"] + prepared["spread_pct"],
            color=_group_color(str(group_name)),
            alpha=0.18,
            linewidth=0)
        ax.plot(
            prepared["window_center_visit"],
            prepared["mean_value_pct"],
            color=_group_color(str(group_name)),
            linewidth=1.0,
            label=str(group_name))

    if chance_level is not None:
        ax.axhline(
            float(chance_level),
            color="#4f4f4f",
            linestyle="--",
            linewidth=1.2,
            label="Chance level")
    ax.set_title(_wrap_title(f"{_title_start(title_label)} across groups ({phase_display_name})"))
    ax.set_xlabel(xlabel or "Visit number")
    _apply_axis_overrides(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        _draw_legend(
            ax,
            ncol=1,
            loc="upper right",
            bbox_to_anchor=(1.0, 0.76),
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)

def plot_onset_violin(
    onset_table: pd.DataFrame,
    *,
    onset_col: str,
    phase_display_name: str,
    title_label: str,
    ylabel: str,
    output_path: Path,
    pairwise_stats: pd.DataFrame | None = None,
    outlier_col: str = "is_outlier",
    reference_line: float | None = None,
    y_limits: tuple[float, float] | None = None,
    show_n: bool = False,
    figsize_cm: tuple[float, float] | None = None,
) -> None:
    """Plot onset distributions per group as violins with point overlays."""

    if onset_table.empty or onset_table[onset_col].dropna().empty:
        return

    _prepare_output_path(output_path)
    plot_data = onset_table.loc[onset_table[onset_col].notna()].copy()
    if outlier_col in plot_data.columns:
        plot_data["__is_outlier"] = plot_data[outlier_col].fillna(False).astype(bool)
    else:
        plot_data["__is_outlier"] = False
    plot_inliers = plot_data.loc[~plot_data["__is_outlier"]].copy()
    group_order = _ordered_group_names_from_series(plot_data["Group"])
    positions = np.arange(1, len(group_order) + 1)
    fig, ax = plt.subplots(figsize=_resolve_figsize_cm(ONSET_FIGSIZE_CM, figsize_cm))

    violin_data = [
        plot_inliers.loc[plot_inliers["Group"].astype(str).eq(group_name), onset_col].to_numpy(dtype=float)
        for group_name in group_order]
    violins = ax.violinplot(violin_data, positions=positions, widths=0.8, showmeans=False, showmedians=True)
    for body, group_name in zip(violins["bodies"], group_order):
        color = _group_color(group_name)
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_linewidth(0.0)
        body.set_alpha(0.25)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in violins:
            violins[key].set_color("#555555")
            violins[key].set_linewidth(1.0)

    for position, group_name in zip(positions, group_order):
        values = plot_inliers.loc[plot_inliers["Group"].astype(str).eq(group_name), onset_col].to_numpy(dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=22,
            color=_group_color(group_name),
            edgecolor="none",
            linewidth=0.0,
            alpha=0.85,
            zorder=3)
        outlier_values = plot_data.loc[
            plot_data["Group"].astype(str).eq(group_name) & plot_data["__is_outlier"],
            onset_col].to_numpy(dtype=float)
        if len(outlier_values) > 0:
            outlier_jitter = np.linspace(-0.08, 0.08, len(outlier_values)) if len(outlier_values) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(outlier_values), position) + outlier_jitter,
                outlier_values,
                s=34,
                color="#c1121f",
                marker="x",
                linewidth=1.0,
                alpha=0.95,
                zorder=4)

    ax.set_xticks(positions)
    if show_n:
        labels = [
            f"{group_name}\n(n={plot_data.loc[plot_data['Group'].astype(str).eq(group_name), 'ET'].nunique()})"
            for group_name in group_order]
    else:
        labels = group_order
    ax.set_xticklabels(labels, rotation=0 if show_n else 20, ha="center" if show_n else "right")
    ax.set_title(f"{_wrap_title(_title_start(title_label), width=34)}\n({phase_display_name})")
    ax.set_ylabel(_wrap_axis_label(ylabel))
    if reference_line is not None:
        ax.axhline(float(reference_line), color="#4f4f4f", linestyle="--", linewidth=1.0, zorder=1)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if pairwise_stats is not None and not pairwise_stats.empty:
        significant_pairs = pairwise_stats.loc[pairwise_stats["p_value"].lt(0.05)].copy()
        if not significant_pairs.empty:
            y_max = float(plot_data[onset_col].max())
            y_min = float(plot_data[onset_col].min())
            data_span = max(1.0, y_max - y_min)
            base_y = y_max + data_span * 0.12
            step_y = data_span * 0.10
            for pair_index, (_, row) in enumerate(significant_pairs.iterrows()):
                left = group_order.index(str(row["group1"])) + 1
                right = group_order.index(str(row["group2"])) + 1
                line_y = base_y + pair_index * step_y
                ax.plot([left, left, right, right], [line_y - 0.03 * data_span, line_y, line_y, line_y - 0.03 * data_span], color="#444444", linewidth=1.0)
                ax.text(
                    (left + right) / 2.0,
                    line_y + 0.02 * data_span,
                    f"p={float(row['p_value']):.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=_font_size(-2.0),
                    color="#444444")
            computed_top = base_y + max(1, len(significant_pairs)) * step_y + data_span * 0.08
            current_bottom = ax.get_ylim()[0]
            if y_limits is None:
                ax.set_ylim(bottom=current_bottom, top=computed_top)
            else:
                bottom = current_bottom if y_limits[0] is None else float(y_limits[0])
                top_override = computed_top if y_limits[1] is None else float(y_limits[1])
                ax.set_ylim(bottom=bottom, top=max(top_override, computed_top))
            _save_figure(fig, output_path)
            return

    if y_limits is not None:
        current_ylim = ax.get_ylim()
        bottom = current_ylim[0] if y_limits[0] is None else float(y_limits[0])
        top = current_ylim[1] if y_limits[1] is None else float(y_limits[1])
        ax.set_ylim(bottom, top)

    _save_figure(fig, output_path)

def plot_phase_activity_boxplot(
    mouse_phase_activity: pd.DataFrame,
    stats_table: pd.DataFrame,
    *,
    phase_display_names: dict[int, str],
    output_path: Path,
    value_col: str = "mean_visits_per_hour",
    title: str = "Mice activity per group and phase",
    ylabel: str = "Mean number of corner visits per hour",
    figsize_cm: tuple[float, float] | None = None,
    show_legend: bool = True,
    legend_loc: str | int | None = None,
    legend_font_size: float | None = None,
    show_n: bool = True,
    xtick_rotation: float = 45.0,
    ylim: tuple[float, float] | None = None,
    yticks: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> None:
    """Plot hourly activity per phase and experimental group."""

    if mouse_phase_activity.empty:
        return
    if value_col not in mouse_phase_activity.columns:
        value_col = "median_visits_per_hour"

    _prepare_output_path(output_path)
    phase_numbers = sorted(mouse_phase_activity["PhaseNumber"].unique())
    group_order = _ordered_group_names_from_series(mouse_phase_activity["Group"])
    fig, ax = plt.subplots(figsize=_resolve_figsize_cm(ACTIVITY_FIGSIZE_CM, figsize_cm))

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
                label=phase_display_names.get(int(phase_number), f"Phase {phase_number}")))

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
                value_col].to_numpy(dtype=float)
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
                boxprops={"linewidth": 1.1})
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

    y_min = max(0.0, float(mouse_phase_activity[value_col].min()) - 0.5)
    data_max = float(mouse_phase_activity[value_col].max())
    data_span = max(1.0, data_max - y_min)
    y_max = data_max + data_span * 0.25
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
        values = mouse_phase_activity.loc[
            mouse_phase_activity["Group"].astype(str).eq(group_name)
            & mouse_phase_activity["PhaseNumber"].eq(phase_number),
            value_col].dropna()
        local_top = float(values.max()) if not values.empty else y_min
        annotation_y = local_top + data_span * 0.06
        y_max = max(y_max, annotation_y + data_span * 0.10)
        ax.text(position, annotation_y, label, color="red", fontsize=_font_size(0.0), ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(y_min, y_max)

    xticks: list[float] = []
    xlabels: list[str] = []
    for group_name in group_order:
        xticks.append(group_centers[group_name])
        n_value = mouse_phase_activity.loc[mouse_phase_activity["Group"].astype(str).eq(group_name), "ET"].nunique()
        xlabels.append(f"{group_name}\n(n={n_value})" if show_n else group_name)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=float(xtick_rotation), ha="right" if xtick_rotation else "center")
    ax.set_title(title)
    ax.set_ylabel(_wrap_axis_label(ylabel))
    _apply_axis_overrides(ax, ylim=ylim, yticks=yticks)
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
        label="Outlier")
    if show_legend:
        _draw_legend(
            ax,
            handles=[*legend_handles, median_handle, outlier_handle],
            loc="upper left",
            legend_loc=legend_loc,
            legend_font_size=legend_font_size)
    _save_figure(fig, output_path)
# %% END
