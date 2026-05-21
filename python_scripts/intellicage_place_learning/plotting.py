"""Plotting helpers for IntelliCage place-learning analyses.

The plotting functions deliberately focus on interpretable summary figures for
poster preparation. Each function expects pre-aggregated tables from the metrics
module and keeps the visual design consistent across pathology groups and
phases. Individual mice are shown together with the group mean whenever that is
helpful for spotting heterogeneity.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import pandas as pd


GROUP_COLORS = {
    "WT": "#264653",
    "tdTomato": "#6c757d",
    "Tau 66-421": "#2a9d8f",
    "Tau 1-421": "#e76f51",
    "Tau 1-441": "#bc4749",
}


def _group_color(group_name: str) -> str:
    """Return a stable display color for one pathology group."""

    return GROUP_COLORS.get(group_name, "#457b9d")


def _prepare_output_path(output_path: Path) -> None:
    """Create the parent directory for one output file if necessary."""

    output_path.parent.mkdir(parents=True, exist_ok=True)


def plot_experiment_overview(
    mouse_bins: pd.DataFrame,
    summary_bins: pd.DataFrame,
    *,
    group_name: str,
    bin_hours: int,
    output_path: Path,
    phase_boundaries: dict[int, float] | None = None,
) -> None:
    """Plot full-experiment visit activity for one pathology group.

    Parameters
    ----------
    mouse_bins:
        Mouse-level bin counts created by
        :func:`intellicage_place_learning.metrics.compute_experiment_visit_bins`.
    summary_bins:
        Group-level mean and SEM table for the same metric.
    group_name:
        Pathology group that should be plotted.
    bin_hours:
        Width of the plotted time bins in hours.
    output_path:
        Target image path.
    phase_boundaries:
        Optional mapping from phase number to experiment-relative start hour.
        When provided, dashed vertical lines are drawn to orient the reader.
    """

    group_mouse = mouse_bins.loc[mouse_bins["Group"].astype(str).eq(group_name)].copy()
    group_summary = summary_bins.loc[summary_bins["Group"].astype(str).eq(group_name)].copy()
    if group_mouse.empty or group_summary.empty:
        return

    _prepare_output_path(output_path)
    color = _group_color(group_name)

    fig, ax = plt.subplots(figsize=(14, 6))
    for et_label, mouse_data in group_mouse.groupby("ETLabel", observed=True):
        ax.plot(
            mouse_data["bin_center_hours"],
            mouse_data["value"],
            linewidth=1.2,
            alpha=0.45,
            label=str(et_label),
        )

    ax.plot(
        group_summary["bin_center_hours"],
        group_summary["mean_value"],
        color=color,
        linewidth=3.0,
        label="Group mean",
    )
    ax.fill_between(
        group_summary["bin_center_hours"],
        group_summary["mean_value"] - group_summary["sem_value"],
        group_summary["mean_value"] + group_summary["sem_value"],
        color=color,
        alpha=0.18,
        linewidth=0,
    )

    max_hour = float(group_summary["bin_end_hours"].max())
    day_ticks = range(0, int(max_hour) + 24, 24)
    for tick in day_ticks:
        ax.axvline(tick, color="#d9d9d9", linewidth=0.8, zorder=0)
    if phase_boundaries:
        for phase_number, start_hour in sorted(phase_boundaries.items()):
            if phase_number > 1:
                ax.axvline(start_hour, color="#7a7a7a", linestyle="--", linewidth=1.2)

    ax.set_title(f"{group_name}: visits across the full experiment ({bin_hours} h bins)")
    ax.set_xlabel("Hours since experiment start")
    ax.set_ylabel("Visits per mouse and bin")
    ax.set_xlim(0, max_hour)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
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
) -> None:
    """Plot the phase-2 adaptation metric as paired bar charts.

    Parameters
    ----------
    primary_summary:
        Group-level phase-2 summary for all visits.
    secondary_summary:
        Group-level phase-2 summary for the selected drinking-related metric.
    group_name:
        Pathology group that should be plotted.
    bin_hours:
        Width of the plotted time bins in hours.
    output_path:
        Target image path.
    secondary_label:
        Human-readable label for the secondary metric shown in the legend.
    """

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

    ax.set_title(f"{group_name}: phase 2 nose-poke adaptation ({bin_hours} h bins)")
    ax.set_xlabel("Hours since start of phase 2")
    ax.set_ylabel("Mean count per mouse and bin")
    ax.set_xlim(0, float(visits_group["bin_end_hours"].max()))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_phase_learning(
    mouse_bins: pd.DataFrame,
    summary_bins: pd.DataFrame,
    *,
    group_name: str,
    phase_number: int,
    bin_hours: int,
    output_path: Path,
    ylabel: str,
) -> None:
    """Plot phase-3 or phase-4 place-learning performance for one group.

    Parameters
    ----------
    mouse_bins:
        Mouse-level binned metric table.
    summary_bins:
        Group-level mean and SEM table for the same metric.
    group_name:
        Pathology group that should be plotted.
    phase_number:
        Target phase number, usually `3` or `4`.
    bin_hours:
        Width of the plotted time bins in hours.
    output_path:
        Target image path.
    ylabel:
        Y-axis label describing the plotted metric.
    """

    group_mouse = mouse_bins.loc[mouse_bins["Group"].astype(str).eq(group_name)].copy()
    group_summary = summary_bins.loc[summary_bins["Group"].astype(str).eq(group_name)].copy()
    if group_mouse.empty or group_summary.empty:
        return

    _prepare_output_path(output_path)
    color = _group_color(group_name)

    fig, ax = plt.subplots(figsize=(14, 6))
    for _, mouse_data in group_mouse.groupby("ETLabel", observed=True):
        ax.plot(
            mouse_data["bin_center_hours"],
            mouse_data["value"],
            color="#9aa0a6",
            linewidth=1.0,
            alpha=0.45,
        )

    ax.plot(
        group_summary["bin_center_hours"],
        group_summary["mean_value"],
        color=color,
        linewidth=3.0,
        label="Group mean",
    )
    ax.fill_between(
        group_summary["bin_center_hours"],
        group_summary["mean_value"] - group_summary["sem_value"],
        group_summary["mean_value"] + group_summary["sem_value"],
        color=color,
        alpha=0.18,
        linewidth=0,
    )

    ax.set_title(f"{group_name}: phase {phase_number} place learning ({bin_hours} h bins)")
    ax.set_xlabel(f"Hours since start of phase {phase_number}")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, float(group_summary["bin_end_hours"].max()))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
