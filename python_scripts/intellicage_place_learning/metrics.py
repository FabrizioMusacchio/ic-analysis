"""Metric computation helpers for IntelliCage place-learning datasets.

The functions in this module convert the merged visit table into analysis-ready
time-bin summaries. They are written around one central idea: first aggregate to
the mouse level, then average across mice within each pathology group. This
preserves individual trajectories, keeps group-level summaries comparable across
unequal sample sizes, and makes it straightforward to overlay individual traces
with the group mean in later plots.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def infer_phase_boundaries(phase_manifest: pd.DataFrame) -> dict[int, float]:
    """Infer experiment-relative phase boundary hours from the manifest.

    Parameters
    ----------
    phase_manifest:
        Phase manifest created by the loader. Each row stores the start time of
        one phase inside one run group.

    Returns
    -------
    dict[int, float]
        Mapping from phase number to the median experiment-relative start hour
        across run groups. The median keeps the result stable even if start
        timestamps differ slightly between runs.
    """

    experiment_starts = (
        phase_manifest.loc[phase_manifest["PhaseNumber"].eq(1), ["RunGroup", "PhaseStart"]]
        .rename(columns={"PhaseStart": "ExperimentStart"})
        .copy()
    )
    aligned = phase_manifest.merge(experiment_starts, on="RunGroup", how="left", validate="many_to_one")
    aligned["phase_start_hour"] = (
        aligned["PhaseStart"] - aligned["ExperimentStart"]
    ).dt.total_seconds() / 3600.0
    boundaries = aligned.groupby("PhaseNumber", observed=True)["phase_start_hour"].median().to_dict()
    return {int(key): float(value) for key, value in boundaries.items()}


def _prepare_mouse_bin_table(
    data: pd.DataFrame,
    *,
    time_col: str,
    bin_hours: int,
    value_col: str = "_value",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate a visit table into mouse-level and group-level time bins.

    Parameters
    ----------
    data:
        Input visit table. The table must already be filtered to the events of
        interest and must contain a numeric column stored in `value_col`.
    time_col:
        Name of the elapsed-time column that should drive the binning. Typical
        values are `experiment_elapsed_hours` or `phase_elapsed_hours`.
    bin_hours:
        Width of one time bin in hours.
    value_col:
        Numeric column that will be summed within each time bin.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Mouse-level bin table and group-level summary table.
    """

    if data.empty:
        empty_mouse = pd.DataFrame(
            columns=[
                "Group",
                "ET",
                "ETLabel",
                "SEX",
                "bin_start_hours",
                "bin_end_hours",
                "bin_center_hours",
                "value",
            ]
        )
        empty_summary = pd.DataFrame(
            columns=[
                "Group",
                "bin_start_hours",
                "bin_end_hours",
                "bin_center_hours",
                "mean_value",
                "std_value",
                "sem_value",
                "mouse_n",
            ]
        )
        return empty_mouse, empty_summary

    work = data.copy()
    work["bin_start_hours"] = np.floor(work[time_col] / float(bin_hours)) * float(bin_hours)

    mouse_counts = (
        work.groupby(["Group", "ET", "ETLabel", "SEX", "bin_start_hours"], observed=True)[value_col]
        .sum()
        .reset_index(name="value")
    )

    mice = work.loc[:, ["Group", "ET", "ETLabel", "SEX"]].drop_duplicates().reset_index(drop=True)
    max_time = float(work[time_col].max())
    max_bin_start = float(np.floor(max_time / float(bin_hours)) * float(bin_hours))
    all_bins = pd.DataFrame(
        {"bin_start_hours": np.arange(0.0, max_bin_start + float(bin_hours), float(bin_hours))}
    )
    all_bins["bin_start_hours"] = all_bins["bin_start_hours"].round(8)
    mice["__key"] = 1
    all_bins["__key"] = 1
    full_index = mice.merge(all_bins, on="__key", how="outer").drop(columns="__key")

    mouse_bins = full_index.merge(
        mouse_counts,
        on=["Group", "ET", "ETLabel", "SEX", "bin_start_hours"],
        how="left",
        validate="one_to_one",
    )
    mouse_bins["value"] = mouse_bins["value"].fillna(0.0)
    mouse_bins["bin_end_hours"] = mouse_bins["bin_start_hours"] + float(bin_hours)
    mouse_bins["bin_center_hours"] = mouse_bins["bin_start_hours"] + float(bin_hours) / 2.0

    summary = (
        mouse_bins.groupby(["Group", "bin_start_hours"], observed=True)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_value", "std": "std_value", "count": "mouse_n"})
    )
    summary["std_value"] = summary["std_value"].fillna(0.0)
    summary["sem_value"] = summary["std_value"] / np.sqrt(summary["mouse_n"].clip(lower=1))
    summary["bin_end_hours"] = summary["bin_start_hours"] + float(bin_hours)
    summary["bin_center_hours"] = summary["bin_start_hours"] + float(bin_hours) / 2.0
    return mouse_bins, summary


def compute_experiment_visit_bins(visits: pd.DataFrame, *, bin_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count total visits per mouse across the full experiment timeline.

    Parameters
    ----------
    visits:
        Merged visit table returned by the loader.
    bin_hours:
        Width of the time bins in hours.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Mouse-level visit counts per bin and the corresponding group-level mean
        and SEM table.
    """

    data = visits.copy()
    data["_value"] = 1.0
    return _prepare_mouse_bin_table(data, time_col="experiment_elapsed_hours", bin_hours=bin_hours)


def compute_phase2_adaptation_bins(
    visits: pd.DataFrame,
    *,
    bin_hours: int,
    secondary_metric: Literal["lick_positive_visits", "lick_count"] = "lick_positive_visits",
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Summarize phase-2 nose-poke adaptation with two complementary metrics.

    Parameters
    ----------
    visits:
        Merged visit table returned by the loader.
    bin_hours:
        Width of the time bins in hours.
    secondary_metric:
        Secondary drinking-related metric. `lick_positive_visits` counts visits
        with at least one lick, whereas `lick_count` sums the raw visit-level
        lick counts.

    Returns
    -------
    dict[str, tuple[pandas.DataFrame, pandas.DataFrame]]
        Dictionary with three entries:

        - `visits`: all visit counts in phase 2
        - `drinking_metric`: selected secondary metric
        - `lick_count`: raw lick counts, always returned for completeness
    """

    phase2 = visits.loc[visits["PhaseNumber"].eq(2)].copy()

    phase2["_value"] = 1.0
    visit_bins = _prepare_mouse_bin_table(phase2, time_col="phase_elapsed_hours", bin_hours=bin_hours)

    drink_visits = phase2.loc[phase2["phase2_drinking_visit"]].copy()
    drink_visits["_value"] = 1.0
    lick_positive_bins = _prepare_mouse_bin_table(
        drink_visits,
        time_col="phase_elapsed_hours",
        bin_hours=bin_hours,
    )

    lick_counts = phase2.copy()
    lick_counts["_value"] = lick_counts["LickNumber"].fillna(0).astype(float)
    lick_count_bins = _prepare_mouse_bin_table(
        lick_counts,
        time_col="phase_elapsed_hours",
        bin_hours=bin_hours,
    )

    chosen_secondary = lick_positive_bins if secondary_metric == "lick_positive_visits" else lick_count_bins
    return {
        "visits": visit_bins,
        "drinking_metric": chosen_secondary,
        "lick_positive_visits": lick_positive_bins,
        "lick_count": lick_count_bins,
    }


def compute_place_learning_bins(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    bin_hours: int,
    strict: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize place-learning performance for phase 3 or phase 4.

    Parameters
    ----------
    visits:
        Merged visit table returned by the loader.
    phase_number:
        Target phase. Only phases 3 and 4 are meaningful for this metric.
    bin_hours:
        Width of the time bins in hours.
    strict:
        If `True`, use the poster-oriented strict metric:
        correct corner plus nose-poke plus licking. If `False`, reproduce the
        simpler MATLAB-compatible metric that only requires `PlaceError == 0`.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Mouse-level and group-level time-bin tables for the requested metric.
    """

    phase_visits = visits.loc[visits["PhaseNumber"].eq(phase_number)].copy()
    metric_mask = phase_visits["rewarded_place_visit"] if strict else phase_visits["correct_place_visit"]
    phase_visits = phase_visits.loc[metric_mask].copy()
    phase_visits["_value"] = 1.0
    return _prepare_mouse_bin_table(
        phase_visits,
        time_col="phase_elapsed_hours",
        bin_hours=bin_hours,
    )
