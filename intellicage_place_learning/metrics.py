"""Metric computation helpers for IntelliCage place-learning datasets.

The functions in this module create analysis-ready tables for time-binned
trajectories, correct-visit rates, phase-wise activity summaries, and helper
metadata such as robust common phase-duration recommendations.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


SpreadMetric = Literal["sem", "std"]


def infer_phase_boundaries(phase_manifest: pd.DataFrame) -> dict[int, float]:
    """Infer experiment-relative phase start hours from the manifest."""

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


def build_phase_time_limit_table(phase_manifest: pd.DataFrame) -> pd.DataFrame:
    """Build a table of observed and robust common phase durations.

    The recommended common duration is the maximum non-outlier duration across
    run groups, using the standard 1.5 IQR rule to detect unusually long phase
    recordings.
    """

    manifest = phase_manifest.copy()
    manifest["duration_hours"] = (
        manifest["PhaseEnd"] - manifest["PhaseStart"]
    ).dt.total_seconds() / 3600.0

    rows: list[dict[str, float | int | bool]] = []
    for phase_number, phase_data in manifest.groupby("PhaseNumber", observed=True):
        durations = phase_data["duration_hours"].to_numpy(dtype=float)
        q1 = float(np.quantile(durations, 0.25))
        q3 = float(np.quantile(durations, 0.75))
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        mask = durations <= upper_fence if iqr > 0 else np.ones_like(durations, dtype=bool)
        recommended = float(durations[mask].max()) if mask.any() else float(durations.max())
        for _, row in phase_data.iterrows():
            rows.append(
                {
                    "RunGroup": row["RunGroup"],
                    "Phase": row["Phase"],
                    "PhaseNumber": int(phase_number),
                    "duration_hours": float(row["duration_hours"]),
                    "upper_outlier_fence_hours": upper_fence,
                    "is_duration_outlier": bool(row["duration_hours"] > upper_fence) if iqr > 0 else False,
                    "recommended_common_limit_hours": recommended,
                }
            )
    return pd.DataFrame(rows).sort_values(["PhaseNumber", "RunGroup"]).reset_index(drop=True)


def suggest_common_phase_limits(phase_manifest: pd.DataFrame) -> dict[int, float]:
    """Return robust common phase-duration limits keyed by phase number."""

    table = build_phase_time_limit_table(phase_manifest)
    return (
        table.groupby("PhaseNumber", observed=True)["recommended_common_limit_hours"]
        .first()
        .astype(float)
        .to_dict()
    )


def filter_visits_by_phase_limits(
    visits: pd.DataFrame,
    phase_max_hours: dict[int, float] | None,
) -> pd.DataFrame:
    """Filter visits so that selected phases stop after a configured hour."""

    if not phase_max_hours:
        return visits.copy()
    filtered = visits.copy()
    keep_mask = pd.Series(True, index=filtered.index)
    for phase_number, max_hours in phase_max_hours.items():
        keep_mask &= ~(
            filtered["PhaseNumber"].eq(int(phase_number)) & filtered["phase_elapsed_hours"].gt(float(max_hours))
        )
    return filtered.loc[keep_mask].copy()


def build_phase_window_table(
    visits: pd.DataFrame,
    phase_boundaries: dict[int, float],
) -> pd.DataFrame:
    """Describe the visible start and end of every phase after filtering."""

    rows: list[dict[str, float | int]] = []
    for phase_number in sorted(phase_boundaries):
        phase_visits = visits.loc[visits["PhaseNumber"].eq(phase_number)]
        if phase_visits.empty:
            continue
        rows.append(
            {
                "PhaseNumber": int(phase_number),
                "start_hours": float(phase_boundaries[phase_number]),
                "end_hours": float(phase_visits["experiment_elapsed_hours"].max()),
                "duration_hours": float(phase_visits["phase_elapsed_hours"].max()),
            }
        )
    return pd.DataFrame(rows)


def _empty_count_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return empty count-shaped mouse and summary tables."""

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
            "median_value",
            "std_value",
            "sem_value",
            "mouse_n",
        ]
    )
    return empty_mouse, empty_summary


def _summarize_mouse_values(mouse_bins: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Summarize mouse-level binned values to group means, medians, and spread."""

    summary = (
        mouse_bins.groupby(["Group", "bin_start_hours"], observed=True)[value_col]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_value",
                "median": "median_value",
                "std": "std_value",
                "count": "mouse_n",
            }
        )
    )
    summary["std_value"] = summary["std_value"].fillna(0.0)
    summary["sem_value"] = summary["std_value"] / np.sqrt(summary["mouse_n"].clip(lower=1))
    return summary


def _build_complete_mouse_bin_frame(
    mice: pd.DataFrame,
    max_time: float,
    bin_hours: int,
) -> pd.DataFrame:
    """Create the full mouse-by-bin grid for count or rate tables."""

    max_bin_start = float(np.floor(max_time / float(bin_hours)) * float(bin_hours))
    all_bins = pd.DataFrame(
        {"bin_start_hours": np.arange(0.0, max_bin_start + float(bin_hours), float(bin_hours))}
    )
    all_bins["bin_start_hours"] = all_bins["bin_start_hours"].round(8)
    mice = mice.copy()
    mice["__key"] = 1
    all_bins["__key"] = 1
    full_index = mice.merge(all_bins, on="__key", how="outer").drop(columns="__key")
    full_index["bin_end_hours"] = full_index["bin_start_hours"] + float(bin_hours)
    full_index["bin_center_hours"] = full_index["bin_start_hours"] + float(bin_hours) / 2.0
    return full_index


def _prepare_count_bins(
    data: pd.DataFrame,
    *,
    time_col: str,
    bin_hours: int,
    value_col: str = "_value",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate event counts into mouse-level and group-level time bins."""

    if data.empty:
        return _empty_count_tables()

    work = data.copy()
    work["bin_start_hours"] = np.floor(work[time_col] / float(bin_hours)) * float(bin_hours)
    mouse_counts = (
        work.groupby(["Group", "ET", "ETLabel", "SEX", "bin_start_hours"], observed=True)[value_col]
        .sum()
        .reset_index(name="value")
    )
    mice = work.loc[:, ["Group", "ET", "ETLabel", "SEX"]].drop_duplicates().reset_index(drop=True)
    full_index = _build_complete_mouse_bin_frame(mice, float(work[time_col].max()), bin_hours)
    mouse_bins = full_index.merge(
        mouse_counts,
        on=["Group", "ET", "ETLabel", "SEX", "bin_start_hours"],
        how="left",
        validate="one_to_one",
    )
    mouse_bins["value"] = mouse_bins["value"].fillna(0.0)
    summary = _summarize_mouse_values(mouse_bins)
    summary["bin_end_hours"] = summary["bin_start_hours"] + float(bin_hours)
    summary["bin_center_hours"] = summary["bin_start_hours"] + float(bin_hours) / 2.0
    return mouse_bins, summary


def _prepare_rate_bins(
    data: pd.DataFrame,
    *,
    time_col: str,
    bin_hours: int,
    success_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate success rates into mouse-level and group-level time bins."""

    if data.empty:
        empty_mouse, empty_summary = _empty_count_tables()
        empty_mouse["correct_visits"] = pd.Series(dtype=float)
        empty_mouse["all_visits"] = pd.Series(dtype=float)
        empty_summary["mean_correct_visits"] = pd.Series(dtype=float)
        empty_summary["mean_all_visits"] = pd.Series(dtype=float)
        return empty_mouse, empty_summary

    work = data.copy()
    work["bin_start_hours"] = np.floor(work[time_col] / float(bin_hours)) * float(bin_hours)
    grouped = (
        work.groupby(["Group", "ET", "ETLabel", "SEX", "bin_start_hours"], observed=True)
        .agg(
            correct_visits=(success_col, "sum"),
            all_visits=("VisitID", "size"),
        )
        .reset_index()
    )
    grouped["value"] = grouped["correct_visits"] / grouped["all_visits"]

    mice = work.loc[:, ["Group", "ET", "ETLabel", "SEX"]].drop_duplicates().reset_index(drop=True)
    full_index = _build_complete_mouse_bin_frame(mice, float(work[time_col].max()), bin_hours)
    mouse_bins = full_index.merge(
        grouped,
        on=["Group", "ET", "ETLabel", "SEX", "bin_start_hours"],
        how="left",
        validate="one_to_one",
    )
    mouse_bins["all_visits"] = mouse_bins["all_visits"].fillna(0.0)
    mouse_bins["correct_visits"] = mouse_bins["correct_visits"].fillna(0.0)
    mouse_bins["value"] = mouse_bins["value"].where(mouse_bins["all_visits"].gt(0), np.nan)

    summary = _summarize_mouse_values(mouse_bins)
    visit_summary = (
        mouse_bins.groupby(["Group", "bin_start_hours"], observed=True)
        .agg(
            mean_correct_visits=("correct_visits", "mean"),
            mean_all_visits=("all_visits", "mean"),
            contributing_mouse_n=("value", lambda values: int(values.notna().sum())),
        )
        .reset_index()
    )
    summary = summary.merge(visit_summary, on=["Group", "bin_start_hours"], how="left", validate="one_to_one")
    summary["bin_end_hours"] = summary["bin_start_hours"] + float(bin_hours)
    summary["bin_center_hours"] = summary["bin_start_hours"] + float(bin_hours) / 2.0
    return mouse_bins, summary


def compute_experiment_visit_bins(visits: pd.DataFrame, *, bin_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count total visits per mouse across the full experiment timeline."""

    data = visits.copy()
    data["_value"] = 1.0
    return _prepare_count_bins(data, time_col="experiment_elapsed_hours", bin_hours=bin_hours)


def compute_experiment_drinking_visit_bins(
    visits: pd.DataFrame,
    *,
    bin_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count lick-positive nose-poke visits across the full experiment timeline."""

    data = visits.loc[visits["has_nosepoke"] & visits["visit_has_lick"]].copy()
    data["_value"] = 1.0
    return _prepare_count_bins(data, time_col="experiment_elapsed_hours", bin_hours=bin_hours)


def compute_phase2_adaptation_bins(
    visits: pd.DataFrame,
    *,
    bin_hours: int,
    secondary_metric: Literal["lick_positive_visits", "lick_count"] = "lick_positive_visits",
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Summarize phase-2 nose-poke adaptation with two complementary metrics."""

    phase2 = visits.loc[visits["PhaseNumber"].eq(2)].copy()
    phase2["_value"] = 1.0
    visit_bins = _prepare_count_bins(phase2, time_col="phase_elapsed_hours", bin_hours=bin_hours)

    drink_visits = phase2.loc[phase2["phase2_drinking_visit"]].copy()
    drink_visits["_value"] = 1.0
    lick_positive_bins = _prepare_count_bins(
        drink_visits,
        time_col="phase_elapsed_hours",
        bin_hours=bin_hours,
    )

    lick_counts = phase2.copy()
    lick_counts["_value"] = lick_counts["LickNumber"].fillna(0).astype(float)
    lick_count_bins = _prepare_count_bins(
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


def compute_place_learning_count_bins(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    bin_hours: int,
    strict: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize correct-visit counts for phase 3 or phase 4."""

    phase_visits = visits.loc[visits["PhaseNumber"].eq(phase_number)].copy()
    success_col = "rewarded_place_visit" if strict else "correct_place_visit"
    phase_visits = phase_visits.loc[phase_visits[success_col]].copy()
    phase_visits["_value"] = 1.0
    return _prepare_count_bins(
        phase_visits,
        time_col="phase_elapsed_hours",
        bin_hours=bin_hours,
    )


def compute_phase_visit_count_bins(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    bin_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize all visit counts for a selected phase."""

    phase_visits = visits.loc[visits["PhaseNumber"].eq(phase_number)].copy()
    phase_visits["_value"] = 1.0
    return _prepare_count_bins(
        phase_visits,
        time_col="phase_elapsed_hours",
        bin_hours=bin_hours,
    )


def compute_place_learning_rate_bins(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    bin_hours: int,
    strict: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize correct-visit rates for phase 3 or phase 4."""

    phase_visits = visits.loc[visits["PhaseNumber"].eq(phase_number)].copy()
    success_col = "rewarded_place_visit" if strict else "correct_place_visit"
    return _prepare_rate_bins(
        phase_visits,
        time_col="phase_elapsed_hours",
        bin_hours=bin_hours,
        success_col=success_col,
    )


def compute_phase_activity_medians(
    visits: pd.DataFrame,
    *,
    hourly_bin_size: int = 1,
) -> pd.DataFrame:
    """Compute per-mouse median hourly visit activity for each phase.

    The calculation follows the interpretation used by the legacy MATLAB figure:
    within each phase, hourly visit counts are built for each mouse, missing
    hours are filled with zeros, and the median across hourly bins is then used
    as the phase activity summary for that mouse.
    """

    phase_visits = visits.copy()
    phase_visits["hour_bin_start"] = np.floor(phase_visits["phase_elapsed_hours"] / float(hourly_bin_size))
    phase_visits["hour_bin_start"] = phase_visits["hour_bin_start"] * float(hourly_bin_size)

    counts = (
        phase_visits.groupby(
            ["RunGroup", "PhaseNumber", "Group", "ET", "ETLabel", "hour_bin_start"],
            observed=True,
        )["VisitID"]
        .size()
        .reset_index(name="visits_in_hour")
    )

    phase_duration_hours = (
        phase_visits.groupby(["RunGroup", "PhaseNumber"], observed=True)["phase_elapsed_hours"]
        .max()
        .reset_index(name="phase_max_hour")
    )
    mice = phase_visits.loc[:, ["RunGroup", "PhaseNumber", "Group", "ET", "ETLabel"]].drop_duplicates()
    complete_rows: list[pd.DataFrame] = []
    for _, duration_row in phase_duration_hours.iterrows():
        run_group = duration_row["RunGroup"]
        phase_number = duration_row["PhaseNumber"]
        max_hour = int(np.floor(float(duration_row["phase_max_hour"]) / float(hourly_bin_size)) * hourly_bin_size)
        phase_mice = mice.loc[mice["RunGroup"].eq(run_group) & mice["PhaseNumber"].eq(phase_number)].copy()
        hour_grid = pd.DataFrame({"hour_bin_start": np.arange(0, max_hour + hourly_bin_size, hourly_bin_size)})
        phase_mice["__key"] = 1
        hour_grid["__key"] = 1
        complete_rows.append(phase_mice.merge(hour_grid, on="__key", how="outer").drop(columns="__key"))
    complete_index = pd.concat(complete_rows, ignore_index=True)
    complete_counts = complete_index.merge(
        counts,
        on=["RunGroup", "PhaseNumber", "Group", "ET", "ETLabel", "hour_bin_start"],
        how="left",
        validate="one_to_one",
    )
    complete_counts["visits_in_hour"] = complete_counts["visits_in_hour"].fillna(0.0)

    mouse_medians = (
        complete_counts.groupby(["Group", "ET", "ETLabel", "PhaseNumber"], observed=True)["visits_in_hour"]
        .median()
        .reset_index(name="median_visits_per_hour")
    )
    return mouse_medians.sort_values(["Group", "PhaseNumber", "ET"]).reset_index(drop=True)


def compute_phase_activity_statistics(mouse_phase_activity: pd.DataFrame) -> pd.DataFrame:
    """Compute legacy-style phase activity statistics per pathology group.

    The old exported CSV files contain an omnibus one-way ANOVA plus pairwise
    phase comparisons. This function reproduces that logic using one-way ANOVA
    across phases 1-4 within a pathology group and Tukey HSD for the pairwise
    post-hoc tests.
    """

    rows: list[dict[str, float | str | int]] = []
    for group_name, group_data in mouse_phase_activity.groupby("Group", observed=True):
        phase_values = {
            phase_number: group_data.loc[group_data["PhaseNumber"].eq(phase_number), "median_visits_per_hour"].to_numpy()
            for phase_number in sorted(group_data["PhaseNumber"].unique())
        }
        valid_arrays = [values for values in phase_values.values() if len(values) > 0]
        anova_p = float(stats.f_oneway(*valid_arrays).pvalue) if len(valid_arrays) >= 2 else np.nan

        tukey_frame = pd.DataFrame(columns=["group1", "group2", "p-adj"])
        if len(group_data["PhaseNumber"].unique()) >= 2:
            tukey = pairwise_tukeyhsd(
                endog=group_data["median_visits_per_hour"].to_numpy(),
                groups=group_data["PhaseNumber"].astype(str).to_numpy(),
            )
            tukey_frame = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            tukey_frame["group1"] = tukey_frame["group1"].astype(str)
            tukey_frame["group2"] = tukey_frame["group2"].astype(str)
            tukey_frame["p-adj"] = tukey_frame["p-adj"].astype(float)

        for phase_number in sorted(group_data["PhaseNumber"].unique()):
            if phase_number == 1:
                continue
            comparison = tukey_frame.loc[
                (
                    tukey_frame["group1"].eq("1") & tukey_frame["group2"].eq(str(phase_number))
                )
                | (
                    tukey_frame["group1"].eq(str(phase_number)) & tukey_frame["group2"].eq("1")
                )
            ]
            pairwise_p = float(comparison["p-adj"].iloc[0]) if not comparison.empty else np.nan
            rows.append(
                {
                    "Group": str(group_name),
                    "PhaseNumber": int(phase_number),
                    "ReferencePhaseNumber": 1,
                    "anova_p_value": anova_p,
                    "pairwise_p_value_vs_phase1": pairwise_p,
                }
            )
    return pd.DataFrame(rows).sort_values(["Group", "PhaseNumber"]).reset_index(drop=True)
