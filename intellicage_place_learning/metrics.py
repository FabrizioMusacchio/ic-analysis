"""Metric computation helpers for IntelliCage place-learning datasets.

The functions in this module create analysis-ready tables for time-binned
trajectories, correct-visit rates, phase-wise activity summaries, and helper
metadata such as robust common phase-duration recommendations.
"""
# %% IMPORTS
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# %%  TYPE ALIASES
SpreadMetric = Literal["sem", "std"]

# %% METRIC COMPUTATION FUNCTIONS
def flag_iqr_outliers(
    data: pd.DataFrame,
    *,
    value_col: str,
    group_cols: list[str],
    fence_factor: float = 1.5,
    min_group_size: int = 4,
) -> pd.DataFrame:
    """Flag IQR-based outliers within user-defined groups.

    The returned frame keeps all original rows and appends the columns
    `is_outlier`, `outlier_lower_fence`, and `outlier_upper_fence`.
    Outlier detection is only applied when a subgroup contains at least
    `min_group_size` non-missing values.
    """

    flagged = data.copy()
    flagged["is_outlier"] = False
    flagged["outlier_lower_fence"] = np.nan
    flagged["outlier_upper_fence"] = np.nan
    if flagged.empty:
        return flagged

    for _, subgroup in flagged.groupby(group_cols, observed=True, dropna=False):
        valid = subgroup[value_col].dropna().to_numpy(dtype=float)
        if len(valid) < min_group_size:
            continue
        q1 = float(np.quantile(valid, 0.25))
        q3 = float(np.quantile(valid, 0.75))
        iqr = q3 - q1
        lower = q1 - float(fence_factor) * iqr
        upper = q3 + float(fence_factor) * iqr
        subgroup_index = subgroup.index
        flagged.loc[subgroup_index, "outlier_lower_fence"] = lower
        flagged.loc[subgroup_index, "outlier_upper_fence"] = upper
        if iqr <= 0:
            continue
        flagged.loc[subgroup_index, "is_outlier"] = (
            flagged.loc[subgroup_index, value_col].lt(lower)
            | flagged.loc[subgroup_index, value_col].gt(upper)
        ).fillna(False)
    return flagged

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
            filtered["AnalysisPhaseNumber"].eq(int(phase_number))
            & filtered["analysis_phase_elapsed_hours"].gt(float(max_hours))
        )
    return filtered.loc[keep_mask].copy()

def build_analysis_phase_window_table(
    visits: pd.DataFrame,
    scheduled_phase_start_hours: dict[int, float],
) -> pd.DataFrame:
    """Describe the visible protocol phase windows on the aligned timeline."""

    sorted_starts = sorted((int(key), float(value)) for key, value in scheduled_phase_start_hours.items())
    if not sorted_starts:
        return pd.DataFrame(columns=["PhaseNumber", "start_hours", "end_hours", "duration_hours"])

    global_end_hour = float(visits["analysis_experiment_elapsed_hours"].max()) if not visits.empty else 0.0
    rows: list[dict[str, float | int]] = []
    for index, (phase_number, start_hour) in enumerate(sorted_starts):
        if phase_number not in {1, 2, 3, 4}:
            continue
        next_start = sorted_starts[index + 1][1] if index + 1 < len(sorted_starts) else global_end_hour
        rows.append(
            {
                "PhaseNumber": phase_number,
                "start_hours": start_hour,
                "end_hours": min(float(next_start), global_end_hour),
                "duration_hours": min(float(next_start), global_end_hour) - float(start_hour),
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
    return _prepare_count_bins(data, time_col="analysis_experiment_elapsed_hours", bin_hours=bin_hours)

def compute_experiment_drinking_visit_bins(
    visits: pd.DataFrame,
    *,
    bin_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count lick-positive nose-poke visits across the full experiment timeline."""

    data = visits.loc[visits["has_nosepoke"] & visits["visit_has_lick"]].copy()
    data["_value"] = 1.0
    return _prepare_count_bins(data, time_col="analysis_experiment_elapsed_hours", bin_hours=bin_hours)

def compute_experiment_nosepoke_count_bins(
    visits: pd.DataFrame,
    *,
    bin_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum nose-poke event counts per mouse across the aligned experiment timeline."""

    data = visits.copy()
    data["_value"] = data["nosepoke_event_count"].fillna(0).astype(float)
    return _prepare_count_bins(data, time_col="analysis_experiment_elapsed_hours", bin_hours=bin_hours)

def compute_experiment_lick_count_bins(
    visits: pd.DataFrame,
    *,
    bin_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum lick counts per mouse across the aligned experiment timeline."""

    data = visits.copy()
    data["_value"] = data["LickNumber"].fillna(0).astype(float)
    return _prepare_count_bins(data, time_col="analysis_experiment_elapsed_hours", bin_hours=bin_hours)

def compute_phase2_adaptation_bins(
    visits: pd.DataFrame,
    *,
    bin_hours: int,
    secondary_metric: Literal["lick_positive_visits", "lick_count"] = "lick_positive_visits",
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Summarize phase-2 nose-poke adaptation with two complementary metrics."""

    phase2 = visits.loc[visits["AnalysisPhaseNumber"].eq(2)].copy()
    phase2["_value"] = 1.0
    visit_bins = _prepare_count_bins(phase2, time_col="analysis_phase_elapsed_hours", bin_hours=bin_hours)

    drink_visits = phase2.loc[phase2["has_nosepoke"] & phase2["visit_has_lick"]].copy()
    drink_visits["_value"] = 1.0
    lick_positive_bins = _prepare_count_bins(
        drink_visits,
        time_col="analysis_phase_elapsed_hours",
        bin_hours=bin_hours,
    )

    lick_counts = phase2.copy()
    lick_counts["_value"] = lick_counts["LickNumber"].fillna(0).astype(float)
    lick_count_bins = _prepare_count_bins(
        lick_counts,
        time_col="analysis_phase_elapsed_hours",
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
    success_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize selected place-learning event counts for phase 3 or phase 4."""

    phase_visits = visits.loc[visits["AnalysisPhaseNumber"].eq(phase_number)].copy()
    phase_visits = phase_visits.loc[phase_visits[success_col]].copy()
    phase_visits["_value"] = 1.0
    return _prepare_count_bins(
        phase_visits,
        time_col="analysis_phase_elapsed_hours",
        bin_hours=bin_hours,
    )

def compute_phase_visit_count_bins(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    bin_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize all visit counts for a selected phase."""

    phase_visits = visits.loc[visits["AnalysisPhaseNumber"].eq(phase_number)].copy()
    phase_visits["_value"] = 1.0
    return _prepare_count_bins(
        phase_visits,
        time_col="analysis_phase_elapsed_hours",
        bin_hours=bin_hours,
    )

def compute_place_learning_rate_bins(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    bin_hours: int,
    success_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize selected place-learning rates for phase 3 or phase 4."""

    phase_visits = visits.loc[visits["AnalysisPhaseNumber"].eq(phase_number)].copy()
    return _prepare_rate_bins(
        phase_visits,
        time_col="analysis_phase_elapsed_hours",
        bin_hours=bin_hours,
        success_col=success_col,
    )

def compute_phase4_reversal_rate_bins(
    visits: pd.DataFrame,
    *,
    bin_hours: int,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Summarize place-reversal corner-choice rates for new, previous, and neutral corners."""

    phase4 = visits.loc[visits["AnalysisPhaseNumber"].eq(4)].copy()
    if not phase4.empty:
        neutral_1 = np.zeros(len(phase4), dtype=bool)
        neutral_2 = np.zeros(len(phase4), dtype=bool)
        for index, row in enumerate(phase4.itertuples(index=False)):
            if pd.isna(row.Corner) or pd.isna(row.CornerPhase3) or pd.isna(row.CornerPhase4):
                continue
            neutrals = [
                corner for corner in (1, 2, 3, 4)
                if corner not in {int(row.CornerPhase3), int(row.CornerPhase4)}
            ]
            if len(neutrals) != 2:
                continue
            if int(row.Corner) == neutrals[0]:
                neutral_1[index] = True
            elif int(row.Corner) == neutrals[1]:
                neutral_2[index] = True
        phase4["neutral_incorrect_corner_1_visit"] = neutral_1
        phase4["neutral_incorrect_corner_2_visit"] = neutral_2

    return {
        "new_correct_corner": compute_place_learning_rate_bins(
            phase4,
            phase_number=4,
            bin_hours=bin_hours,
            success_col="correct_corner_visit",
        ),
        "previous_correct_corner": compute_place_learning_rate_bins(
            phase4,
            phase_number=4,
            bin_hours=bin_hours,
            success_col="previous_correct_corner_visit",
        ),
        "neutral_incorrect_corner_1": compute_place_learning_rate_bins(
            phase4,
            phase_number=4,
            bin_hours=bin_hours,
            success_col="neutral_incorrect_corner_1_visit",
        ),
        "neutral_incorrect_corner_2": compute_place_learning_rate_bins(
            phase4,
            phase_number=4,
            bin_hours=bin_hours,
            success_col="neutral_incorrect_corner_2_visit",
        ),
    }

def compute_phase4_reversal_count_bins(
    visits: pd.DataFrame,
    *,
    bin_hours: int,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Summarize phase-4 reversal component counts for new, old, and neutral corners."""

    phase4 = visits.loc[visits["AnalysisPhaseNumber"].eq(4)].copy()
    if not phase4.empty:
        neutral_1 = np.zeros(len(phase4), dtype=bool)
        neutral_2 = np.zeros(len(phase4), dtype=bool)
        for index, row in enumerate(phase4.itertuples(index=False)):
            if pd.isna(row.Corner) or pd.isna(row.CornerPhase3) or pd.isna(row.CornerPhase4):
                continue
            neutrals = [
                corner for corner in (1, 2, 3, 4)
                if corner not in {int(row.CornerPhase3), int(row.CornerPhase4)}
            ]
            if len(neutrals) != 2:
                continue
            if int(row.Corner) == neutrals[0]:
                neutral_1[index] = True
            elif int(row.Corner) == neutrals[1]:
                neutral_2[index] = True
        phase4["neutral_incorrect_corner_1_visit"] = neutral_1
        phase4["neutral_incorrect_corner_2_visit"] = neutral_2

    return {
        "new_correct_corner": compute_place_learning_count_bins(
            phase4,
            phase_number=4,
            bin_hours=bin_hours,
            success_col="correct_corner_visit",
        ),
        "previous_correct_corner": compute_place_learning_count_bins(
            phase4,
            phase_number=4,
            bin_hours=bin_hours,
            success_col="previous_correct_corner_visit",
        ),
        "neutral_incorrect_corner_1": compute_place_learning_count_bins(
            phase4,
            phase_number=4,
            bin_hours=bin_hours,
            success_col="neutral_incorrect_corner_1_visit",
        ),
        "neutral_incorrect_corner_2": compute_place_learning_count_bins(
            phase4,
            phase_number=4,
            bin_hours=bin_hours,
            success_col="neutral_incorrect_corner_2_visit",
        ),
    }

def _phase_segment_table(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    origin_clock_hour: float,
    awake_start_clock_hour: float,
    awake_end_clock_hour: float,
) -> pd.DataFrame:
    """Annotate visits with mouse-day aligned phase-day and awake/sleep segment labels."""

    phase_visits = visits.loc[visits["AnalysisPhaseNumber"].eq(phase_number)].copy()
    if phase_visits.empty:
        return phase_visits

    phase_offset_hours = origin_clock_hour - awake_start_clock_hour
    phase_visits["mouse_day_aligned_hours"] = phase_visits["analysis_phase_elapsed_hours"] + phase_offset_hours
    phase_visits["segment_day"] = np.floor(phase_visits["mouse_day_aligned_hours"] / 24.0).astype(int) + 1
    phase_visits["segment_clock_hour"] = np.mod(phase_visits["mouse_day_aligned_hours"], 24.0)
    awake_duration = awake_end_clock_hour - awake_start_clock_hour
    phase_visits["segment_name"] = np.where(
        phase_visits["segment_clock_hour"] < awake_duration,
        "awake",
        "sleep",
    )
    phase_visits["segment_order"] = (phase_visits["segment_day"] - 1) * 2 + np.where(
        phase_visits["segment_name"].eq("awake"),
        1,
        2,
    )
    phase_visits["segment_label"] = (
        "Day "
        + phase_visits["segment_day"].astype(str)
        + " "
        + phase_visits["segment_name"].astype(str)
    )
    return phase_visits

def compute_phase_segment_rate_tables(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    success_col: str,
    origin_clock_hour: float,
    awake_start_clock_hour: float,
    awake_end_clock_hour: float,
    max_days: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mouse-level and group-level rates for awake/sleep segments within one phase."""

    phase_visits = _phase_segment_table(
        visits,
        phase_number=phase_number,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )
    phase_visits = phase_visits.loc[phase_visits["segment_day"].between(1, max_days)].copy()
    if phase_visits.empty:
        return _empty_count_tables()

    grouped = (
        phase_visits.groupby(
            ["Group", "ET", "ETLabel", "SEX", "segment_day", "segment_name", "segment_order", "segment_label"],
            observed=True,
        )
        .agg(
            correct_visits=(success_col, "sum"),
            all_visits=("VisitID", "size"),
        )
        .reset_index()
    )
    grouped["value"] = grouped["correct_visits"] / grouped["all_visits"]

    mice = phase_visits.loc[:, ["Group", "ET", "ETLabel", "SEX"]].drop_duplicates().reset_index(drop=True)
    segment_rows = []
    for day in range(1, max_days + 1):
        segment_rows.append(
            {
                "segment_day": day,
                "segment_name": "awake",
                "segment_order": (day - 1) * 2 + 1,
                "segment_label": f"Day {day} awake",
            }
        )
        segment_rows.append(
            {
                "segment_day": day,
                "segment_name": "sleep",
                "segment_order": (day - 1) * 2 + 2,
                "segment_label": f"Day {day} sleep",
            }
        )
    segment_frame = pd.DataFrame(segment_rows)
    mice["__key"] = 1
    segment_frame["__key"] = 1
    full_index = mice.merge(segment_frame, on="__key", how="outer").drop(columns="__key")
    mouse_table = full_index.merge(
        grouped,
        on=["Group", "ET", "ETLabel", "SEX", "segment_day", "segment_name", "segment_order", "segment_label"],
        how="left",
        validate="one_to_one",
    )
    mouse_table["all_visits"] = mouse_table["all_visits"].fillna(0.0)
    mouse_table["correct_visits"] = mouse_table["correct_visits"].fillna(0.0)
    mouse_table["value"] = mouse_table["value"].where(mouse_table["all_visits"].gt(0), np.nan)

    summary = (
        mouse_table.groupby(["Group", "segment_order", "segment_day", "segment_name", "segment_label"], observed=True)
        .agg(
            mean_value=("value", "mean"),
            median_value=("value", "median"),
            std_value=("value", "std"),
            mouse_n=("ET", "nunique"),
            contributing_mouse_n=("value", lambda values: int(values.notna().sum())),
            mean_correct_visits=("correct_visits", "mean"),
            mean_all_visits=("all_visits", "mean"),
        )
        .reset_index()
    )
    summary["std_value"] = summary["std_value"].fillna(0.0)
    summary["sem_value"] = summary["std_value"] / np.sqrt(summary["mouse_n"].clip(lower=1))
    return mouse_table, summary

def compute_awake_day_rate_tables(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    success_col: str,
    origin_clock_hour: float,
    awake_start_clock_hour: float,
    awake_end_clock_hour: float,
    max_days: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-mouse awake-only daily rates for violin and day-wise analyses."""

    mouse_table, summary = compute_phase_segment_rate_tables(
        visits,
        phase_number=phase_number,
        success_col=success_col,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        max_days=max_days,
    )
    mouse_table = mouse_table.loc[mouse_table["segment_name"].eq("awake")].copy()
    summary = summary.loc[summary["segment_name"].eq("awake")].copy()
    mouse_table = mouse_table.rename(
        columns={
            "segment_day": "phase_day",
            "segment_name": "segment",
            "segment_order": "segment_order",
            "segment_label": "segment_label",
        }
    )
    summary = summary.rename(
        columns={
            "segment_day": "phase_day",
            "segment_name": "segment",
            "segment_order": "segment_order",
            "segment_label": "segment_label",
        }
    )
    return mouse_table, summary

def compute_awake_day_ratio_tables(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    numerator_col: str,
    denominator_col: str,
    origin_clock_hour: float,
    awake_start_clock_hour: float,
    awake_end_clock_hour: float,
    max_days: int = 3,
    pseudocount: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute awake-only daily ratios from two visit-level indicator columns."""

    phase_visits = _phase_segment_table(
        visits,
        phase_number=phase_number,
        origin_clock_hour=origin_clock_hour,
        awake_start_clock_hour=awake_start_clock_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )
    phase_visits = phase_visits.loc[
        phase_visits["segment_day"].between(1, max_days) & phase_visits["segment_name"].eq("awake")
    ].copy()
    if phase_visits.empty:
        return _empty_count_tables()

    grouped = (
        phase_visits.groupby(
            ["Group", "ET", "ETLabel", "SEX", "segment_day", "segment_name", "segment_order", "segment_label"],
            observed=True,
        )
        .agg(
            numerator_count=(numerator_col, "sum"),
            denominator_count=(denominator_col, "sum"),
        )
        .reset_index()
    )
    grouped["value"] = (grouped["numerator_count"] + float(pseudocount)) / (
        grouped["denominator_count"] + float(pseudocount)
    )
    if pseudocount == 0.0:
        grouped["value"] = grouped["value"].where(grouped["denominator_count"].gt(0), np.nan)

    mice = phase_visits.loc[:, ["Group", "ET", "ETLabel", "SEX"]].drop_duplicates().reset_index(drop=True)
    day_frame = pd.DataFrame(
        [
            {
                "segment_day": day,
                "segment_name": "awake",
                "segment_order": (day - 1) * 2 + 1,
                "segment_label": f"Day {day} awake",
            }
            for day in range(1, max_days + 1)
        ]
    )
    mice["__key"] = 1
    day_frame["__key"] = 1
    full_index = mice.merge(day_frame, on="__key", how="outer").drop(columns="__key")
    mouse_table = full_index.merge(
        grouped,
        on=["Group", "ET", "ETLabel", "SEX", "segment_day", "segment_name", "segment_order", "segment_label"],
        how="left",
        validate="one_to_one",
    )
    mouse_table["numerator_count"] = mouse_table["numerator_count"].fillna(0.0)
    mouse_table["denominator_count"] = mouse_table["denominator_count"].fillna(0.0)
    if pseudocount == 0.0:
        mouse_table["value"] = mouse_table["value"].where(mouse_table["denominator_count"].gt(0), np.nan)
    else:
        mouse_table["value"] = (
            mouse_table["numerator_count"] + float(pseudocount)
        ) / (mouse_table["denominator_count"] + float(pseudocount))
    mouse_table = mouse_table.rename(
        columns={
            "segment_day": "phase_day",
            "segment_name": "segment",
        }
    )

    summary = (
        mouse_table.groupby(["Group", "phase_day", "segment", "segment_order", "segment_label"], observed=True)
        .agg(
            mean_value=("value", "mean"),
            median_value=("value", "median"),
            std_value=("value", "std"),
            mouse_n=("ET", "nunique"),
            contributing_mouse_n=("value", lambda values: int(values.notna().sum())),
            mean_numerator_count=("numerator_count", "mean"),
            mean_denominator_count=("denominator_count", "mean"),
        )
        .reset_index()
    )
    summary["std_value"] = summary["std_value"].fillna(0.0)
    summary["sem_value"] = summary["std_value"] / np.sqrt(summary["mouse_n"].clip(lower=1))
    return mouse_table, summary

def compute_first_hours_rate_table(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    success_col: str,
    first_hours: float = 24.0,
) -> pd.DataFrame:
    """Aggregate per-mouse success counts and rates within the first phase hours.

    The resulting table is intended for early-learning analyses that use the
    natural binomial structure of the IntelliCage data instead of only the
    derived percentage values.
    """

    phase_visits = visits.loc[
        visits["AnalysisPhaseNumber"].eq(phase_number)
        & visits["analysis_phase_elapsed_hours"].ge(0.0)
        & visits["analysis_phase_elapsed_hours"].lt(float(first_hours))
    ].copy()
    if phase_visits.empty:
        return pd.DataFrame()

    grouped = (
        phase_visits.groupby(["Group", "ET", "ETLabel", "SEX"], observed=True)
        .agg(
            correct_visits=(success_col, "sum"),
            all_visits=("VisitID", "size"),
        )
        .reset_index()
    )
    grouped["value"] = np.where(
        grouped["all_visits"].gt(0),
        grouped["correct_visits"] / grouped["all_visits"],
        np.nan,
    )
    grouped["PhaseNumber"] = int(phase_number)
    grouped["first_hours"] = float(first_hours)
    return grouped

def compute_threshold_responder_table(
    onset_table: pd.DataFrame,
    *,
    phase_number: int,
    threshold_pct: float,
    horizons_hours: tuple[float, ...] = (24.0, 48.0, 72.0),
) -> pd.DataFrame:
    """Convert threshold-onset times into responder yes/no labels by horizon."""

    if onset_table.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in onset_table.iterrows():
        onset = row.get("onset_hours", np.nan)
        for horizon_hours in horizons_hours:
            reached = bool(pd.notna(onset) and float(onset) <= float(horizon_hours))
            rows.append(
                {
                    "PhaseNumber": int(phase_number),
                    "threshold_pct": float(threshold_pct),
                    "horizon_hours": float(horizon_hours),
                    "Group": row["Group"],
                    "ET": row["ET"],
                    "ETLabel": row["ETLabel"],
                    "SEX": row["SEX"],
                    "onset_hours": onset,
                    "reached_criterion": reached,
                }
            )
    return pd.DataFrame(rows)

def compute_responder_group_statistics(
    responder_table: pd.DataFrame,
    *,
    phase_number: int,
    metric_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute responder summaries and group-comparison statistics.

    Omnibus testing uses Fisher's exact test for two groups and chi-square for
    more than two groups. Pairwise comparisons use Fisher's exact test with
    FDR-BH correction.
    """

    if responder_table.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary = (
        responder_table.groupby(["PhaseNumber", "threshold_pct", "horizon_hours", "Group"], observed=True)
        .agg(
            responder_n=("reached_criterion", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
            mouse_n=("ET", "nunique"),
        )
        .reset_index()
    )
    summary["non_responder_n"] = summary["mouse_n"] - summary["responder_n"]
    summary["responder_rate"] = np.where(
        summary["mouse_n"].gt(0),
        summary["responder_n"] / summary["mouse_n"],
        np.nan,
    )

    omnibus_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    for (threshold_pct, horizon_hours), subset in summary.groupby(["threshold_pct", "horizon_hours"], observed=True):
        groups = [str(group_name) for group_name in subset["Group"].astype(str)]
        if len(groups) < 2:
            continue
        contingency = subset.loc[:, ["responder_n", "non_responder_n"]].to_numpy(dtype=int)
        if len(groups) == 2:
            _, omnibus_p = stats.fisher_exact(contingency)
            omnibus_test = "fisher_exact"
        else:
            try:
                _, omnibus_p, _, _ = stats.chi2_contingency(contingency)
                omnibus_test = "chi2"
            except ValueError:
                # Sparse responder tables can contain structural zeros. In that
                # case we apply a small Haldane-Anscombe continuity correction
                # so the omnibus comparison remains computable.
                _, omnibus_p, _, _ = stats.chi2_contingency(contingency + 0.5)
                omnibus_test = "chi2_haldane"
        omnibus_rows.append(
            {
                "PhaseNumber": int(phase_number),
                "Metric": metric_name,
                "threshold_pct": float(threshold_pct),
                "horizon_hours": float(horizon_hours),
                "test": omnibus_test,
                "p_value": float(omnibus_p),
                "group_n": len(groups),
            }
        )

        raw_ps: list[float] = []
        pair_rows: list[tuple[str, str]] = []
        for left_index, left_group in enumerate(groups):
            left_row = subset.loc[subset["Group"].astype(str).eq(left_group)].iloc[0]
            for right_group in groups[left_index + 1 :]:
                right_row = subset.loc[subset["Group"].astype(str).eq(right_group)].iloc[0]
                table = np.array(
                    [
                        [int(left_row["responder_n"]), int(left_row["non_responder_n"])],
                        [int(right_row["responder_n"]), int(right_row["non_responder_n"])],
                    ]
                )
                raw_ps.append(float(stats.fisher_exact(table)[1]))
                pair_rows.append((left_group, right_group))
        adjusted_ps = _fdr_bh_adjust(raw_ps)
        for (left_group, right_group), p_value in zip(pair_rows, adjusted_ps):
            pairwise_rows.append(
                {
                    "PhaseNumber": int(phase_number),
                    "Metric": metric_name,
                    "threshold_pct": float(threshold_pct),
                    "horizon_hours": float(horizon_hours),
                    "test": "fisher_exact_fdr_bh",
                    "group1": left_group,
                    "group2": right_group,
                    "p_value": float(p_value),
                }
            )

    return summary, pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)

def compute_binomial_glm_group_statistics(
    data: pd.DataFrame,
    *,
    phase_number: int,
    metric_name: str,
    success_col: str,
    total_col: str,
    subset_col: str | None = None,
    subset_label: str = "subset",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit count-based binomial GLMs and derive omnibus/pairwise group tests.

    The model uses mouse-level success counts with the corresponding visit
    totals as binomial weights. If available and non-degenerate, `SEX` is
    included as a covariate. The function fits separate models per subset, for
    example `phase_day` or `first_hours`.
    """

    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = data.copy()
    work = work.loc[work[total_col].fillna(0).gt(0)].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work["prop"] = work[success_col] / work[total_col]

    subset_values = [("all", work)] if subset_col is None else [
        (subset_value, subset_frame.copy())
        for subset_value, subset_frame in work.groupby(subset_col, observed=True)
    ]

    omnibus_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for subset_value, subset_frame in subset_values:
        subset_frame = subset_frame.copy()
        group_order = [str(group_name) for group_name in subset_frame["Group"].astype(str).dropna().unique()]
        categories = getattr(subset_frame["Group"].dtype, "categories", None)
        if categories is not None:
            group_order = [str(group_name) for group_name in categories if str(group_name) in group_order]
        if len(group_order) < 2:
            continue
        reference_group = group_order[0]
        subset_frame["Group"] = pd.Categorical(subset_frame["Group"].astype(str), categories=group_order, ordered=True)

        formula = f"prop ~ C(Group, Treatment(reference='{reference_group}'))"
        if "SEX" in subset_frame.columns and subset_frame["SEX"].dropna().astype(str).nunique() > 1:
            formula += " + C(SEX)"

        try:
            model = smf.glm(
                formula=formula,
                data=subset_frame,
                family=sm.families.Binomial(),
                freq_weights=subset_frame[total_col].astype(float),
            )
            result = model.fit()
        except Exception as exc:
            omnibus_rows.append(
                {
                    "PhaseNumber": int(phase_number),
                    "Metric": metric_name,
                    subset_label: subset_value,
                    "test": "binomial_glm",
                    "p_value": np.nan,
                    "group_n": len(group_order),
                    "fit_failed": True,
                    "error": str(exc),
                }
            )
            continue

        param_names = list(result.params.index)
        group_param_names = [name for name in param_names if name.startswith("C(Group, Treatment(")]
        if group_param_names:
            restriction = np.zeros((len(group_param_names), len(param_names)), dtype=float)
            for row_index, param_name in enumerate(group_param_names):
                restriction[row_index, param_names.index(param_name)] = 1.0
            omnibus_p = float(np.asarray(result.wald_test(restriction).pvalue).reshape(-1)[0])
        else:
            omnibus_p = np.nan

        omnibus_rows.append(
            {
                "PhaseNumber": int(phase_number),
                "Metric": metric_name,
                subset_label: subset_value,
                "test": "binomial_glm_wald",
                "p_value": omnibus_p,
                "group_n": len(group_order),
                "fit_failed": False,
                "n_obs": int(len(subset_frame)),
            }
        )

        raw_ps: list[float] = []
        effect_rows: list[dict[str, object]] = []
        for left_index, left_group in enumerate(group_order):
            for right_group in group_order[left_index + 1 :]:
                contrast = np.zeros(len(param_names), dtype=float)
                if left_group != reference_group:
                    left_name = f"C(Group, Treatment(reference='{reference_group}'))[T.{left_group}]"
                    if left_name in param_names:
                        contrast[param_names.index(left_name)] = -1.0
                if right_group != reference_group:
                    right_name = f"C(Group, Treatment(reference='{reference_group}'))[T.{right_group}]"
                    if right_name in param_names:
                        contrast[param_names.index(right_name)] = 1.0
                test_result = result.t_test(contrast)
                raw_p = float(np.asarray(test_result.pvalue).reshape(-1)[0])
                effect = float(np.asarray(test_result.effect).reshape(-1)[0])
                raw_ps.append(raw_p)
                effect_rows.append(
                    {
                        "PhaseNumber": int(phase_number),
                        "Metric": metric_name,
                        subset_label: subset_value,
                        "test": "binomial_glm_pairwise",
                        "group1": left_group,
                        "group2": right_group,
                        "logit_difference": effect,
                        "odds_ratio": float(np.exp(effect)),
                    }
                )
        adjusted_ps = _fdr_bh_adjust(raw_ps)
        for row, p_value in zip(effect_rows, adjusted_ps):
            row["p_value"] = float(p_value)
            pairwise_rows.append(row)

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)

def compute_clustered_binomial_gee_group_statistics(
    data: pd.DataFrame,
    *,
    phase_number: int,
    metric_name: str,
    success_col: str,
    total_col: str,
    cluster_col: str = "ET",
    subset_col: str | None = None,
    subset_label: str = "subset",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit mouse-clustered binomial GEEs and derive omnibus/pairwise tests.

    Compared with the simpler weighted GLM, this approach retains repeated
    within-mouse observations, for example 1 h bins, and uses a clustered
    sandwich covariance via GEE. This yields more conservative standard errors
    when mice contribute many bins with correlated behavior.
    """

    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = data.copy()
    work = work.loc[work[total_col].fillna(0).gt(0)].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work["prop"] = work[success_col] / work[total_col]

    subset_values = [("all", work)] if subset_col is None else [
        (subset_value, subset_frame.copy())
        for subset_value, subset_frame in work.groupby(subset_col, observed=True)
    ]

    omnibus_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for subset_value, subset_frame in subset_values:
        subset_frame = subset_frame.copy()
        group_order = [str(group_name) for group_name in subset_frame["Group"].astype(str).dropna().unique()]
        categories = getattr(subset_frame["Group"].dtype, "categories", None)
        if categories is not None:
            group_order = [str(group_name) for group_name in categories if str(group_name) in group_order]
        if len(group_order) < 2:
            continue

        reference_group = group_order[0]
        subset_frame["Group"] = pd.Categorical(
            subset_frame["Group"].astype(str),
            categories=group_order,
            ordered=True,
        )

        formula = f"prop ~ C(Group, Treatment(reference='{reference_group}'))"
        if "SEX" in subset_frame.columns and subset_frame["SEX"].dropna().astype(str).nunique() > 1:
            formula += " + C(SEX)"

        try:
            model = smf.gee(
                formula=formula,
                groups=cluster_col,
                data=subset_frame,
                family=sm.families.Binomial(),
                weights=subset_frame[total_col].astype(float),
                cov_struct=sm.cov_struct.Exchangeable(),
            )
            result = model.fit()
        except Exception as exc:
            omnibus_rows.append(
                {
                    "PhaseNumber": int(phase_number),
                    "Metric": metric_name,
                    subset_label: subset_value,
                    "test": "binomial_gee",
                    "p_value": np.nan,
                    "group_n": len(group_order),
                    "fit_failed": True,
                    "error": str(exc),
                }
            )
            continue

        param_names = list(result.params.index)
        group_param_names = [name for name in param_names if name.startswith("C(Group, Treatment(")]
        if group_param_names:
            restriction = np.zeros((len(group_param_names), len(param_names)), dtype=float)
            for row_index, param_name in enumerate(group_param_names):
                restriction[row_index, param_names.index(param_name)] = 1.0
            omnibus_p = float(np.asarray(result.wald_test(restriction).pvalue).reshape(-1)[0])
        else:
            omnibus_p = np.nan

        omnibus_rows.append(
            {
                "PhaseNumber": int(phase_number),
                "Metric": metric_name,
                subset_label: subset_value,
                "test": "binomial_gee_wald",
                "p_value": omnibus_p,
                "group_n": len(group_order),
                "fit_failed": False,
                "n_obs": int(len(subset_frame)),
                "cluster_n": int(subset_frame[cluster_col].astype(str).nunique()),
            }
        )

        raw_ps: list[float] = []
        effect_rows: list[dict[str, object]] = []
        for left_index, left_group in enumerate(group_order):
            for right_group in group_order[left_index + 1 :]:
                contrast = np.zeros(len(param_names), dtype=float)
                if left_group != reference_group:
                    left_name = f"C(Group, Treatment(reference='{reference_group}'))[T.{left_group}]"
                    if left_name in param_names:
                        contrast[param_names.index(left_name)] = -1.0
                if right_group != reference_group:
                    right_name = f"C(Group, Treatment(reference='{reference_group}'))[T.{right_group}]"
                    if right_name in param_names:
                        contrast[param_names.index(right_name)] = 1.0
                test_result = result.t_test(contrast)
                raw_p = float(np.asarray(test_result.pvalue).reshape(-1)[0])
                effect = float(np.asarray(test_result.effect).reshape(-1)[0])
                raw_ps.append(raw_p)
                effect_rows.append(
                    {
                        "PhaseNumber": int(phase_number),
                        "Metric": metric_name,
                        subset_label: subset_value,
                        "test": "binomial_gee_pairwise",
                        "group1": left_group,
                        "group2": right_group,
                        "logit_difference": effect,
                        "odds_ratio": float(np.exp(effect)),
                    }
                )

        adjusted_ps = _fdr_bh_adjust(raw_ps)
        for row, p_value in zip(effect_rows, adjusted_ps):
            row["p_value"] = float(p_value)
            pairwise_rows.append(row)

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)

def _fdr_bh_adjust(p_values: list[float]) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction to a sequence of p-values."""

    if not p_values:
        return []
    _, adjusted, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    return adjusted.tolist()

def compute_group_day_violin_statistics(
    mouse_day_rates: pd.DataFrame,
    *,
    phase_number: int,
    metric_name: str,
    chance_level: float,
    exclude_outliers: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute omnibus, pairwise, and chance-level statistics for one phase/day-rate panel set."""

    phase_data = mouse_day_rates.loc[mouse_day_rates["PhaseNumber"].eq(phase_number)].copy()
    if exclude_outliers and "is_outlier" in phase_data.columns:
        phase_data = phase_data.loc[~phase_data["is_outlier"].fillna(False)].copy()
    omnibus_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    chance_rows: list[dict[str, object]] = []
    if phase_data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for phase_day, day_data in phase_data.groupby("phase_day", observed=True):
        grouped = {
            str(group_name): group_frame["value"].dropna().to_numpy(dtype=float)
            for group_name, group_frame in day_data.groupby("Group", observed=True)
        }
        grouped = {key: values for key, values in grouped.items() if len(values) > 0}
        if len(grouped) < 2:
            continue

        shapiro_ok = True
        for values in grouped.values():
            if len(values) < 3:
                shapiro_ok = False
                break
            if len(values) <= 5000 and stats.shapiro(values).pvalue <= 0.05:
                shapiro_ok = False
                break

        levene_p = np.nan
        if all(len(values) >= 2 for values in grouped.values()):
            levene_p = float(stats.levene(*grouped.values()).pvalue)
        normal_path = bool(shapiro_ok and not np.isnan(levene_p) and levene_p > 0.05)

        if normal_path:
            if len(grouped) == 2:
                group_names = list(grouped)
                omnibus_p = float(
                    stats.ttest_ind(
                        grouped[group_names[0]],
                        grouped[group_names[1]],
                        equal_var=bool(levene_p > 0.05) if not np.isnan(levene_p) else True,
                    ).pvalue
                )
                omnibus_test = "ttest_ind"
                pairwise_rows.append(
                    {
                        "PhaseNumber": phase_number,
                        "Metric": metric_name,
                        "phase_day": int(phase_day),
                        "test": "ttest_ind",
                        "group1": group_names[0],
                        "group2": group_names[1],
                        "p_value": float(omnibus_p),
                    }
                )
            else:
                omnibus_p = float(stats.f_oneway(*grouped.values()).pvalue)
                omnibus_test = "anova"
                tukey = pairwise_tukeyhsd(
                    endog=day_data.loc[day_data["value"].notna(), "value"].to_numpy(),
                    groups=day_data.loc[day_data["value"].notna(), "Group"].astype(str).to_numpy(),
                )
                tukey_table = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                tukey_table["group1"] = tukey_table["group1"].astype(str)
                tukey_table["group2"] = tukey_table["group2"].astype(str)
                tukey_table["p_value"] = tukey_table["p-adj"].astype(float)
                for _, row in tukey_table.iterrows():
                    pairwise_rows.append(
                        {
                            "PhaseNumber": phase_number,
                            "Metric": metric_name,
                            "phase_day": int(phase_day),
                            "test": "tukey_hsd",
                            "group1": row["group1"],
                            "group2": row["group2"],
                            "p_value": float(row["p_value"]),
                        }
                    )
        else:
            group_names = list(grouped)
            if len(grouped) == 2:
                omnibus_p = float(
                    stats.mannwhitneyu(
                        grouped[group_names[0]],
                        grouped[group_names[1]],
                        alternative="two-sided",
                    ).pvalue
                )
                omnibus_test = "mannwhitney"
                pairwise_rows.append(
                    {
                        "PhaseNumber": phase_number,
                        "Metric": metric_name,
                        "phase_day": int(phase_day),
                        "test": "mannwhitney",
                        "group1": group_names[0],
                        "group2": group_names[1],
                        "p_value": float(omnibus_p),
                    }
                )
            else:
                omnibus_p = float(stats.kruskal(*grouped.values()).pvalue)
                omnibus_test = "kruskal"
                raw_ps: list[float] = []
                pairs: list[tuple[str, str]] = []
                for left_index, left_group in enumerate(group_names):
                    for right_group in group_names[left_index + 1 :]:
                        raw_ps.append(
                            float(
                                stats.mannwhitneyu(
                                    grouped[left_group],
                                    grouped[right_group],
                                    alternative="two-sided",
                                ).pvalue
                            )
                        )
                        pairs.append((left_group, right_group))
                adjusted = _fdr_bh_adjust(raw_ps)
                for (left_group, right_group), p_value in zip(pairs, adjusted):
                    pairwise_rows.append(
                        {
                            "PhaseNumber": phase_number,
                            "Metric": metric_name,
                            "phase_day": int(phase_day),
                            "test": "mannwhitney_fdr_bh",
                            "group1": left_group,
                            "group2": right_group,
                            "p_value": float(p_value),
                        }
                    )

        omnibus_rows.append(
            {
                "PhaseNumber": phase_number,
                "Metric": metric_name,
                "phase_day": int(phase_day),
                "test": omnibus_test,
                "p_value": float(omnibus_p),
                "group_n": len(grouped),
            }
        )

        if {"correct_visits", "all_visits"}.issubset(day_data.columns):
            for group_name, group_frame in day_data.groupby("Group", observed=True):
                success_count = int(group_frame["correct_visits"].sum())
                total_count = int(group_frame["all_visits"].sum())
                p_value = np.nan
                if total_count > 0:
                    p_value = float(binomtest(success_count, total_count, chance_level, alternative="greater").pvalue)
                chance_rows.append(
                    {
                        "PhaseNumber": phase_number,
                        "Metric": metric_name,
                        "phase_day": int(phase_day),
                        "Group": str(group_name),
                        "success_count": success_count,
                        "total_count": total_count,
                        "chance_level": chance_level,
                        "p_value": p_value,
                    }
                )

    return (
        pd.DataFrame(omnibus_rows),
        pd.DataFrame(pairwise_rows),
        pd.DataFrame(chance_rows),
    )

def compute_role_cumulative_curves(
    visits: pd.DataFrame,
    *,
    pre_phase_hours: float = 24.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute event-based cumulative corner-role visits across late NPA, PL, and PR.

    This version is intentionally not time-binned. Each cumulative step is
    aligned to the actual visit-event times, which keeps the trajectories as
    close as possible to the underlying behavioral sequence.
    """

    phase3_start = (
        visits.loc[visits["AnalysisPhaseNumber"].eq(3), "analysis_phase_start_hours"].dropna().min()
    )
    if pd.isna(phase3_start):
        return pd.DataFrame(), pd.DataFrame()

    phase_visits = visits.copy()
    phase_visits["combined_phase_elapsed_hours"] = phase_visits["analysis_experiment_elapsed_hours"] - float(phase3_start)
    phase_visits = phase_visits.loc[
        phase_visits["AnalysisPhaseNumber"].between(1, 4)
        & phase_visits["combined_phase_elapsed_hours"].le(144.0)
    ].copy()
    if phase_visits.empty:
        return pd.DataFrame(), pd.DataFrame()

    def role_for_row(row: pd.Series) -> str | None:
        all_corners = [1, 2, 3, 4]
        pl_corner = row["CornerPhase3"]
        pr_corner = row["CornerPhase4"]
        if pd.isna(row["Corner"]) or pd.isna(pl_corner) or pd.isna(pr_corner):
            return None
        neutrals = [corner for corner in all_corners if corner not in {int(pl_corner), int(pr_corner)}]
        role_map = {
            int(pl_corner): "PL target corner",
            int(pr_corner): "PR target corner",
            int(neutrals[0]): "Neutral corner 1",
            int(neutrals[1]): "Neutral corner 2",
        }
        return role_map.get(int(row["Corner"]))

    phase_visits["corner_role"] = phase_visits.apply(role_for_row, axis=1)
    phase_visits = phase_visits.loc[phase_visits["corner_role"].notna()].copy()
    visible_phase_visits = phase_visits.loc[
        phase_visits["combined_phase_elapsed_hours"].between(-float(pre_phase_hours), 144.0, inclusive="both")
    ].copy()
    if visible_phase_visits.empty:
        return pd.DataFrame(), pd.DataFrame()
    role_levels = ["PL target corner", "PR target corner", "Neutral corner 1", "Neutral corner 2"]
    mouse_rows: list[dict[str, object]] = []
    max_time = float(visible_phase_visits["combined_phase_elapsed_hours"].max())
    min_time = -float(pre_phase_hours)

    for group_name, group_data in phase_visits.groupby("Group", observed=True):
        visible_group_data = group_data.loc[
            group_data["combined_phase_elapsed_hours"].between(min_time, max_time, inclusive="both")
        ].copy()
        if visible_group_data.empty:
            continue
        timeline = np.unique(
            np.concatenate(
                (
                    np.array([min_time], dtype=float),
                    visible_group_data["combined_phase_elapsed_hours"].to_numpy(dtype=float),
                    np.array([max_time], dtype=float),
                )
            )
        )
        timeline.sort()
        if len(timeline) == 0:
            continue
        end_times = np.append(timeline[1:], timeline[-1])

        for (et, et_label, sex), mouse_data in group_data.groupby(["ET", "ETLabel", "SEX"], observed=True):
            mouse_data = mouse_data.loc[mouse_data["combined_phase_elapsed_hours"].le(max_time)].copy()
            if mouse_data.empty:
                continue
            mouse_data = mouse_data.sort_values("combined_phase_elapsed_hours").copy()
            mouse_data["cumulative_all_visits"] = np.arange(1, len(mouse_data) + 1, dtype=float)
            for role_name in role_levels:
                mouse_data[f"cum__{role_name}"] = mouse_data["corner_role"].eq(role_name).astype(float).cumsum()

            collapsed = (
                mouse_data.groupby("combined_phase_elapsed_hours", observed=True)[
                    ["cumulative_all_visits", *[f"cum__{role_name}" for role_name in role_levels]]
                ]
                .max()
                .sort_index()
            )
            expanded = collapsed.reindex(timeline).ffill().fillna(0.0)
            expanded = expanded.reset_index().rename(columns={"index": "bin_start_hours", "combined_phase_elapsed_hours": "bin_start_hours"})
            expanded["bin_start_hours"] = timeline
            expanded["bin_end_hours"] = end_times
            expanded["bin_center_hours"] = (expanded["bin_start_hours"] + expanded["bin_end_hours"]) / 2.0

            for role_name in role_levels:
                role_values = expanded[f"cum__{role_name}"].to_numpy(dtype=float)
                total_values = expanded["cumulative_all_visits"].to_numpy(dtype=float)
                relative_values = np.full_like(total_values, np.nan, dtype=float)
                np.divide(role_values, total_values, out=relative_values, where=total_values > 0)
                for idx in range(len(expanded)):
                    mouse_rows.append(
                        {
                            "Group": str(group_name),
                            "ET": et,
                            "ETLabel": str(et_label),
                            "SEX": sex,
                            "corner_role": role_name,
                            "bin_start_hours": float(expanded["bin_start_hours"].iloc[idx]),
                            "bin_end_hours": float(expanded["bin_end_hours"].iloc[idx]),
                            "bin_center_hours": float(expanded["bin_center_hours"].iloc[idx]),
                            "cumulative_value": float(role_values[idx]),
                            "cumulative_all_visits": float(total_values[idx]),
                            "relative_cumulative_value": float(relative_values[idx]) if not np.isnan(relative_values[idx]) else np.nan,
                        }
                    )

    mouse_counts = pd.DataFrame(mouse_rows)
    if mouse_counts.empty:
        return pd.DataFrame(), pd.DataFrame()

    absolute_summary = (
        mouse_counts.groupby(["Group", "corner_role", "bin_start_hours", "bin_end_hours", "bin_center_hours"], observed=True)[
            "cumulative_value"
        ]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_value", "median": "median_value", "std": "std_value", "count": "mouse_n"})
    )
    absolute_summary["std_value"] = absolute_summary["std_value"].fillna(0.0)
    absolute_summary["sem_value"] = absolute_summary["std_value"] / np.sqrt(absolute_summary["mouse_n"].clip(lower=1))

    relative_summary = (
        mouse_counts.groupby(["Group", "corner_role", "bin_start_hours", "bin_end_hours", "bin_center_hours"], observed=True)[
            "relative_cumulative_value"
        ]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_value", "median": "median_value", "std": "std_value", "count": "mouse_n"})
    )
    relative_summary["std_value"] = relative_summary["std_value"].fillna(0.0)
    relative_summary["sem_value"] = relative_summary["std_value"] / np.sqrt(relative_summary["mouse_n"].clip(lower=1))
    return mouse_counts, absolute_summary.merge(
        relative_summary,
        on=["Group", "corner_role", "bin_start_hours", "bin_end_hours", "bin_center_hours"],
        how="left",
        suffixes=("_absolute", "_relative"),
    )

def compute_time_window_learning_curves(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    success_col: str,
    window_hours: float = 1.0,
    step_hours: float = 0.5,
    min_visits: int = 5,
    threshold: float = 0.40,
    consecutive_windows: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute sliding-window clock-time learning curves and onset times."""

    phase_visits = visits.loc[visits["AnalysisPhaseNumber"].eq(phase_number)].copy()
    if phase_visits.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    curve_rows: list[dict[str, object]] = []
    onset_rows: list[dict[str, object]] = []
    for (group_name, et, et_label, sex), mouse_data in phase_visits.groupby(
        ["Group", "ET", "ETLabel", "SEX"], observed=True
    ):
        mouse_data = mouse_data.sort_values("analysis_phase_elapsed_hours").copy()
        max_hour = float(mouse_data["analysis_phase_elapsed_hours"].max())
        if max_hour < window_hours:
            continue
        window_starts = np.arange(0.0, max_hour - window_hours + step_hours, step_hours)
        probabilities: list[float] = []
        for start_hour in window_starts:
            mask = mouse_data["analysis_phase_elapsed_hours"].between(start_hour, start_hour + window_hours, inclusive="left")
            window = mouse_data.loc[mask]
            total_visits = int(len(window))
            success_count = int(window[success_col].sum())
            probability = np.nan
            if total_visits >= min_visits:
                probability = success_count / total_visits
            probabilities.append(probability)
            curve_rows.append(
                {
                    "Group": str(group_name),
                    "ET": et,
                    "ETLabel": str(et_label),
                    "SEX": sex,
                    "window_start_hours": float(start_hour),
                    "window_end_hours": float(start_hour + window_hours),
                    "window_center_hours": float(start_hour + window_hours / 2.0),
                    "all_visits": total_visits,
                    "correct_visits": success_count,
                    "value": probability,
                }
            )
        onset_hour = np.nan
        for index in range(0, max(0, len(probabilities) - consecutive_windows + 1)):
            candidate = probabilities[index : index + consecutive_windows]
            if all(not np.isnan(value) and value > threshold for value in candidate):
                onset_hour = float(window_starts[index])
                break
        onset_rows.append(
            {
                "Group": str(group_name),
                "ET": et,
                "ETLabel": str(et_label),
                "SEX": sex,
                "onset_hours": onset_hour,
            }
        )

    curve_table = pd.DataFrame(curve_rows)
    onset_table = pd.DataFrame(onset_rows)
    if curve_table.empty:
        return curve_table, pd.DataFrame(), onset_table
    summary = (
        curve_table.groupby(["Group", "window_start_hours", "window_end_hours", "window_center_hours"], observed=True)["value"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_value", "median": "median_value", "std": "std_value", "count": "mouse_n"})
    )
    summary["std_value"] = summary["std_value"].fillna(0.0)
    summary["sem_value"] = summary["std_value"] / np.sqrt(summary["mouse_n"].clip(lower=1))
    return curve_table, summary, onset_table

def compute_visit_window_learning_curves(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    success_col: str,
    window_visits: int = 20,
    min_visits: int = 20,
    threshold: float = 0.40,
    consecutive_windows: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute rolling learning curves over visit number and the resulting onset by experience."""

    phase_visits = visits.loc[visits["AnalysisPhaseNumber"].eq(phase_number)].copy()
    if phase_visits.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    curve_rows: list[dict[str, object]] = []
    onset_rows: list[dict[str, object]] = []
    for (group_name, et, et_label, sex), mouse_data in phase_visits.groupby(
        ["Group", "ET", "ETLabel", "SEX"], observed=True
    ):
        mouse_data = mouse_data.sort_values("Start").copy().reset_index(drop=True)
        successes = mouse_data[success_col].astype(int).to_numpy(dtype=int)
        probabilities: list[float] = []
        centers: list[float] = []
        for start_index in range(0, max(0, len(mouse_data) - window_visits + 1)):
            window = successes[start_index : start_index + window_visits]
            total_visits = len(window)
            success_count = int(window.sum())
            probability = np.nan
            if total_visits >= min_visits:
                probability = success_count / total_visits
            center_visit = start_index + window_visits / 2.0
            probabilities.append(probability)
            centers.append(center_visit)
            curve_rows.append(
                {
                    "Group": str(group_name),
                    "ET": et,
                    "ETLabel": str(et_label),
                    "SEX": sex,
                    "window_start_visit": int(start_index + 1),
                    "window_end_visit": int(start_index + window_visits),
                    "window_center_visit": float(center_visit),
                    "all_visits": total_visits,
                    "correct_visits": success_count,
                    "value": probability,
                }
            )
        onset_visit = np.nan
        for index in range(0, max(0, len(probabilities) - consecutive_windows + 1)):
            candidate = probabilities[index : index + consecutive_windows]
            if all(not np.isnan(value) and value > threshold for value in candidate):
                onset_visit = float(centers[index])
                break
        onset_rows.append(
            {
                "Group": str(group_name),
                "ET": et,
                "ETLabel": str(et_label),
                "SEX": sex,
                "onset_visit": onset_visit,
            }
        )

    curve_table = pd.DataFrame(curve_rows)
    onset_table = pd.DataFrame(onset_rows)
    if curve_table.empty:
        return curve_table, pd.DataFrame(), onset_table
    summary = (
        curve_table.groupby(["Group", "window_start_visit", "window_end_visit", "window_center_visit"], observed=True)["value"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_value", "median": "median_value", "std": "std_value", "count": "mouse_n"})
    )
    summary["std_value"] = summary["std_value"].fillna(0.0)
    summary["sem_value"] = summary["std_value"] / np.sqrt(summary["mouse_n"].clip(lower=1))
    return curve_table, summary, onset_table

def compute_onset_group_statistics(
    onset_table: pd.DataFrame,
    *,
    onset_col: str,
    phase_number: int,
    metric_name: str,
    exclude_outliers: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute omnibus and pairwise group statistics for one onset metric."""

    data = onset_table.loc[onset_table[onset_col].notna()].copy()
    if exclude_outliers and "is_outlier" in data.columns:
        data = data.loc[~data["is_outlier"].fillna(False)].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    grouped = {
        str(group_name): group_frame[onset_col].to_numpy(dtype=float)
        for group_name, group_frame in data.groupby("Group", observed=True)
    }
    grouped = {key: values for key, values in grouped.items() if len(values) > 0}
    if len(grouped) < 2:
        return pd.DataFrame(), pd.DataFrame()

    shapiro_ok = True
    for values in grouped.values():
        if len(values) < 3:
            shapiro_ok = False
            break
        if len(values) <= 5000 and stats.shapiro(values).pvalue <= 0.05:
            shapiro_ok = False
            break

    levene_p = np.nan
    if all(len(values) >= 2 for values in grouped.values()):
        levene_p = float(stats.levene(*grouped.values()).pvalue)
    normal_path = bool(shapiro_ok and not np.isnan(levene_p) and levene_p > 0.05)

    pairwise_rows: list[dict[str, object]] = []
    if normal_path:
        if len(grouped) == 2:
            group_names = list(grouped)
            omnibus_p = float(
                stats.ttest_ind(
                    grouped[group_names[0]],
                    grouped[group_names[1]],
                    equal_var=bool(levene_p > 0.05) if not np.isnan(levene_p) else True,
                ).pvalue
            )
            omnibus_test = "ttest_ind"
            pairwise_rows.append(
                {
                    "PhaseNumber": phase_number,
                    "Metric": metric_name,
                    "OnsetColumn": onset_col,
                    "test": "ttest_ind",
                    "group1": group_names[0],
                    "group2": group_names[1],
                    "p_value": float(omnibus_p),
                }
            )
        else:
            omnibus_p = float(stats.f_oneway(*grouped.values()).pvalue)
            omnibus_test = "anova"
            tukey = pairwise_tukeyhsd(
                endog=data[onset_col].to_numpy(dtype=float),
                groups=data["Group"].astype(str).to_numpy(),
            )
            tukey_table = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            tukey_table["group1"] = tukey_table["group1"].astype(str)
            tukey_table["group2"] = tukey_table["group2"].astype(str)
            tukey_table["p_value"] = tukey_table["p-adj"].astype(float)
            for _, row in tukey_table.iterrows():
                pairwise_rows.append(
                    {
                        "PhaseNumber": phase_number,
                        "Metric": metric_name,
                        "OnsetColumn": onset_col,
                        "test": "tukey_hsd",
                        "group1": row["group1"],
                        "group2": row["group2"],
                        "p_value": float(row["p_value"]),
                    }
                )
    else:
        group_names = list(grouped)
        if len(grouped) == 2:
            omnibus_p = float(
                stats.mannwhitneyu(
                    grouped[group_names[0]],
                    grouped[group_names[1]],
                    alternative="two-sided",
                ).pvalue
            )
            omnibus_test = "mannwhitney"
            pairwise_rows.append(
                {
                    "PhaseNumber": phase_number,
                    "Metric": metric_name,
                    "OnsetColumn": onset_col,
                    "test": "mannwhitney",
                    "group1": group_names[0],
                    "group2": group_names[1],
                    "p_value": float(omnibus_p),
                }
            )
        else:
            omnibus_p = float(stats.kruskal(*grouped.values()).pvalue)
            omnibus_test = "kruskal"
            raw_ps: list[float] = []
            pairs: list[tuple[str, str]] = []
            for left_index, left_group in enumerate(group_names):
                for right_group in group_names[left_index + 1 :]:
                    raw_ps.append(
                        float(
                            stats.mannwhitneyu(
                                grouped[left_group],
                                grouped[right_group],
                                alternative="two-sided",
                            ).pvalue
                        )
                    )
                    pairs.append((left_group, right_group))
            adjusted = _fdr_bh_adjust(raw_ps)
            for (left_group, right_group), p_value in zip(pairs, adjusted):
                pairwise_rows.append(
                    {
                        "PhaseNumber": phase_number,
                        "Metric": metric_name,
                        "OnsetColumn": onset_col,
                        "test": "mannwhitney_fdr_bh",
                        "group1": left_group,
                        "group2": right_group,
                        "p_value": float(p_value),
                    }
                )

    omnibus = pd.DataFrame(
        [
            {
                "PhaseNumber": phase_number,
                "Metric": metric_name,
                "OnsetColumn": onset_col,
                "test": omnibus_test,
                "p_value": float(omnibus_p),
                "group_n": len(grouped),
            }
        ]
    )
    return omnibus, pd.DataFrame(pairwise_rows)

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
    phase_visits = phase_visits.loc[phase_visits["AnalysisPhaseNumber"].between(1, 4)].copy()
    phase_visits["hour_bin_start"] = np.floor(
        phase_visits["analysis_phase_elapsed_hours"] / float(hourly_bin_size)
    )
    phase_visits["hour_bin_start"] = phase_visits["hour_bin_start"] * float(hourly_bin_size)

    counts = (
        phase_visits.groupby(
            ["RunGroup", "AnalysisPhaseNumber", "Group", "ET", "ETLabel", "hour_bin_start"],
            observed=True,
        )["VisitID"]
        .size()
        .reset_index(name="visits_in_hour")
    )

    phase_duration_hours = (
        phase_visits.groupby(["RunGroup", "AnalysisPhaseNumber"], observed=True)["analysis_phase_elapsed_hours"]
        .max()
        .reset_index(name="phase_max_hour")
    )
    mice = phase_visits.loc[:, ["RunGroup", "AnalysisPhaseNumber", "Group", "ET", "ETLabel"]].drop_duplicates()
    complete_rows: list[pd.DataFrame] = []
    for _, duration_row in phase_duration_hours.iterrows():
        run_group = duration_row["RunGroup"]
        phase_number = duration_row["AnalysisPhaseNumber"]
        max_hour = int(np.floor(float(duration_row["phase_max_hour"]) / float(hourly_bin_size)) * hourly_bin_size)
        phase_mice = mice.loc[
            mice["RunGroup"].eq(run_group) & mice["AnalysisPhaseNumber"].eq(phase_number)
        ].copy()
        hour_grid = pd.DataFrame({"hour_bin_start": np.arange(0, max_hour + hourly_bin_size, hourly_bin_size)})
        phase_mice["__key"] = 1
        hour_grid["__key"] = 1
        complete_rows.append(phase_mice.merge(hour_grid, on="__key", how="outer").drop(columns="__key"))
    complete_index = pd.concat(complete_rows, ignore_index=True)
    complete_counts = complete_index.merge(
        counts,
        on=["RunGroup", "AnalysisPhaseNumber", "Group", "ET", "ETLabel", "hour_bin_start"],
        how="left",
        validate="one_to_one",
    )
    complete_counts["visits_in_hour"] = complete_counts["visits_in_hour"].fillna(0.0)

    mouse_medians = (
        complete_counts.groupby(["Group", "ET", "ETLabel", "AnalysisPhaseNumber"], observed=True)["visits_in_hour"]
        .median()
        .reset_index(name="median_visits_per_hour")
        .rename(columns={"AnalysisPhaseNumber": "PhaseNumber"})
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
# %% END
