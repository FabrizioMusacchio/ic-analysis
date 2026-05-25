"""Run the IntelliCage workflow for the BioMedX 10-month cohort.

This user-facing script mirrors the 4-month analysis pipeline but adapts the
defaults to the 10-month cohort:

- seven IntelliCage run folders, including WT sex-specific runs
- mouse day starts at 07:00 and the awake period ends at 19:00
- protocol phase timing follows the protocol-aligned 10-month schedule
- experiment day counting follows the mouse-day definition, so day 0 begins
  at 07:00 and captures any pre-day-1 placement interval
- an additional sugar-preference analysis is performed on SP day 2 only

The phase 1-4 place-learning analysis is intentionally kept separate from the
SP analysis so the additional sugar-preference folders do not alter the
overview and learning plots of the main experiment.
"""
# %% IMPORTS
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intellicage_place_learning.loader import load_cohort_data
from intellicage_place_learning.metrics import compute_onset_group_statistics, flag_iqr_outliers
from intellicage_place_learning.plotting import (
    configure_plot_style,
    plot_onset_violin,
    set_group_colors,
    set_figure_size_presets,
)
from user_scripts.analyze_4month_cohort import (
    DEFAULT_GROUP_COLORS,
    apply_group_preferences,
    resolved_group_colors,
    run_analysis,
    save_table,
)

# %% PARAMETERS AND DEFAULTS
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Data IntelliCage" / "BioMedX_10MonthCohort_2019"
DEFAULT_RESULTS_SUBDIR = Path("results")
DEFAULT_PHASE_DISPLAY_NAMES = {
    1: "Free Hab",
    2: "NPA",
    3: "PL",
    4: "PR",
}
DEFAULT_PHASE_NAME_MAP = {
    "Phase1": 1,
    "Phase2": 2,
    "Phase3": 3,
    "Phase4": 4,
}
DEFAULT_PHASE_NAME_MAP_WITH_SP = {
    "Phase1": 1,
    "Phase2": 2,
    "Phase3": 3,
    "Phase4": 4,
    "SP1": 5,
    "SP2": 6,
}
DEFAULT_OPTIONAL_PHASE_NAMES_WITH_SP = {"SP2"}
DEFAULT_PHASE_MAX_HOURS = {
    3: 71.0,
    4: 71.0,
}
DEFAULT_FIGSIZE_CM = { # always a pair of (width, height)
    "LONG_FIGSIZE_CM": (18.2, 7.4),
    "LONG_FIGSIZE_2_CM": (15.2, 7.4),
    "PHASE2_FIGSIZE_CM": (10.4, 7.0),
    "MEDIUM_FIGSIZE_CM": (11.8, 7.6),
    "MEDIUM_WIDE_FIGSIZE_CM": (12.8, 8.0),
    "SEGMENT_FIGSIZE_CM": (12.6, 7.9),
    "VIOLIN_FIGSIZE_CM": (5.8, 7.2),
    "ONSET_FIGSIZE_CM": (5.8, 7.0),
    "ACTIVITY_FIGSIZE_CM": (8.8, 8.1),
    "WIDE_GROUP_FIGSIZE_CM": (18.2, 7.4),
}
DEFAULT_SCHEDULED_PHASE_START_HOURS = {
    1: 0.0,
    2: 74.0,
    3: 122.0,
    4: 194.0,
    5: 266.0,
    6: 290.0,
}
DEFAULT_GROUP_RENAMES = {
    "WT":           "WT",
    "WT_female":    "WT female",
    "WT_male":      "WT male",
    "tdTomato":     "tdTomato",
    "Tau 1-441":    "Tau 1-441",
    "Tau 1-421":    "Tau 1-421",
    "Tau 66-421":   "Tau 66-421",
}

USER_DATASET_ROOT = DEFAULT_DATASET_ROOT
USER_RESULTS_SUBDIR = DEFAULT_RESULTS_SUBDIR
USER_BIN_HOURS = [1, 2]
USER_PHASE2_SECONDARY_METRIC = "lick_positive_visits"
USER_SPREAD_METRIC = "sem"
USER_PLOT_STYLE = "line"
USER_PHASE2_PLOT_STYLE = "line"
USER_PHASE_MAX_HOURS = DEFAULT_PHASE_MAX_HOURS.copy()
USER_PHASE_NAME_MAP = DEFAULT_PHASE_NAME_MAP.copy()
USER_PHASE_NAME_MAP_WITH_SP = DEFAULT_PHASE_NAME_MAP_WITH_SP.copy()
USER_OPTIONAL_PHASE_NAMES_WITH_SP = DEFAULT_OPTIONAL_PHASE_NAMES_WITH_SP.copy()
USER_DROP_UNMATCHED_VISITS = True
USER_EXCLUDED_GROUPS: list[str] = ["WT"]
USER_GROUP_RENAMES = DEFAULT_GROUP_RENAMES.copy()
USER_GROUP_COLORS = {
    "WT":        "#264653",
    "tdTomato":  "#6c757d",
    "Tau 1-441": "#4ade80",
    "Tau 1-421": "#e9a820",
    "Tau 66-421":"#2a9d8f",
    "WT female": "#264653",
    "WT male":   "#5194AE",
}
USER_FIGSIZE_CM = DEFAULT_FIGSIZE_CM.copy()
USER_MOUSE_DAY_START_HOUR = 7.0
USER_AWAKE_DURATION_HOURS = 12.0
USER_EXPERIMENT_DAY0_START_HOUR = None
USER_SCHEDULE_ANCHOR_PHASE_NUMBER = 2
USER_SCHEDULED_PHASE_START_HOURS = DEFAULT_SCHEDULED_PHASE_START_HOURS.copy()
USER_BASE_FONT_SIZE = 10.0
USER_EXCLUDE_VIOLIN_OUTLIERS = True
# %% HELPER FUNCTIONS
def compute_sp2_preference_table(
    visits: pd.DataFrame,
    nosepokes: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize sugar-side versus water-side licking for SP day 2.

    The legacy MATLAB script derived sugar preference from `Nosepokes.txt`
    rather than from visit-level totals because the bottle side matters during
    the sugar-preference test. This function follows the same logic:

    - restrict to raw phase number 6 (`SP2`)
    - keep only nose-pokes with a lick response
    - classify the licks by side correctness / side condition
    - compute per-mouse sugar preference ratio as
      `sugar_licks / (sugar_licks + water_licks) * 100`
    """

    sp2_visits = visits.loc[visits["PhaseNumber"].eq(6)].copy()
    if sp2_visits.empty:
        return pd.DataFrame(), pd.DataFrame()

    visit_keys = (
        sp2_visits.loc[:, ["RunGroup", "Phase", "PhaseNumber", "VisitID", "AnimalTag", "Group", "ET", "ETLabel", "SEX"]]
        .drop_duplicates()
        .rename(columns={"AnimalTag": "RFID"})
    )
    sp2_nosepokes = nosepokes.loc[nosepokes["PhaseNumber"].eq(6)].copy()
    if sp2_nosepokes.empty:
        return pd.DataFrame(), pd.DataFrame()

    sp2_nosepokes = sp2_nosepokes.merge(
        visit_keys,
        on=["RunGroup", "Phase", "PhaseNumber", "VisitID"],
        how="inner",
        validate="many_to_one",
    )
    lick_positive = sp2_nosepokes["LickNumber"].fillna(0).gt(0) | sp2_nosepokes["LickDuration"].fillna(0).gt(0)
    sp2_nosepokes = sp2_nosepokes.loc[lick_positive].copy()
    if sp2_nosepokes.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "SideCondition" in sp2_nosepokes.columns:
        sugar_mask = sp2_nosepokes["SideCondition"].eq(1)
        water_mask = sp2_nosepokes["SideCondition"].eq(-1)
    else:
        sugar_mask = sp2_nosepokes["SideError"].eq(0)
        water_mask = sp2_nosepokes["SideError"].eq(1)

    sp2_nosepokes["sugar_licks"] = np.where(sugar_mask, sp2_nosepokes["LickNumber"].fillna(0), 0)
    sp2_nosepokes["water_licks"] = np.where(water_mask, sp2_nosepokes["LickNumber"].fillna(0), 0)
    sp2_nosepokes["sugar_side_lick_event"] = sugar_mask.astype(int)
    sp2_nosepokes["water_side_lick_event"] = water_mask.astype(int)

    mouse_summary = (
        sp2_nosepokes.groupby(["Group", "ET", "ETLabel", "SEX"], observed=True)
        .agg(
            sugar_licks=("sugar_licks", "sum"),
            water_licks=("water_licks", "sum"),
            sugar_side_lick_events=("sugar_side_lick_event", "sum"),
            water_side_lick_events=("water_side_lick_event", "sum"),
            total_nosepoke_rows=("VisitID", "size"),
        )
        .reset_index()
    )
    mouse_summary["total_licks"] = mouse_summary["sugar_licks"] + mouse_summary["water_licks"]
    mouse_summary["sugar_preference_ratio_pct"] = np.where(
        mouse_summary["total_licks"].gt(0),
        mouse_summary["sugar_licks"] / mouse_summary["total_licks"] * 100.0,
        np.nan,
    )

    mouse_index = metadata.loc[:, ["Group", "ET", "ETLabel", "SEX"]].drop_duplicates().reset_index(drop=True)
    mouse_summary = mouse_index.merge(
        mouse_summary,
        on=["Group", "ET", "ETLabel", "SEX"],
        how="left",
        validate="one_to_one",
    )
    fill_zero_cols = [
        "sugar_licks",
        "water_licks",
        "sugar_side_lick_events",
        "water_side_lick_events",
        "total_nosepoke_rows",
        "total_licks",
    ]
    for column in fill_zero_cols:
        mouse_summary[column] = mouse_summary[column].fillna(0.0)
    mouse_summary["has_sp2_data"] = mouse_summary["total_nosepoke_rows"].gt(0)

    group_summary = (
        mouse_summary.groupby("Group", observed=True)
        .agg(
            mean_preference_ratio_pct=("sugar_preference_ratio_pct", "mean"),
            median_preference_ratio_pct=("sugar_preference_ratio_pct", "median"),
            std_preference_ratio_pct=("sugar_preference_ratio_pct", "std"),
            mouse_n=("ET", "nunique"),
            contributing_mouse_n=("sugar_preference_ratio_pct", lambda values: int(values.notna().sum())),
            mean_sugar_licks=("sugar_licks", "mean"),
            mean_water_licks=("water_licks", "mean"),
        )
        .reset_index()
    )
    group_summary["std_preference_ratio_pct"] = group_summary["std_preference_ratio_pct"].fillna(0.0)
    group_summary["sem_preference_ratio_pct"] = (
        group_summary["std_preference_ratio_pct"] / np.sqrt(group_summary["mouse_n"].clip(lower=1))
    )
    return mouse_summary, group_summary

def render_sp2_sugar_preference_analysis(
    *,
    dataset_root: Path,
    output_root: Path,
    excluded_groups: list[str] | None,
    group_renames: dict[str, str] | None,
    group_colors: dict[str, str] | None,
    base_font_size: float,
    exclude_outliers: bool,
) -> None:
    """Load SP1/SP2 data and render the SP day-2 sugar-preference summary."""

    cohort = load_cohort_data(
        dataset_root,
        phase_name_map=USER_PHASE_NAME_MAP_WITH_SP,
        optional_phase_names=USER_OPTIONAL_PHASE_NAMES_WITH_SP,
        drop_unmatched_visits=USER_DROP_UNMATCHED_VISITS,
    )
    selected_visits, selected_metadata, selected_nosepokes = apply_group_preferences(
        cohort.visits,
        cohort.metadata,
        cohort.nosepokes,
        excluded_groups=excluded_groups,
        group_renames=group_renames,
    )

    configure_plot_style(font_size=base_font_size, font_family="Arial")
    set_group_colors(
        resolved_group_colors(
            group_renames=group_renames or {},
            group_colors=group_colors,
        )
    )
    set_figure_size_presets(USER_FIGSIZE_CM)

    mouse_summary, group_summary = compute_sp2_preference_table(
        selected_visits,
        selected_nosepokes,
        selected_metadata,
    )
    if mouse_summary.empty:
        return

    save_table(mouse_summary, output_root / "sp2_sugar_preference_mouse.tsv")
    save_table(group_summary, output_root / "sp2_sugar_preference_group_summary.tsv")

    mouse_with_outliers = flag_iqr_outliers(
        mouse_summary,
        value_col="sugar_preference_ratio_pct",
        group_cols=["Group"],
    )
    save_table(mouse_with_outliers, output_root / "sp2_sugar_preference_mouse_with_outliers.tsv")

    omnibus_stats, pairwise_stats = compute_onset_group_statistics(
        mouse_with_outliers,
        onset_col="sugar_preference_ratio_pct",
        phase_number=6,
        metric_name="sugar_preference_ratio",
        exclude_outliers=exclude_outliers,
    )
    save_table(omnibus_stats, output_root / "sp2_sugar_preference_omnibus_stats.tsv")
    save_table(pairwise_stats, output_root / "sp2_sugar_preference_pairwise_stats.tsv")

    plot_onset_violin(
        mouse_with_outliers,
        onset_col="sugar_preference_ratio_pct",
        phase_display_name="SP day 2",
        title_label="Sugar preference ratio",
        ylabel="Sugar preference ratio [%]",
        output_path=output_root / "sp2_sugar_preference_ratio_violin.png",
        pairwise_stats=pairwise_stats,
        outlier_col="is_outlier",
        reference_line=50.0,
    )

# %% MAIN WORKFLOW
def main() -> None:
    """Run the 10-month place-learning and SP day-2 analyses."""

    output_root = run_analysis(
        dataset_root=USER_DATASET_ROOT,
        results_subdir=USER_RESULTS_SUBDIR,
        bin_hours=USER_BIN_HOURS,
        phase2_secondary_metric=USER_PHASE2_SECONDARY_METRIC,
        spread_metric=USER_SPREAD_METRIC,
        plot_style=USER_PLOT_STYLE,
        phase2_plot_style=USER_PHASE2_PLOT_STYLE,
        phase_max_hours=USER_PHASE_MAX_HOURS,
        phase_name_map=USER_PHASE_NAME_MAP,
        drop_unmatched_visits=USER_DROP_UNMATCHED_VISITS,
        excluded_groups=USER_EXCLUDED_GROUPS,
        group_renames=USER_GROUP_RENAMES,
        group_colors=USER_GROUP_COLORS,
        figure_size_cm=USER_FIGSIZE_CM,
        mouse_day_start_hour=USER_MOUSE_DAY_START_HOUR,
        awake_duration_hours=USER_AWAKE_DURATION_HOURS,
        experiment_day0_start_hour=USER_EXPERIMENT_DAY0_START_HOUR,
        schedule_anchor_phase_number=USER_SCHEDULE_ANCHOR_PHASE_NUMBER,
        scheduled_phase_start_hours=USER_SCHEDULED_PHASE_START_HOURS,
        base_font_size=USER_BASE_FONT_SIZE,
        exclude_violin_outliers=USER_EXCLUDE_VIOLIN_OUTLIERS,
    )

    render_sp2_sugar_preference_analysis(
        dataset_root=USER_DATASET_ROOT,
        output_root=output_root,
        excluded_groups=USER_EXCLUDED_GROUPS,
        group_renames=USER_GROUP_RENAMES,
        group_colors=USER_GROUP_COLORS,
        base_font_size=USER_BASE_FONT_SIZE,
        exclude_outliers=USER_EXCLUDE_VIOLIN_OUTLIERS,
    )

# %% ENTRY POINT
if __name__ == "__main__":
    main()
# %% END
