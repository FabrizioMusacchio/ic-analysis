"""Run the IntelliCage place-learning workflow on the synthetic 
Group A/B data.

author: Fabrizio Musacchio
date: Aug 2026
"""
# %% IMPORTS
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

from ic_placelearning.loader import attach_analysis_time_columns
from ic_placelearning.loader import load_cohort_data
from ic_placelearning.metrics import (
    build_analysis_phase_window_table,
    compute_experiment_visit_bins,
    compute_phase4_reversal_rate_bins,
    compute_place_learning_rate_bins,
    filter_visits_by_phase_limits)
from ic_placelearning.plotting import (
    configure_plot_style,
    plot_experiment_overview_groups,
    plot_phase4_reversal_components,
    plot_phase_learning_rate_groups,
    set_group_colors,
    set_figure_size_presets)
# %% PARAMETERS AND DEFAULTS
USER_DATASET_ROOT = PROJECT_ROOT / "example_data" / "synthetic_group_ab_place_learning"
USER_RESULTS_SUBDIR = Path("results")
USER_BIN_HOURS = [1, 2]
USER_PHASE2_SECONDARY_METRIC = "lick_positive_visits"
USER_SPREAD_METRIC = "sem"
USER_PLOT_STYLE = "line"
USER_PHASE2_PLOT_STYLE = "line"
USER_PHASE_MAX_HOURS = {
    3: 72.0,
    4: 72.0}
USER_EXCLUDED_GROUPS: list[str] = []
USER_GROUP_RENAMES = {
    "Group A": "Group A",
    "Group B": "Group B"}
USER_GROUP_COLORS = {
    "Group A": "#267d8f",
    "Group B": "#c7523f"}
USER_FIGSIZE_CM = {
    "LONG_FIGSIZE_CM": (18.2, 7.4),
    "MEDIUM_FIGSIZE_CM": (11.8, 7.6)}
USER_MOUSE_DAY_START_HOUR = 6.0
USER_AWAKE_DURATION_HOURS = 12.0
USER_SCHEDULED_PHASE_START_HOURS = {
    1: 0.0,
    2: 74.0,
    3: 122.0,
    4: 194.0,
    5: 266.0}
USER_BASE_FONT_SIZE = 10.0
USER_EXCLUDE_VIOLIN_OUTLIERS = True
USER_RATE_THRESHOLD_PCTS = [50.0, 60.0, 70.0, 80.0]
USER_THRESHOLD_ONSET_BIN_HOURS = 1
USER_RESPONDER_HORIZONS_HOURS = [24.0, 48.0, 72.0]
USER_BINOMIAL_MODEL_FIRST_HOURS = 24.0
USER_SUMMARY_RESPONDER_HORIZON_HOURS = 24.0

# %% SOME HELPER FUNCTIONS
def save_table(dataframe, output_path: Path) -> None:
    """Save a DataFrame as a tab-separated text file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, sep="\t", index=False)

def prepare_output_root(dataset_root: Path, results_subdir: Path) -> Path:
    """Create the output directory below the dataset root."""

    if results_subdir.is_absolute():
        raise ValueError("USER_RESULTS_SUBDIR must be relative to USER_DATASET_ROOT.")
    output_root = dataset_root / results_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root

def apply_demo_group_names(dataframe):
    """Rename demo groups while preserving their display order."""

    renamed = dataframe.copy()
    renamed["Group"] = renamed["Group"].astype(str).map(USER_GROUP_RENAMES).fillna(renamed["Group"].astype(str))
    renamed["Group"] = renamed["Group"].astype("category")
    renamed["Group"] = renamed["Group"].cat.set_categories(list(USER_GROUP_RENAMES.values()), ordered=True)
    return renamed
# %% MAIN FUNCTION
def main() -> None:
    """Run a compact synthetic Group A/B demo analysis."""

    configure_plot_style(font_size=USER_BASE_FONT_SIZE, font_family="Arial")
    set_group_colors(USER_GROUP_COLORS)
    set_figure_size_presets(USER_FIGSIZE_CM)

    output_root = prepare_output_root(USER_DATASET_ROOT, USER_RESULTS_SUBDIR)
    cohort = load_cohort_data(
        USER_DATASET_ROOT,
        group_names=list(USER_GROUP_RENAMES.values()))
    visits = attach_analysis_time_columns(
        cohort.visits,
        cohort.phase_manifest,
        scheduled_phase_start_hours=USER_SCHEDULED_PHASE_START_HOURS,
        mouse_day_start_hour=USER_MOUSE_DAY_START_HOUR)
    visits = filter_visits_by_phase_limits(visits, USER_PHASE_MAX_HOURS)
    visits = apply_demo_group_names(visits)
    metadata = apply_demo_group_names(cohort.metadata)

    save_table(metadata, output_root / "mouse_metadata.tsv")
    save_table(cohort.phase_manifest, output_root / "phase_manifest.tsv")
    save_table(visits, output_root / "merged_visits.tsv")

    phase_windows = build_analysis_phase_window_table(
        visits,
        USER_SCHEDULED_PHASE_START_HOURS)
    save_table(phase_windows, output_root / "analysis_phase_windows.tsv")

    visit_mouse_bins, visit_summary_bins = compute_experiment_visit_bins(
        visits,
        bin_hours=2)
    save_table(visit_mouse_bins, output_root / "overview_visits_mouse_bins_2h.tsv")
    save_table(visit_summary_bins, output_root / "overview_visits_group_summary_2h.tsv")
    plot_experiment_overview_groups(
        visit_summary_bins,
        output_path=output_root / "overview_all_phases_visits_all_groups_2h.png",
        phase_window_table=phase_windows,
        phase_display_names={1: "Free Hab", 2: "NPA", 3: "PL", 4: "PR"},
        spread_metric=USER_SPREAD_METRIC,
        plot_style=USER_PLOT_STYLE,
        origin_clock_hour=USER_MOUSE_DAY_START_HOUR,
        awake_start_clock_hour=USER_MOUSE_DAY_START_HOUR,
        awake_end_clock_hour=USER_MOUSE_DAY_START_HOUR + USER_AWAKE_DURATION_HOURS)

    for phase_number in (3, 4):
        mouse_bins, summary_bins = compute_place_learning_rate_bins(
            visits,
            phase_number=phase_number,
            bin_hours=2,
            success_col="rewarded_correct_corner_visit")
        save_table(mouse_bins, output_root / f"phase{phase_number}_rewarded_correct_corner_rate_mouse_bins_2h.tsv")
        save_table(summary_bins, output_root / f"phase{phase_number}_rewarded_correct_corner_rate_summary_2h.tsv")
        plot_phase_learning_rate_groups(
            summary_bins,
            output_path=output_root / f"phase{phase_number}_rewarded_correct_corner_rate_all_groups_2h.png",
            phase_number=phase_number,
            phase_display_name="PL" if phase_number == 3 else "PR",
            title_label="Rewarded correct-corner visit rate",
            ylabel="Rewarded correct-corner visit rate [%]",
            bin_hours=2,
            spread_metric=USER_SPREAD_METRIC,
            chance_level=25.0,
            plot_style=USER_PLOT_STYLE,
            origin_clock_hour=(
                USER_MOUSE_DAY_START_HOUR
                + USER_SCHEDULED_PHASE_START_HOURS[phase_number]) % 24,
            starting_day=int(USER_SCHEDULED_PHASE_START_HOURS[phase_number] // 24),
            awake_start_clock_hour=USER_MOUSE_DAY_START_HOUR,
            awake_end_clock_hour=USER_MOUSE_DAY_START_HOUR + USER_AWAKE_DURATION_HOURS)

    reversal_rates = compute_phase4_reversal_rate_bins(
        visits,
        bin_hours=2)
    reversal_summaries = {
        name: frames[1]
        for name, frames in reversal_rates.items()}
    for group_name in USER_GROUP_RENAMES.values():
        plot_phase4_reversal_components(
            reversal_summaries,
            group_name=group_name,
            phase_display_name="PR",
            output_path=output_root / f"phase4_reversal_corner_components_{group_name.replace(' ', '_')}_2h.png",
            bin_hours=2,
            spread_metric=USER_SPREAD_METRIC,
            plot_style=USER_PLOT_STYLE,
            origin_clock_hour=(USER_MOUSE_DAY_START_HOUR + USER_SCHEDULED_PHASE_START_HOURS[4]) % 24,
            starting_day=int(USER_SCHEDULED_PHASE_START_HOURS[4] // 24),
            awake_start_clock_hour=USER_MOUSE_DAY_START_HOUR,
            awake_end_clock_hour=USER_MOUSE_DAY_START_HOUR + USER_AWAKE_DURATION_HOURS)
    print(f"Synthetic demo analysis complete. Final output directory: {output_root}")

# %% ENTRY POINT
if __name__ == "__main__":
    main()
# %% END
