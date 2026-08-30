"""Run the IntelliCage place-learning workflow on the synthetic Group A/B data."""
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

from user_scripts.analyze_4month_cohort import (
    DEFAULT_FIGSIZE_CM,
    render_target_group_summary_panels,
    resolved_group_colors,
    run_analysis)

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
USER_FIGSIZE_CM = DEFAULT_FIGSIZE_CM.copy()
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

# %% MAIN FUNCTION
def main() -> None:
    """Run the synthetic Group A/B demo analysis."""

    output_root = run_analysis(
        dataset_root=USER_DATASET_ROOT,
        results_subdir=USER_RESULTS_SUBDIR,
        bin_hours=USER_BIN_HOURS,
        phase2_secondary_metric=USER_PHASE2_SECONDARY_METRIC,
        spread_metric=USER_SPREAD_METRIC,
        plot_style=USER_PLOT_STYLE,
        phase2_plot_style=USER_PHASE2_PLOT_STYLE,
        phase_max_hours=USER_PHASE_MAX_HOURS,
        excluded_groups=USER_EXCLUDED_GROUPS,
        group_renames=USER_GROUP_RENAMES,
        group_colors=USER_GROUP_COLORS,
        figure_size_cm=USER_FIGSIZE_CM,
        mouse_day_start_hour=USER_MOUSE_DAY_START_HOUR,
        awake_duration_hours=USER_AWAKE_DURATION_HOURS,
        scheduled_phase_start_hours=USER_SCHEDULED_PHASE_START_HOURS,
        base_font_size=USER_BASE_FONT_SIZE,
        exclude_violin_outliers=USER_EXCLUDE_VIOLIN_OUTLIERS,
        rate_threshold_pcts=USER_RATE_THRESHOLD_PCTS,
        threshold_onset_bin_hours=USER_THRESHOLD_ONSET_BIN_HOURS,
        responder_horizons_hours=USER_RESPONDER_HORIZONS_HOURS,
        binomial_model_first_hours=USER_BINOMIAL_MODEL_FIRST_HOURS)
    all_groups_order = tuple(USER_GROUP_RENAMES.values())
    render_target_group_summary_panels(
        output_root=output_root,
        all_groups_order=all_groups_order,
        threshold_pcts=USER_RATE_THRESHOLD_PCTS,
        responder_horizon_hours=USER_SUMMARY_RESPONDER_HORIZON_HOURS,
        group_colors=resolved_group_colors(
            group_renames=USER_GROUP_RENAMES,
            group_colors=USER_GROUP_COLORS),
        awake_duration_hours=USER_AWAKE_DURATION_HOURS)
    print(f"Synthetic demo analysis complete. Final output directory: {output_root}")

# %% ENTRY POINT
if __name__ == "__main__":
    main()
