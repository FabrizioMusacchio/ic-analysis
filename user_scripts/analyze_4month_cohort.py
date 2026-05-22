"""Run the IntelliCage place-learning workflow for the BioMedX 4-month cohort.

This user-facing script is the main entry point for the current poster
workflow. It reads the four IntelliCage runs, merges metadata and behavior
tables, computes activity and place-learning metrics, and renders poster-ready
plots into a results directory that always lives inside the selected dataset
directory.
"""
# %% IMPORTS
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intellicage_place_learning.loader import load_cohort_data
from intellicage_place_learning.loader import attach_analysis_time_columns
from intellicage_place_learning.metrics import (
    build_phase_time_limit_table,
    build_analysis_phase_window_table,
    compute_experiment_drinking_visit_bins,
    compute_experiment_lick_count_bins,
    compute_experiment_nosepoke_count_bins,
    compute_experiment_visit_bins,
    compute_phase4_reversal_rate_bins,
    compute_phase2_adaptation_bins,
    compute_phase_activity_medians,
    compute_phase_activity_statistics,
    compute_phase_visit_count_bins,
    compute_place_learning_count_bins,
    compute_place_learning_rate_bins,
    filter_visits_by_phase_limits,
    suggest_common_phase_limits,
)
from intellicage_place_learning.plotting import (
    plot_experiment_dual_metric_bars,
    plot_experiment_overview,
    plot_experiment_overview_groups,
    plot_phase2_adaptation,
    plot_phase_activity_boxplot,
    plot_phase_learning_counts,
    plot_phase_learning_counts_groups,
    plot_phase_learning_rate,
    plot_phase_learning_rate_groups,
    plot_phase4_reversal_components,
    sanitize_filename_part,
)
# %% PARAMETERS AND DEFAULTS

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Data IntelliCage" / "BioMedX_4MonthCohort_2019"
DEFAULT_PHASE_DISPLAY_NAMES = {
    1: "Free Hab",
    2: "NPA",
    3: "PL",
    4: "PR",
}
DEFAULT_PHASE_MAX_HOURS = {
    3: 72.0,
    4: 72.0,
}
DEFAULT_EXCLUDED_GROUPS = ["WT"]
DEFAULT_GROUP_RENAMES = {
    "WT": "WT",
    "tdTomato": "tdTomato",
    "Tau 66-421": "Tau 66-421",
    "Tau 1-421": "Tau 1-421",
    "Tau 1-441": "Tau 1-441",
}
DEFAULT_PHASE2_PLOT_STYLE = "line"
DEFAULT_MOUSE_DAY_START_HOUR = 6.0
DEFAULT_AWAKE_DURATION_HOURS = 12.0
DEFAULT_SCHEDULED_PHASE_START_HOURS = {
    1: 0.0,
    2: 74.0,
    3: 122.0,
    4: 194.0,
    5: 266.0,
}

# %% FUNCTIONS
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the cohort analysis script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the cohort directory that contains Gruppe1-4.",
    )
    parser.add_argument(
        "--results-subdir",
        type=Path,
        default=Path("results"),
        help="Relative results directory below the dataset root, for example `results`.",
    )
    parser.add_argument(
        "--bin-hours",
        type=int,
        nargs="+",
        default=[1, 2],
        help="One or more hour-bin sizes to render, for example `--bin-hours 1 2`.",
    )
    parser.add_argument(
        "--phase2-secondary-metric",
        choices=["lick_positive_visits", "lick_count"],
        default="lick_positive_visits",
        help="Secondary phase-2 metric. The default uses lick-positive visits.",
    )
    parser.add_argument(
        "--spread-metric",
        choices=["sem", "std"],
        default="sem",
        help="Spread statistic used for shaded areas around the mean.",
    )
    parser.add_argument(
        "--plot-style",
        choices=["step", "line"],
        default="line",
        help="Mean-trace style to render in the user-facing plots.",
    )
    parser.add_argument(
        "--phase-max-hours",
        nargs="*",
        default=[],
        help="Optional phase limits such as `3=72 4=72`. These override the user-script defaults.",
    )
    parser.add_argument(
        "--exclude-groups",
        nargs="*",
        default=None,
        help="Optional pathology groups to exclude from the analysis.",
    )
    parser.add_argument(
        "--group-rename",
        nargs="*",
        default=None,
        help="Optional group renaming entries such as `tdTomato=Control`.",
    )
    parser.add_argument(
        "--phase2-plot-style",
        choices=["bar", "line"],
        default=DEFAULT_PHASE2_PLOT_STYLE,
        help="Display style for the phase-2 visits-versus-drinking plots.",
    )
    parser.add_argument(
        "--mouse-day-start-hour",
        type=float,
        default=DEFAULT_MOUSE_DAY_START_HOUR,
        help="Clock hour that defines the beginning of day 0 on the aligned mouse-day timeline.",
    )
    parser.add_argument(
        "--awake-duration-hours",
        type=float,
        default=DEFAULT_AWAKE_DURATION_HOURS,
        help="Duration of the active mouse period per day on the aligned timeline.",
    )
    parser.add_argument(
        "--scheduled-phase-start-hours",
        nargs="*",
        default=[],
        help="Optional scheduled phase start hours such as `1=0 2=74 3=122 4=194 5=266`.",
    )
    return parser.parse_args()


def parse_numeric_mapping(raw_items: list[str]) -> dict[int, float]:
    """Parse `key=value` CLI strings into a numeric dictionary."""

    limits: dict[int, float] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid mapping '{item}'. Use the form KEY=VALUE.")
        phase_text, hour_text = item.split("=", 1)
        limits[int(phase_text)] = float(hour_text)
    return limits


def parse_group_rename_mapping(raw_items: list[str] | None) -> dict[str, str]:
    """Parse `old=new` group-renaming entries into a string dictionary."""

    if not raw_items:
        return {}
    mapping: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid group rename '{item}'. Use the form OLD=NEW.")
        old_name, new_name = item.split("=", 1)
        mapping[old_name] = new_name
    return mapping


def resolve_output_root(dataset_root: Path, results_subdir: Path) -> Path:
    """Resolve a results directory that is always relative to the dataset root."""

    if results_subdir.is_absolute():
        raise ValueError("`--results-subdir` must be a relative path below the dataset root.")
    return dataset_root / results_subdir


def ordered_group_names(visits) -> list[str]:
    """Extract pathology-group names in their categorical display order."""

    categories = getattr(visits["Group"].dtype, "categories", None)
    if categories is not None:
        return [str(category) for category in categories if str(category) != "nan"]
    return sorted(visits["Group"].dropna().astype(str).unique())


def active_period_bounds(mouse_day_start_hour: float, awake_duration_hours: float) -> tuple[float, float]:
    """Return the absolute clock bounds of the configured awake period."""

    awake_end_clock_hour = mouse_day_start_hour + awake_duration_hours
    if awake_end_clock_hour <= mouse_day_start_hour or awake_end_clock_hour > 24.0:
        raise ValueError(
            "The current plotting helpers support a same-day awake window. "
            "Please choose `mouse_day_start_hour + awake_duration_hours <= 24`."
        )
    return mouse_day_start_hour, awake_end_clock_hour


def phase_origin_clock_hour(mouse_day_start_hour: float, scheduled_phase_start_hour: float) -> float:
    """Return the wall-clock hour that corresponds to phase-relative time zero."""

    return float((mouse_day_start_hour + (scheduled_phase_start_hour % 24.0)) % 24.0)


def save_table(dataframe, output_path: Path) -> None:
    """Save a DataFrame as a tab-separated text file with parent creation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, sep="\t", index=False)


def apply_group_preferences(
    visits: pd.DataFrame,
    metadata: pd.DataFrame,
    nosepokes: pd.DataFrame,
    *,
    excluded_groups: list[str] | None,
    group_renames: dict[str, str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply optional group exclusion and renaming to all analysis tables."""

    visits = visits.copy()
    metadata = metadata.copy()
    nosepokes = nosepokes.copy()

    visits["GroupOriginal"] = visits["Group"].astype(str)
    metadata["GroupOriginal"] = metadata["Group"].astype(str)

    if excluded_groups:
        exclude_set = {str(group_name) for group_name in excluded_groups}
        visits = visits.loc[~visits["GroupOriginal"].isin(exclude_set)].copy()
        metadata = metadata.loc[~metadata["GroupOriginal"].isin(exclude_set)].copy()

    if group_renames:
        visits["Group"] = visits["GroupOriginal"].map(group_renames).fillna(visits["GroupOriginal"])
        metadata["Group"] = metadata["GroupOriginal"].map(group_renames).fillna(metadata["GroupOriginal"])
    else:
        visits["Group"] = visits["GroupOriginal"]
        metadata["Group"] = metadata["GroupOriginal"]

    if visits.empty:
        raise ValueError("No visits remain after applying the group-selection settings.")

    seen: list[str] = []
    for group_name in visits["Group"].astype(str):
        if group_name not in seen:
            seen.append(group_name)
    visits["Group"] = pd.Categorical(visits["Group"], categories=seen, ordered=True)
    metadata["Group"] = pd.Categorical(metadata["Group"], categories=seen, ordered=True)

    kept_visit_keys = visits.loc[:, ["RunGroup", "Phase", "PhaseNumber", "VisitID"]].drop_duplicates()
    nosepokes = nosepokes.merge(
        kept_visit_keys,
        on=["RunGroup", "Phase", "PhaseNumber", "VisitID"],
        how="inner",
        validate="many_to_one",
    )
    return visits, metadata, nosepokes


def render_overview_plots(
    visits,
    output_dir: Path,
    *,
    bin_hours: int,
    group_names: list[str],
    phase_window_table,
    phase_display_names: dict[int, str],
    spread_metric: str,
    plot_style: str,
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Create full-experiment visit-activity plots for every pathology group."""

    mouse_bins, summary_bins = compute_experiment_visit_bins(visits, bin_hours=bin_hours)
    save_table(mouse_bins, output_dir / f"overview_visits_mouse_bins_{bin_hours}h.tsv")
    save_table(summary_bins, output_dir / f"overview_visits_group_summary_{bin_hours}h.tsv")
    group_end_hours = (
        visits.groupby("Group", observed=True)["analysis_experiment_elapsed_hours"].max() + float(bin_hours)
    ).astype(float).to_dict()

    for group_name in group_names:
        plot_experiment_overview(
            mouse_bins,
            summary_bins,
            group_name=group_name,
            bin_hours=bin_hours,
            output_path=output_dir
            / "plots"
            / f"overview_all_phases_visits_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
            phase_window_table=phase_window_table,
            phase_display_names=phase_display_names,
            spread_metric=spread_metric,
            x_end_hours=group_end_hours.get(group_name),
            plot_style=plot_style,
            origin_clock_hour=mouse_day_start_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )

    plot_experiment_overview_groups(
        summary_bins,
        output_path=output_dir / "plots" / f"overview_all_phases_visits_all_groups_{bin_hours}h.png",
        phase_window_table=phase_window_table,
        phase_display_names=phase_display_names,
        spread_metric=spread_metric,
        plot_style=plot_style,
        origin_clock_hour=mouse_day_start_hour,
        awake_start_clock_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )


def render_experiment_metric_plots(
    mouse_bins,
    summary_bins,
    output_dir: Path,
    *,
    file_stub: str,
    title_label: str,
    group_title_label: str,
    ylabel: str,
    bin_hours: int,
    group_names: list[str],
    phase_window_table,
    phase_display_names: dict[int, str],
    spread_metric: str,
    plot_style: str,
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Render one full-experiment metric in the same visual style as the overview."""

    save_table(mouse_bins, output_dir / f"{file_stub}_mouse_bins_{bin_hours}h.tsv")
    save_table(summary_bins, output_dir / f"{file_stub}_group_summary_{bin_hours}h.tsv")
    group_end_hours = (
        summary_bins.groupby("Group", observed=True)["bin_end_hours"].max().astype(float).to_dict()
    )

    for group_name in group_names:
        plot_experiment_overview(
            mouse_bins,
            summary_bins,
            group_name=group_name,
            bin_hours=bin_hours,
            output_path=output_dir / "plots" / f"{file_stub}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
            phase_window_table=phase_window_table,
            phase_display_names=phase_display_names,
            spread_metric=spread_metric,
            x_end_hours=group_end_hours.get(group_name),
            plot_style=plot_style,
            show_individual_labels=False,
            title_label=title_label,
            ylabel=ylabel,
            origin_clock_hour=mouse_day_start_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )

    plot_experiment_overview_groups(
        summary_bins,
        output_path=output_dir / "plots" / f"{file_stub}_all_groups_{bin_hours}h.png",
        phase_window_table=phase_window_table,
        phase_display_names=phase_display_names,
        spread_metric=spread_metric,
        plot_style=plot_style,
        title_label=group_title_label,
        ylabel=ylabel,
        origin_clock_hour=mouse_day_start_hour,
        awake_start_clock_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )


def render_phase2_plots(
    visits,
    output_dir: Path,
    *,
    bin_hours: int,
    group_names: list[str],
    secondary_metric: str,
    phase_display_names: dict[int, str],
    plot_style: str,
    spread_metric: str,
    phase_origin_hour: float,
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Create phase-2 adaptation summaries and paired-bar plots."""

    metric_tables = compute_phase2_adaptation_bins(
        visits,
        bin_hours=bin_hours,
        secondary_metric=secondary_metric,
    )
    primary_mouse, primary_summary = metric_tables["visits"]
    secondary_mouse, secondary_summary = metric_tables["drinking_metric"]
    lick_mouse, lick_summary = metric_tables["lick_count"]
    lick_positive_mouse, lick_positive_summary = metric_tables["lick_positive_visits"]

    save_table(primary_mouse, output_dir / f"phase2_visits_mouse_bins_{bin_hours}h.tsv")
    save_table(primary_summary, output_dir / f"phase2_visits_group_summary_{bin_hours}h.tsv")
    save_table(secondary_mouse, output_dir / f"phase2_{secondary_metric}_mouse_bins_{bin_hours}h.tsv")
    save_table(secondary_summary, output_dir / f"phase2_{secondary_metric}_group_summary_{bin_hours}h.tsv")
    save_table(lick_positive_mouse, output_dir / f"phase2_lick_positive_visits_mouse_bins_{bin_hours}h.tsv")
    save_table(lick_positive_summary, output_dir / f"phase2_lick_positive_visits_group_summary_{bin_hours}h.tsv")
    save_table(lick_mouse, output_dir / f"phase2_lick_count_mouse_bins_{bin_hours}h.tsv")
    save_table(lick_summary, output_dir / f"phase2_lick_count_group_summary_{bin_hours}h.tsv")

    secondary_label = "Drinking visits" if secondary_metric == "lick_positive_visits" else "Lick count"
    phase2_end_hours = (
        visits.loc[visits["AnalysisPhaseNumber"].eq(2)]
        .groupby("Group", observed=True)["analysis_phase_elapsed_hours"]
        .max()
        .add(float(bin_hours))
        .astype(float)
        .to_dict()
    )
    for group_name in group_names:
        plot_phase2_adaptation(
            primary_summary,
            secondary_summary,
            group_name=group_name,
            bin_hours=bin_hours,
            output_path=output_dir
            / "plots"
            / f"phase2_visits_vs_{sanitize_filename_part(secondary_metric)}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
            secondary_label=secondary_label,
            phase_display_name=phase_display_names[2],
            plot_style=plot_style,
            spread_metric=spread_metric,
            x_end_hours=phase2_end_hours.get(group_name),
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )


def render_phase2_control_plots(
    visits,
    output_dir: Path,
    *,
    bin_hours: int,
    group_names: list[str],
    phase_window_table,
    phase_display_names: dict[int, str],
    plot_style: str,
    spread_metric: str,
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Create full-experiment control plots for visits versus drinking visits."""

    primary_mouse, primary_summary = compute_experiment_visit_bins(visits, bin_hours=bin_hours)
    drinking_mouse, drinking_summary = compute_experiment_drinking_visit_bins(visits, bin_hours=bin_hours)
    nosepoke_mouse, nosepoke_summary = compute_experiment_nosepoke_count_bins(visits, bin_hours=bin_hours)
    lick_mouse, lick_summary = compute_experiment_lick_count_bins(visits, bin_hours=bin_hours)
    save_table(primary_mouse, output_dir / f"phase2_control_all_phases_visits_mouse_bins_{bin_hours}h.tsv")
    save_table(primary_summary, output_dir / f"phase2_control_all_phases_visits_group_summary_{bin_hours}h.tsv")
    save_table(drinking_mouse, output_dir / f"phase2_control_all_phases_drinking_visits_mouse_bins_{bin_hours}h.tsv")
    save_table(drinking_summary, output_dir / f"phase2_control_all_phases_drinking_visits_group_summary_{bin_hours}h.tsv")

    for group_name in group_names:
        plot_experiment_dual_metric_bars(
            primary_summary,
            drinking_summary,
            group_name=group_name,
            bin_hours=bin_hours,
            output_path=output_dir
            / "plots"
            / f"phase2_control_all_phases_visits_vs_drinking_visits_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
            secondary_label="Drinking visits",
            phase_window_table=phase_window_table,
            phase_display_names=phase_display_names,
            plot_style=plot_style,
            spread_metric=spread_metric,
            origin_clock_hour=mouse_day_start_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )

    render_experiment_metric_plots(
        nosepoke_mouse,
        nosepoke_summary,
        output_dir,
        file_stub="phase2_control_all_phases_nosepoke_counts",
        title_label="nose-poke counts across all phases",
        group_title_label="Nose-poke counts across all phases by group",
        ylabel="Nose pokes per mouse and bin",
        bin_hours=bin_hours,
        group_names=group_names,
        phase_window_table=phase_window_table,
        phase_display_names=phase_display_names,
        spread_metric=spread_metric,
        plot_style="line",
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )
    render_experiment_metric_plots(
        lick_mouse,
        lick_summary,
        output_dir,
        file_stub="phase2_control_all_phases_lick_counts",
        title_label="lick counts across all phases",
        group_title_label="Lick counts across all phases by group",
        ylabel="Licks per mouse and bin",
        bin_hours=bin_hours,
        group_names=group_names,
        phase_window_table=phase_window_table,
        phase_display_names=phase_display_names,
        spread_metric=spread_metric,
        plot_style="line",
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )


def render_phase_learning_plots(
    visits,
    output_dir: Path,
    *,
    bin_hours: int,
    group_names: list[str],
    phase_display_names: dict[int, str],
    spread_metric: str,
    plot_style: str,
    scheduled_phase_start_hours: dict[int, float],
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Create phase-3 and phase-4 count and rate plots."""

    phase_group_end_hours = (
        visits.groupby(["Group", "AnalysisPhaseNumber"], observed=True)["analysis_phase_elapsed_hours"].max()
        + float(bin_hours)
    ).astype(float).to_dict()
    phase_end_hours = (
        visits.groupby("AnalysisPhaseNumber", observed=True)["analysis_phase_elapsed_hours"].max() + float(bin_hours)
    ).astype(float).to_dict()

    metric_specs = [
        {
            "file_stub": "correct_corner_visit_rate",
            "success_col": "correct_corner_visit",
            "title_label": "correct-corner visit rate",
            "ylabel": "Correct-corner visit rate [%]",
            "chance_level": 25.0,
            "count_file_stub": "correct_corner_visits_absolute",
            "count_title_label": "correct-corner visits",
            "count_ylabel": "Correct-corner visits per mouse and bin",
        },
        {
            "file_stub": "correct_np_visit_rate",
            "success_col": "correct_np_visit",
            "title_label": "correct NP visit rate",
            "ylabel": "Correct NP visit rate [%]",
            "chance_level": 25.0,
            "count_file_stub": "correct_np_visits_absolute",
            "count_title_label": "correct NP visits",
            "count_ylabel": "Correct NP visits per mouse and bin",
        },
        {
            "file_stub": "rewarded_correct_corner_visit_rate",
            "success_col": "rewarded_correct_corner_visit",
            "title_label": "rewarded correct-corner visit rate",
            "ylabel": "Rewarded correct-corner visit rate [%]",
            "chance_level": 25.0,
            "count_file_stub": "rewarded_correct_corner_visits_absolute",
            "count_title_label": "rewarded correct-corner visits",
            "count_ylabel": "Rewarded correct-corner visits per mouse and bin",
        },
    ]

    for phase_number in (3, 4):
        phase_origin_hour = phase_origin_clock_hour(
            mouse_day_start_hour,
            scheduled_phase_start_hours[phase_number],
        )
        phase_visit_mouse, phase_visit_summary = compute_phase_visit_count_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
        )
        rewarded_count_mouse, rewarded_count_summary = compute_place_learning_count_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
            success_col="rewarded_correct_corner_visit",
        )
        matlab_rate_mouse, matlab_rate_summary = compute_place_learning_rate_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
            success_col="correct_place_visit",
        )

        save_table(
            phase_visit_mouse,
            output_dir / f"phase{phase_number}_all_visit_counts_mouse_bins_{bin_hours}h.tsv",
        )
        save_table(
            phase_visit_summary,
            output_dir / f"phase{phase_number}_all_visit_counts_group_summary_{bin_hours}h.tsv",
        )
        save_table(
            rewarded_count_mouse,
            output_dir / f"phase{phase_number}_rewarded_correct_corner_visits_absolute_mouse_bins_{bin_hours}h.tsv",
        )
        save_table(
            rewarded_count_summary,
            output_dir / f"phase{phase_number}_rewarded_correct_corner_visits_absolute_group_summary_{bin_hours}h.tsv",
        )
        save_table(
            matlab_rate_mouse,
            output_dir / f"phase{phase_number}_matlab_placeerror_only_rate_mouse_bins_{bin_hours}h.tsv",
        )
        save_table(
            matlab_rate_summary,
            output_dir / f"phase{phase_number}_matlab_placeerror_only_rate_group_summary_{bin_hours}h.tsv",
        )

        for metric_spec in metric_specs:
            metric_count_mouse, metric_count_summary = compute_place_learning_count_bins(
                visits,
                phase_number=phase_number,
                bin_hours=bin_hours,
                success_col=metric_spec["success_col"],
            )
            metric_mouse, metric_summary = compute_place_learning_rate_bins(
                visits,
                phase_number=phase_number,
                bin_hours=bin_hours,
                success_col=metric_spec["success_col"],
            )
            save_table(
                metric_mouse,
                output_dir / f"phase{phase_number}_{metric_spec['file_stub']}_mouse_bins_{bin_hours}h.tsv",
            )
            save_table(
                metric_summary,
                output_dir / f"phase{phase_number}_{metric_spec['file_stub']}_group_summary_{bin_hours}h.tsv",
            )
            save_table(
                metric_count_mouse,
                output_dir / f"phase{phase_number}_{metric_spec['count_file_stub']}_mouse_bins_{bin_hours}h.tsv",
            )
            save_table(
                metric_count_summary,
                output_dir / f"phase{phase_number}_{metric_spec['count_file_stub']}_group_summary_{bin_hours}h.tsv",
            )

            for group_name in group_names:
                plot_phase_learning_counts(
                    metric_count_mouse,
                    metric_count_summary,
                    group_name=group_name,
                    phase_number=phase_number,
                    phase_display_name=phase_display_names[phase_number],
                    bin_hours=bin_hours,
                    output_path=output_dir
                    / "plots"
                    / f"phase{phase_number}_{metric_spec['count_file_stub']}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    title_label=metric_spec["count_title_label"],
                    ylabel=metric_spec["count_ylabel"],
                    x_end_hours=phase_group_end_hours.get((group_name, phase_number)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                )
                plot_phase_learning_rate(
                    metric_mouse,
                    metric_summary,
                    group_name=group_name,
                    phase_number=phase_number,
                    phase_display_name=phase_display_names[phase_number],
                    bin_hours=bin_hours,
                    output_path=output_dir
                    / "plots"
                    / f"phase{phase_number}_{metric_spec['file_stub']}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    title_label=metric_spec["title_label"],
                    ylabel=metric_spec["ylabel"],
                    chance_level=metric_spec["chance_level"],
                    x_end_hours=phase_group_end_hours.get((group_name, phase_number)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                )

            plot_phase_learning_counts_groups(
                metric_count_summary,
                phase_display_name=phase_display_names[phase_number],
                bin_hours=bin_hours,
                output_path=output_dir
                / "plots"
                / f"phase{phase_number}_{metric_spec['count_file_stub']}_all_groups_{bin_hours}h.png",
                spread_metric=spread_metric,
                x_end_hours=phase_end_hours.get(phase_number),
                plot_style=plot_style,
                title_prefix=f"{metric_spec['count_title_label'].capitalize()} across groups",
                ylabel=metric_spec["count_ylabel"],
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=mouse_day_start_hour,
                awake_end_clock_hour=awake_end_clock_hour,
            )
            plot_phase_learning_rate_groups(
                metric_summary,
                phase_number=phase_number,
                phase_display_name=phase_display_names[phase_number],
                bin_hours=bin_hours,
                output_path=output_dir
                / "plots"
                / f"phase{phase_number}_{metric_spec['file_stub']}_all_groups_{bin_hours}h.png",
                spread_metric=spread_metric,
                title_label=metric_spec["title_label"],
                ylabel=metric_spec["ylabel"],
                chance_level=metric_spec["chance_level"],
                x_end_hours=phase_end_hours.get(phase_number),
                plot_style=plot_style,
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=mouse_day_start_hour,
                awake_end_clock_hour=awake_end_clock_hour,
            )

        plot_phase_learning_counts_groups(
            phase_visit_summary,
            phase_display_name=phase_display_names[phase_number],
            bin_hours=bin_hours,
            output_path=output_dir
            / "plots"
            / f"phase{phase_number}_all_visit_counts_all_groups_{bin_hours}h.png",
            spread_metric=spread_metric,
            x_end_hours=phase_end_hours.get(phase_number),
            plot_style=plot_style,
            title_prefix="All visit counts across groups",
            ylabel="All visits per mouse and bin",
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )

        if phase_number == 4:
            reversal_rate_tables = compute_phase4_reversal_rate_bins(visits, bin_hours=bin_hours)
            reversal_group_summaries: dict[str, pd.DataFrame] = {}
            for component_name, (component_mouse, component_summary) in reversal_rate_tables.items():
                save_table(
                    component_mouse,
                    output_dir / f"phase4_{component_name}_visit_rate_mouse_bins_{bin_hours}h.tsv",
                )
                save_table(
                    component_summary,
                    output_dir / f"phase4_{component_name}_visit_rate_group_summary_{bin_hours}h.tsv",
                )
                reversal_group_summaries[component_name] = component_summary
            for group_name in group_names:
                plot_phase4_reversal_components(
                    reversal_group_summaries,
                    group_name=group_name,
                    phase_display_name=phase_display_names[4],
                    bin_hours=bin_hours,
                    output_path=output_dir
                    / "plots"
                    / f"phase4_reversal_corner_components_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    x_end_hours=phase_group_end_hours.get((group_name, 4)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                )


def render_phase_activity_plot(
    visits,
    output_dir: Path,
    *,
    phase_display_names: dict[int, str],
) -> None:
    """Render the median hourly activity boxplot across phases and groups."""

    mouse_phase_activity = compute_phase_activity_medians(visits)
    activity_stats = compute_phase_activity_statistics(mouse_phase_activity)
    save_table(mouse_phase_activity, output_dir / "phase_activity_median_visits_per_hour_mouse.tsv")
    save_table(activity_stats, output_dir / "phase_activity_median_visits_per_hour_stats.tsv")

    plot_phase_activity_boxplot(
        mouse_phase_activity,
        activity_stats,
        phase_display_names=phase_display_names,
        output_path=output_dir / "plots" / "phase_activity_median_visits_per_hour_boxplot.png",
    )


def run_analysis(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    results_subdir: Path = Path("results"),
    bin_hours: list[int] | tuple[int, ...] = (1, 2),
    phase2_secondary_metric: str = "lick_positive_visits",
    spread_metric: str = "sem",
    plot_style: str = "line",
    phase2_plot_style: str = DEFAULT_PHASE2_PLOT_STYLE,
    phase_max_hours: dict[int, float] | None = None,
    excluded_groups: list[str] | None = None,
    group_renames: dict[str, str] | None = None,
    mouse_day_start_hour: float = DEFAULT_MOUSE_DAY_START_HOUR,
    awake_duration_hours: float = DEFAULT_AWAKE_DURATION_HOURS,
    scheduled_phase_start_hours: dict[int, float] | None = None,
) -> Path:
    """Run the 4-month cohort pipeline from a normal Python function call.

    This function mirrors the CLI configuration but can be imported and called
    directly from any user script without argument parsing.
    """

    merged_phase_limits = DEFAULT_PHASE_MAX_HOURS.copy()
    if phase_max_hours:
        merged_phase_limits.update(phase_max_hours)
    merged_scheduled_phase_starts = DEFAULT_SCHEDULED_PHASE_START_HOURS.copy()
    if scheduled_phase_start_hours:
        merged_scheduled_phase_starts.update(scheduled_phase_start_hours)
    selected_excluded_groups = DEFAULT_EXCLUDED_GROUPS if excluded_groups is None else list(excluded_groups)
    selected_group_renames = DEFAULT_GROUP_RENAMES.copy()
    if group_renames:
        selected_group_renames.update(group_renames)
    awake_start_clock_hour, awake_end_clock_hour = active_period_bounds(
        mouse_day_start_hour,
        awake_duration_hours,
    )
    output_root = resolve_output_root(dataset_root, results_subdir)
    output_root.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort_data(dataset_root)
    aligned_visits = attach_analysis_time_columns(
        cohort.visits,
        cohort.phase_manifest,
        scheduled_phase_start_hours=merged_scheduled_phase_starts,
        mouse_day_start_hour=mouse_day_start_hour,
    )
    selected_visits, selected_metadata, selected_nosepokes = apply_group_preferences(
        aligned_visits,
        cohort.metadata,
        cohort.nosepokes,
        excluded_groups=selected_excluded_groups,
        group_renames=selected_group_renames,
    )
    limit_table = build_phase_time_limit_table(cohort.phase_manifest)
    save_table(selected_metadata, output_root / "mouse_metadata.tsv")
    save_table(cohort.phase_manifest, output_root / "phase_manifest.tsv")
    save_table(limit_table, output_root / "phase_time_limit_recommendations.tsv")
    save_table(
        pd.DataFrame(
            {
                "Setting": [
                    "mouse_day_start_hour",
                    "awake_duration_hours",
                    "phase2_plot_style",
                    "exclude_groups",
                    "group_rename_mapping",
                ],
                "Value": [
                    mouse_day_start_hour,
                    awake_duration_hours,
                    phase2_plot_style,
                    ",".join(selected_excluded_groups) if selected_excluded_groups else "",
                    ";".join(f"{key}={value}" for key, value in selected_group_renames.items()),
                ],
            }
        ),
        output_root / "analysis_settings.tsv",
    )
    save_table(
        pd.DataFrame(
            {
                "PhaseNumber": list(sorted(merged_scheduled_phase_starts)),
                "ScheduledStartHours": [
                    merged_scheduled_phase_starts[key] for key in sorted(merged_scheduled_phase_starts)
                ],
            }
        ),
        output_root / "scheduled_phase_start_hours.tsv",
    )

    filtered_visits = filter_visits_by_phase_limits(selected_visits, merged_phase_limits)
    filtered_visits.to_csv(output_root / "merged_visits.tsv.gz", sep="\t", index=False, compression="gzip")
    selected_nosepokes.to_csv(output_root / "merged_nosepokes.tsv.gz", sep="\t", index=False, compression="gzip")

    group_names = ordered_group_names(filtered_visits)
    phase_window_table = build_analysis_phase_window_table(filtered_visits, merged_scheduled_phase_starts)
    save_table(phase_window_table, output_root / "analysis_phase_windows.tsv")
    suggested_limits = suggest_common_phase_limits(cohort.phase_manifest)
    save_table(
        pd.DataFrame(
            {
                "PhaseNumber": list(sorted(suggested_limits)),
                "SuggestedCommonLimitHours": [suggested_limits[key] for key in sorted(suggested_limits)],
            }
        ),
        output_root / "suggested_common_phase_limits.tsv",
    )

    for current_bin_hours in sorted(set(bin_hours)):
        bin_dir = output_root / f"{current_bin_hours}h_bins"
        render_overview_plots(
            filtered_visits,
            bin_dir,
            bin_hours=current_bin_hours,
            group_names=group_names,
            phase_window_table=phase_window_table,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
            spread_metric=spread_metric,
            plot_style=plot_style,
            mouse_day_start_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )
        render_phase2_plots(
            filtered_visits,
            bin_dir,
            bin_hours=current_bin_hours,
            group_names=group_names,
            secondary_metric=phase2_secondary_metric,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
            plot_style=phase2_plot_style,
            spread_metric=spread_metric,
            phase_origin_hour=phase_origin_clock_hour(
                mouse_day_start_hour,
                merged_scheduled_phase_starts[2],
            ),
            mouse_day_start_hour=awake_start_clock_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )
        render_phase2_control_plots(
            filtered_visits,
            bin_dir,
            bin_hours=current_bin_hours,
            group_names=group_names,
            phase_window_table=phase_window_table,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
            plot_style=phase2_plot_style,
            spread_metric=spread_metric,
            mouse_day_start_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )
        render_phase_learning_plots(
            filtered_visits,
            bin_dir,
            bin_hours=current_bin_hours,
            group_names=group_names,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
            spread_metric=spread_metric,
            plot_style=plot_style,
            scheduled_phase_start_hours=merged_scheduled_phase_starts,
            mouse_day_start_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )

    render_phase_activity_plot(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
    )
    return output_root

# %% MAIN FUNCTION
def main() -> None:
    """Run the full 4-month place-learning workflow from CLI arguments."""

    args = parse_args()
    run_analysis(
        dataset_root=args.dataset_root,
        results_subdir=args.results_subdir,
        bin_hours=args.bin_hours,
        phase2_secondary_metric=args.phase2_secondary_metric,
        spread_metric=args.spread_metric,
        plot_style=args.plot_style,
        phase2_plot_style=args.phase2_plot_style,
        phase_max_hours=parse_numeric_mapping(args.phase_max_hours),
        excluded_groups=DEFAULT_EXCLUDED_GROUPS if args.exclude_groups is None else list(args.exclude_groups),
        group_renames={**DEFAULT_GROUP_RENAMES, **parse_group_rename_mapping(args.group_rename)},
        mouse_day_start_hour=args.mouse_day_start_hour,
        awake_duration_hours=args.awake_duration_hours,
        scheduled_phase_start_hours=parse_numeric_mapping(args.scheduled_phase_start_hours),
    )

# %% ENTRY POINT
if __name__ == "__main__":
    main()
# %% END