"""Run the IntelliCage place-learning workflow for the BioMedX 4-month cohort.

This user-facing script is the main entry point for the current poster
workflow. It reads the four IntelliCage runs, merges metadata and behavior
tables, computes activity and place-learning metrics, and renders poster-ready
plots into a results directory that always lives inside the selected dataset
directory.
"""
# %% IMPORTS
from __future__ import annotations

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
    compute_awake_day_rate_tables,
    compute_experiment_drinking_visit_bins,
    compute_experiment_lick_count_bins,
    compute_experiment_nosepoke_count_bins,
    compute_experiment_visit_bins,
    compute_group_day_violin_statistics,
    compute_onset_group_statistics,
    compute_phase4_reversal_rate_bins,
    compute_phase4_reversal_count_bins,
    compute_phase2_adaptation_bins,
    compute_phase_activity_medians,
    compute_phase_activity_statistics,
    compute_phase_segment_rate_tables,
    compute_phase_visit_count_bins,
    compute_place_learning_count_bins,
    compute_place_learning_rate_bins,
    compute_role_cumulative_curves,
    compute_time_window_learning_curves,
    compute_visit_window_learning_curves,
    filter_visits_by_phase_limits,
    flag_iqr_outliers,
    suggest_common_phase_limits,
)
from intellicage_place_learning.plotting import (
    configure_plot_style,
    plot_experiment_dual_metric_bars,
    plot_experiment_overview,
    plot_experiment_overview_groups,
    plot_cumulative_role_curves,
    plot_group_day_violin,
    plot_onset_violin,
    plot_phase2_adaptation,
    plot_phase_activity_boxplot,
    plot_phase_learning_counts,
    plot_phase_learning_counts_groups,
    plot_phase_learning_rate,
    plot_phase_learning_rate_groups,
    plot_phase_segment_rate_groups,
    plot_phase4_reversal_components,
    plot_visit_learning_curve_groups,
    sanitize_filename_part,
    set_group_colors,
    set_figure_size_presets,
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
DEFAULT_GROUP_COLORS = {
    "WT": "#264653",
    "tdTomato": "#6c757d",
    "Tau 66-421": "#2a9d8f",
    "Tau 1-421": "#e9a820",
    "Tau 1-441": "#4ade80",
}
DEFAULT_FIGSIZE_CM = {# always a pair of (width, height)
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
USER_DATASET_ROOT = DEFAULT_DATASET_ROOT
USER_RESULTS_SUBDIR = Path("results")
USER_BIN_HOURS = [1, 2]
USER_PHASE2_SECONDARY_METRIC = "lick_positive_visits"
USER_SPREAD_METRIC = "sem"
USER_PLOT_STYLE = "line"
USER_PHASE2_PLOT_STYLE = "line"
USER_PHASE_MAX_HOURS = DEFAULT_PHASE_MAX_HOURS.copy()
USER_EXCLUDED_GROUPS = DEFAULT_EXCLUDED_GROUPS.copy()
USER_GROUP_RENAMES = DEFAULT_GROUP_RENAMES.copy()
USER_GROUP_COLORS = DEFAULT_GROUP_COLORS.copy()
USER_FIGSIZE_CM = DEFAULT_FIGSIZE_CM.copy()
USER_MOUSE_DAY_START_HOUR = DEFAULT_MOUSE_DAY_START_HOUR
USER_AWAKE_DURATION_HOURS = DEFAULT_AWAKE_DURATION_HOURS
USER_SCHEDULED_PHASE_START_HOURS = DEFAULT_SCHEDULED_PHASE_START_HOURS.copy()
USER_BASE_FONT_SIZE = 10.0
USER_EXCLUDE_VIOLIN_OUTLIERS = True
# %% FUNCTIONS
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

def resolved_group_colors(
    *,
    group_renames: dict[str, str],
    group_colors: dict[str, str] | None,
) -> dict[str, str]:
    """Resolve group colors against the active rename mapping for plotting."""

    base_colors = DEFAULT_GROUP_COLORS.copy()
    if group_colors:
        base_colors.update(group_colors)
    resolved: dict[str, str] = {}
    for original_name, color in base_colors.items():
        resolved[group_renames.get(original_name, original_name)] = color
    if group_colors:
        for key, value in group_colors.items():
            resolved.setdefault(str(key), str(value))
    return resolved

def phase_origin_clock_hour(mouse_day_start_hour: float, scheduled_phase_start_hour: float) -> float:
    """Return the wall-clock hour that corresponds to phase-relative time zero."""

    return float((mouse_day_start_hour + (scheduled_phase_start_hour % 24.0)) % 24.0)


def experiment_day_from_scheduled_start(scheduled_phase_start_hour: float) -> int:
    """Convert one scheduled phase-start hour to its global experiment-day number."""

    return int(scheduled_phase_start_hour // 24.0)

def save_table(dataframe, output_path: Path) -> None:
    """Save a DataFrame as a tab-separated text file with parent creation."""

    target_path = output_path.parent / "csv" / output_path.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(target_path, sep="\t", index=False)

def csv_output_path(output_path: Path) -> Path:
    """Return the standardized CSV destination below a local `csv` subfolder."""

    return output_path.parent / "csv" / output_path.name

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
    preferred_order = [str(name) for name in (group_renames or {}).values()]
    available_groups = set(visits["Group"].astype(str))
    for group_name in preferred_order:
        if group_name in available_groups and group_name not in seen:
            seen.append(group_name)
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
            output_path=output_dir / f"overview_all_phases_visits_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
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
        output_path=output_dir / f"overview_all_phases_visits_all_groups_{bin_hours}h.png",
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
            output_path=output_dir / f"{file_stub}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
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
        output_path=output_dir / f"{file_stub}_all_groups_{bin_hours}h.png",
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
    phase_start_day: int,
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
            output_path=output_dir / f"phase2_visits_vs_{sanitize_filename_part(secondary_metric)}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
            secondary_label=secondary_label,
            phase_display_name=phase_display_names[2],
            plot_style=plot_style,
            spread_metric=spread_metric,
            x_end_hours=phase2_end_hours.get(group_name),
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
            starting_day=phase_start_day,
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
            output_path=output_dir / f"phase2_control_all_phases_visits_vs_drinking_visits_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
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
        phase_start_day = experiment_day_from_scheduled_start(scheduled_phase_start_hours[phase_number])
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
                    output_path=output_dir / f"phase{phase_number}_{metric_spec['count_file_stub']}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    title_label=metric_spec["count_title_label"],
                    ylabel=metric_spec["count_ylabel"],
                    x_end_hours=phase_group_end_hours.get((group_name, phase_number)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                    starting_day=phase_start_day,
                )
                plot_phase_learning_rate(
                    metric_mouse,
                    metric_summary,
                    group_name=group_name,
                    phase_number=phase_number,
                    phase_display_name=phase_display_names[phase_number],
                    bin_hours=bin_hours,
                    output_path=output_dir / f"phase{phase_number}_{metric_spec['file_stub']}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    title_label=metric_spec["title_label"],
                    ylabel=metric_spec["ylabel"],
                    chance_level=metric_spec["chance_level"],
                    x_end_hours=phase_group_end_hours.get((group_name, phase_number)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                    starting_day=phase_start_day,
                )

            plot_phase_learning_counts_groups(
                metric_count_summary,
                phase_display_name=phase_display_names[phase_number],
                bin_hours=bin_hours,
                output_path=output_dir / f"phase{phase_number}_{metric_spec['count_file_stub']}_all_groups_{bin_hours}h.png",
                spread_metric=spread_metric,
                x_end_hours=phase_end_hours.get(phase_number),
                plot_style=plot_style,
                title_prefix=f"{metric_spec['count_title_label'].capitalize()} across groups",
                ylabel=metric_spec["count_ylabel"],
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=mouse_day_start_hour,
                awake_end_clock_hour=awake_end_clock_hour,
                starting_day=phase_start_day,
            )
            plot_phase_learning_rate_groups(
                metric_summary,
                phase_number=phase_number,
                phase_display_name=phase_display_names[phase_number],
                bin_hours=bin_hours,
                output_path=output_dir / f"phase{phase_number}_{metric_spec['file_stub']}_all_groups_{bin_hours}h.png",
                spread_metric=spread_metric,
                title_label=metric_spec["title_label"],
                ylabel=metric_spec["ylabel"],
                chance_level=metric_spec["chance_level"],
                x_end_hours=phase_end_hours.get(phase_number),
                plot_style=plot_style,
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=mouse_day_start_hour,
                awake_end_clock_hour=awake_end_clock_hour,
                starting_day=phase_start_day,
            )

        plot_phase_learning_counts_groups(
            phase_visit_summary,
            phase_display_name=phase_display_names[phase_number],
            bin_hours=bin_hours,
            output_path=output_dir / f"phase{phase_number}_all_visit_counts_all_groups_{bin_hours}h.png",
            spread_metric=spread_metric,
            x_end_hours=phase_end_hours.get(phase_number),
            plot_style=plot_style,
            title_prefix="All visit counts across groups",
            ylabel="All visits per mouse and bin",
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
            starting_day=phase_start_day,
        )

        if phase_number == 4:
            reversal_rate_tables = compute_phase4_reversal_rate_bins(visits, bin_hours=bin_hours)
            reversal_count_tables = compute_phase4_reversal_count_bins(visits, bin_hours=bin_hours)
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
            for component_name, (component_mouse, component_summary) in reversal_count_tables.items():
                save_table(
                    component_mouse,
                    output_dir / f"phase4_{component_name}_visits_absolute_mouse_bins_{bin_hours}h.tsv",
                )
                save_table(
                    component_summary,
                    output_dir / f"phase4_{component_name}_visits_absolute_group_summary_{bin_hours}h.tsv",
                )
                chance_level = 25.0
                title_map = {
                    "new_correct_corner": "new correct-corner visit rate",
                    "previous_correct_corner": "previous correct-corner visit rate",
                    "neutral_incorrect_corner_1": "neutral incorrect-corner 1 visit rate",
                    "neutral_incorrect_corner_2": "neutral incorrect-corner 2 visit rate",
                }
                ylabel_map = {
                    "new_correct_corner": "New correct-corner visit rate [%]",
                    "previous_correct_corner": "Previous correct-corner visit rate [%]",
                    "neutral_incorrect_corner_1": "Neutral incorrect-corner 1 visit rate [%]",
                    "neutral_incorrect_corner_2": "Neutral incorrect-corner 2 visit rate [%]",
                }
                count_title_map = {
                    "new_correct_corner": "new correct-corner visits",
                    "previous_correct_corner": "previous correct-corner visits",
                    "neutral_incorrect_corner_1": "neutral incorrect-corner 1 visits",
                    "neutral_incorrect_corner_2": "neutral incorrect-corner 2 visits",
                }
                count_ylabel_map = {
                    "new_correct_corner": "New correct-corner visits per mouse and bin",
                    "previous_correct_corner": "Previous correct-corner visits per mouse and bin",
                    "neutral_incorrect_corner_1": "Neutral incorrect-corner 1 visits per mouse and bin",
                    "neutral_incorrect_corner_2": "Neutral incorrect-corner 2 visits per mouse and bin",
                }
                for group_name in group_names:
                    plot_phase_learning_rate(
                        reversal_rate_tables[component_name][0],
                        reversal_rate_tables[component_name][1],
                        group_name=group_name,
                        phase_number=4,
                        phase_display_name=phase_display_names[4],
                        bin_hours=bin_hours,
                        output_path=output_dir / f"phase4_{component_name}_visit_rate_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                        spread_metric=spread_metric,
                        title_label=title_map[component_name],
                        ylabel=ylabel_map[component_name],
                        chance_level=chance_level,
                        x_end_hours=phase_group_end_hours.get((group_name, 4)),
                        plot_style=plot_style,
                        origin_clock_hour=phase_origin_hour,
                        awake_start_clock_hour=mouse_day_start_hour,
                        awake_end_clock_hour=awake_end_clock_hour,
                        starting_day=phase_start_day,
                    )
                    plot_phase_learning_counts(
                        component_mouse,
                        component_summary,
                        group_name=group_name,
                        phase_number=4,
                        phase_display_name=phase_display_names[4],
                        bin_hours=bin_hours,
                        output_path=output_dir / f"phase4_{component_name}_visits_absolute_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                        spread_metric=spread_metric,
                        title_label=count_title_map[component_name],
                        ylabel=count_ylabel_map[component_name],
                        x_end_hours=phase_group_end_hours.get((group_name, 4)),
                        plot_style=plot_style,
                        origin_clock_hour=phase_origin_hour,
                        awake_start_clock_hour=mouse_day_start_hour,
                        awake_end_clock_hour=awake_end_clock_hour,
                        starting_day=phase_start_day,
                    )
                plot_phase_learning_rate_groups(
                    reversal_rate_tables[component_name][1],
                    phase_number=4,
                    phase_display_name=phase_display_names[4],
                    bin_hours=bin_hours,
                    output_path=output_dir / f"phase4_{component_name}_visit_rate_all_groups_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    title_label=title_map[component_name],
                    ylabel=ylabel_map[component_name],
                    chance_level=chance_level,
                    x_end_hours=phase_end_hours.get(4),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                    starting_day=phase_start_day,
                )
                plot_phase_learning_counts_groups(
                    component_summary,
                    phase_display_name=phase_display_names[4],
                    bin_hours=bin_hours,
                    output_path=output_dir / f"phase4_{component_name}_visits_absolute_all_groups_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    x_end_hours=phase_end_hours.get(4),
                    plot_style=plot_style,
                    title_prefix=f"{count_title_map[component_name].capitalize()} across groups",
                    ylabel=count_ylabel_map[component_name],
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                    starting_day=phase_start_day,
                )
            for group_name in group_names:
                plot_phase4_reversal_components(
                    reversal_group_summaries,
                    group_name=group_name,
                    phase_display_name=phase_display_names[4],
                    bin_hours=bin_hours,
                    output_path=output_dir / f"phase4_reversal_corner_components_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    x_end_hours=phase_group_end_hours.get((group_name, 4)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                    starting_day=phase_start_day,
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
        output_path=output_dir / "phase_activity_median_visits_per_hour_boxplot.png",
    )

def render_phase_segment_plots(
    visits,
    output_dir: Path,
    *,
    phase_display_names: dict[int, str],
    spread_metric: str,
    scheduled_phase_start_hours: dict[int, float],
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Render awake/sleep-segment learning trajectories for PL and PR across groups."""

    segment_metrics = [
        ("correct_corner_visit", "correct-corner visit rate", "Correct-corner visit rate [%]"),
        ("correct_np_visit", "correct NP visit rate", "Correct NP visit rate [%]"),
        ("rewarded_correct_corner_visit", "rewarded correct-corner visit rate", "Rewarded correct-corner visit rate [%]"),
    ]
    for phase_number in (3, 4):
        phase_origin_hour = phase_origin_clock_hour(mouse_day_start_hour, scheduled_phase_start_hours[phase_number])
        phase_start_day = experiment_day_from_scheduled_start(scheduled_phase_start_hours[phase_number])
        for success_col, title_label, ylabel in segment_metrics:
            mouse_table, summary = compute_phase_segment_rate_tables(
                visits,
                phase_number=phase_number,
                success_col=success_col,
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=mouse_day_start_hour,
                awake_end_clock_hour=awake_end_clock_hour,
                max_days=3,
            )
            metric_stub = success_col.replace("_visit", "")
            save_table(
                mouse_table,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_rate_mouse.tsv",
            )
            save_table(
                summary,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_rate_group_summary.tsv",
            )
            plot_phase_segment_rate_groups(
                summary,
                phase_number=phase_number,
                phase_display_name=phase_display_names[phase_number],
                title_label=title_label,
                ylabel=ylabel,
                output_path=output_dir / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_rate_all_groups.png",
                spread_metric=spread_metric,
                starting_day=phase_start_day,
                chance_level=25.0,
            )

def render_awake_day_violin_plots(
    visits,
    output_dir: Path,
    *,
    phase_display_names: dict[int, str],
    scheduled_phase_start_hours: dict[int, float],
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
    exclude_outliers: bool,
) -> None:
    """Render awake-only daily violin plots plus omnibus, pairwise, and chance statistics."""

    violin_metrics = [
        ("correct_corner_visit", "correct_corner", "correct-corner visit rate", "Correct-corner visit rate [%]", 25.0),
        ("correct_np_visit", "correct_np", "correct NP visit rate", "Correct NP visit rate [%]", 25.0),
        (
            "rewarded_correct_corner_visit",
            "rewarded_correct_corner",
            "rewarded correct-corner visit rate",
            "Rewarded correct-corner visit rate [%]",
            25.0,
        ),
    ]
    for phase_number in (3, 4):
        phase_origin_hour = phase_origin_clock_hour(mouse_day_start_hour, scheduled_phase_start_hours[phase_number])
        for success_col, metric_stub, title_label, ylabel, chance_level in violin_metrics:
            mouse_table, _ = compute_awake_day_rate_tables(
                visits,
                phase_number=phase_number,
                success_col=success_col,
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=mouse_day_start_hour,
                awake_end_clock_hour=awake_end_clock_hour,
                max_days=3,
            )
            mouse_table["PhaseNumber"] = phase_number
            mouse_table = flag_iqr_outliers(
                mouse_table,
                value_col="value",
                group_cols=["phase_day", "Group"],
            )
            omnibus, pairwise, chance = compute_group_day_violin_statistics(
                mouse_table,
                phase_number=phase_number,
                metric_name=metric_stub,
                chance_level=chance_level / 100.0,
                exclude_outliers=exclude_outliers,
            )
            save_table(
                mouse_table,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_day_rate_mouse.tsv",
            )
            save_table(
                omnibus,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_day_rate_omnibus_stats.tsv",
            )
            save_table(
                pairwise,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_day_rate_pairwise_stats.tsv",
            )
            save_table(
                chance,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_day_rate_chance_stats.tsv",
            )
            for phase_day in (1, 2, 3):
                plot_group_day_violin(
                    mouse_table,
                    phase_number=phase_number,
                    phase_display_name=phase_display_names[phase_number],
                    phase_day=phase_day,
                    metric_title=title_label,
                    ylabel=ylabel,
                    pairwise_stats=pairwise,
                    chance_stats=chance,
                    output_path=output_dir / f"phase{phase_number}_{metric_stub}_awake_day{phase_day}_violin.png",
                    outlier_col="is_outlier",
                )

def render_cumulative_role_plots(
    visits,
    output_dir: Path,
    *,
    group_names: list[str],
    plot_style: str,
    phase_display_names: dict[int, str],
    spread_metric: str,
    scheduled_phase_start_hours: dict[int, float],
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Render event-based cumulative and relative cumulative PL-to-PR corner-role plots."""

    pre_phase_hours = 24.0
    mouse_counts, summary = compute_role_cumulative_curves(visits, pre_phase_hours=pre_phase_hours)
    if mouse_counts.empty or summary.empty:
        return

    save_table(mouse_counts, output_dir / "pl_pr_cumulative_corner_roles_mouse_events.tsv")
    save_table(summary, output_dir / "pl_pr_cumulative_corner_roles_group_summary.tsv")

    phase_window_table = pd.DataFrame(
        [
            {"PhaseNumber": 2, "start_hours": -pre_phase_hours, "end_hours": 0.0},
            {"PhaseNumber": 3, "start_hours": 0.0, "end_hours": 72.0},
            {"PhaseNumber": 4, "start_hours": 72.0, "end_hours": 144.0},
        ]
    )
    origin_clock_hour = phase_origin_clock_hour(mouse_day_start_hour, scheduled_phase_start_hours[3])
    aligned_x_start = -pre_phase_hours + (mouse_day_start_hour - origin_clock_hour)
    cumulative_start_day = max(0, experiment_day_from_scheduled_start(scheduled_phase_start_hours[3]) - 1)
    pl_curve_mouse, _, pl_onset = compute_time_window_learning_curves(
        visits,
        phase_number=3,
        success_col="correct_corner_visit",
    )
    pr_curve_mouse, _, pr_onset = compute_time_window_learning_curves(
        visits,
        phase_number=4,
        success_col="correct_corner_visit",
    )
    absolute_summary = summary.rename(
        columns={
            "mean_value_absolute": "mean_value",
            "sem_value_absolute": "sem_value",
            "std_value_absolute": "std_value",
        }
    )
    relative_summary = summary.rename(
        columns={
            "mean_value_relative": "mean_value",
            "sem_value_relative": "sem_value",
            "std_value_relative": "std_value",
        }
    )
    relative_summary["mean_value_rate"] = relative_summary["mean_value"] * 100.0
    relative_summary["sem_value_rate"] = relative_summary["sem_value"] * 100.0
    relative_summary["std_value_rate"] = relative_summary["std_value"] * 100.0

    for group_name in group_names:
        onset_points: list[dict[str, float | str]] = []
        pl_onset_group = pl_onset.loc[pl_onset["Group"].astype(str).eq(group_name), "onset_hours"].dropna()
        pr_onset_group = pr_onset.loc[pr_onset["Group"].astype(str).eq(group_name), "onset_hours"].dropna()
        if not pl_onset_group.empty:
            onset_points.append(
                {
                    "corner_role": "PL target corner",
                    "x_hours": float(pl_onset_group.median()),
                }
            )
        if not pr_onset_group.empty:
            onset_points.append(
                {
                    "corner_role": "PR target corner",
                    "x_hours": 72.0 + float(pr_onset_group.median()),
                }
            )
        plot_cumulative_role_curves(
            absolute_summary,
            group_name=group_name,
            output_path=output_dir / f"pl_pr_cumulative_corner_roles_absolute_{sanitize_filename_part(group_name)}.png",
            title_label="cumulative corner-role visits from late NPA to PR",
            ylabel="Cumulative visits per mouse",
            value_col="mean_value",
            spread_col=f"{spread_metric}_value",
            plot_style=plot_style,
            phase_window_table=phase_window_table,
            phase_display_names={2: phase_display_names[2], 3: phase_display_names[3], 4: phase_display_names[4]},
            origin_clock_hour=origin_clock_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
            x_start_hours=aligned_x_start,
            onset_points=onset_points,
            starting_day=cumulative_start_day,
        )
        plot_cumulative_role_curves(
            relative_summary,
            group_name=group_name,
            output_path=output_dir / f"pl_pr_cumulative_corner_roles_relative_{sanitize_filename_part(group_name)}.png",
            title_label="relative cumulative corner-role visits from late NPA to PR",
            ylabel="Relative cumulative visits [%]",
            value_col="mean_value_rate",
            spread_col="sem_value_rate" if spread_metric == "sem" else "std_value_rate",
            plot_style=plot_style,
            phase_window_table=phase_window_table,
            phase_display_names={2: phase_display_names[2], 3: phase_display_names[3], 4: phase_display_names[4]},
            origin_clock_hour=origin_clock_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
            x_start_hours=aligned_x_start,
            onset_points=onset_points,
            starting_day=cumulative_start_day,
        )

def render_experience_learning_plots(
    visits,
    output_dir: Path,
    *,
    phase_display_names: dict[int, str],
    spread_metric: str,
    exclude_outliers: bool,
) -> None:
    """Render experience-dependent learning curves and onset violins for PL and PR."""

    learning_metrics = [
        ("correct_corner_visit", "correct_corner", "correct-corner learning by visit number"),
        ("correct_np_visit", "correct_np", "correct NP learning by visit number"),
        ("rewarded_correct_corner_visit", "rewarded_correct_corner", "rewarded correct-corner learning by visit number"),
    ]
    for phase_number in (3, 4):
        for success_col, metric_stub, title_label in learning_metrics:
            curve_mouse, curve_summary, onset_visits = compute_visit_window_learning_curves(
                visits,
                phase_number=phase_number,
                success_col=success_col,
            )
            save_table(
                curve_mouse,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_mouse.tsv",
            )
            save_table(
                curve_summary,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_group_summary.tsv",
            )
            save_table(
                onset_visits,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_onset.tsv",
            )
            onset_visits = flag_iqr_outliers(
                onset_visits,
                value_col="onset_visit",
                group_cols=["Group"],
            )
            save_table(
                onset_visits,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_onset_with_outliers.tsv",
            )
            onset_omnibus, onset_pairwise = compute_onset_group_statistics(
                onset_visits,
                onset_col="onset_visit",
                phase_number=phase_number,
                metric_name=metric_stub,
                exclude_outliers=exclude_outliers,
            )
            save_table(
                onset_omnibus,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_onset_omnibus_stats.tsv",
            )
            save_table(
                onset_pairwise,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_onset_pairwise_stats.tsv",
            )
            plot_visit_learning_curve_groups(
                curve_summary,
                phase_display_name=phase_display_names[phase_number],
                title_label=title_label,
                ylabel="Success probability [%]",
                output_path=output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_all_groups.png",
                spread_metric=spread_metric,
            )
            plot_onset_violin(
                onset_visits,
                onset_col="onset_visit",
                phase_display_name=phase_display_names[phase_number],
                title_label=f"{title_label} onset",
                ylabel="Learning onset [visit number]",
                output_path=output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_onset_violin.png",
                pairwise_stats=onset_pairwise,
                outlier_col="is_outlier",
            )

            time_curve_mouse, time_curve_summary, onset_hours = compute_time_window_learning_curves(
                visits,
                phase_number=phase_number,
                success_col=success_col,
            )
            save_table(
                time_curve_mouse,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_mouse.tsv",
            )
            save_table(
                time_curve_summary,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_group_summary.tsv",
            )
            save_table(
                onset_hours,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_onset.tsv",
            )
            onset_hours = flag_iqr_outliers(
                onset_hours,
                value_col="onset_hours",
                group_cols=["Group"],
            )
            save_table(
                onset_hours,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_onset_with_outliers.tsv",
            )
            onset_hour_omnibus, onset_hour_pairwise = compute_onset_group_statistics(
                onset_hours,
                onset_col="onset_hours",
                phase_number=phase_number,
                metric_name=metric_stub,
                exclude_outliers=exclude_outliers,
            )
            save_table(
                onset_hour_omnibus,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_onset_omnibus_stats.tsv",
            )
            save_table(
                onset_hour_pairwise,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_onset_pairwise_stats.tsv",
            )
            plot_onset_violin(
                onset_hours,
                onset_col="onset_hours",
                phase_display_name=phase_display_names[phase_number],
                title_label=f"{title_label} onset",
                ylabel="Learning onset [hours]",
                output_path=output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_onset_violin.png",
                pairwise_stats=onset_hour_pairwise,
                outlier_col="is_outlier",
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
    phase_name_map: dict[str, int] | None = None,
    optional_phase_names: list[str] | tuple[str, ...] | set[str] | None = None,
    drop_unmatched_visits: bool = False,
    excluded_groups: list[str] | None = None,
    group_renames: dict[str, str] | None = None,
    group_colors: dict[str, str] | None = None,
    figure_size_cm: dict[str, tuple[float, float]] | None = None,
    mouse_day_start_hour: float = DEFAULT_MOUSE_DAY_START_HOUR,
    awake_duration_hours: float = DEFAULT_AWAKE_DURATION_HOURS,
    experiment_day0_start_hour: float | None = None,
    schedule_anchor_phase_number: int | None = None,
    scheduled_phase_start_hours: dict[int, float] | None = None,
    base_font_size: float = 10.0,
    exclude_violin_outliers: bool = True,
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
    selected_group_colors = resolved_group_colors(
        group_renames=selected_group_renames,
        group_colors=group_colors,
    )
    configure_plot_style(font_size=base_font_size, font_family="Arial")
    set_group_colors(selected_group_colors)
    set_figure_size_presets(figure_size_cm)
    awake_start_clock_hour, awake_end_clock_hour = active_period_bounds(
        mouse_day_start_hour,
        awake_duration_hours,
    )
    output_root = resolve_output_root(dataset_root, results_subdir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading cohort data from {dataset_root}...")
    cohort = load_cohort_data(
        dataset_root,
        phase_name_map=phase_name_map,
        optional_phase_names=optional_phase_names,
        drop_unmatched_visits=drop_unmatched_visits,
    )
    aligned_visits = attach_analysis_time_columns(
        cohort.visits,
        cohort.phase_manifest,
        scheduled_phase_start_hours=merged_scheduled_phase_starts,
        mouse_day_start_hour=mouse_day_start_hour,
        experiment_day0_start_hour=experiment_day0_start_hour,
        schedule_anchor_phase_number=schedule_anchor_phase_number,
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
                    "experiment_day0_start_hour",
                    "schedule_anchor_phase_number",
                    "phase2_plot_style",
                        "exclude_groups",
                        "phase_name_map",
                        "optional_phase_names",
                        "drop_unmatched_visits",
                    "group_rename_mapping",
                    "group_color_mapping",
                    "figure_size_cm",
                    "base_font_size",
                    "exclude_violin_outliers",
                ],
                "Value": [
                    mouse_day_start_hour,
                    awake_duration_hours,
                        "" if experiment_day0_start_hour is None else experiment_day0_start_hour,
                        "" if schedule_anchor_phase_number is None else schedule_anchor_phase_number,
                        phase2_plot_style,
                        ",".join(selected_excluded_groups) if selected_excluded_groups else "",
                        ";".join(f"{key}={value}" for key, value in sorted((phase_name_map or {}).items())),
                        ",".join(sorted(str(name) for name in (optional_phase_names or []))),
                        drop_unmatched_visits,
                        ";".join(f"{key}={value}" for key, value in selected_group_renames.items()),
                        ";".join(f"{key}={value}" for key, value in selected_group_colors.items()),
                        ";".join(f"{key}={value}" for key, value in (figure_size_cm or {}).items()),
                        base_font_size,
                        exclude_violin_outliers,
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
    print(f"done. Output root directory: {output_root}")
    
    print("Filtering visits by phase time limits...")
    filtered_visits = filter_visits_by_phase_limits(selected_visits, merged_phase_limits)
    filtered_visits.to_csv(csv_output_path(output_root / "merged_visits.tsv.gz"), sep="\t", index=False, compression="gzip")
    selected_nosepokes.to_csv(csv_output_path(output_root / "merged_nosepokes.tsv.gz"), sep="\t", index=False, compression="gzip")
    print("done.")

    print("Rendering analysis plots...")
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
    print(f"done. Rendered plots will be saved to {output_root} in subdirectories by binning parameter.")
    
    print("Rendering phase-aligned learning curves and visit counts across groups and binning parameters...")
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
            phase_start_day=experiment_day_from_scheduled_start(merged_scheduled_phase_starts[2]),
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
    print("Rendering phase activity, segment, awake-day, experience-learning, and cumulative role plots...")
    render_phase_activity_plot(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
    )
    print("done. Rendering awake/sleep segment plots...")
    render_phase_segment_plots(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        spread_metric=spread_metric,
        scheduled_phase_start_hours=merged_scheduled_phase_starts,
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )
    print("done. Rendering awake-day violin plots...")
    render_awake_day_violin_plots(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        scheduled_phase_start_hours=merged_scheduled_phase_starts,
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        exclude_outliers=exclude_violin_outliers,
    )
    print("done. Rendering experience-learning plots...")
    render_experience_learning_plots(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        spread_metric=spread_metric,
        exclude_outliers=exclude_violin_outliers,
    )
    print("done. Rendering cumulative role plots...")
    render_cumulative_role_plots(
        filtered_visits,
        output_root,
        group_names=group_names,
        plot_style=plot_style,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        spread_metric=spread_metric,
        scheduled_phase_start_hours=merged_scheduled_phase_starts,
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )
    print("done. All analysis plots rendered.")
    return output_root
# %% MAIN FUNCTION
def main() -> None:
    """Run the full 4-month place-learning workflow from in-script settings."""

    run_analysis(
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
    )

# %% ENTRY POINT
if __name__ == "__main__":
    main()
# %% END
