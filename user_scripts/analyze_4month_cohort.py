"""Run the IntelliCage place-learning workflow for the BioMedX 4-month cohort.

This user-facing script is the main entry point for the current poster
workflow. It reads the four IntelliCage runs, merges metadata and behavior
tables, computes activity and place-learning metrics, and renders poster-ready
plots into a results directory that always lives inside the selected dataset
directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intellicage_place_learning.loader import load_cohort_data
from intellicage_place_learning.metrics import (
    build_phase_time_limit_table,
    build_phase_window_table,
    compute_experiment_drinking_visit_bins,
    compute_experiment_visit_bins,
    compute_phase2_adaptation_bins,
    compute_phase_activity_medians,
    compute_phase_activity_statistics,
    compute_phase_visit_count_bins,
    compute_place_learning_count_bins,
    compute_place_learning_rate_bins,
    filter_visits_by_phase_limits,
    infer_phase_boundaries,
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
    sanitize_filename_part,
)


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
    return parser.parse_args()


def parse_phase_limits(raw_limits: list[str]) -> dict[int, float]:
    """Parse `phase=max_hours` CLI strings into a numeric dictionary."""

    limits: dict[int, float] = {}
    for item in raw_limits:
        if "=" not in item:
            raise ValueError(f"Invalid phase limit '{item}'. Use the form PHASE=HOURS, e.g. 4=74.")
        phase_text, hour_text = item.split("=", 1)
        limits[int(phase_text)] = float(hour_text)
    return limits


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


def save_table(dataframe, output_path: Path) -> None:
    """Save a DataFrame as a tab-separated text file with parent creation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, sep="\t", index=False)


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
) -> None:
    """Create full-experiment visit-activity plots for every pathology group."""

    mouse_bins, summary_bins = compute_experiment_visit_bins(visits, bin_hours=bin_hours)
    save_table(mouse_bins, output_dir / f"overview_visits_mouse_bins_{bin_hours}h.tsv")
    save_table(summary_bins, output_dir / f"overview_visits_group_summary_{bin_hours}h.tsv")
    group_end_hours = (
        visits.groupby("Group", observed=True)["experiment_elapsed_hours"].max() + float(bin_hours)
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
        )

    plot_experiment_overview_groups(
        summary_bins,
        output_path=output_dir / "plots" / f"overview_all_phases_visits_all_groups_{bin_hours}h.png",
        phase_window_table=phase_window_table,
        phase_display_names=phase_display_names,
        spread_metric=spread_metric,
        plot_style=plot_style,
    )


def render_phase2_plots(
    visits,
    output_dir: Path,
    *,
    bin_hours: int,
    group_names: list[str],
    secondary_metric: str,
    phase_display_names: dict[int, str],
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
        )


def render_phase2_control_plots(
    visits,
    output_dir: Path,
    *,
    bin_hours: int,
    group_names: list[str],
    phase_window_table,
    phase_display_names: dict[int, str],
) -> None:
    """Create full-experiment control plots for visits versus drinking visits."""

    primary_mouse, primary_summary = compute_experiment_visit_bins(visits, bin_hours=bin_hours)
    drinking_mouse, drinking_summary = compute_experiment_drinking_visit_bins(visits, bin_hours=bin_hours)
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
) -> None:
    """Create phase-3 and phase-4 count and rate plots."""

    phase_group_end_hours = (
        visits.groupby(["Group", "PhaseNumber"], observed=True)["phase_elapsed_hours"].max() + float(bin_hours)
    ).astype(float).to_dict()
    phase_end_hours = (visits.groupby("PhaseNumber", observed=True)["phase_elapsed_hours"].max() + float(bin_hours)).astype(float).to_dict()

    for phase_number in (3, 4):
        phase_visit_mouse, phase_visit_summary = compute_phase_visit_count_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
        )
        strict_count_mouse, strict_count_summary = compute_place_learning_count_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
            strict=True,
        )
        strict_rate_mouse, strict_rate_summary = compute_place_learning_rate_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
            strict=True,
        )
        matlab_rate_mouse, matlab_rate_summary = compute_place_learning_rate_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
            strict=False,
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
            strict_count_mouse,
            output_dir / f"phase{phase_number}_correct_rewarded_visits_absolute_mouse_bins_{bin_hours}h.tsv",
        )
        save_table(
            strict_count_summary,
            output_dir / f"phase{phase_number}_correct_rewarded_visits_absolute_group_summary_{bin_hours}h.tsv",
        )
        save_table(
            strict_rate_mouse,
            output_dir / f"phase{phase_number}_correct_rewarded_visit_rate_mouse_bins_{bin_hours}h.tsv",
        )
        save_table(
            strict_rate_summary,
            output_dir / f"phase{phase_number}_correct_rewarded_visit_rate_group_summary_{bin_hours}h.tsv",
        )
        save_table(
            matlab_rate_mouse,
            output_dir / f"phase{phase_number}_matlab_placeerror_only_rate_mouse_bins_{bin_hours}h.tsv",
        )
        save_table(
            matlab_rate_summary,
            output_dir / f"phase{phase_number}_matlab_placeerror_only_rate_group_summary_{bin_hours}h.tsv",
        )

        for group_name in group_names:
            plot_phase_learning_counts(
                strict_count_mouse,
                strict_count_summary,
                group_name=group_name,
                phase_number=phase_number,
                phase_display_name=phase_display_names[phase_number],
                bin_hours=bin_hours,
                output_path=output_dir
                / "plots"
                / f"phase{phase_number}_correct_rewarded_visits_absolute_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                spread_metric=spread_metric,
                x_end_hours=phase_group_end_hours.get((group_name, phase_number)),
                plot_style=plot_style,
            )
            plot_phase_learning_rate(
                strict_rate_mouse,
                strict_rate_summary,
                group_name=group_name,
                phase_number=phase_number,
                phase_display_name=phase_display_names[phase_number],
                bin_hours=bin_hours,
                output_path=output_dir
                / "plots"
                / f"phase{phase_number}_correct_rewarded_visit_rate_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                spread_metric=spread_metric,
                x_end_hours=phase_group_end_hours.get((group_name, phase_number)),
                plot_style=plot_style,
            )

        plot_phase_learning_rate_groups(
            strict_rate_summary,
            phase_number=phase_number,
            phase_display_name=phase_display_names[phase_number],
            bin_hours=bin_hours,
            output_path=output_dir
            / "plots"
            / f"phase{phase_number}_correct_rewarded_visit_rate_all_groups_{bin_hours}h.png",
            spread_metric=spread_metric,
            x_end_hours=phase_end_hours.get(phase_number),
            plot_style=plot_style,
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


def main() -> None:
    """Run the full 4-month place-learning workflow from raw data to plots."""

    args = parse_args()
    phase_limits = DEFAULT_PHASE_MAX_HOURS.copy()
    phase_limits.update(parse_phase_limits(args.phase_max_hours))
    dataset_root = args.dataset_root
    output_root = resolve_output_root(dataset_root, args.results_subdir)
    output_root.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort_data(dataset_root)
    limit_table = build_phase_time_limit_table(cohort.phase_manifest)
    save_table(cohort.metadata, output_root / "mouse_metadata.tsv")
    save_table(cohort.phase_manifest, output_root / "phase_manifest.tsv")
    save_table(limit_table, output_root / "phase_time_limit_recommendations.tsv")

    filtered_visits = filter_visits_by_phase_limits(cohort.visits, phase_limits)
    filtered_visits.to_csv(output_root / "merged_visits.tsv.gz", sep="\t", index=False, compression="gzip")
    cohort.nosepokes.to_csv(output_root / "merged_nosepokes.tsv.gz", sep="\t", index=False, compression="gzip")

    group_names = ordered_group_names(filtered_visits)
    phase_boundaries = infer_phase_boundaries(cohort.phase_manifest)
    phase_window_table = build_phase_window_table(filtered_visits, phase_boundaries)
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

    for bin_hours in sorted(set(args.bin_hours)):
        bin_dir = output_root / f"{bin_hours}h_bins"
        render_overview_plots(
            filtered_visits,
            bin_dir,
            bin_hours=bin_hours,
            group_names=group_names,
            phase_window_table=phase_window_table,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
            spread_metric=args.spread_metric,
            plot_style=args.plot_style,
        )
        render_phase2_plots(
            filtered_visits,
            bin_dir,
            bin_hours=bin_hours,
            group_names=group_names,
            secondary_metric=args.phase2_secondary_metric,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        )
        render_phase2_control_plots(
            filtered_visits,
            bin_dir,
            bin_hours=bin_hours,
            group_names=group_names,
            phase_window_table=phase_window_table,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        )
        render_phase_learning_plots(
            filtered_visits,
            bin_dir,
            bin_hours=bin_hours,
            group_names=group_names,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
            spread_metric=args.spread_metric,
            plot_style=args.plot_style,
        )

    render_phase_activity_plot(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
    )


if __name__ == "__main__":
    main()
