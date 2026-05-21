"""Run the Python place-learning workflow for the BioMedX 4-month cohort.

This script is the main entry point for the first IntelliCage poster workflow.
It performs four tasks end-to-end:

1. Read and merge all four run groups of the 4-month cohort.
2. Save a harmonized visit-level analysis table plus metadata side tables.
3. Compute the main poster metrics with flexible hour-bin sizes.
4. Create group-wise plots for the full experiment, phase 2 adaptation, and
   phase 3/4 place-learning performance.

The script intentionally exposes a few command-line options so that the same
code can later be reused for alternative bin sizes or related cohorts without
editing the implementation itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from intellicage_place_learning.loader import load_cohort_data
from intellicage_place_learning.metrics import (
    compute_experiment_visit_bins,
    compute_phase2_adaptation_bins,
    compute_place_learning_bins,
    infer_phase_boundaries,
)
from intellicage_place_learning.plotting import (
    plot_experiment_overview,
    plot_phase2_adaptation,
    plot_phase_learning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Data IntelliCage" / "BioMedX_4MonthCohort_2019"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "python_scripts" / "results" / "BioMedX_4MonthCohort_2019"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the cohort analysis script.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with validated defaults that point to the
        local project layout.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the cohort directory that contains Gruppe1-4.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where merged tables and plots will be written.",
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
        help=(
            "Secondary phase-2 metric. The default focuses on successful "
            "drinking visits rather than raw lick counts."
        ),
    )
    return parser.parse_args()


def _ordered_group_names(visits) -> list[str]:
    """Extract pathology-group names in their categorical display order."""

    categories = getattr(visits["Group"].dtype, "categories", None)
    if categories is not None:
        return [str(category) for category in categories if str(category) != "nan"]
    return sorted(visits["Group"].dropna().astype(str).unique())


def _save_table(dataframe, output_path: Path) -> None:
    """Save a DataFrame as a tab-separated text file with parent creation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, sep="\t", index=False)


def _render_overview_plots(visits, output_dir: Path, bin_hours: int, group_names: list[str], phase_boundaries) -> None:
    """Create full-experiment visit-activity plots for every pathology group."""

    mouse_bins, summary_bins = compute_experiment_visit_bins(visits, bin_hours=bin_hours)
    _save_table(mouse_bins, output_dir / f"overview_mouse_bins_{bin_hours}h.tsv")
    _save_table(summary_bins, output_dir / f"overview_group_summary_{bin_hours}h.tsv")

    for group_name in group_names:
        plot_experiment_overview(
            mouse_bins,
            summary_bins,
            group_name=group_name,
            bin_hours=bin_hours,
            output_path=output_dir / "plots" / f"overview_{group_name.replace(' ', '_')}_{bin_hours}h.png",
            phase_boundaries=phase_boundaries,
        )


def _render_phase2_plots(
    visits,
    output_dir: Path,
    bin_hours: int,
    group_names: list[str],
    secondary_metric: str,
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

    _save_table(primary_mouse, output_dir / f"phase2_visits_mouse_bins_{bin_hours}h.tsv")
    _save_table(primary_summary, output_dir / f"phase2_visits_group_summary_{bin_hours}h.tsv")
    _save_table(secondary_mouse, output_dir / f"phase2_{secondary_metric}_mouse_bins_{bin_hours}h.tsv")
    _save_table(secondary_summary, output_dir / f"phase2_{secondary_metric}_group_summary_{bin_hours}h.tsv")
    _save_table(lick_positive_mouse, output_dir / f"phase2_lick_positive_visits_mouse_bins_{bin_hours}h.tsv")
    _save_table(lick_positive_summary, output_dir / f"phase2_lick_positive_visits_group_summary_{bin_hours}h.tsv")
    _save_table(lick_mouse, output_dir / f"phase2_lick_count_mouse_bins_{bin_hours}h.tsv")
    _save_table(lick_summary, output_dir / f"phase2_lick_count_group_summary_{bin_hours}h.tsv")

    secondary_label = (
        "Drinking visits"
        if secondary_metric == "lick_positive_visits"
        else "Lick count"
    )
    for group_name in group_names:
        plot_phase2_adaptation(
            primary_summary,
            secondary_summary,
            group_name=group_name,
            bin_hours=bin_hours,
            output_path=output_dir / "plots" / f"phase2_{group_name.replace(' ', '_')}_{bin_hours}h.png",
            secondary_label=secondary_label,
        )


def _render_place_learning_plots(visits, output_dir: Path, bin_hours: int, group_names: list[str]) -> None:
    """Create phase-3 and phase-4 place-learning summaries and plots."""

    for phase_number in (3, 4):
        strict_mouse, strict_summary = compute_place_learning_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
            strict=True,
        )
        matlab_mouse, matlab_summary = compute_place_learning_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
            strict=False,
        )

        _save_table(
            strict_mouse,
            output_dir / f"phase{phase_number}_strict_rewarded_mouse_bins_{bin_hours}h.tsv",
        )
        _save_table(
            strict_summary,
            output_dir / f"phase{phase_number}_strict_rewarded_group_summary_{bin_hours}h.tsv",
        )
        _save_table(
            matlab_mouse,
            output_dir / f"phase{phase_number}_matlab_placeerror_only_mouse_bins_{bin_hours}h.tsv",
        )
        _save_table(
            matlab_summary,
            output_dir / f"phase{phase_number}_matlab_placeerror_only_group_summary_{bin_hours}h.tsv",
        )

        ylabel = "Rewarded correct visits per mouse and bin"
        for group_name in group_names:
            plot_phase_learning(
                strict_mouse,
                strict_summary,
                group_name=group_name,
                phase_number=phase_number,
                bin_hours=bin_hours,
                output_path=output_dir
                / "plots"
                / f"phase{phase_number}_{group_name.replace(' ', '_')}_{bin_hours}h.png",
                ylabel=ylabel,
            )


def main() -> None:
    """Run the full 4-month place-learning workflow from raw data to plots."""

    args = parse_args()
    cohort = load_cohort_data(args.dataset_root)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    _save_table(cohort.metadata, output_root / "mouse_metadata.tsv")
    _save_table(cohort.phase_manifest, output_root / "phase_manifest.tsv")
    cohort.visits.to_csv(output_root / "merged_visits.tsv.gz", sep="\t", index=False, compression="gzip")
    cohort.nosepokes.to_csv(output_root / "merged_nosepokes.tsv.gz", sep="\t", index=False, compression="gzip")

    group_names = _ordered_group_names(cohort.visits)
    phase_boundaries = infer_phase_boundaries(cohort.phase_manifest)

    for bin_hours in sorted(set(args.bin_hours)):
        bin_dir = output_root / f"{bin_hours}h_bins"
        _render_overview_plots(cohort.visits, bin_dir, bin_hours, group_names, phase_boundaries)
        _render_phase2_plots(
            cohort.visits,
            bin_dir,
            bin_hours,
            group_names,
            args.phase2_secondary_metric,
        )
        _render_place_learning_plots(cohort.visits, bin_dir, bin_hours, group_names)


if __name__ == "__main__":
    main()
