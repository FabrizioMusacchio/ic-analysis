"""Command-line entry point for the BioMedX 4-month IntelliCage workflow.

This wrapper keeps the full CLI available while the main user script remains a
normal Python script with editable in-file settings and a direct function call.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from user_scripts.analyze_4month_cohort import (
    DEFAULT_AWAKE_DURATION_HOURS,
    DEFAULT_DATASET_ROOT,
    DEFAULT_EXCLUDED_GROUPS,
    DEFAULT_GROUP_RENAMES,
    DEFAULT_MOUSE_DAY_START_HOUR,
    DEFAULT_PHASE2_PLOT_STYLE,
    parse_group_rename_mapping,
    parse_numeric_mapping,
    run_analysis,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the 4-month cohort analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-subdir", type=Path, default=Path("results"))
    parser.add_argument("--bin-hours", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--phase2-secondary-metric",
        choices=["lick_positive_visits", "lick_count"],
        default="lick_positive_visits",
    )
    parser.add_argument("--spread-metric", choices=["sem", "std"], default="sem")
    parser.add_argument("--plot-style", choices=["step", "line"], default="line")
    parser.add_argument("--phase-max-hours", nargs="*", default=[])
    parser.add_argument("--exclude-groups", nargs="*", default=None)
    parser.add_argument("--group-rename", nargs="*", default=None)
    parser.add_argument("--phase2-plot-style", choices=["bar", "line"], default=DEFAULT_PHASE2_PLOT_STYLE)
    parser.add_argument("--mouse-day-start-hour", type=float, default=DEFAULT_MOUSE_DAY_START_HOUR)
    parser.add_argument("--awake-duration-hours", type=float, default=DEFAULT_AWAKE_DURATION_HOURS)
    parser.add_argument("--scheduled-phase-start-hours", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    """Run the workflow from the command line."""

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


if __name__ == "__main__":
    main()
