from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ic_placelearning.metrics import (
    build_analysis_phase_window_table,
    compute_awake_day_rate_tables,
    compute_experiment_drinking_visit_bins,
    compute_experiment_visit_bins,
    compute_group_day_violin_statistics,
    compute_onset_group_statistics,
    compute_phase2_adaptation_bins,
    compute_phase4_reversal_rate_bins,
    compute_phase_activity_medians,
    compute_phase_activity_statistics,
    compute_phase_segment_rate_tables,
    compute_phase_visit_count_bins,
    compute_place_learning_count_bins,
    compute_place_learning_rate_bins,
    compute_role_cumulative_curves,
    compute_time_window_learning_curves,
    compute_visit_window_learning_curves,
    flag_iqr_outliers)
from ic_placelearning.plotting import (
    configure_plot_style,
    plot_cumulative_role_curves,
    plot_experiment_dual_metric_bars,
    plot_experiment_dual_metric_groups,
    plot_experiment_overview,
    plot_experiment_overview_groups,
    plot_group_day_violin,
    plot_onset_violin,
    plot_phase2_adaptation,
    plot_phase2_adaptation_groups,
    plot_phase4_reversal_components,
    plot_phase_activity_boxplot,
    plot_phase_learning_counts,
    plot_phase_learning_counts_groups,
    plot_phase_learning_rate,
    plot_phase_learning_rate_groups,
    plot_phase_segment_rate_groups,
    plot_visit_learning_curve_groups,
    sanitize_filename_part,
    set_group_colors,
    set_figure_size_presets)

SCHEDULED_PHASE_START_HOURS = {
    1: 0.0,
    2: 74.0,
    3: 122.0,
    4: 194.0,
    5: 266.0}
PHASE_DISPLAY_NAMES = {
    1: "Free Hab",
    2: "NPA",
    3: "PL",
    4: "PR"}
GROUP_NAMES = ["Group A", "Group B"]

def assert_written(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0

def test_plotting_public_functions_smoke(aligned_visits, tmp_path: Path) -> None:
    configure_plot_style(font_size=8.0, font_family="DejaVu Sans")
    set_group_colors(
        {
            "Group A": "#267d8f",
            "Group B": "#c7523f"})
    set_figure_size_presets(
        {
            "LONG_FIGSIZE_CM": (10.0, 5.0),
            "LONG_FIGSIZE_2_CM": (10.0, 5.0),
            "PHASE2_FIGSIZE_CM": (8.0, 5.0),
            "MEDIUM_FIGSIZE_CM": (8.0, 5.0),
            "MEDIUM_WIDE_FIGSIZE_CM": (9.0, 5.0),
            "SEGMENT_FIGSIZE_CM": (9.0, 5.0),
            "VIOLIN_FIGSIZE_CM": (5.0, 5.0),
            "ONSET_FIGSIZE_CM": (5.0, 5.0),
            "ACTIVITY_FIGSIZE_CM": (8.0, 5.0),
            "WIDE_GROUP_FIGSIZE_CM": (10.0, 5.0)})

    phase_windows = build_analysis_phase_window_table(
        aligned_visits,
        SCHEDULED_PHASE_START_HOURS)
    visit_mouse, visit_summary = compute_experiment_visit_bins(
        aligned_visits,
        bin_hours=12)
    drinking_mouse, drinking_summary = compute_experiment_drinking_visit_bins(
        aligned_visits,
        bin_hours=12)
    phase2 = compute_phase2_adaptation_bins(
        aligned_visits,
        bin_hours=6,
        secondary_metric="lick_positive_visits")
    count_mouse, count_summary = compute_place_learning_count_bins(
        aligned_visits,
        phase_number=3,
        bin_hours=6,
        success_col="rewarded_correct_corner_visit")
    rate_mouse, rate_summary = compute_place_learning_rate_bins(
        aligned_visits,
        phase_number=3,
        bin_hours=6,
        success_col="rewarded_correct_corner_visit")
    phase_visit_mouse, phase_visit_summary = compute_phase_visit_count_bins(
        aligned_visits,
        phase_number=3,
        bin_hours=6)
    reversal_rate_tables = compute_phase4_reversal_rate_bins(
        aligned_visits,
        bin_hours=6)
    reversal_summaries = {
        name: frames[1]
        for name, frames in reversal_rate_tables.items()}
    segment_mouse, segment_summary = compute_phase_segment_rate_tables(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0)
    awake_mouse, _ = compute_awake_day_rate_tables(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0)
    awake_mouse["PhaseNumber"] = 3
    flagged_awake = flag_iqr_outliers(
        awake_mouse,
        value_col="value",
        group_cols=["Group", "phase_day"])
    day_omnibus, day_pairwise, chance_stats = compute_group_day_violin_statistics(
        flagged_awake,
        phase_number=3,
        metric_name="rewarded_correct_corner",
        chance_level=0.25)
    time_curves, time_summary, time_onset = compute_time_window_learning_curves(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        window_hours=6.0,
        step_hours=6.0,
        min_visits=2,
        threshold=0.35,
        consecutive_windows=1)
    visit_curves, visit_summary_windows, visit_onset = compute_visit_window_learning_curves(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        window_visits=8,
        min_visits=8,
        threshold=0.35,
        consecutive_windows=1)
    onset_omnibus, onset_pairwise = compute_onset_group_statistics(
        time_onset,
        onset_col="onset_hours",
        phase_number=3,
        metric_name="time_window_onset")
    role_mouse, role_summary = compute_role_cumulative_curves(aligned_visits)
    activity = compute_phase_activity_medians(
        aligned_visits,
        hourly_bin_size=6)
    activity_stats = compute_phase_activity_statistics(activity)

    assert sanitize_filename_part("Group A / 1") == "Group_A___1"
    assert not drinking_mouse.empty
    assert not phase_visit_mouse.empty
    assert not segment_mouse.empty
    assert not day_omnibus.empty
    assert not time_curves.empty
    assert not visit_curves.empty
    assert not visit_onset.empty
    assert not role_mouse.empty
    assert not onset_omnibus.empty

    plot_experiment_overview(
        visit_mouse,
        visit_summary,
        group_name="Group A",
        bin_hours=12,
        output_path=tmp_path / "overview_group.png",
        phase_window_table=phase_windows,
        phase_display_names=PHASE_DISPLAY_NAMES,
        spread_metric="sem",
        plot_style="line",
        show_individual_labels=False)
    plot_experiment_overview_groups(
        visit_summary,
        output_path=tmp_path / "overview_groups.png",
        phase_window_table=phase_windows,
        phase_display_names=PHASE_DISPLAY_NAMES,
        spread_metric="sem",
        plot_style="line")
    plot_experiment_dual_metric_bars(
        visit_summary,
        drinking_summary,
        group_name="Group A",
        bin_hours=12,
        output_path=tmp_path / "dual_bars.png",
        secondary_label="Drinking visits",
        phase_window_table=phase_windows,
        phase_display_names=PHASE_DISPLAY_NAMES,
        plot_style="bar",
        spread_metric="sem")
    plot_experiment_dual_metric_groups(
        visit_summary,
        drinking_summary,
        group_names=GROUP_NAMES,
        bin_hours=12,
        output_path=tmp_path / "dual_groups.png",
        secondary_label="Drinking visits",
        phase_window_table=phase_windows,
        phase_display_names=PHASE_DISPLAY_NAMES)
    plot_phase2_adaptation(
        phase2["visits"][1],
        phase2["drinking_metric"][1],
        group_name="Group A",
        bin_hours=6,
        output_path=tmp_path / "phase2_group.png",
        secondary_label="Drinking visits",
        phase_display_name="NPA",
        plot_style="line",
        spread_metric="sem")
    plot_phase2_adaptation_groups(
        phase2["visits"][1],
        phase2["drinking_metric"][1],
        group_names=GROUP_NAMES,
        bin_hours=6,
        output_path=tmp_path / "phase2_groups.png",
        secondary_label="Drinking visits",
        phase_display_name="NPA")
    plot_phase_learning_counts(
        count_mouse,
        count_summary,
        group_name="Group A",
        phase_number=3,
        phase_display_name="PL",
        bin_hours=6,
        output_path=tmp_path / "count_group.png",
        spread_metric="sem",
        title_label="Rewarded visits",
        ylabel="Rewarded visits",
        plot_style="line")
    plot_phase_learning_counts_groups(
        count_summary,
        phase_display_name="PL",
        bin_hours=6,
        output_path=tmp_path / "count_groups.png",
        spread_metric="sem",
        plot_style="line")
    plot_phase_learning_rate(
        rate_mouse,
        rate_summary,
        group_name="Group A",
        phase_number=3,
        phase_display_name="PL",
        bin_hours=6,
        output_path=tmp_path / "rate_group.png",
        spread_metric="sem",
        title_label="Rewarded correct-corner rate",
        ylabel="Rewarded correct-corner rate [%]",
        chance_level=25.0,
        plot_style="line")
    plot_phase_learning_rate_groups(
        rate_summary,
        phase_number=3,
        phase_display_name="PL",
        bin_hours=6,
        output_path=tmp_path / "rate_groups.png",
        spread_metric="sem",
        title_label="Rewarded correct-corner rate",
        ylabel="Rewarded correct-corner rate [%]",
        chance_level=25.0,
        plot_style="line")
    plot_phase4_reversal_components(
        reversal_summaries,
        group_name="Group B",
        phase_display_name="PR",
        bin_hours=6,
        output_path=tmp_path / "reversal_components.png",
        spread_metric="sem",
        plot_style="line")
    plot_phase_segment_rate_groups(
        segment_summary,
        phase_number=3,
        phase_display_name="PL",
        title_label="Rewarded correct-corner rate",
        ylabel="Rewarded correct-corner rate [%]",
        output_path=tmp_path / "segment_groups.png",
        spread_metric="sem",
        chance_level=25.0)
    plot_group_day_violin(
        flagged_awake,
        phase_number=3,
        phase_display_name="PL",
        phase_day=1,
        metric_title="Rewarded correct-corner rate",
        ylabel="Rewarded correct-corner rate [%]",
        pairwise_stats=day_pairwise,
        chance_stats=chance_stats,
        output_path=tmp_path / "day_violin.png")
    plot_cumulative_role_curves(
        role_summary,
        group_name="Group A",
        output_path=tmp_path / "role_curves.png",
        title_label="Cumulative corner roles",
        ylabel="Cumulative visits",
        value_col="mean_value_absolute",
        spread_col="sem_value_absolute",
        plot_style="line",
        phase_window_table=phase_windows,
        phase_display_names=PHASE_DISPLAY_NAMES,
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0)
    plot_visit_learning_curve_groups(
        visit_summary_windows,
        phase_display_name="PL",
        title_label="Experience learning",
        ylabel="Rewarded correct-corner rate [%]",
        output_path=tmp_path / "visit_learning.png",
        spread_metric="sem")
    plot_onset_violin(
        time_onset,
        onset_col="onset_hours",
        phase_display_name="PL",
        title_label="Learning onset",
        ylabel="Onset [hours]",
        output_path=tmp_path / "onset_violin.png",
        pairwise_stats=onset_pairwise)
    plot_phase_activity_boxplot(
        activity,
        activity_stats,
        phase_display_names=PHASE_DISPLAY_NAMES,
        output_path=tmp_path / "activity.png")

    for path in tmp_path.glob("*.png"):
        assert_written(path)
