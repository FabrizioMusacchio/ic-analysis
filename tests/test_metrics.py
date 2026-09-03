from __future__ import annotations

import numpy as np
import pandas as pd

from ic_analysis.metrics import (
    build_analysis_phase_window_table,
    build_phase_time_limit_table,
    compute_awake_day_error_rate_tables,
    compute_awake_day_rate_tables,
    compute_awake_day_ratio_tables,
    compute_binomial_glm_group_statistics,
    compute_clustered_binomial_gee_group_statistics,
    compute_bottle_preference_bins,
    compute_experiment_drinking_visit_bins,
    compute_experiment_lick_count_bins,
    compute_experiment_nosepoke_count_bins,
    compute_experiment_visit_bins,
    compute_first_hours_rate_table,
    compute_group_day_violin_statistics,
    compute_onset_group_statistics,
    compute_phase2_adaptation_bins,
    compute_phase4_reversal_count_bins,
    compute_phase4_reversal_rate_bins,
    compute_phase_activity_medians,
    compute_phase_activity_statistics,
    compute_phase_segment_error_rate_tables,
    compute_phase_segment_rate_tables,
    compute_phase_visit_count_bins,
    compute_place_learning_count_bins,
    compute_place_learning_rate_bins,
    compute_responder_group_statistics,
    compute_role_cumulative_curves,
    compute_threshold_responder_table,
    compute_time_window_learning_curves,
    compute_visit_window_learning_curves,
    filter_by_dayphase,
    filter_visits_by_phase_limits,
    flag_iqr_outliers,
    infer_phase_boundaries,
    suggest_common_phase_limits)

SCHEDULED_PHASE_START_HOURS = {
    1: 0.0,
    2: 74.0,
    3: 122.0,
    4: 194.0,
    5: 266.0}

def test_phase_boundary_helpers(synthetic_cohort, aligned_visits) -> None:
    inferred = infer_phase_boundaries(synthetic_cohort.phase_manifest)
    assert inferred[1] == 0.0
    assert inferred[3] > inferred[2]

    limit_table = build_phase_time_limit_table(synthetic_cohort.phase_manifest)
    assert set(limit_table["PhaseNumber"]) == {1, 2, 3, 4}
    assert suggest_common_phase_limits(synthetic_cohort.phase_manifest)[4] > 60.0

    phase_windows = build_analysis_phase_window_table(
        aligned_visits,
        SCHEDULED_PHASE_START_HOURS)
    assert phase_windows["duration_hours"].min() > 0

def test_binned_count_and_rate_metrics(aligned_visits) -> None:
    filtered = filter_visits_by_phase_limits(
        aligned_visits,
        {3: 48.0, 4: 48.0})
    assert filtered["analysis_phase_elapsed_hours"].max() <= aligned_visits["analysis_phase_elapsed_hours"].max()

    for function in (
        compute_experiment_visit_bins,
        compute_experiment_drinking_visit_bins,
        compute_experiment_nosepoke_count_bins,
        compute_experiment_lick_count_bins):
        mouse_bins, summary_bins = function(filtered, bin_hours=6)
        assert not mouse_bins.empty
        assert not summary_bins.empty
        assert summary_bins["mouse_n"].max() >= 1

    phase2 = compute_phase2_adaptation_bins(
        filtered,
        bin_hours=6,
        secondary_metric="lick_count")
    assert {"visits", "drinking_metric", "lick_positive_visits", "lick_count"} == set(phase2)
    assert not phase2["drinking_metric"][1].empty

    count_mouse, count_summary = compute_place_learning_count_bins(
        filtered,
        phase_number=3,
        bin_hours=6,
        success_col="rewarded_correct_corner_visit")
    rate_mouse, rate_summary = compute_place_learning_rate_bins(
        filtered,
        phase_number=3,
        bin_hours=6,
        success_col="rewarded_correct_corner_visit")
    all_mouse, all_summary = compute_phase_visit_count_bins(
        filtered,
        phase_number=3,
        bin_hours=6)
    assert count_mouse["value"].sum() > 0
    assert rate_summary["mean_value"].between(0, 1).all()
    assert all_summary["mean_value"].max() > count_summary["mean_value"].max()

def test_dayphase_filter_uses_configured_active_period(aligned_visits) -> None:
    day_visits = filter_by_dayphase(
        aligned_visits,
        dayphase="day",
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0)
    night_visits = filter_by_dayphase(
        aligned_visits,
        dayphase="night",
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0)
    all_visits = filter_by_dayphase(
        aligned_visits,
        dayphase="all",
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0)

    assert len(day_visits) + len(night_visits) == len(all_visits)
    assert day_visits["Start"].dt.hour.between(6, 17).all()
    assert not night_visits["Start"].dt.hour.between(6, 17).all()

def test_bottle_preference_bins_capture_synthetic_group_difference(synthetic_cohort, aligned_visits) -> None:
    keys = ["RunGroup", "Phase", "PhaseNumber", "VisitID"]
    visit_timing = aligned_visits.loc[
        :,
        keys + [
            "AnimalID",
            "Group",
            "ET",
            "ETLabel",
            "SEX",
            "AnalysisPhaseNumber",
            "analysis_experiment_elapsed_hours"]].drop_duplicates(subset=keys)
    nosepokes = synthetic_cohort.nosepokes.merge(
        visit_timing,
        on=keys,
        how="inner",
        validate="many_to_one")

    mouse_bins, summary = compute_bottle_preference_bins(
        nosepokes,
        phases="all",
        bin_h=24,
        left_sides=(1, 3, 5, 7),
        right_sides=(2, 4, 6, 8),
        calc="right_bottle/left_bottle")
    group_means = summary.groupby("Group", observed=True)["mean_value"].mean()

    assert not mouse_bins.empty
    assert not summary.empty
    assert group_means["Group A"] > group_means["Group B"]
    assert group_means["Group A"] > 0.6
    assert group_means["Group B"] < 0.5

    left_mouse_bins, left_summary = compute_bottle_preference_bins(
        nosepokes,
        phases=(3, 4),
        bin_h=7 * 24,
        calc="left_bottle")
    assert left_mouse_bins["left_bottle_consumption"].sum() >= left_mouse_bins["value"].sum()
    assert left_summary["mean_value"].notna().any()

def test_reversal_segment_and_awake_day_metrics(aligned_visits) -> None:
    rate_tables = compute_phase4_reversal_rate_bins(
        aligned_visits,
        bin_hours=6)
    count_tables = compute_phase4_reversal_count_bins(
        aligned_visits,
        bin_hours=6)
    assert {"new_correct_corner", "previous_correct_corner"}.issubset(rate_tables)
    assert not rate_tables["new_correct_corner"][1].empty
    assert count_tables["previous_correct_corner"][0]["value"].sum() > 0

    segment_mouse, segment_summary = compute_phase_segment_rate_tables(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0)
    awake_mouse, awake_summary = compute_awake_day_rate_tables(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0)
    ratio_mouse, ratio_summary = compute_awake_day_ratio_tables(
        aligned_visits,
        phase_number=4,
        numerator_col="correct_corner_visit",
        denominator_col="previous_correct_corner_visit",
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0,
        pseudocount=0.5)
    assert set(segment_mouse["segment_name"]) == {"awake", "sleep"}
    assert set(awake_mouse["segment"]) == {"awake"}
    assert not segment_summary.empty
    assert not awake_summary.empty
    assert ratio_summary["mean_value"].notna().any()

    sleep_mouse, _ = compute_awake_day_rate_tables(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0,
        dayphase_segment="sleep")
    full_day_mouse, _ = compute_awake_day_rate_tables(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0,
        dayphase_segment="all")
    assert set(sleep_mouse["segment"]) == {"sleep"}
    assert set(full_day_mouse["segment"]) == {"all"}
    first_full_day = full_day_mouse.loc[full_day_mouse["phase_day"].eq(1)]
    first_segments = segment_mouse.loc[segment_mouse["segment_day"].eq(1)]
    assert first_full_day["all_visits"].sum() == first_segments["all_visits"].sum()
    assert first_full_day["correct_visits"].sum() == first_segments["correct_visits"].sum()

    segment_error_mouse, segment_error_summary = compute_phase_segment_error_rate_tables(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0)
    awake_error_mouse, awake_error_summary = compute_awake_day_error_rate_tables(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0,
        dayphase_segment="all")
    valid_segment_rows = segment_error_mouse["all_visits"].gt(0)
    valid_awake_rows = awake_error_mouse["all_visits"].gt(0)
    assert not segment_error_summary.empty
    assert not awake_error_summary.empty
    assert np.allclose(
        segment_error_mouse.loc[valid_segment_rows, "success_visits"] + segment_error_mouse.loc[valid_segment_rows, "error_visits"],
        segment_error_mouse.loc[valid_segment_rows, "all_visits"])
    assert np.allclose(
        segment_error_mouse.loc[valid_segment_rows, "value"],
        1.0 - segment_mouse.loc[valid_segment_rows, "value"])
    assert set(awake_error_mouse["segment"]) == {"all"}
    assert np.allclose(
        awake_error_mouse.loc[valid_awake_rows, "success_visits"] + awake_error_mouse.loc[valid_awake_rows, "error_visits"],
        awake_error_mouse.loc[valid_awake_rows, "all_visits"])

def test_onset_responder_and_statistical_metrics(aligned_visits) -> None:
    first_hours = compute_first_hours_rate_table(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        first_hours=24.0)
    assert set(first_hours["Group"].astype(str)) == {"Group A", "Group B"}

    glm_omnibus, glm_pairwise = compute_binomial_glm_group_statistics(
        first_hours,
        phase_number=3,
        metric_name="rewarded_correct_corner_first24h",
        success_col="correct_visits",
        total_col="all_visits")
    assert not glm_omnibus.empty
    assert not glm_pairwise.empty

    binned_mouse, _ = compute_place_learning_rate_bins(
        aligned_visits,
        phase_number=3,
        bin_hours=6,
        success_col="rewarded_correct_corner_visit")
    gee_omnibus, gee_pairwise = compute_clustered_binomial_gee_group_statistics(
        binned_mouse,
        phase_number=3,
        metric_name="rewarded_correct_corner_6h",
        success_col="correct_visits",
        total_col="all_visits")
    assert not gee_omnibus.empty
    assert not gee_pairwise.empty

    curves, summary, onset = compute_time_window_learning_curves(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        window_hours=6.0,
        step_hours=3.0,
        min_visits=2,
        threshold=0.35,
        consecutive_windows=2)
    visit_curves, visit_summary, visit_onset = compute_visit_window_learning_curves(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        window_visits=6,
        min_visits=6,
        threshold=0.35,
        consecutive_windows=2)
    assert not curves.empty
    assert not summary.empty
    assert not onset.empty
    assert not visit_curves.empty
    assert not visit_summary.empty
    assert not visit_onset.empty

    responders = compute_threshold_responder_table(
        onset,
        phase_number=3,
        threshold_pct=35.0,
        horizons_hours=(24.0, 48.0))
    responder_summary, responder_omnibus, responder_pairwise = compute_responder_group_statistics(
        responders,
        phase_number=3,
        metric_name="time_window_onset")
    assert not responder_summary.empty
    assert not responder_omnibus.empty
    assert not responder_pairwise.empty

    onset_omnibus, onset_pairwise = compute_onset_group_statistics(
        onset,
        onset_col="onset_hours",
        phase_number=3,
        metric_name="time_window_onset")
    assert not onset_omnibus.empty
    assert not onset_pairwise.empty

def test_group_day_activity_and_outlier_helpers(aligned_visits) -> None:
    awake_mouse, _ = compute_awake_day_rate_tables(
        aligned_visits,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0)
    awake_mouse["PhaseNumber"] = 3
    flagged = flag_iqr_outliers(
        awake_mouse,
        value_col="value",
        group_cols=["Group", "phase_day"])
    assert "is_outlier" in flagged.columns

    omnibus, pairwise, chance = compute_group_day_violin_statistics(
        flagged,
        phase_number=3,
        metric_name="rewarded_correct_corner",
        chance_level=0.25,
        exclude_outliers=True)
    assert not omnibus.empty
    assert not pairwise.empty
    assert not chance.empty

    mouse_counts, role_summary = compute_role_cumulative_curves(aligned_visits)
    assert not mouse_counts.empty
    assert not role_summary.empty

    activity = compute_phase_activity_medians(
        aligned_visits,
        hourly_bin_size=6)
    stats = compute_phase_activity_statistics(activity)
    assert set(activity["PhaseNumber"]) == {1, 2, 3, 4}
    assert not stats.empty

def test_empty_inputs_return_empty_tables(aligned_visits) -> None:
    empty = aligned_visits.loc[aligned_visits["AnalysisPhaseNumber"].eq(99)].copy()
    assert compute_phase_segment_rate_tables(
        empty,
        phase_number=3,
        success_col="rewarded_correct_corner_visit",
        origin_clock_hour=8.0,
        awake_start_clock_hour=6.0,
        awake_end_clock_hour=18.0)[0].empty
    assert compute_first_hours_rate_table(
        empty,
        phase_number=3,
        success_col="rewarded_correct_corner_visit").empty
    assert compute_threshold_responder_table(
        pd.DataFrame(),
        phase_number=3,
        threshold_pct=50.0).empty
    assert compute_responder_group_statistics(
        pd.DataFrame(),
        phase_number=3,
        metric_name="empty")[0].empty
    assert compute_binomial_glm_group_statistics(
        pd.DataFrame(),
        phase_number=3,
        metric_name="empty",
        success_col="correct_visits",
        total_col="all_visits")[0].empty
    assert compute_clustered_binomial_gee_group_statistics(
        pd.DataFrame(),
        phase_number=3,
        metric_name="empty",
        success_col="correct_visits",
        total_col="all_visits")[0].empty

def test_three_group_statistical_branches() -> None:
    rows = []
    for group_index, group_name in enumerate(["A", "B", "C"]):
        for mouse_index in range(5):
            rows.append(
                {
                    "PhaseNumber": 3,
                    "Metric": "demo",
                    "phase_day": 1,
                    "Group": group_name,
                    "ET": f"{group_name}{mouse_index}",
                    "ETLabel": f"{group_name}{mouse_index}",
                    "SEX": "male",
                    "value": 0.2 + group_index * 0.1 + mouse_index * 0.01,
                    "correct_visits": 2 + group_index,
                    "all_visits": 10})
    day_rates = pd.DataFrame(rows)
    day_rates["Group"] = pd.Categorical(day_rates["Group"], categories=["A", "B", "C"], ordered=True)
    omnibus, pairwise, chance = compute_group_day_violin_statistics(
        day_rates,
        phase_number=3,
        metric_name="demo",
        chance_level=0.25)
    assert omnibus.loc[0, "group_n"] == 3
    assert len(pairwise) == 3
    assert len(chance) == 3

    onset = day_rates.loc[:, ["Group", "ET", "ETLabel", "SEX"]].copy()
    onset["onset_hours"] = np.tile([5.0, 6.0, 7.0, 8.0, 9.0], 3) + np.repeat([0.0, 5.0, 10.0], 5)
    onset_omnibus, onset_pairwise = compute_onset_group_statistics(
        onset,
        onset_col="onset_hours",
        phase_number=3,
        metric_name="demo")
    assert onset_omnibus.loc[0, "group_n"] == 3
    assert len(onset_pairwise) == 3
