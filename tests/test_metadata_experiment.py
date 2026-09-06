from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ic_analysis as ic
from ic_analysis import ExperimentMetadata, PhaseMetadata, SubjectRegistry
from ic_analysis.loader import load_cohort_data

def _experiment(root: Path, results: Path) -> ExperimentMetadata:
    phases = {
        1: PhaseMetadata(number=1, short_name="Hab", long_name="Habituation"),
        2: PhaseMetadata(number=2, short_name="NPA", long_name="Nose-poke adaptation"),
        3: PhaseMetadata(number=3, short_name="PL", long_name="Place learning"),
        4: PhaseMetadata(number=4, short_name="PR", long_name="Place reversal")}
    return ExperimentMetadata(
        name="Synthetic test experiment",
        root_data_path=root,
        results_data_path=results,
        phases=phases,
        group_names=["Script Group A", "Script Group B"],
        group_colors={
            "Script Group A": "#267d8f",
            "Script Group B": "#c7523f"})

def _subjects() -> SubjectRegistry:
    group_a_windows = {
        1: ("2026-01-05 06:00:00", "2026-01-08 08:00:00"),
        2: ("2026-01-08 08:00:00", "2026-01-10 08:00:00"),
        3: ("2026-01-10 08:00:00", "2026-01-13 08:00:00"),
        4: ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}
    group_b_windows = {
        1: ("2026-01-17 06:00:00", "2026-01-20 08:00:00"),
        2: ("2026-01-20 08:00:00", "2026-01-22 08:00:00"),
        3: ("2026-01-22 08:00:00", "2026-01-25 08:00:00"),
        4: ("2026-01-25 08:00:00", "2026-01-28 08:00:00")}
    return SubjectRegistry.from_mapping({
        "910200000001000": {
            "group": "Script Group A",
            "sex": "male",
            "true_id": "A-script",
            "date_of_birth": "2025-09-01",
            "phases": {
                phase_number: {"time_window": window}
                for phase_number, window in group_a_windows.items()},
            "corner_assignments": {
                3: 1,
                4: 3}},
        "910200000002000": {
            "group": "Script Group B",
            "sex": "male",
            "true_id": "B-script",
            "date_of_birth": "2025-09-01",
            "phases": {
                phase_number: {"time_window": window}
                for phase_number, window in group_b_windows.items()},
            "corner_assignments": {
                3: 1,
                4: 3}}})

def test_subject_registry_builds_loader_frame_and_schedule(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    subjects = _subjects()

    frame = subjects.to_loader_frame(experiment)
    schedule = subjects.scheduled_phase_start_hours(experiment)

    assert set(frame["AnimalID"]) == {"910200000001000", "910200000002000"}
    assert schedule == {
        1: 0.0,
        2: 74.0,
        3: 122.0,
        4: 194.0,
        5: 266.0}

def test_loader_uses_subject_metadata_as_inclusion_policy(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    subject_frame = _subjects().to_loader_frame(experiment)

    cohort = load_cohort_data(
        synthetic_dataset_root,
        group_names=experiment.group_names,
        subject_metadata=subject_frame)

    assert set(cohort.metadata["ET"]) == {"A-script", "B-script"}
    assert set(cohort.visits["ET"]) == {"A-script", "B-script"}
    assert set(cohort.visits["Group"].astype(str)) == {"Script Group A", "Script Group B"}
    assert cohort.visits["AnimalID"].nunique() == 2

def test_loader_can_reject_unregistered_raw_animals(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    subject_frame = _subjects().to_loader_frame(experiment)

    with pytest.raises(ValueError, match="without subject metadata"):
        load_cohort_data(
            synthetic_dataset_root,
            group_names=experiment.group_names,
            subject_metadata=subject_frame,
            drop_unregistered_subjects=False)

def test_experiment_object_loads_and_computes_overview(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    loaded = my_exp.load()
    mouse_bins, summary = ic.metric.compute_experiment_visit_bins(my_exp.visits, bin_hours=2)

    assert loaded is None
    assert my_exp.visits is not None
    assert not mouse_bins.empty
    assert not summary.empty
    assert set(my_exp.group_names) == {"Script Group A", "Script Group B"}
    assert (my_exp.results_data_path / "csv" / "loaded_visits.tsv.gz").exists()
    assert (my_exp.results_data_path / "experiment.yaml").exists()

def test_experiment_object_plots_bottle_preference(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_bottle_preference(
        phases="all",
        left_bottle="plain water",
        right_bottle="saccharin",
        calc="right_bottle/left_bottle",
        bin_h=24,
        x_unit="days",
        indicate_dots=True,
        calc_stats=True,
        day_night_indicator=("aw", "sl"),
        base_font_size=8.0)

    assert output_dir.name == "24h_day_bins"
    assert (output_dir / "bottle_preference_right_bottle_over_left_bottle_all_phases_24h_all_groups.png").exists()
    assert (
        output_dir
        / "pdf"
        / "bottle_preference_right_bottle_over_left_bottle_all_phases_24h_all_groups.pdf").exists()
    assert (
        output_dir
        / "csv"
        / "bottle_preference_right_bottle_over_left_bottle_all_phases_24h_group_summary.tsv").exists()
    stats = pd.read_csv(
        output_dir / "csv" / "bottle_preference_right_bottle_over_left_bottle_all_phases_24h_binwise_group_tests.tsv",
        sep="\t")
    assert {"test", "p_value", "significance"}.issubset(stats.columns)
    assert stats["test"].isin(["Welch t-test", "Mann-Whitney U"]).any()

def test_phase_activity_summary_uses_general_output_folder(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_phase_activity_summary(
        dayphase="all",
        base_font_size=8.0,
        figsize_cm=(6.0, 5.0),
        median_line_width=0.8,
        median_marker_size=2.5,
        plot_layout={"legend": False})

    assert output_dir.name == "visit_activity_summary"
    assert (output_dir / "phase_activity_mean_visits_per_hour_boxplot.png").exists()
    assert (output_dir / "csv" / "phase_activity_mean_visits_per_hour_mouse.tsv").exists()

def test_bottle_preference_day_violin_outputs_endpoint(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_bottle_preference_day(
        experiment_day=5,
        phases="all",
        dayphase="day",
        calc="all",
        calc_stats=True,
        base_font_size=8.0,
        figsize_cm=(6.0, 5.0))

    assert output_dir.name == "bottle_preference_summary"
    assert (output_dir / "bottle_preference_all_all_phases_day5_awake_violin.png").exists()
    assert (output_dir / "csv" / "bottle_preference_all_all_phases_day5_awake_mouse.tsv").exists()
    stats = pd.read_csv(
        output_dir / "csv" / "bottle_preference_all_all_phases_day5_awake_group_test.tsv",
        sep="\t")
    assert {"test", "p_value", "significance"}.issubset(stats.columns)

def test_reversal_components_accept_day_night_indicator(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_plr_reversal_components(
        phase_number=4,
        bin_hours=6,
        dayphase="all",
        day_night_indicator=("aw", "sl"),
        base_font_size=8.0,
        figsize_cm=(8.0, 5.0),
        plot_layout={"legend": True})

    assert (output_dir / "phase4_reversal_corner_components_Script_Group_A_6h.png").exists()

def test_prepare_analysis_returns_none_and_keeps_state(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    prepared = my_exp.prepare_analysis(phase_max_hours={3: 24.0, 4: 24.0})

    assert prepared is None
    assert my_exp.analysis_visits is not None

def test_experiment_age_plot_supports_time_units(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    my_exp.plot_ages(
        time_unit="days",
        show_N=True,
        plot_layout={
            "ylim": (0.0, 300.0)},
        base_font_size=8.0,
        figsize_cm=(5.0, 5.0))

    assert (my_exp.results_data_path / "mouse_age_at_phase1_start_days_violin.png").exists()
    assert (my_exp.results_data_path / "csv" / "mouse_age_at_phase1_start_days_mouse.tsv").exists()

def test_np_control_plots_use_compact_selected_phase_axis(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_NP_counts(
        phases=(2, 4),
        bin_hours=12,
        dayphase="all",
        plot_layout={
            "legend": False},
        base_font_size=8.0,
        figsize_cm=(8.0, 5.0),
        day_night_indicator=("aw", "sl"))
    summary = pd.read_csv(
        output_dir / "csv" / "np_counts_phases_2_4_group_summary_12h.tsv",
        sep="\t")

    assert summary["bin_start_hours"].min() >= 72.0
    assert summary["bin_end_hours"].max() <= 276.0
    assert (output_dir / "np_counts_phases_2_4_all_groups_12h.png").exists()

def test_np_adaptation_phase_subset_aligns_staggered_cage_starts(
    synthetic_dataset_root: Path,
    tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_NP_adaptation(
        phases=2,
        bin_hours=1,
        dayphase="all",
        base_font_size=8.0,
        figsize_cm=(8.0, 5.0))
    summary = pd.read_csv(
        output_dir / "csv" / "np_adaptation_phase2_visits_vs_drinking_visits_visits_group_summary_1h.tsv",
        sep="\t")
    first_active_bins = (
        summary.loc[summary["mean_value"].gt(5.0)]
        .groupby("Group", observed=True)["bin_start_hours"]
        .min())

    assert first_active_bins["Script Group A"] == 74.0
    assert first_active_bins["Script Group B"] == 74.0

def test_mice_activity_phase_subset_uses_compact_phase_axis(
    synthetic_dataset_root: Path,
    tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_mice_activity(
        phases=2,
        bin_hours=1,
        dayphase="all",
        base_font_size=8.0,
        figsize_cm=(8.0, 5.0))
    summary = pd.read_csv(
        output_dir / "csv" / "mice_activity_phase2_group_summary_1h.tsv",
        sep="\t")
    first_active_bins = (
        summary.loc[summary["mean_value"].gt(5.0)]
        .groupby("Group", observed=True)["bin_start_hours"]
        .min())

    assert summary["bin_start_hours"].min() == 74.0
    assert summary["bin_end_hours"].max() <= 122.0
    assert first_active_bins["Script Group A"] == 74.0
    assert first_active_bins["Script Group B"] == 74.0

def test_phase2_subset_axis_preserves_experiment_clock_position(
    synthetic_dataset_root: Path,
    tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    selected_visits, selected_phases, _ = my_exp._selected_phase_visits(
        phases=2,
        dayphase="all")
    compact_visits, phase_windows, origin_clock_hour = my_exp._compact_phase_timeline(
        selected_visits,
        selected_phases=selected_phases,
        bin_hours=1)
    assert set(phase_windows["PhaseNumber"].astype(int)) == {2}
    assert phase_windows["start_hours"].tolist() == [74.0]
    assert phase_windows["end_hours"].tolist() == [122.0]
    assert origin_clock_hour == 6.0
    assert np.allclose(
        compact_visits.loc[selected_visits.index, "analysis_experiment_elapsed_hours"].astype(float),
        selected_visits["analysis_experiment_elapsed_hours"].astype(float))

def test_compact_phase1_axis_keeps_late_cage_start_gap(
    synthetic_dataset_root: Path,
    tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    selected_visits, selected_phases, _ = my_exp._selected_phase_visits(
        phases=1,
        dayphase="all")
    compact_visits, phase_windows, _ = my_exp._compact_phase_timeline(
        selected_visits,
        selected_phases=selected_phases,
        bin_hours=1)
    _, summary = ic.metric.compute_experiment_visit_bins(compact_visits, bin_hours=1)
    first_active_bins = (
        summary.loc[summary["mean_value"].gt(5.0)]
        .groupby("Group", observed=True)["bin_start_hours"]
        .min())

    assert set(phase_windows["PhaseNumber"].astype(int)) == {1}
    assert phase_windows["start_hours"].tolist() == [0.0]
    assert first_active_bins["Script Group A"] == 0.0
    assert 6.0 <= first_active_bins["Script Group B"] <= 8.0
    assert summary["bin_end_hours"].max() <= 83.0

def test_bottle_preference_phase_subset_uses_compact_phase_axis(
    synthetic_dataset_root: Path,
    tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_bottle_preference(
        phases=3,
        dayphase="all",
        calc="right_bottle/left_bottle",
        bin_h=24,
        base_font_size=8.0,
        figsize_cm=(8.0, 5.0))
    summary = pd.read_csv(
        output_dir / "csv" / "bottle_preference_right_bottle_over_left_bottle_phase3_24h_group_summary.tsv",
        sep="\t")
    selected_visits, selected_phases, _ = my_exp._selected_phase_visits(
        phases=3,
        dayphase="all")
    _, phase_windows, _ = my_exp._compact_phase_timeline(
        selected_visits,
        selected_phases=selected_phases,
        bin_hours=24)

    assert summary["bin_start_hours"].min() == 120.0
    assert summary["bin_end_hours"].max() <= 216.0
    assert set(summary["Group"].astype(str)) == {"Script Group A", "Script Group B"}
    assert set(phase_windows["PhaseNumber"].astype(int)) == {3}
    assert phase_windows["start_hours"].tolist() == [122.0]
    assert phase_windows["end_hours"].tolist() == [194.0]

def test_plr_learning_rate_uses_phase_relative_axis_for_late_cage_starts(
    synthetic_dataset_root: Path,
    tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_plr_learning_rate(
        phase_number=3,
        metric="correct_corner_visit",
        bin_hours=1,
        dayphase="all",
        base_font_size=8.0,
        figsize_cm=(8.0, 5.0),
        plot_layout={
            "legend": False})
    summary = pd.read_csv(
        output_dir / "csv" / "phase3_correct_corner_visit_rate_group_summary_1h.tsv",
        sep="\t")
    first_sampled_bins = (
        summary.loc[summary["contributing_mouse_n"].gt(0)]
        .groupby("Group", observed=True)["bin_start_hours"]
        .min())

    assert first_sampled_bins["Script Group A"] == 0.0
    assert first_sampled_bins["Script Group B"] == 0.0
    assert summary["bin_end_hours"].max() <= 73.0

def test_np_adaptation_all_phases_reuses_overview_experiment_axis(
    synthetic_dataset_root: Path,
    tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_mice_activity(
        phases="all",
        bin_hours=1,
        dayphase="all",
        base_font_size=8.0,
        figsize_cm=(8.0, 5.0))
    my_exp.plot_NP_adaptation(
        phases="all",
        bin_hours=1,
        dayphase="all",
        base_font_size=8.0,
        figsize_cm=(8.0, 5.0))
    overview = pd.read_csv(
        output_dir / "csv" / "overview_all_phases_visits_group_summary_1h.tsv",
        sep="\t")
    np_adaptation = pd.read_csv(
        output_dir / "csv" / "np_adaptation_all_phases_visits_vs_drinking_visits_visits_group_summary_1h.tsv",
        sep="\t")
    columns = ["Group", "bin_start_hours", "mean_value"]
    overview = overview.loc[:, columns].sort_values(columns[:2]).reset_index(drop=True)
    np_adaptation = np_adaptation.loc[:, columns].sort_values(columns[:2]).reset_index(drop=True)

    pd.testing.assert_frame_equal(overview, np_adaptation)

def test_plr_error_rate_plot_methods_write_outputs(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    segment_dir = my_exp.plot_plr_phase_segment_error_rate(
        phase_number=3,
        metric="correct_corner_visit",
        dayphase="all",
        base_font_size=8.0,
        figsize_cm=(8.0, 5.0),
        plot_layout={
            "legend": False})
    endpoint_dir = my_exp.plot_plr_awake_day_error_rate(
        phase_number=3,
        metric="correct_corner_visit",
        phase_day=1,
        dayphase="all",
        base_font_size=8.0,
        figsize_cm=(5.0, 5.0))

    assert (segment_dir / "phase3_correct_corner_error_awake_sleep_segment_error_rate_all_groups.png").exists()
    assert (segment_dir / "csv" / "phase3_correct_corner_error_awake_sleep_segment_error_rate_mouse.tsv").exists()
    assert (endpoint_dir / "phase3_correct_corner_error_full_day1_error_violin.png").exists()
    assert (endpoint_dir / "csv" / "phase3_correct_corner_error_full_day_error_rate_mouse.tsv").exists()

def test_derived_ratio_accepts_plot_layout(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_plr_derived_ratio(
        phase_number=4,
        numerator_col="correct_corner_visit",
        denominator_col="new_or_previous_correct_corner_visit",
        metric_name="reversal_preference_index",
        title="reversal preference index",
        ylabel="New / (new + previous)",
        phase_day=1,
        dayphase="all",
        value_scale=1.0,
        format_as_percent=False,
        reference_line=0.5,
        base_font_size=8.0,
        figsize_cm=(5.0, 5.0),
        plot_layout={
            "ylim": (0.0, 3.0)})

    assert (output_dir / "phase4_reversal_preference_index_full_day1_violin.png").exists()
    assert (output_dir / "csv" / "phase4_reversal_preference_index_full_day_rate_mouse_with_outliers.tsv").exists()

def test_cumulative_preferences_accept_day_night_indicator(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    my_exp = ic.experiment(EXPERIMENT=experiment, SUBJECTS=_subjects())

    my_exp.load()
    output_dir = my_exp.plot_plr_cumulative_preferences(
        phases=(2, 3),
        dayphase="day",
        phase_max_hours={
            3: 24.0,
            4: 24.0},
        day_night_indicator=("aw", "sl"),
        base_font_size=8.0,
        figsize_cm=(8.0, 5.0),
        plot_layout={
            "legend": False})

    assert (output_dir / "pl_pr_cumulative_corner_roles_relative_Script_Group_A.png").exists()

def test_experiment_factory_uses_subject_phase_time_windows(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    phases = {
        1: {
            "shortname": "Hab",
            "longname": "Habituation"},
        2: {
            "shortname": "NPA",
            "longname": "Nose-poke adaptation"},
        3: {
            "shortname": "PL",
            "longname": "Place learning"},
        4: {
            "shortname": "PR",
            "longname": "Place reversal"}}
    subjects = {
        "910200000001000": {
            "Group": "Script Group A",
            "sex": "male",
            "true_id": "A-script",
            "date_of_birth": "2025-09-01",
            "corner_assignments": {
                3: 1,
                4: 3},
            "phases": {
                1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
                2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")},
                3: {"time_window": ("2026-01-10 08:00:00", "2026-01-13 08:00:00")},
                4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}},
        "910200000002000": {
            "Group": "Script Group B",
            "sex": "male",
            "true_id": "B-script",
            "date_of_birth": "2025-09-01",
            "corner_assignments": {
                3: 1,
                4: 3},
            "phases": {
                1: {"time_window": ("2026-01-17 06:00:00", "2026-01-20 08:00:00")},
                2: {"time_window": ("2026-01-20 08:00:00", "2026-01-22 08:00:00")},
                3: {"time_window": ("2026-01-22 08:00:00", "2026-01-25 08:00:00")},
                4: {"time_window": ("2026-01-25 08:00:00", "2026-01-28 08:00:00")}}}}
    my_exp = ic.experiment(
        EXPERIMENT={
            "name": "Facade test",
            "root_data_path": synthetic_dataset_root,
            "results_data_path": tmp_path / "results",
            "group_names": ["Script Group A", "Script Group B"],
            "mouse_day": {
                "start": "06:00",
                "end": "18:00"}},
        PHASES=phases,
        SUBJECTS=subjects)

    my_exp.load()

    assert my_exp.experiment.awake_duration_hours == 12.0
    assert my_exp.phases[1].short_name == "Hab"
    assert set(my_exp.subjects.subjects) == {"910200000001000", "910200000002000"}

def test_global_phase_windows_are_rejected(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="per subject"):
        ic.experiment(
            EXPERIMENT={
                "name": "Bad global windows",
                "root_data_path": synthetic_dataset_root,
                "results_data_path": tmp_path / "results",
                "group_names": ["Script Group A"]},
            PHASES={
                1: {
                    "short_name": "Hab",
                    "long_name": "Habituation",
                    "window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")}},
            SUBJECTS={})

def test_subject_yaml_template_and_loader(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    target_path = tmp_path / "subjects.yaml"

    written = ic.create_subjects_yaml_template(
        EXPERIMENT={
            "root_data_path": synthetic_dataset_root},
        number_of_phases=4,
        output_path=target_path)
    subjects = ic.load_subjects_yaml(written)

    assert target_path.exists()
    assert "910200000001000" in subjects
    assert "phases" in subjects["910200000001000"]
    assert subjects["910200000002000"]["phases"][1]["time_window"][0].startswith("2026-01-17")

def test_subject_registry_validates_missing_phase(synthetic_dataset_root: Path, tmp_path: Path) -> None:
    experiment = _experiment(synthetic_dataset_root, tmp_path / "results")
    subjects = SubjectRegistry.from_mapping({
        "910200000001000": {
            "group": "Script Group A",
            "sex": "male",
            "true_id": "A-script",
            "phases": {
                1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")}},
            "corner_assignments": {
                3: 1,
                4: 3}}})

    with pytest.raises(ValueError, match="missing phase windows"):
        subjects.validate_for_experiment(experiment)
