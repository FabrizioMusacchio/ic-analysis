from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd
import pytest

from additional_scripts.generate_synthetic_group_ab_data import write_dataset
from ic_placelearning.loader import attach_analysis_time_columns
from ic_placelearning.loader import load_cohort_data
from ic_placelearning.loader import read_mice_metadata
from ic_placelearning.loader import summarize_nosepokes_by_visit

SCHEDULED_PHASE_START_HOURS = {
    1: 0.0,
    2: 74.0,
    3: 122.0,
    4: 194.0,
    5: 266.0}

def test_load_synthetic_dataset_has_expected_structure(synthetic_cohort) -> None:
    assert synthetic_cohort.metadata["RFID"].nunique() == 12
    assert set(synthetic_cohort.metadata["Group"].astype(str)) == {"Group A", "Group B"}
    assert set(synthetic_cohort.phase_manifest["PhaseNumber"]) == {1, 2, 3, 4}
    assert {"rewarded_correct_corner_visit", "phase2_drinking_visit"}.issubset(
        synthetic_cohort.visits.columns)
    assert synthetic_cohort.nosepokes["VisitID"].notna().all()

def test_load_synthetic_dataset_preserves_group_difference(synthetic_cohort) -> None:
    phase3 = synthetic_cohort.visits.loc[synthetic_cohort.visits["PhaseNumber"].eq(3)]
    rates = phase3.groupby("Group", observed=True)["rewarded_correct_corner_visit"].mean()
    assert rates["Group A"] > rates["Group B"] + 0.15

def test_load_cohort_data_uses_generic_group_names_by_default(synthetic_dataset_root: Path) -> None:
    cohort = load_cohort_data(synthetic_dataset_root)
    assert list(cohort.metadata["Group"].cat.categories) == ["Group 1", "Group 2"]
    assert set(cohort.visits["Group"].astype(str)) == {"Group 1", "Group 2"}

def test_load_cohort_data_fills_partially_supplied_group_names(tmp_path: Path) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    combined_root = tmp_path / "combined"
    write_dataset(
        left_root,
        mouse_count_per_group=1,
        random_seed=1,
        overwrite=False)
    write_dataset(
        right_root,
        mouse_count_per_group=1,
        random_seed=2,
        overwrite=False)
    combined_root.mkdir()
    shutil.copytree(left_root / "GroupA", combined_root / "GroupA")
    shutil.copytree(left_root / "GroupB", combined_root / "GroupB")
    shutil.copytree(right_root / "GroupA", combined_root / "GroupC")
    shutil.copytree(right_root / "GroupB", combined_root / "GroupD")
    for index, run_group in enumerate(["GroupA", "GroupB", "GroupC", "GroupD"]):
        mice_path = combined_root / run_group / "Mice.txt"
        mice = pd.read_csv(mice_path, sep="\t")
        mice["VIRUS"] = f"Raw {index + 1}"
        mice["RFID"] = mice["RFID"] + index * 10000
        mice.to_csv(mice_path, sep="\t", index=False)
        for visits_path in (combined_root / run_group).glob("Phase*/IntelliCage/Visits.txt"):
            visits = pd.read_csv(visits_path, sep="\t")
            visits["AnimalTag"] = visits["AnimalTag"] + index * 10000
            visits.to_csv(visits_path, sep="\t", index=False)

    cohort = load_cohort_data(
        combined_root,
        group_names=["Control", "Control", "Treatment"])
    assert list(cohort.metadata["Group"].cat.categories) == [
        "Control",
        "Treatment",
        "Group 3",
        "Group 4"]
    assert set(cohort.visits["Group"].astype(str)) == {
        "Control",
        "Treatment",
        "Group 3",
        "Group 4"}

def test_attach_analysis_time_columns_assigns_protocol_windows(synthetic_cohort) -> None:
    visits = attach_analysis_time_columns(
        synthetic_cohort.visits,
        synthetic_cohort.phase_manifest,
        scheduled_phase_start_hours=SCHEDULED_PHASE_START_HOURS,
        mouse_day_start_hour=6.0,
        schedule_anchor_phase_number=2)
    assert visits["AnalysisPhaseNumber"].dropna().isin([1, 2, 3, 4]).all()
    assert visits.loc[visits["AnalysisPhaseNumber"].eq(3), "AnalysisAssignedCorner"].notna().all()
    phase4 = visits.loc[visits["AnalysisPhaseNumber"].eq(4)]
    assert phase4["previous_correct_corner_visit"].any()
    assert phase4["neutral_incorrect_corner_visit"].any()

def test_read_mice_metadata_accepts_legacy_corner_columns(tmp_path: Path) -> None:
    mice_path = tmp_path / "Mice.txt"
    pd.DataFrame(
        [
            {
                "RFID": 123,
                "SEX": "male",
                "DOB": "01.01.26",
                "ET": "1",
                "VIRUS": "Group A",
                "Corner Phase 1": 2,
                "Corner Phase 2": 4}]).to_csv(mice_path, sep="\t", index=False)
    metadata = read_mice_metadata(mice_path, "GroupA")
    assert metadata.loc[0, "ETLabel"] == "ET1"
    assert int(metadata.loc[0, "CornerPhase3"]) == 2
    assert int(metadata.loc[0, "CornerPhase4"]) == 4

def test_summarize_nosepokes_by_visit_handles_licks() -> None:
    nosepokes = pd.DataFrame(
        [
            {
                "RunGroup": "GroupA",
                "Phase": "Phase3",
                "PhaseNumber": 3,
                "VisitID": 1,
                "Side": 1,
                "LickNumber": 2,
                "LickDuration": 0.5,
                "ConditionError": 0},
            {
                "RunGroup": "GroupA",
                "Phase": "Phase3",
                "PhaseNumber": 3,
                "VisitID": 1,
                "Side": 3,
                "LickNumber": 0,
                "LickDuration": 0.0,
                "ConditionError": 1}])
    summary = summarize_nosepokes_by_visit(nosepokes)
    assert summary.loc[0, "nosepoke_event_count"] == 2
    assert summary.loc[0, "nosepoke_side_count"] == 2
    assert bool(summary.loc[0, "has_nosepoke_lick"])

def test_load_cohort_data_reports_missing_run_group(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No run-group directories"):
        load_cohort_data(tmp_path)
