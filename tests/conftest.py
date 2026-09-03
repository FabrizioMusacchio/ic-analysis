from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from additional_scripts.generate_synthetic_group_ab_data import write_dataset
from ic_analysis.loader import attach_analysis_time_columns
from ic_analysis.loader import load_cohort_data

SCHEDULED_PHASE_START_HOURS = {
    1: 0.0,
    2: 74.0,
    3: 122.0,
    4: 194.0,
    5: 266.0}

def synthetic_subject_metadata(dataset_root: Path) -> pd.DataFrame:
    """Build explicit test subject metadata from synthetic visit exports."""

    rows = []
    default_group_names = {
        "GroupA": "Group A",
        "GroupB": "Group B"}
    for run_group_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        run_group = run_group_dir.name
        group_name = default_group_names.get(run_group, run_group)
        phase_windows = {}
        for phase_number in range(1, 5):
            visits_path = run_group_dir / f"Phase{phase_number}" / "IntelliCage" / "Visits.txt"
            visits = pd.read_csv(visits_path, sep="\t")
            phase_windows[f"Phase{phase_number}Start"] = pd.to_datetime(visits["Start"]).min()
            phase_windows[f"Phase{phase_number}End"] = pd.to_datetime(visits["End"]).max()
        first_visits = pd.read_csv(run_group_dir / "Phase1" / "IntelliCage" / "Visits.txt", sep="\t")
        for mouse_index, animal_id in enumerate(sorted(first_visits["AnimalTag"].astype(str).unique())):
            rows.append({
                "AnimalID": animal_id,
                "DetectedRunGroup": run_group,
                "Group": group_name,
                "ET": f"{group_name[-1]}{mouse_index + 1:02d}",
                "ETLabel": f"{group_name[-1]}{mouse_index + 1:02d}",
                "SEX": "female" if mouse_index % 2 else "male",
                "DOB": pd.Timestamp("2025-09-01"),
                "age_months": 4.1,
                "CornerPhase3": int((mouse_index % 4) + 1),
                "CornerPhase4": int(((mouse_index + 2) % 4) + 1),
                **phase_windows})
    return pd.DataFrame(rows)

@pytest.fixture(scope="session")
def synthetic_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ic_analysis") / "synthetic_group_ab"
    write_dataset(
        root,
        mouse_count_per_group=6,
        random_seed=2701,
        overwrite=False)
    return root

@pytest.fixture(scope="session")
def synthetic_cohort(synthetic_dataset_root: Path):
    return load_cohort_data(
        synthetic_dataset_root,
        group_names=["Group A", "Group B"],
        subject_metadata=synthetic_subject_metadata(synthetic_dataset_root))

@pytest.fixture(scope="session")
def aligned_visits(synthetic_cohort):
    return attach_analysis_time_columns(
        synthetic_cohort.visits,
        synthetic_cohort.phase_manifest,
        scheduled_phase_start_hours=SCHEDULED_PHASE_START_HOURS,
        mouse_day_start_hour=6.0)
