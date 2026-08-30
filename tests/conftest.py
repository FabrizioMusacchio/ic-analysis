from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from additional_scripts.generate_synthetic_group_ab_data import write_dataset
from ic_placelearning.loader import attach_analysis_time_columns
from ic_placelearning.loader import load_cohort_data

SCHEDULED_PHASE_START_HOURS = {
    1: 0.0,
    2: 74.0,
    3: 122.0,
    4: 194.0,
    5: 266.0}

@pytest.fixture(scope="session")
def synthetic_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ic_placelearning") / "synthetic_group_ab"
    write_dataset(
        root,
        mouse_count_per_group=6,
        random_seed=2701,
        overwrite=False)
    return root

@pytest.fixture(scope="session")
def synthetic_cohort(synthetic_dataset_root: Path):
    return load_cohort_data(synthetic_dataset_root)

@pytest.fixture(scope="session")
def aligned_visits(synthetic_cohort):
    return attach_analysis_time_columns(
        synthetic_cohort.visits,
        synthetic_cohort.phase_manifest,
        scheduled_phase_start_hours=SCHEDULED_PHASE_START_HOURS,
        mouse_day_start_hour=6.0)
