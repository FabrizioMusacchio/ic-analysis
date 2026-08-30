from __future__ import annotations

from pathlib import Path

import pandas as pd

from additional_scripts.generate_synthetic_group_ab_data import write_dataset
from user_scripts.analyze_synthetic_group_ab import main as run_synthetic_analysis

def test_generator_writes_intellicage_export_shape(tmp_path: Path) -> None:
    dataset_root = tmp_path / "synthetic_group_ab"
    write_dataset(
        dataset_root,
        mouse_count_per_group=3,
        random_seed=11,
        overwrite=False)
    assert (dataset_root / "GroupA" / "Mice.txt").exists()
    assert (dataset_root / "GroupB" / "Phase4" / "IntelliCage" / "Visits.txt").exists()

    manifest = pd.read_csv(dataset_root / "synthetic_dataset_manifest.tsv", sep="\t")
    assert manifest["MouseCount"].unique().tolist() == [3]
    assert set(manifest["Group"]) == {"Group A", "Group B"}

def test_public_demo_script_writes_compact_results(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "synthetic_group_ab"
    write_dataset(
        dataset_root,
        mouse_count_per_group=4,
        random_seed=22,
        overwrite=False)

    import user_scripts.analyze_synthetic_group_ab as script

    monkeypatch.setattr(script, "USER_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(script, "USER_RESULTS_SUBDIR", Path("demo_results"))
    run_synthetic_analysis()

    results = dataset_root / "demo_results"
    assert (results / "mouse_metadata.tsv").exists()
    assert (results / "phase3_rewarded_correct_corner_rate_summary_2h.tsv").exists()
    assert (results / "phase4_reversal_corner_components_Group_A_2h.png").exists()
