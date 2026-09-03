from __future__ import annotations

from pathlib import Path

import pandas as pd

from additional_scripts.generate_synthetic_group_ab_data import write_dataset

def test_generator_writes_intellicage_export_shape(tmp_path: Path) -> None:
    dataset_root = tmp_path / "synthetic_group_ab"
    write_dataset(
        dataset_root,
        mouse_count_per_group=3,
        random_seed=11,
        overwrite=False)
    assert not (dataset_root / "GroupA" / "Mice.txt").exists()
    assert (dataset_root / "GroupB" / "Phase4" / "IntelliCage" / "Visits.txt").exists()

    manifest = pd.read_csv(dataset_root / "synthetic_dataset_manifest.tsv", sep="\t")
    group_a_nosepokes = pd.read_csv(dataset_root / "GroupA" / "Phase3" / "IntelliCage" / "Nosepokes.txt", sep="\t")
    group_b_nosepokes = pd.read_csv(dataset_root / "GroupB" / "Phase3" / "IntelliCage" / "Nosepokes.txt", sep="\t")
    group_a_saccharin_fraction = group_a_nosepokes["Bottle"].eq("saccharin").mean()
    group_b_saccharin_fraction = group_b_nosepokes["Bottle"].eq("saccharin").mean()

    assert manifest["MouseCount"].unique().tolist() == [3]
    assert set(manifest["Group"]) == {"Group A", "Group B"}
    assert set(group_a_nosepokes["Bottle"]).issubset({"plain water", "saccharin"})
    assert group_a_saccharin_fraction > group_b_saccharin_fraction
    assert group_a_saccharin_fraction > 0.6
    assert group_b_saccharin_fraction < 0.5
