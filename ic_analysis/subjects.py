"""Subject-template helpers for user-defined IntelliCage analyses.

author: Fabrizio Musacchio
date: May/August 2026
"""
# %% IMPORTS
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .metadata import ExperimentMetadata, PhaseMetadata
# %% HELPERS
def create_subjects_yaml_template(
    *,
    EXPERIMENT: ExperimentMetadata | dict[str, Any],
    PHASES: dict[int | str, PhaseMetadata | dict[str, Any]] | None = None,
    number_of_phases: int | None = None,
    output_path: Path | str | None = None) -> Path:
    """Create or update a subject YAML template from detected raw animal IDs.

    The helper scans ``Visits.txt`` files below the experiment data root,
    detects unique raw IntelliCage animal IDs, and writes editable subject
    entries to YAML. Existing entries are preserved; newly detected IDs are
    appended. IDs present in the YAML but absent from current raw data are
    reported but not deleted.

    :param EXPERIMENT: Experiment metadata object or dictionary. It must define
        ``root_data_path`` so raw exports can be scanned.
    :param PHASES: Optional phase mapping. Default is ``None`` when using
        ``number_of_phases``. If provided, phase ``folder_name`` values are
        used only to prefill phase windows when export-block folder names happen
        to match biological phase names.
    :param number_of_phases: Optional number of default ``PhaseN`` folders to
        scan. Default is ``None``. Use this only when no ``PHASES`` dictionary
        is available yet.
    :param output_path: Optional YAML path. Default is
        ``<root_data_path>/subjects.yaml``.
    :returns: The written YAML path.
    :raises ValueError: If neither ``PHASES`` nor ``number_of_phases`` is
        provided, or if an existing YAML file is not a subject mapping.
    """

    dataset_root = _experiment_root(EXPERIMENT)
    phase_map = _phase_name_map(PHASES=PHASES, number_of_phases=number_of_phases)
    target_path = Path(output_path) if output_path is not None else dataset_root / "subjects.yaml"
    detected_subjects = _detect_subjects(dataset_root, phase_map)
    existing_payload = _load_yaml_payload(target_path) if target_path.exists() else {}
    if "subjects" in existing_payload:
        existing_subjects = existing_payload["subjects"]
    else:
        existing_subjects = existing_payload
        existing_payload = {"subjects": existing_subjects}
    if not isinstance(existing_subjects, dict):
        raise ValueError("Subject YAML must contain a subject mapping.")
    existing_ids = {str(animal_id) for animal_id in existing_subjects}
    detected_ids = {str(animal_id) for animal_id in detected_subjects}

    missing_from_raw = sorted(existing_ids.difference(detected_ids))
    if missing_from_raw:
        print(
            "Subject YAML contains AnimalID values not currently present in the raw data: "
            + ", ".join(missing_from_raw))

    added_ids: list[str] = []
    for animal_id, subject in detected_subjects.items():
        if animal_id in existing_subjects:
            continue
        existing_subjects[animal_id] = subject
        added_ids.append(animal_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        yaml.safe_dump(existing_payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8")
    if added_ids:
        print(f"Added {len(added_ids)} detected subject(s) to {target_path}.")
    else:
        print(f"No new subjects detected. Subject YAML is up to date: {target_path}.")
    return target_path

def load_subjects_yaml(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load a user-edited subject YAML file as a ``SUBJECTS`` dictionary.

    The YAML may either be a direct subject mapping or contain a top-level
    ``subjects`` mapping. The returned dictionary can be passed directly to
    ``ic.experiment(EXPERIMENT=..., PHASES=..., SUBJECTS=...)``.

    :param path: Path to the subject YAML file.
    :returns: A ``dict`` keyed by raw IntelliCage animal ID.
    :raises ValueError: If the YAML file does not contain a subject mapping.
    """

    payload = _load_yaml_payload(Path(path))
    subjects = payload.get("subjects", payload)
    if not isinstance(subjects, dict):
        raise ValueError("Subject YAML must contain a mapping or a top-level `subjects` mapping.")
    return {
        str(animal_id): subject
        for animal_id, subject in subjects.items()}

def _experiment_root(experiment: ExperimentMetadata | dict[str, Any]) -> Path:
    """Return the raw data root from an experiment metadata object or dict."""

    if isinstance(experiment, ExperimentMetadata):
        return Path(experiment.root_data_path)
    value = experiment.get("root_data_path") or experiment.get("root") or experiment.get("data_path")
    if value is None:
        raise ValueError("`EXPERIMENT` must define `root_data_path` for subject auto-detection.")
    return Path(value)

def _phase_name_map(
    *,
    PHASES: dict[int | str, PhaseMetadata | dict[str, Any]] | None,
    number_of_phases: int | None) -> dict[str, int]:
    """Resolve raw phase-folder names to phase numbers."""

    if PHASES is None:
        if number_of_phases is None:
            raise ValueError("Provide `PHASES` or `number_of_phases` for subject template generation.")
        return {f"Phase{phase_number}": phase_number for phase_number in range(1, int(number_of_phases) + 1)}
    phase_map: dict[str, int] = {}
    for phase_number, phase in PHASES.items():
        key = int(phase_number)
        if isinstance(phase, PhaseMetadata):
            folder_name = phase.raw_folder_name
        else:
            folder_name = (
                phase.get("folder_name")
                or phase.get("folder")
                or phase.get("folderName")
                or f"Phase{key}")
        phase_map[str(folder_name)] = key
    return phase_map

def _detect_subjects(dataset_root: Path, phase_map: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Detect raw AnimalID values and prepare editable subject entries."""

    subjects: dict[str, dict[str, Any]] = {}
    for run_group_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        phase_windows = _detect_run_group_phase_windows(run_group_dir, phase_map)
        detected_ids = _detect_run_group_animal_ids(run_group_dir, phase_map)
        for animal_id in detected_ids:
            subjects[animal_id] = {
                "group": "",
                "sex": "",
                "true_id": "",
                "date_of_birth": "",
                "corner_assignments": {},
                "phases": {
                    phase_number: {
                        "time_window": phase_windows.get(phase_number, ["", ""])}
                    for phase_number in sorted(phase_map.values())}}
    return subjects

def _detect_run_group_animal_ids(run_group_dir: Path, phase_map: dict[str, int]) -> list[str]:
    """Detect unique animal IDs from raw visit exports in one run-group folder."""

    del phase_map
    animal_ids: set[str] = set()
    for visits_path in sorted(run_group_dir.rglob("IntelliCage/Visits.txt")):
        visits = pd.read_csv(visits_path, sep="\t", usecols=["AnimalTag"])
        animal_ids.update(visits["AnimalTag"].dropna().astype(str))
    return sorted(animal_ids)

def _detect_run_group_phase_windows(run_group_dir: Path, phase_map: dict[str, int]) -> dict[int, list[str]]:
    """Detect observed start/end timestamps for each raw phase folder."""

    windows: dict[int, list[str]] = {}
    for phase_folder, phase_number in phase_map.items():
        visits_path = run_group_dir / phase_folder / "IntelliCage" / "Visits.txt"
        if not visits_path.exists():
            continue
        visits = pd.read_csv(visits_path, sep="\t", usecols=["Start", "End"])
        visits["Start"] = pd.to_datetime(visits["Start"], errors="raise")
        visits["End"] = pd.to_datetime(visits["End"], errors="raise")
        if visits.empty:
            continue
        windows[int(phase_number)] = [
            str(visits["Start"].min()),
            str(visits["End"].max())]
    return windows

def _format_date(value: Any) -> str:
    """Format optional date values for YAML output."""

    if value is None or pd.isna(value):
        return ""
    return str(pd.to_datetime(value).date())

def _load_yaml_payload(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Subject YAML must contain a mapping: {path}")
    return payload
# %% END
