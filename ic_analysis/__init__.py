"""IntelliCage Analysis Toolkit.

The package provides reusable utilities to read IntelliCage exports, define
script-level experiment and subject metadata, compute behavior metrics, and
create publication-oriented plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import metrics as metrics
from .experiment import IntelliCageExperiment as Experiment
from .loader import CohortData, load_cohort_data
from .metadata import ExperimentMetadata, PhaseMetadata, SubjectMetadata, SubjectRegistry
from .subjects import create_subjects_yaml_template, load_subjects_yaml

metric = metrics

__all__ = [
    "CohortData",
    "Experiment",
    "ExperimentMetadata",
    "PhaseMetadata",
    "SubjectMetadata",
    "SubjectRegistry",
    "create_subjects_yaml_template",
    "experiment",
    "load_subjects_yaml",
    "load_cohort_data",
    "metric",
    "metrics"]
__version__ = "0.1.0"

def experiment(
    *,
    EXPERIMENT: ExperimentMetadata | dict[str, Any],
    PHASES: dict[int | str, PhaseMetadata | dict[str, Any]] | None = None,
    SUBJECTS: SubjectRegistry | dict[str | int, SubjectMetadata | dict[str, Any]] | None = None) -> Experiment:
    """Build a generic IntelliCage experiment object from user-script metadata.

    This is the recommended public entry point after ``import ic_analysis as
    ic``. It validates experiment-level metadata, phase definitions, and the
    subject registry, creates the configured results folder, and returns an
    :class:`ic_analysis.experiment.IntelliCageExperiment` object ready for
    ``load()``, ``prepare_analysis()``, and modular plotting calls.

    :param EXPERIMENT: Experiment metadata as an
        :class:`ExperimentMetadata` instance or a dictionary. Required keys for
        dictionaries are ``name``, ``root_data_path``, ``results_data_path``,
        and ``group_names``. The optional ``mouse_day`` dictionary can define
        ``{"start": "06:00", "end": "18:00"}``.
    :param PHASES: Phase definitions as a mapping from phase number to
        :class:`PhaseMetadata` or dictionary. Default is ``None`` when phases
        are embedded in ``EXPERIMENT``. Each phase can define ``short_name``,
        ``long_name``, ``folder_name``, ``color``, and
        ``scheduled_start_hour``.
    :param SUBJECTS: Subject definitions as a
        :class:`SubjectRegistry` or dictionary. No default is allowed: the
        toolkit deliberately analyzes only explicitly declared animals.
    :returns: A configured :class:`ic_analysis.experiment.IntelliCageExperiment`.
    :raises ValueError: If phases, groups, or subjects are missing or invalid.
    """

    experiment_metadata = _coerce_experiment_metadata(EXPERIMENT, PHASES)
    if SUBJECTS is None:
        raise ValueError("`SUBJECTS` must define the animals included in the analysis.")
    if isinstance(SUBJECTS, SubjectRegistry):
        subjects = SUBJECTS
    else:
        subjects = SubjectRegistry.from_mapping(SUBJECTS)
    experiment_metadata.results_data_path.mkdir(parents=True, exist_ok=True)
    return Experiment(experiment_metadata, subjects)

def _coerce_experiment_metadata(
    experiment_value: ExperimentMetadata | dict[str, Any],
    phases_value: dict[int | str, PhaseMetadata | dict[str, Any]] | None) -> ExperimentMetadata:
    """Normalize public ``ic.experiment`` metadata inputs."""

    if isinstance(experiment_value, ExperimentMetadata):
        return experiment_value
    data = dict(experiment_value)
    phases = phases_value or data.pop("phases", None) or data.pop("PHASES", None)
    if phases is None:
        raise ValueError("`PHASES` must define at least one experiment phase.")
    mouse_day = data.pop("mouse_day", None) or data.pop("MouseDay", None)
    if isinstance(mouse_day, dict):
        if "start" in mouse_day and "mouse_day_start_time" not in data:
            data["mouse_day_start_time"] = mouse_day["start"]
        if "end" in mouse_day and "mouse_day_end_time" not in data:
            data["mouse_day_end_time"] = mouse_day["end"]
    normalized_phases = _normalize_phase_definitions(phases)
    phase_colors = data.get("phase_colors") or {
        int(number): phase.get("color")
        for number, phase in normalized_phases.items()
        if isinstance(phase, dict) and phase.get("color") is not None}
    group_names = data.get("group_names") or data.get("groups") or data.get("GROUPS")
    if group_names is None:
        raise ValueError("`EXPERIMENT` must define `group_names`.")
    return ExperimentMetadata(
        name=str(data.get("name") or data.get("experiment_name") or data.get("ExperimentName")),
        root_data_path=Path(data.get("root_data_path") or data.get("root") or data.get("data_path")),
        results_data_path=Path(data.get("results_data_path") or data.get("results") or data.get("output_path")),
        phases=normalized_phases,
        group_names=list(group_names),
        group_colors=data.get("group_colors") or {},
        phase_colors=phase_colors,
        optional_phase_numbers=set(data.get("optional_phase_numbers") or []),
        mouse_day_start_hour=float(data.get("mouse_day_start_hour", 6.0)),
        awake_duration_hours=float(data.get("awake_duration_hours", 12.0)),
        mouse_day_start_time=data.get("mouse_day_start_time"),
        mouse_day_end_time=data.get("mouse_day_end_time"),
        experiment_day0_start_hour=data.get("experiment_day0_start_hour"),
        schedule_anchor_phase_number=data.get("schedule_anchor_phase_number"))

def _normalize_phase_definitions(
    phases: dict[int | str, PhaseMetadata | dict[str, Any]]) -> dict[int, PhaseMetadata | dict[str, Any]]:
    """Normalize public phase definitions to integer keys."""

    normalized: dict[int, PhaseMetadata | dict[str, Any]] = {}
    for phase_number, phase in phases.items():
        normalized[int(phase_number)] = phase
    return normalized
