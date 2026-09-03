"""Experiment-definition classes for script-defined IntelliCage analyses.

The toolkit treats metadata declared in user scripts as the authoritative
analysis layer. Raw IntelliCage exports provide event tables; experiment and
subject definitions describe which phases exist, which animals should be
included, and how groups, time windows, and task-specific assignments should be
interpreted.
"""
# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# %% TYPE ALIASES
DateLike = str | datetime | pd.Timestamp
PhaseWindow = tuple[DateLike, DateLike]

# %% HELPERS
def _parse_datetime(value: DateLike) -> pd.Timestamp:
    """Parse a user-supplied date/time value."""

    if isinstance(value, pd.Timestamp):
        return value
    return pd.to_datetime(value, errors="raise")

def _normalize_animal_id(value: str | int) -> str:
    """Convert an IntelliCage animal identifier into a stable string key."""

    return str(value).strip()

def parse_clock_time_to_hour(value: str | float | int) -> float:
    """Convert ``HH:MM`` or decimal hour values into a floating clock hour."""

    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    hour_text, minute_text = text.split(":", 1)
    return int(hour_text) + int(minute_text) / 60.0

def _normalize_phase_mapping(phase: dict[str, Any]) -> dict[str, Any]:
    """Accept concise user-script phase keys and normalize them."""

    aliases = {
        "shortname": "short_name",
        "shortName": "short_name",
        "longname": "long_name",
        "longName": "long_name",
        "folder": "folder_name",
        "folderName": "folder_name"}
    normalized = dict(phase)
    forbidden_windows = {"window", "phase_window", "phaseWindow", "time_window", "begin_end", "start_end"}
    if forbidden_windows.intersection(normalized):
        raise ValueError("Phase time windows must be defined per subject under `SUBJECTS[animal_id]['phases']`.")
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized.pop(source)
    return normalized

def _normalize_subject_mapping(subject: dict[str, Any]) -> dict[str, Any]:
    """Accept concise user-script subject keys and normalize them."""

    aliases = {
        "Group": "group",
        "SEX": "sex",
        "Sex": "sex",
        "TrueID": "true_id",
        "true id": "true_id",
        "trueID": "true_id",
        "ET": "true_id",
        "age": "age_months",
        "AgeMonths": "age_months",
        "DOB": "date_of_birth",
        "dateOfBirth": "date_of_birth",
        "phaseWindows": "phase_windows",
        "PhaseWindows": "phase_windows",
        "phases": "phase_windows",
        "corners": "corner_assignments",
        "cornerAssignments": "corner_assignments"}
    normalized = dict(subject)
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized.pop(source)
    if "phase_windows" in normalized and normalized["phase_windows"] is not None:
        normalized["phase_windows"] = _normalize_subject_phase_windows(normalized["phase_windows"])
    if "corner_assignments" in normalized and normalized["corner_assignments"] is not None:
        normalized["corner_assignments"] = {
            int(phase_number): int(corner)
            for phase_number, corner in normalized["corner_assignments"].items()}
    return normalized

def _normalize_subject_phase_windows(phases: dict[int | str, Any]) -> dict[int, PhaseWindow]:
    """Normalize subject-specific phase windows from compact user mappings."""

    normalized: dict[int, PhaseWindow] = {}
    for phase_number, phase_value in phases.items():
        if isinstance(phase_value, dict):
            window = (
                phase_value.get("time_window")
                or phase_value.get("begin_end")
                or phase_value.get("start_end")
                or phase_value.get("window")
                or phase_value.get("phase_window"))
        else:
            window = phase_value
        if window is None or len(window) != 2:
            raise ValueError(f"Subject phase {phase_number} must define a two-value `time_window`.")
        normalized[int(phase_number)] = (window[0], window[1])
    return normalized

# %% DATA CLASSES
@dataclass(frozen=True)
class PhaseMetadata:
    """Description of one experiment phase."""

    number: int
    short_name: str
    long_name: str
    folder_name: str | None = None
    color: str | None = None
    scheduled_start_hour: float | None = None

    @property
    def raw_folder_name(self) -> str:
        """Return the IntelliCage export folder expected for this phase."""

        return self.folder_name or f"Phase{int(self.number)}"

@dataclass(frozen=True)
class ExperimentMetadata:
    """Experiment-level metadata defined by the user script."""

    name: str
    root_data_path: Path | str
    results_data_path: Path | str
    phases: dict[int, PhaseMetadata | dict[str, Any]]
    group_names: list[str] | tuple[str, ...]
    group_colors: dict[str, str] = field(default_factory=dict)
    phase_colors: dict[int, str] = field(default_factory=dict)
    optional_phase_numbers: set[int] = field(default_factory=set)
    mouse_day_start_hour: float = 6.0
    awake_duration_hours: float = 12.0
    mouse_day_start_time: str | None = None
    mouse_day_end_time: str | None = None
    experiment_day0_start_hour: float | None = None
    schedule_anchor_phase_number: int | None = None

    def __post_init__(self) -> None:
        """Normalize path and phase values after dataclass initialization."""

        object.__setattr__(self, "root_data_path", Path(self.root_data_path))
        object.__setattr__(self, "results_data_path", Path(self.results_data_path))
        normalized_phases: dict[int, PhaseMetadata] = {}
        for phase_number, phase in self.phases.items():
            key = int(phase_number)
            if isinstance(phase, PhaseMetadata):
                normalized = phase
            else:
                normalized = PhaseMetadata(number=key, **_normalize_phase_mapping(phase))
            if int(normalized.number) != key:
                normalized = PhaseMetadata(
                    number=key,
                    short_name=normalized.short_name,
                    long_name=normalized.long_name,
                    folder_name=normalized.folder_name,
                    color=normalized.color,
                    scheduled_start_hour=normalized.scheduled_start_hour)
            normalized_phases[key] = normalized
        if not normalized_phases:
            raise ValueError("ExperimentMetadata requires at least one phase.")
        object.__setattr__(self, "phases", dict(sorted(normalized_phases.items())))
        object.__setattr__(self, "group_names", [str(group) for group in self.group_names])
        if self.mouse_day_start_time is not None:
            object.__setattr__(self, "mouse_day_start_hour", parse_clock_time_to_hour(self.mouse_day_start_time))
        if self.mouse_day_end_time is not None:
            end_hour = parse_clock_time_to_hour(self.mouse_day_end_time)
            start_hour = float(self.mouse_day_start_hour)
            if end_hour <= start_hour:
                end_hour += 24.0
            object.__setattr__(self, "awake_duration_hours", end_hour - start_hour)

    @property
    def phase_name_map(self) -> dict[str, int]:
        """Return a loader-compatible mapping from folder names to phase numbers."""

        return {phase.raw_folder_name: number for number, phase in self.phases.items()}

    @property
    def optional_phase_names(self) -> set[str]:
        """Return optional phase folder names for the loader."""

        return {
            self.phases[int(phase_number)].raw_folder_name
            for phase_number in self.optional_phase_numbers
            if int(phase_number) in self.phases}

    @property
    def phase_display_names(self) -> dict[int, str]:
        """Return short phase labels for figures."""

        return {number: phase.short_name for number, phase in self.phases.items()}

    def explicit_scheduled_phase_start_hours(self) -> dict[int, float]:
        """Return scheduled start hours defined directly on phases."""

        return {
            number: float(phase.scheduled_start_hour)
            for number, phase in self.phases.items()
            if phase.scheduled_start_hour is not None}

@dataclass(frozen=True)
class SubjectMetadata:
    """Subject-level metadata declared in a user script."""

    animal_id: str | int
    group: str
    sex: str
    true_id: str
    phase_windows: dict[int, PhaseWindow] = field(default_factory=dict)
    age_months: float | None = None
    date_of_birth: DateLike | None = None
    corner_assignments: dict[int, int] = field(default_factory=dict)

    @property
    def animal_id_key(self) -> str:
        """Return the normalized raw IntelliCage animal ID."""

        return _normalize_animal_id(self.animal_id)

    def phase_start(self, phase_number: int) -> pd.Timestamp:
        """Return the start timestamp for one phase."""

        return _parse_datetime(self.phase_windows[int(phase_number)][0])

    def phase_end(self, phase_number: int) -> pd.Timestamp:
        """Return the end timestamp for one phase."""

        return _parse_datetime(self.phase_windows[int(phase_number)][1])

    def to_loader_record(self, phase_numbers: list[int]) -> dict[str, Any]:
        """Return one row of loader-compatible subject metadata."""

        record: dict[str, Any] = {
            "AnimalID": self.animal_id_key,
            "RFID": self.animal_id_key,
            "Group": str(self.group),
            "ET": str(self.true_id),
            "ETLabel": str(self.true_id),
            "SEX": str(self.sex),
            "age_months": self.age_months}
        if self.date_of_birth is not None:
            record["DOB"] = _parse_datetime(self.date_of_birth)
        for phase_number in phase_numbers:
            start, end = self.phase_windows[int(phase_number)]
            record[f"Phase{int(phase_number)}Start"] = _parse_datetime(start)
            record[f"Phase{int(phase_number)}End"] = _parse_datetime(end)
        for phase_number, corner in self.corner_assignments.items():
            record[f"CornerPhase{int(phase_number)}"] = int(corner)
        return record

@dataclass(frozen=True)
class SubjectRegistry:
    """Validated collection of subject metadata."""

    subjects: dict[str, SubjectMetadata]

    @classmethod
    def from_mapping(
        cls,
        data: dict[str | int, SubjectMetadata | dict[str, Any]],
        *,
        default_phase_windows: dict[int, PhaseWindow] | None = None) -> "SubjectRegistry":
        """Build a registry from a user-script dictionary."""

        subjects: dict[str, SubjectMetadata] = {}
        for animal_id, entry in data.items():
            if isinstance(entry, SubjectMetadata):
                subject = entry
            else:
                normalized = _normalize_subject_mapping(entry)
                if not normalized.get("phase_windows") and default_phase_windows:
                    normalized["phase_windows"] = default_phase_windows
                subject = SubjectMetadata(animal_id=animal_id, **normalized)
            subjects[subject.animal_id_key] = subject
        return cls(subjects=subjects)

    def validate_for_experiment(self, experiment: ExperimentMetadata) -> None:
        """Check that every subject defines every required experiment phase."""

        required_phases = set(experiment.phases)
        if not self.subjects:
            raise ValueError("SubjectRegistry must contain at least one subject.")
        for animal_id, subject in self.subjects.items():
            missing = sorted(required_phases.difference(int(phase) for phase in subject.phase_windows))
            if missing:
                raise ValueError(f"Subject {animal_id} is missing phase windows for phases {missing}.")
            for phase_number in required_phases:
                start = subject.phase_start(phase_number)
                end = subject.phase_end(phase_number)
                if end <= start:
                    raise ValueError(f"Subject {animal_id} has a non-positive phase-{phase_number} window.")

    def to_loader_frame(self, experiment: ExperimentMetadata) -> pd.DataFrame:
        """Return all subjects as a loader-compatible DataFrame."""

        self.validate_for_experiment(experiment)
        phase_numbers = list(experiment.phases)
        rows = [
            subject.to_loader_record(phase_numbers)
            for subject in self.subjects.values()]
        return pd.DataFrame(rows)

    def scheduled_phase_start_hours(self, experiment: ExperimentMetadata) -> dict[int, float]:
        """Infer protocol start hours from subject-level phase windows."""

        explicit = experiment.explicit_scheduled_phase_start_hours()
        if len(explicit) == len(experiment.phases):
            schedule = explicit.copy()
        else:
            self.validate_for_experiment(experiment)
            rows: list[dict[str, float | int]] = []
            for subject in self.subjects.values():
                phase1_start = subject.phase_start(min(experiment.phases))
                for phase_number in experiment.phases:
                    elapsed_hours = (subject.phase_start(phase_number) - phase1_start).total_seconds() / 3600.0
                    rows.append({"PhaseNumber": int(phase_number), "start_hours": float(elapsed_hours)})
            frame = pd.DataFrame(rows)
            schedule = {
                int(phase_number): float(values["start_hours"].median())
                for phase_number, values in frame.groupby("PhaseNumber", observed=True)}
        last_phase = max(experiment.phases)
        terminal_key = last_phase + 1
        if terminal_key not in schedule:
            terminal_offsets: list[float] = []
            for subject in self.subjects.values():
                phase1_start = subject.phase_start(min(experiment.phases))
                terminal_offsets.append((subject.phase_end(last_phase) - phase1_start).total_seconds() / 3600.0)
            schedule[terminal_key] = float(pd.Series(terminal_offsets).median())
        return dict(sorted(schedule.items()))
# %% END
