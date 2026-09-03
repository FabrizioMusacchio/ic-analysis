"""Data loading utilities for IntelliCage experiments.

This module turns the raw IntelliCage exports into harmonized pandas
DataFrames. It reads:

1. `Visits.txt` for visit-level behavior summaries.
2. `Nosepokes.txt` for visit-linked nose-poke events.

Subject metadata are expected from the user-defined `SUBJECTS` input and are
matched to raw `AnimalTag` values from `Visits.txt`.

author: Fabrizio Musacchio
date: May 2026
"""
# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
# %% CONSTANTS
DEFAULT_PHASE_NAME_MAP: dict[str, int] = {}
DEFAULT_GROUP_NAMES: tuple[str, ...] = tuple(f"Group {index}" for index in range(1, 11))
PHASE_DISPLAY_LABELS: dict[int, str] = {
    1: "Phase1",
    2: "Phase2",
    3: "Phase3",
    4: "Phase4"}
# %% DATA CLASSES
@dataclass(frozen=True)
class CohortData:
    """Container that keeps the loaded cohort tables together."""

    visits: pd.DataFrame
    metadata: pd.DataFrame
    nosepokes: pd.DataFrame
    phase_manifest: pd.DataFrame

class ExportBlock(NamedTuple):
    """One technical IntelliCage export block inside a cage-run folder."""

    run_group: str
    block_name: str
    block_number: int
    visits_path: Path
    nosepokes_path: Path

# %% HELPER FUNCTIONS
def _first_visit_start(visits_path: Path) -> pd.Timestamp:
    """Return the first visit timestamp in one export block."""

    starts = pd.read_csv(visits_path, sep="\t", usecols=["Start"])
    if starts.empty:
        raise ValueError(f"`Visits.txt` contains no rows: {visits_path}")
    return pd.to_datetime(starts["Start"], errors="raise").min()

def _find_export_block_candidates(run_group_dir: Path) -> list[tuple[str, Path, Path, pd.Timestamp]]:
    """Find raw IntelliCage export blocks below one cage-run folder."""

    candidates: list[tuple[str, Path, Path, pd.Timestamp]] = []
    direct_intellicage_dir = run_group_dir / "IntelliCage"
    direct_visits_path = direct_intellicage_dir / "Visits.txt"
    direct_nosepokes_path = direct_intellicage_dir / "Nosepokes.txt"
    if direct_visits_path.exists():
        candidates.append((
            "Export_Block_1",
            direct_visits_path,
            direct_nosepokes_path,
            _first_visit_start(direct_visits_path)))
    for child_dir in sorted(path for path in run_group_dir.iterdir() if path.is_dir()):
        intellicage_dir = child_dir / "IntelliCage"
        visits_path = intellicage_dir / "Visits.txt"
        nosepokes_path = intellicage_dir / "Nosepokes.txt"
        if visits_path.exists():
            candidates.append((
                child_dir.name,
                visits_path,
                nosepokes_path,
                _first_visit_start(visits_path)))
    return sorted(candidates, key=lambda item: (item[3], item[0]))

def discover_export_blocks(dataset_root: Path | str) -> list[ExportBlock]:
    """Discover all IntelliCage export blocks below one experiment data root."""

    dataset_root = Path(dataset_root)
    blocks: list[ExportBlock] = []
    for run_group_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        candidates = _find_export_block_candidates(run_group_dir)
        for block_index, (block_name, visits_path, nosepokes_path, _) in enumerate(candidates, start=1):
            blocks.append(ExportBlock(
                run_group=run_group_dir.name,
                block_name=block_name,
                block_number=block_index,
                visits_path=visits_path,
                nosepokes_path=nosepokes_path))
    return blocks

def _ordered_raw_group_names(groups: pd.Series) -> list[str]:
    """Return raw group labels in first-seen order."""

    ordered: list[str] = []
    for group_name in groups.dropna().astype(str):
        if group_name not in ordered:
            ordered.append(group_name)
    return ordered

def _complete_group_names(group_count: int, group_names: list[str] | tuple[str, ...] | None) -> list[str]:
    """Fill a user-supplied group-name list with generic defaults."""

    selected: list[str] = []
    for group_name in group_names or []:
        display_name = str(group_name)
        if display_name and display_name not in selected:
            selected.append(display_name)
    next_index = len(selected) + 1
    for default_name in DEFAULT_GROUP_NAMES[len(selected):]:
        if len(selected) >= group_count:
            break
        if default_name not in selected:
            selected.append(default_name)
        next_index += 1
    while len(selected) < group_count:
        candidate = f"Group {next_index}"
        if candidate not in selected:
            selected.append(candidate)
        next_index += 1
    return selected[:group_count]

def _resolve_group_name_mapping(
    groups: pd.Series,
    group_names: list[str] | tuple[str, ...] | None) -> tuple[dict[str, str], list[str]]:
    """Map raw dataset group labels to public display labels."""

    raw_groups = _ordered_raw_group_names(groups)
    display_names = _complete_group_names(len(raw_groups), group_names)
    return dict(zip(raw_groups, display_names)), display_names

def read_visits_file(visits_path: Path, run_group: str, phase_name: str, phase_number: int) -> pd.DataFrame:
    """Read one IntelliCage `Visits.txt` file into a typed DataFrame."""

    visits = pd.read_csv(visits_path, sep="\t")
    visits["RunGroup"] = run_group
    visits["Phase"] = phase_name
    visits["PhaseNumber"] = int(phase_number)
    visits["Start"] = pd.to_datetime(visits["Start"], errors="raise")
    visits["End"] = pd.to_datetime(visits["End"], errors="raise")
    visits["AnimalTag"] = pd.to_numeric(visits["AnimalTag"], errors="raise").astype("Int64")
    visits["VisitID"] = pd.to_numeric(visits["VisitID"], errors="raise").astype("Int64")
    visits["VisitDurationSeconds"] = (visits["End"] - visits["Start"]).dt.total_seconds()
    visits["visit_has_lick"] = visits["LickNumber"].fillna(0).gt(0) | visits["LickDuration"].fillna(0).gt(0)
    return visits

def read_nosepokes_file(
    nosepokes_path: Path,
    run_group: str,
    phase_name: str,
    phase_number: int,
) -> pd.DataFrame:
    """Read one IntelliCage `Nosepokes.txt` file into a typed DataFrame."""

    nosepokes = pd.read_csv(nosepokes_path, sep="\t")
    nosepokes["RunGroup"] = run_group
    nosepokes["Phase"] = phase_name
    nosepokes["PhaseNumber"] = int(phase_number)
    nosepokes["Start"] = pd.to_datetime(nosepokes["Start"], errors="raise")
    nosepokes["End"] = pd.to_datetime(nosepokes["End"], errors="raise")
    nosepokes["VisitID"] = pd.to_numeric(nosepokes["VisitID"], errors="raise").astype("Int64")
    if "LickStartTime" in nosepokes.columns:
        nosepokes["LickStartTime"] = pd.to_datetime(nosepokes["LickStartTime"], errors="coerce")
    return nosepokes

def summarize_nosepokes_by_visit(nosepokes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw nose-poke rows to one row per visit."""

    summary = (
        nosepokes.groupby(["RunGroup", "Phase", "PhaseNumber", "VisitID"], observed=True)
        .agg(
            nosepoke_event_count=("VisitID", "size"),
            nosepoke_side_count=("Side", "nunique"),
            nosepoke_lick_count=("LickNumber", "sum"),
            nosepoke_lick_duration=("LickDuration", "sum"),
            nosepoke_condition_error_count=("ConditionError", "sum"),
        ).reset_index())
    summary["has_nosepoke"] = summary["nosepoke_event_count"].gt(0)
    summary["has_nosepoke_lick"] = summary["nosepoke_lick_count"].gt(0) | summary["nosepoke_lick_duration"].gt(0)
    return summary

def _build_phase_manifest(visits: pd.DataFrame) -> pd.DataFrame:
    """Create a manifest with the temporal extent of every phase."""

    return (
        visits.groupby(["RunGroup", "Phase", "PhaseNumber"], observed=True)
        .agg(
            PhaseStart=("Start", "min"),
            PhaseEnd=("End", "max"),
            VisitCount=("VisitID", "size"),
            MouseCount=("AnimalTag", "nunique"))
        .reset_index()
        .sort_values(["RunGroup", "PhaseNumber"])
        .reset_index(drop=True))

def _attach_time_reference_columns(visits: pd.DataFrame, phase_manifest: pd.DataFrame) -> pd.DataFrame:
    """Add experiment-relative and phase-relative timing columns."""

    experiment_starts = (
        phase_manifest.loc[phase_manifest["PhaseNumber"].eq(1), ["RunGroup", "PhaseStart"]]
        .rename(columns={"PhaseStart": "ExperimentStart"})
        .copy())
    phase_starts = phase_manifest.loc[:, ["RunGroup", "Phase", "PhaseNumber", "PhaseStart"]].copy()

    enriched = visits.merge(experiment_starts, on="RunGroup", how="left", validate="many_to_one")
    enriched = enriched.merge(
        phase_starts,
        on=["RunGroup", "Phase", "PhaseNumber"],
        how="left",
        validate="many_to_one")
    enriched["experiment_elapsed_hours"] = (enriched["Start"] - enriched["ExperimentStart"]).dt.total_seconds() / 3600.0
    enriched["phase_elapsed_hours"] = (enriched["Start"] - enriched["PhaseStart"]).dt.total_seconds() / 3600.0
    enriched["experiment_day"] = np.floor(enriched["experiment_elapsed_hours"] / 24.0).astype(int)
    enriched["phase_day"] = np.floor(enriched["phase_elapsed_hours"] / 24.0).astype(int) + 1
    return enriched

def attach_analysis_time_columns(
    visits: pd.DataFrame,
    phase_manifest: pd.DataFrame,
    *,
    scheduled_phase_start_hours: dict[int, float],
    mouse_day_start_hour: float,
    experiment_day0_start_hour: float | None = None,
    schedule_anchor_phase_number: int | None = None,
) -> pd.DataFrame:
    """Attach globally aligned analysis-time columns.

    Raw IntelliCage exports are treated as technical export blocks. Their file
    boundaries do not have to match the experimental protocol. For cross-group
    comparisons we therefore create a second time axis:

    - experimental time starts at the mouse-day onset of day 0
    - protocol phase windows are assigned from subject-level absolute phase
      timestamps when available
    - phase-relative elapsed time is then computed from the subject-level
      phase start

    Parameters
    ----------
    visits:
        Visit-level table returned by :func:`load_cohort_data`.
    phase_manifest:
        Manifest with the observed temporal range of each raw export block.
    scheduled_phase_start_hours:
        Mapping from phase number to global experiment-relative start hour.
        An additional trailing marker, for example ``5=266``, can be provided
        to define the exclusive end of phase 4.
    mouse_day_start_hour:
        Clock time that defines the beginning of the mouse day on day 0.
    experiment_day0_start_hour:
        Optional independent wall-clock hour that defines experiment elapsed
        time zero on day 0. When omitted, the experiment timeline continues to
        start at ``mouse_day_start_hour`` for backward compatibility. Set this
        separately when day counting should start before the awake phase, for
        example at midnight while the mouse day still begins at 07:00.
    schedule_anchor_phase_number:
        Optional legacy raw export-block number whose observed start should be
        aligned to the configured scheduled start hour. This is only used when
        subject-level phase windows are absent.
    """

    enriched = visits.copy()
    has_subject_phase_windows = "Phase1Start" in enriched.columns and "Phase1End" in enriched.columns
    if has_subject_phase_windows:
        enriched["Phase1ObservedStart"] = pd.to_datetime(enriched["Phase1Start"], errors="raise")
    else:
        phase1_starts = (
            phase_manifest.loc[phase_manifest["PhaseNumber"].eq(1), ["RunGroup", "PhaseStart"]]
            .rename(columns={"PhaseStart": "Phase1ObservedStart"})
            .copy())
        enriched = enriched.merge(phase1_starts, on="RunGroup", how="left", validate="many_to_one")
        if enriched["Phase1ObservedStart"].isna().any():
            raise ValueError("Could not determine the observed first export-block start for every run group.")

    analysis_origin_hour = (
        float(mouse_day_start_hour)
        if experiment_day0_start_hour is None
        else float(experiment_day0_start_hour))
    phase1_floor_day = enriched["Phase1ObservedStart"].dt.floor("D")
    tentative_start = phase1_floor_day + pd.to_timedelta(analysis_origin_hour, unit="h")
    starts_before_day_anchor = enriched["Phase1ObservedStart"] < tentative_start
    enriched["AnalysisExperimentStart"] = tentative_start.where(
        ~starts_before_day_anchor,
        tentative_start - pd.to_timedelta(1, unit="D"))
    enriched["analysis_experiment_elapsed_hours"] = (enriched["Start"] - enriched["AnalysisExperimentStart"]).dt.total_seconds() / 3600.0

    sorted_phase_starts = sorted((int(key), float(value)) for key, value in scheduled_phase_start_hours.items())
    if not sorted_phase_starts:
        raise ValueError("`scheduled_phase_start_hours` must contain at least one phase start.")

    if schedule_anchor_phase_number is not None and not has_subject_phase_windows:
        anchor_phase_number = int(schedule_anchor_phase_number)
        if anchor_phase_number not in scheduled_phase_start_hours:
            raise ValueError("`schedule_anchor_phase_number` must be present in `scheduled_phase_start_hours`.")
        anchor_rows = phase_manifest.loc[
            phase_manifest["PhaseNumber"].eq(anchor_phase_number),
            ["RunGroup", "PhaseStart"]].rename(columns={"PhaseStart": "AnchorPhaseObservedStart"})
        if anchor_rows.empty:
            raise ValueError("Could not find the requested anchor phase in the phase manifest.")
        enriched = enriched.merge(anchor_rows, on="RunGroup", how="left", validate="many_to_one")
        if enriched["AnchorPhaseObservedStart"].isna().any():
            raise ValueError("Could not determine the observed anchor-phase start for every run group.")
        enriched["anchor_phase_observed_hours"] = (enriched["AnchorPhaseObservedStart"] - enriched["AnalysisExperimentStart"]).dt.total_seconds() / 3600.0
        enriched["schedule_alignment_offset_hours"] = (float(scheduled_phase_start_hours[anchor_phase_number]) - enriched["anchor_phase_observed_hours"])
        enriched["analysis_experiment_elapsed_hours"] = (enriched["analysis_experiment_elapsed_hours"] + enriched["schedule_alignment_offset_hours"])
    else:
        enriched["schedule_alignment_offset_hours"] = 0.0

    enriched["analysis_experiment_day"] = np.floor(enriched["analysis_experiment_elapsed_hours"] / 24.0).astype(int)

    phase_rows: list[dict[str, float | int | str]] = []
    for index, (phase_number, start_hour) in enumerate(sorted_phase_starts):
        next_start = sorted_phase_starts[index + 1][1] if index + 1 < len(sorted_phase_starts) else np.inf
        phase_rows.append(
            {
                "AnalysisPhaseNumber": phase_number,
                "analysis_phase_start_hours": start_hour,
                "analysis_phase_end_hours": next_start,
                "AnalysisPhase": PHASE_DISPLAY_LABELS.get(phase_number, f"Phase{phase_number}"),
            })

    phase_table = pd.DataFrame(phase_rows)
    valid_phase_table = phase_table.copy()
    phase_name_lookup = valid_phase_table.set_index("AnalysisPhaseNumber")["AnalysisPhase"]
    if has_subject_phase_windows:
        enriched["AnalysisPhaseNumber"] = pd.Series(pd.NA, index=enriched.index, dtype="Int64")
        enriched["analysis_phase_start_hours"] = np.nan
        enriched["analysis_phase_elapsed_hours"] = np.nan
        for phase_number in valid_phase_table["AnalysisPhaseNumber"].astype(int):
            start_col = f"Phase{phase_number}Start"
            end_col = f"Phase{phase_number}End"
            if start_col not in enriched.columns or end_col not in enriched.columns:
                continue
            phase_start = pd.to_datetime(enriched[start_col], errors="raise")
            phase_end = pd.to_datetime(enriched[end_col], errors="raise")
            mask = enriched["Start"].ge(phase_start) & enriched["Start"].lt(phase_end)
            enriched.loc[mask, "AnalysisPhaseNumber"] = int(phase_number)
            enriched.loc[mask, "analysis_phase_start_hours"] = (
                phase_start.loc[mask] - enriched.loc[mask, "AnalysisExperimentStart"]).dt.total_seconds() / 3600.0
            enriched.loc[mask, "analysis_phase_elapsed_hours"] = (
                enriched.loc[mask, "Start"] - phase_start.loc[mask]).dt.total_seconds() / 3600.0
    else:
        bins = [-np.inf, *valid_phase_table["analysis_phase_end_hours"].tolist()]
        labels = valid_phase_table["AnalysisPhaseNumber"].tolist()
        enriched["AnalysisPhaseNumber"] = pd.cut(
            enriched["analysis_experiment_elapsed_hours"],
            bins=bins,
            labels=labels,
            right=False).astype("Float64")
        enriched["AnalysisPhaseNumber"] = enriched["AnalysisPhaseNumber"].astype("Int64")
        phase_start_lookup = valid_phase_table.set_index("AnalysisPhaseNumber")["analysis_phase_start_hours"]
        enriched["analysis_phase_start_hours"] = enriched["AnalysisPhaseNumber"].map(phase_start_lookup)
        enriched["analysis_phase_elapsed_hours"] = (enriched["analysis_experiment_elapsed_hours"] - enriched["analysis_phase_start_hours"])
    enriched["analysis_phase_day"] = np.floor(enriched["analysis_phase_elapsed_hours"] / 24.0).astype("Int64") + 1
    enriched["AnalysisPhase"] = enriched["AnalysisPhaseNumber"].map(phase_name_lookup).astype("string")
    enriched["AnalysisAssignedCorner"] = pd.Series(pd.NA, index=enriched.index, dtype="Int64")
    enriched.loc[enriched["AnalysisPhaseNumber"].eq(3), "AnalysisAssignedCorner"] = enriched.loc[enriched["AnalysisPhaseNumber"].eq(3), "CornerPhase3"]
    enriched.loc[enriched["AnalysisPhaseNumber"].eq(4), "AnalysisAssignedCorner"] = enriched.loc[enriched["AnalysisPhaseNumber"].eq(4), "CornerPhase4"]
    enriched["correct_corner_visit"] = enriched["Corner"].eq(enriched["AnalysisAssignedCorner"])
    enriched["correct_np_visit"] = enriched["correct_corner_visit"] & enriched["has_nosepoke"]
    enriched["rewarded_correct_corner_visit"] = (enriched["correct_corner_visit"] & enriched["has_nosepoke"] & enriched["visit_has_lick"])
    enriched["phase2_drinking_visit"] = (enriched["AnalysisPhaseNumber"].eq(2) & enriched["has_nosepoke"] & enriched["visit_has_lick"])
    enriched["previous_correct_corner_visit"] = (enriched["AnalysisPhaseNumber"].eq(4) & enriched["Corner"].eq(enriched["CornerPhase3"]))
    enriched["neutral_incorrect_corner_visit"] = (
        enriched["AnalysisPhaseNumber"].eq(4)
        & enriched["Corner"].notna()
        & ~enriched["Corner"].eq(enriched["CornerPhase4"])
        & ~enriched["Corner"].eq(enriched["CornerPhase3"]))
    return enriched

def load_cohort_data(
    dataset_root: Path | str,
    *,
    phase_name_map: dict[str, int] | None = None,
    optional_phase_names: set[str] | list[str] | tuple[str, ...] | None = None,
    drop_unmatched_visits: bool = False,
    group_names: list[str] | tuple[str, ...] | None = None,
    subject_metadata: pd.DataFrame | None = None,
    drop_unregistered_subjects: bool = True) -> CohortData:
    """Load and harmonize one IntelliCage cohort directory.

    Parameters
    ----------
    dataset_root:
        Root directory of one IntelliCage cohort.
    phase_name_map:
        Legacy argument kept for configuration compatibility. Raw subfolders
        are now treated as technical export blocks and no longer define
        analysis phases.
    optional_phase_names:
        Legacy argument kept for configuration compatibility. Missing
        biological phase folders are no longer checked during loading.
    group_names:
        Optional group display order. Missing entries are filled with generic
        `Group N` names so partially specified group-name lists remain valid.
    subject_metadata:
        Script-defined subject table. Rows are matched by `AnimalID` against
        raw IntelliCage `AnimalTag` values.
    drop_unregistered_subjects:
        If `subject_metadata` is provided, control whether raw animals without
        a subject entry are silently excluded. This is the recommended public
        policy for script-defined analyses.
    """

    dataset_root = Path(dataset_root)
    if subject_metadata is None:
        raise ValueError("`subject_metadata` is required. Define subjects in the user script or load them from YAML.")
    export_blocks = discover_export_blocks(dataset_root)
    if not export_blocks:
        raise FileNotFoundError(f"No cage-run directories with IntelliCage `Visits.txt` files found below {dataset_root}")

    visit_frames: list[pd.DataFrame] = []
    nosepoke_frames: list[pd.DataFrame] = []

    for export_block in export_blocks:
        if not export_block.nosepokes_path.exists():
            raise FileNotFoundError(f"Missing nose-poke file: {export_block.nosepokes_path}")
        visit_frames.append(read_visits_file(
            export_block.visits_path,
            export_block.run_group,
            export_block.block_name,
            export_block.block_number))
        nosepoke_frames.append(read_nosepokes_file(
            export_block.nosepokes_path,
            export_block.run_group,
            export_block.block_name,
            export_block.block_number))

    if not visit_frames:
        raise FileNotFoundError(f"No `Visits.txt` files found below {dataset_root}")
    visits = pd.concat(visit_frames, ignore_index=True)
    nosepokes = pd.concat(nosepoke_frames, ignore_index=True)
    visits["AnimalID"] = visits["AnimalTag"].astype(str)

    subject_frame = subject_metadata.copy()
    if "AnimalID" not in subject_frame.columns:
        raise ValueError("`subject_metadata` must contain an `AnimalID` column.")
    subject_frame["AnimalID"] = subject_frame["AnimalID"].astype(str)
    duplicated_subjects = subject_frame["AnimalID"].duplicated(keep=False)
    if duplicated_subjects.any():
        duplicated_ids = sorted(subject_frame.loc[duplicated_subjects, "AnimalID"].unique())
        raise ValueError(f"`subject_metadata` contains duplicated AnimalID values: {duplicated_ids}")

    registered_ids = set(subject_frame["AnimalID"])
    raw_ids = set(visits["AnimalID"].dropna().astype(str))
    unregistered_ids = sorted(raw_ids.difference(registered_ids))
    if unregistered_ids and not drop_unregistered_subjects:
        raise ValueError(
            "Raw IntelliCage data contain AnimalID values without subject metadata: "
            f"{unregistered_ids}")
    keep_ids = raw_ids.intersection(registered_ids)
    if not keep_ids:
        raise ValueError("No raw IntelliCage AnimalID values match `subject_metadata`.")

    visits = visits.loc[visits["AnimalID"].isin(keep_ids)].copy()
    valid_visit_keys = visits.loc[:, ["RunGroup", "Phase", "PhaseNumber", "VisitID"]].drop_duplicates()
    nosepokes = nosepokes.merge(
        valid_visit_keys,
        on=["RunGroup", "Phase", "PhaseNumber", "VisitID"],
        how="inner",
        validate="many_to_one")

    observed_subjects = visits.loc[:, ["RunGroup", "AnimalID"]].drop_duplicates()
    metadata = observed_subjects.merge(subject_frame, on="AnimalID", how="left", validate="many_to_one")
    metadata["RFID"] = pd.to_numeric(metadata["AnimalID"], errors="raise").astype("Int64")

    nosepoke_summary = summarize_nosepokes_by_visit(nosepokes)
    visits = visits.merge(
        metadata,
        on=["RunGroup", "AnimalID"],
        how="left",
        validate="many_to_one")
    if visits["ET"].isna().any():
        missing_rows = visits.loc[visits["ET"].isna(), ["RunGroup", "AnimalTag"]].drop_duplicates()
        if not drop_unmatched_visits:
            raise ValueError(
                "Some visits could not be matched to `subject_metadata`. "
                f"Missing pairs: {missing_rows.to_dict(orient='records')}")
        visits = visits.loc[visits["ET"].notna()].copy()
        valid_visit_keys = visits.loc[:, ["RunGroup", "Phase", "PhaseNumber", "VisitID"]].drop_duplicates()
        nosepokes = nosepokes.merge(
            valid_visit_keys,
            on=["RunGroup", "Phase", "PhaseNumber", "VisitID"],
            how="inner",
            validate="many_to_one")

    visits = visits.merge(
        nosepoke_summary,
        on=["RunGroup", "Phase", "PhaseNumber", "VisitID"],
        how="left",
        validate="many_to_one")
    nosepoke_columns = [
        "nosepoke_event_count",
        "nosepoke_side_count",
        "nosepoke_lick_count",
        "nosepoke_lick_duration",
        "nosepoke_condition_error_count"]
    for column in nosepoke_columns:
        visits[column] = visits[column].fillna(0)
    for column in ["has_nosepoke", "has_nosepoke_lick"]:
        visits[column] = visits[column].fillna(False)

    visits["AssignedCorner"] = pd.Series(pd.NA, index=visits.index, dtype="Int64")
    visits.loc[visits["PhaseNumber"].eq(3), "AssignedCorner"] = visits.loc[visits["PhaseNumber"].eq(3), "CornerPhase3"]
    visits.loc[visits["PhaseNumber"].eq(4), "AssignedCorner"] = visits.loc[visits["PhaseNumber"].eq(4), "CornerPhase4"]
    visits["assigned_corner_visit"] = visits["Corner"].eq(visits["AssignedCorner"])
    visits["correct_place_visit"] = visits["PlaceError"].eq(0)
    visits["correct_corner_visit"] = visits["assigned_corner_visit"]
    visits["correct_np_visit"] = visits["assigned_corner_visit"] & visits["has_nosepoke"]
    visits["rewarded_place_visit"] = (visits["correct_place_visit"] & visits["has_nosepoke"] & visits["visit_has_lick"])
    visits["rewarded_correct_corner_visit"] = (visits["assigned_corner_visit"] & visits["has_nosepoke"] & visits["visit_has_lick"])
    visits["previous_correct_corner_visit"] = visits["PhaseNumber"].eq(4) & visits["Corner"].eq(visits["CornerPhase3"])
    visits["neutral_incorrect_corner_visit"] = (
        visits["PhaseNumber"].eq(4)
        & visits["Corner"].notna()
        & ~visits["Corner"].eq(visits["CornerPhase4"])
        & ~visits["Corner"].eq(visits["CornerPhase3"]))
    visits["phase2_drinking_visit"] = (visits["PhaseNumber"].eq(2) & visits["has_nosepoke"] & visits["visit_has_lick"])

    group_name_mapping, group_categories = _resolve_group_name_mapping(metadata["Group"], group_names)
    visits["Group"] = visits["Group"].astype(str).map(group_name_mapping).fillna(visits["Group"].astype(str))
    metadata["Group"] = metadata["Group"].astype(str).map(group_name_mapping).fillna(metadata["Group"].astype(str))
    visits["Group"] = pd.Categorical(visits["Group"], categories=group_categories, ordered=True)
    metadata["Group"] = pd.Categorical(metadata["Group"], categories=group_categories, ordered=True)

    phase_manifest = _build_phase_manifest(visits)
    visits = _attach_time_reference_columns(visits, phase_manifest)
    visits = visits.sort_values(["RunGroup", "PhaseNumber", "Start", "VisitID"]).reset_index(drop=True)
    metadata = metadata.sort_values(["Group", "ET"]).reset_index(drop=True)
    nosepokes = nosepokes.sort_values(["RunGroup", "PhaseNumber", "Start", "VisitID"]).reset_index(drop=True)

    return CohortData(
        visits=visits,
        metadata=metadata,
        nosepokes=nosepokes,
        phase_manifest=phase_manifest)
# %% END
