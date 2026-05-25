"""Data loading utilities for IntelliCage place-learning experiments.

This module turns the raw IntelliCage exports into harmonized pandas
DataFrames. It reads:

1. `Mice.txt` for mouse metadata and assigned corners.
2. `Visits.txt` for visit-level behavior summaries.
3. `Nosepokes.txt` for visit-linked nose-poke events.

The returned visit table contains both the old MATLAB-compatible metric
(`PlaceError == 0`) and the stricter poster-oriented metric that requires a
correct place visit with at least one associated nose-poke and at least one
lick.
"""
# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# %% CONSTANTS
PHASE_NAMES: tuple[str, ...] = ("Phase1", "Phase2", "Phase3", "Phase4")
DEFAULT_PHASE_NAME_MAP: dict[str, int] = {
    "Phase1": 1,
    "Phase2": 2,
    "Phase3": 3,
    "Phase4": 4,
}
PREFERRED_GROUP_ORDER: tuple[str, ...] = (
    "WT",
    "tdTomato",
    "Tau 66-421",
    "Tau 1-421",
    "Tau 1-441",
)
PHASE_DISPLAY_LABELS: dict[int, str] = {
    1: "Phase1",
    2: "Phase2",
    3: "Phase3",
    4: "Phase4",
}

# %% DATA CLASSES
@dataclass(frozen=True)
class CohortData:
    """Container that keeps the loaded cohort tables together."""

    visits: pd.DataFrame
    metadata: pd.DataFrame
    nosepokes: pd.DataFrame
    phase_manifest: pd.DataFrame

# %% HELPER FUNCTIONS
def _ordered_group_categories(groups: pd.Series) -> list[str]:
    """Return a stable categorical order for pathology groups."""

    present = [group for group in PREFERRED_GROUP_ORDER if group in set(groups.dropna())]
    extras = sorted(set(groups.dropna()) - set(present))
    return present + extras

def read_mice_metadata(mice_path: Path, run_group: str) -> pd.DataFrame:
    """Read one `Mice.txt` file and normalize its metadata columns."""

    metadata = pd.read_csv(mice_path, sep="\t")
    if "Corner Phase 3" not in metadata.columns and "Corner Phase 1" in metadata.columns:
        metadata = metadata.rename(columns={"Corner Phase 1": "Corner Phase 3"})
    if "Corner Phase 4" not in metadata.columns and "Corner Phase 2" in metadata.columns:
        metadata = metadata.rename(columns={"Corner Phase 2": "Corner Phase 4"})
    metadata = metadata.rename(
        columns={
            "VIRUS": "Group",
            "Corner Phase 3": "CornerPhase3",
            "Corner Phase 4": "CornerPhase4",
        }
    )
    metadata["RunGroup"] = run_group
    metadata["RFID"] = pd.to_numeric(metadata["RFID"], errors="raise").astype("Int64")
    metadata["ET"] = metadata["ET"].astype("string").str.strip()
    metadata["ETLabel"] = np.where(
        metadata["ET"].str.match(r"^(ET|Lo)", case=False, na=False),
        metadata["ET"],
        "ET" + metadata["ET"],
    )
    metadata["DOB"] = pd.to_datetime(metadata["DOB"], format="%d.%m.%y", dayfirst=True, errors="coerce")
    if "CornerPhase3" not in metadata.columns:
        metadata["CornerPhase3"] = pd.Series(pd.NA, index=metadata.index, dtype="Int64")
    else:
        metadata["CornerPhase3"] = pd.to_numeric(metadata["CornerPhase3"], errors="coerce").astype("Int64")
    if "CornerPhase4" not in metadata.columns:
        metadata["CornerPhase4"] = pd.Series(pd.NA, index=metadata.index, dtype="Int64")
    else:
        metadata["CornerPhase4"] = pd.to_numeric(metadata["CornerPhase4"], errors="coerce").astype("Int64")
    metadata["SEX"] = metadata["SEX"].astype("string")
    metadata["Group"] = metadata["Group"].astype("string")
    return metadata

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
        )
        .reset_index()
    )
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
            MouseCount=("AnimalTag", "nunique"),
        )
        .reset_index()
        .sort_values(["RunGroup", "PhaseNumber"])
        .reset_index(drop=True)
    )

def _attach_time_reference_columns(visits: pd.DataFrame, phase_manifest: pd.DataFrame) -> pd.DataFrame:
    """Add experiment-relative and phase-relative timing columns."""

    experiment_starts = (
        phase_manifest.loc[phase_manifest["PhaseNumber"].eq(1), ["RunGroup", "PhaseStart"]]
        .rename(columns={"PhaseStart": "ExperimentStart"})
        .copy()
    )
    phase_starts = phase_manifest.loc[:, ["RunGroup", "Phase", "PhaseNumber", "PhaseStart"]].copy()

    enriched = visits.merge(experiment_starts, on="RunGroup", how="left", validate="many_to_one")
    enriched = enriched.merge(
        phase_starts,
        on=["RunGroup", "Phase", "PhaseNumber"],
        how="left",
        validate="many_to_one",
    )
    enriched["experiment_elapsed_hours"] = (
        enriched["Start"] - enriched["ExperimentStart"]
    ).dt.total_seconds() / 3600.0
    enriched["phase_elapsed_hours"] = (
        enriched["Start"] - enriched["PhaseStart"]
    ).dt.total_seconds() / 3600.0
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

    The raw IntelliCage exports store visits in phase-specific files and the
    actual file boundaries can differ by a small amount from the intended
    protocol timing. For poster-style cross-group comparisons we therefore
    create a second time axis:

    - experimental time starts at the mouse-day onset of day 0
    - protocol phase windows are assigned from a global schedule in elapsed
      hours rather than from the raw file boundary
    - phase-relative elapsed time is then computed from this scheduled phase
      start

    Parameters
    ----------
    visits:
        Visit-level table returned by :func:`load_cohort_data`.
    phase_manifest:
        Manifest with the observed temporal range of each raw phase file.
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
        Optional raw phase number whose observed start should be aligned to the
        configured scheduled start hour for every run group. This is useful
        when early free-hab durations vary between runs, but all later phases
        should be synchronized to the protocol transition point, for example
        the observed start of NPA.
    """

    enriched = visits.copy()
    phase1_starts = (
        phase_manifest.loc[phase_manifest["PhaseNumber"].eq(1), ["RunGroup", "PhaseStart"]]
        .rename(columns={"PhaseStart": "Phase1ObservedStart"})
        .copy()
    )
    enriched = enriched.merge(phase1_starts, on="RunGroup", how="left", validate="many_to_one")
    if enriched["Phase1ObservedStart"].isna().any():
        raise ValueError("Could not determine the observed phase-1 start for every run group.")

    analysis_origin_hour = (
        float(mouse_day_start_hour)
        if experiment_day0_start_hour is None
        else float(experiment_day0_start_hour)
    )
    phase1_floor_day = enriched["Phase1ObservedStart"].dt.floor("D")
    tentative_start = phase1_floor_day + pd.to_timedelta(analysis_origin_hour, unit="h")
    starts_before_day_anchor = enriched["Phase1ObservedStart"] < tentative_start
    enriched["AnalysisExperimentStart"] = tentative_start.where(
        ~starts_before_day_anchor,
        tentative_start - pd.to_timedelta(1, unit="D"),
    )
    enriched["analysis_experiment_elapsed_hours"] = (
        enriched["Start"] - enriched["AnalysisExperimentStart"]
    ).dt.total_seconds() / 3600.0

    sorted_phase_starts = sorted((int(key), float(value)) for key, value in scheduled_phase_start_hours.items())
    if not sorted_phase_starts:
        raise ValueError("`scheduled_phase_start_hours` must contain at least one phase start.")

    if schedule_anchor_phase_number is not None:
        anchor_phase_number = int(schedule_anchor_phase_number)
        if anchor_phase_number not in scheduled_phase_start_hours:
            raise ValueError(
                "`schedule_anchor_phase_number` must be present in `scheduled_phase_start_hours`."
            )
        anchor_rows = phase_manifest.loc[
            phase_manifest["PhaseNumber"].eq(anchor_phase_number),
            ["RunGroup", "PhaseStart"],
        ].rename(columns={"PhaseStart": "AnchorPhaseObservedStart"})
        if anchor_rows.empty:
            raise ValueError("Could not find the requested anchor phase in the phase manifest.")
        enriched = enriched.merge(anchor_rows, on="RunGroup", how="left", validate="many_to_one")
        if enriched["AnchorPhaseObservedStart"].isna().any():
            raise ValueError("Could not determine the observed anchor-phase start for every run group.")
        enriched["anchor_phase_observed_hours"] = (
            enriched["AnchorPhaseObservedStart"] - enriched["AnalysisExperimentStart"]
        ).dt.total_seconds() / 3600.0
        enriched["schedule_alignment_offset_hours"] = (
            float(scheduled_phase_start_hours[anchor_phase_number]) - enriched["anchor_phase_observed_hours"]
        )
        enriched["analysis_experiment_elapsed_hours"] = (
            enriched["analysis_experiment_elapsed_hours"] + enriched["schedule_alignment_offset_hours"]
        )
    else:
        enriched["schedule_alignment_offset_hours"] = 0.0

    enriched["analysis_experiment_day"] = np.floor(
        enriched["analysis_experiment_elapsed_hours"] / 24.0
    ).astype(int)

    phase_rows: list[dict[str, float | int | str]] = []
    for index, (phase_number, start_hour) in enumerate(sorted_phase_starts):
        next_start = sorted_phase_starts[index + 1][1] if index + 1 < len(sorted_phase_starts) else np.inf
        phase_rows.append(
            {
                "AnalysisPhaseNumber": phase_number,
                "analysis_phase_start_hours": start_hour,
                "analysis_phase_end_hours": next_start,
                "AnalysisPhase": PHASE_DISPLAY_LABELS.get(phase_number, f"Phase{phase_number}"),
            }
        )

    phase_table = pd.DataFrame(phase_rows)
    valid_phase_table = phase_table.loc[phase_table["AnalysisPhaseNumber"].between(1, 4)].copy()
    bins = [-np.inf, *valid_phase_table["analysis_phase_end_hours"].tolist()]
    labels = valid_phase_table["AnalysisPhaseNumber"].tolist()
    enriched["AnalysisPhaseNumber"] = pd.cut(
        enriched["analysis_experiment_elapsed_hours"],
        bins=bins,
        labels=labels,
        right=False,
    ).astype("Float64")
    enriched["AnalysisPhaseNumber"] = enriched["AnalysisPhaseNumber"].astype("Int64")
    phase_start_lookup = valid_phase_table.set_index("AnalysisPhaseNumber")["analysis_phase_start_hours"]
    phase_name_lookup = valid_phase_table.set_index("AnalysisPhaseNumber")["AnalysisPhase"]
    enriched["analysis_phase_start_hours"] = enriched["AnalysisPhaseNumber"].map(phase_start_lookup)
    enriched["analysis_phase_elapsed_hours"] = (
        enriched["analysis_experiment_elapsed_hours"] - enriched["analysis_phase_start_hours"]
    )
    enriched["analysis_phase_day"] = np.floor(enriched["analysis_phase_elapsed_hours"] / 24.0).astype("Int64") + 1
    enriched["AnalysisPhase"] = enriched["AnalysisPhaseNumber"].map(phase_name_lookup).astype("string")
    enriched["AnalysisAssignedCorner"] = pd.Series(pd.NA, index=enriched.index, dtype="Int64")
    enriched.loc[enriched["AnalysisPhaseNumber"].eq(3), "AnalysisAssignedCorner"] = enriched.loc[
        enriched["AnalysisPhaseNumber"].eq(3), "CornerPhase3"
    ]
    enriched.loc[enriched["AnalysisPhaseNumber"].eq(4), "AnalysisAssignedCorner"] = enriched.loc[
        enriched["AnalysisPhaseNumber"].eq(4), "CornerPhase4"
    ]
    enriched["correct_corner_visit"] = enriched["Corner"].eq(enriched["AnalysisAssignedCorner"])
    enriched["correct_np_visit"] = enriched["correct_corner_visit"] & enriched["has_nosepoke"]
    enriched["rewarded_correct_corner_visit"] = (
        enriched["correct_corner_visit"] & enriched["has_nosepoke"] & enriched["visit_has_lick"]
    )
    enriched["previous_correct_corner_visit"] = (
        enriched["AnalysisPhaseNumber"].eq(4) & enriched["Corner"].eq(enriched["CornerPhase3"])
    )
    enriched["neutral_incorrect_corner_visit"] = (
        enriched["AnalysisPhaseNumber"].eq(4)
        & enriched["Corner"].notna()
        & ~enriched["Corner"].eq(enriched["CornerPhase4"])
        & ~enriched["Corner"].eq(enriched["CornerPhase3"])
    )
    return enriched

def load_cohort_data(
    dataset_root: Path | str,
    *,
    phase_name_map: dict[str, int] | None = None,
    optional_phase_names: set[str] | list[str] | tuple[str, ...] | None = None,
    drop_unmatched_visits: bool = False,
) -> CohortData:
    """Load and harmonize one IntelliCage cohort directory.

    Parameters
    ----------
    dataset_root:
        Root directory of one IntelliCage cohort.
    phase_name_map:
        Mapping from subfolder names such as ``Phase1`` or ``SP2`` to the
        raw phase numbers that should be assigned during loading.
    optional_phase_names:
        Folder names that may be absent for some run groups without causing
        the loader to fail.
    """

    dataset_root = Path(dataset_root)
    selected_phase_map = DEFAULT_PHASE_NAME_MAP.copy()
    if phase_name_map:
        selected_phase_map = {str(key): int(value) for key, value in phase_name_map.items()}
    optional_phase_name_set = {str(name) for name in (optional_phase_names or set())}
    run_group_dirs = sorted(
        path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("Gruppe")
    )
    if not run_group_dirs:
        raise FileNotFoundError(f"No run-group directories found below {dataset_root}")

    metadata_frames: list[pd.DataFrame] = []
    visit_frames: list[pd.DataFrame] = []
    nosepoke_frames: list[pd.DataFrame] = []

    for run_group_dir in run_group_dirs:
        mice_path = run_group_dir / "Mice.txt"
        if not mice_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {mice_path}")
        metadata_frames.append(read_mice_metadata(mice_path, run_group_dir.name))

        for phase_name, phase_number in selected_phase_map.items():
            visits_path = run_group_dir / phase_name / "IntelliCage" / "Visits.txt"
            nosepokes_path = run_group_dir / phase_name / "IntelliCage" / "Nosepokes.txt"
            if not visits_path.exists():
                if phase_name in optional_phase_name_set:
                    continue
                raise FileNotFoundError(f"Missing visit file: {visits_path}")
            if not nosepokes_path.exists():
                if phase_name in optional_phase_name_set:
                    continue
                raise FileNotFoundError(f"Missing nose-poke file: {nosepokes_path}")
            visit_frames.append(read_visits_file(visits_path, run_group_dir.name, phase_name, phase_number))
            nosepoke_frames.append(read_nosepokes_file(nosepokes_path, run_group_dir.name, phase_name, phase_number))

    metadata = pd.concat(metadata_frames, ignore_index=True)
    visits = pd.concat(visit_frames, ignore_index=True)
    nosepokes = pd.concat(nosepoke_frames, ignore_index=True)

    nosepoke_summary = summarize_nosepokes_by_visit(nosepokes)
    visits = visits.merge(
        metadata,
        left_on=["RunGroup", "AnimalTag"],
        right_on=["RunGroup", "RFID"],
        how="left",
        validate="many_to_one",
    )
    if visits["ET"].isna().any():
        missing_rows = visits.loc[visits["ET"].isna(), ["RunGroup", "AnimalTag"]].drop_duplicates()
        if not drop_unmatched_visits:
            raise ValueError(
                "Some visits could not be matched to `Mice.txt` metadata. "
                f"Missing pairs: {missing_rows.to_dict(orient='records')}"
            )
        visits = visits.loc[visits["ET"].notna()].copy()
        valid_visit_keys = visits.loc[:, ["RunGroup", "Phase", "PhaseNumber", "VisitID"]].drop_duplicates()
        nosepokes = nosepokes.merge(
            valid_visit_keys,
            on=["RunGroup", "Phase", "PhaseNumber", "VisitID"],
            how="inner",
            validate="many_to_one",
        )

    visits = visits.merge(
        nosepoke_summary,
        on=["RunGroup", "Phase", "PhaseNumber", "VisitID"],
        how="left",
        validate="many_to_one",
    )
    nosepoke_columns = [
        "nosepoke_event_count",
        "nosepoke_side_count",
        "nosepoke_lick_count",
        "nosepoke_lick_duration",
        "nosepoke_condition_error_count",
    ]
    for column in nosepoke_columns:
        visits[column] = visits[column].fillna(0)
    for column in ["has_nosepoke", "has_nosepoke_lick"]:
        visits[column] = visits[column].fillna(False)

    visits["AssignedCorner"] = pd.Series(pd.NA, index=visits.index, dtype="Int64")
    visits.loc[visits["PhaseNumber"].eq(3), "AssignedCorner"] = visits.loc[
        visits["PhaseNumber"].eq(3), "CornerPhase3"
    ]
    visits.loc[visits["PhaseNumber"].eq(4), "AssignedCorner"] = visits.loc[
        visits["PhaseNumber"].eq(4), "CornerPhase4"
    ]
    visits["assigned_corner_visit"] = visits["Corner"].eq(visits["AssignedCorner"])
    visits["correct_place_visit"] = visits["PlaceError"].eq(0)
    visits["correct_corner_visit"] = visits["assigned_corner_visit"]
    visits["correct_np_visit"] = visits["assigned_corner_visit"] & visits["has_nosepoke"]
    visits["rewarded_place_visit"] = (
        visits["correct_place_visit"] & visits["has_nosepoke"] & visits["visit_has_lick"]
    )
    visits["rewarded_correct_corner_visit"] = (
        visits["assigned_corner_visit"] & visits["has_nosepoke"] & visits["visit_has_lick"]
    )
    visits["previous_correct_corner_visit"] = visits["PhaseNumber"].eq(4) & visits["Corner"].eq(visits["CornerPhase3"])
    visits["neutral_incorrect_corner_visit"] = (
        visits["PhaseNumber"].eq(4)
        & visits["Corner"].notna()
        & ~visits["Corner"].eq(visits["CornerPhase4"])
        & ~visits["Corner"].eq(visits["CornerPhase3"])
    )
    visits["phase2_drinking_visit"] = (
        visits["PhaseNumber"].eq(2) & visits["has_nosepoke"] & visits["visit_has_lick"]
    )

    group_categories = _ordered_group_categories(visits["Group"])
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
        phase_manifest=phase_manifest,
    )
# %% END
