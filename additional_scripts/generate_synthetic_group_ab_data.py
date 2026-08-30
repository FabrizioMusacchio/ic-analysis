"""Generate a synthetic IntelliCage place-learning example dataset.

The generated files mimic the subset of IntelliCage text exports consumed by
the current Python pipeline:

- `Gruppe*/Mice.txt`
- `Gruppe*/Phase*/IntelliCage/Visits.txt`
- `Gruppe*/Phase*/IntelliCage/Nosepokes.txt`

The data are intentionally pseudo-data. They are useful for documentation,
tests, and demos, but they are not derived from real mice.
"""
# %% IMPORTS
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# %% PATHS AND CONSTANTS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "example_data" / "synthetic_group_ab_place_learning"

PHASE_START_HOURS = {
    1: 0.0,
    2: 74.0,
    3: 122.0,
    4: 194.0}
PHASE_DURATIONS_HOURS = {
    1: 74.0,
    2: 48.0,
    3: 72.0,
    4: 72.0}
PHASE_NAMES = {
    1: "Phase1",
    2: "Phase2",
    3: "Phase3",
    4: "Phase4"}
MODULE_NAMES = {
    1: "Free Habituation (Phase 1)",
    2: "Nosepoke Adaptation (Phase 2)",
    3: "Place Learning Test (Phase 3)",
    4: "Place Reversal Test (Phase 4)"}
VISITS_COLUMNS = [
    "VisitID",
    "AnimalTag",
    "Start",
    "End",
    "ModuleName",
    "Cage",
    "Corner",
    "CornerCondition",
    "PlaceError",
    "AntennaNumber",
    "AntennaDuration",
    "PresenceNumber",
    "PresenceDuration",
    "VisitSolution",
    "LickNumber",
    "LickContactTime",
    "LickDuration"]
NOSEPOKE_COLUMNS = [
    "VisitID",
    "Start",
    "End",
    "Side",
    "SideCondition",
    "SideError",
    "TimeError",
    "ConditionError",
    "LickNumber",
    "LickContactTime",
    "LickDuration",
    "AirState",
    "DoorState",
    "LED1State",
    "LED2State",
    "LED3State",
    "LickStartTime"]
CORNER_TO_SIDE = {
    1: 1,
    2: 3,
    3: 5,
    4: 7}

# %% DATA CLASSES
@dataclass(frozen=True)
class GroupProfile:
    """Behavioral parameters for one synthetic group."""

    run_group: str
    group_name: str
    rfid_start: int
    visit_rate_scale: float
    phase2_lick_probability: float
    phase3_asymptote: float
    phase3_tau_hours: float
    phase4_asymptote: float
    phase4_tau_hours: float
    phase4_old_corner_floor: float
    phase4_old_corner_tau_hours: float
    reward_probability: float

# %% SYNTHETIC BEHAVIOR FUNCTIONS
def logistic_learning_probability(
    elapsed_hours: float,
    *,
    chance_level: float,
    asymptote: float,
    tau_hours: float,
    mouse_shift: float) -> float:
    """Return a smooth learning curve bounded between chance and asymptote."""

    learned_fraction = 1.0 - np.exp(-max(elapsed_hours, 0.0) / float(tau_hours))
    probability = chance_level + (float(asymptote) - chance_level) * learned_fraction + mouse_shift
    return float(np.clip(probability, 0.05, 0.96))

def sample_elapsed_hours(
    rng: np.random.Generator,
    *,
    duration_hours: float,
    phase_start_clock_hour: float,
    awake_start_hour: float = 6.0,
    awake_end_hour: float = 18.0) -> float:
    """Sample one phase-relative visit time with a simple day/night rhythm."""

    while True:
        elapsed_hours = float(rng.uniform(0.0, duration_hours))
        clock_hour = (phase_start_clock_hour + elapsed_hours) % 24.0
        acceptance_probability = 1.0 if awake_start_hour <= clock_hour < awake_end_hour else 0.25
        if rng.random() <= acceptance_probability:
            return elapsed_hours

def choose_incorrect_corner(
    rng: np.random.Generator,
    *,
    correct_corner: int) -> int:
    """Choose any corner except the current correct corner."""

    candidates = [corner for corner in (1, 2, 3, 4) if corner != int(correct_corner)]
    return int(rng.choice(candidates))

def choose_phase_corner(
    rng: np.random.Generator,
    *,
    phase_number: int,
    elapsed_hours: float,
    corner_phase3: int,
    corner_phase4: int,
    profile: GroupProfile,
    mouse_shift: float) -> int:
    """Choose the visited corner for one synthetic visit."""

    if phase_number in {1, 2}:
        return int(rng.integers(1, 5))

    if phase_number == 3:
        correct_probability = logistic_learning_probability(
            elapsed_hours,
            chance_level=0.25,
            asymptote=profile.phase3_asymptote,
            tau_hours=profile.phase3_tau_hours,
            mouse_shift=mouse_shift)
        if rng.random() <= correct_probability:
            return int(corner_phase3)
        return choose_incorrect_corner(rng, correct_corner=corner_phase3)

    old_corner_probability = (
        profile.phase4_old_corner_floor
        + 0.38 * np.exp(-max(elapsed_hours, 0.0) / profile.phase4_old_corner_tau_hours))
    new_corner_probability = logistic_learning_probability(
        elapsed_hours,
        chance_level=0.25,
        asymptote=profile.phase4_asymptote,
        tau_hours=profile.phase4_tau_hours,
        mouse_shift=mouse_shift * 0.7)
    old_corner_probability = float(np.clip(old_corner_probability, 0.02, 0.75))
    new_corner_probability = float(np.clip(new_corner_probability, 0.05, 0.92))
    if old_corner_probability + new_corner_probability > 0.94:
        scale = 0.94 / (old_corner_probability + new_corner_probability)
        old_corner_probability *= scale
        new_corner_probability *= scale

    draw = rng.random()
    if draw <= new_corner_probability:
        return int(corner_phase4)
    if draw <= new_corner_probability + old_corner_probability:
        return int(corner_phase3)

    neutral_corners = [
        corner
        for corner in (1, 2, 3, 4)
        if corner not in {int(corner_phase3), int(corner_phase4)}]
    return int(rng.choice(neutral_corners))

def visit_has_nosepoke(
    rng: np.random.Generator,
    *,
    phase_number: int,
    is_correct_corner: bool,
    profile: GroupProfile) -> bool:
    """Sample whether a visit contains at least one nose-poke event."""

    if phase_number == 1:
        probability = 0.58
    elif phase_number == 2:
        probability = 0.78
    elif is_correct_corner:
        probability = 0.92
    else:
        probability = 0.42
    if profile.group_name == "Group B" and phase_number in {3, 4}:
        probability -= 0.08
    return bool(rng.random() <= np.clip(probability, 0.05, 0.98))

def visit_lick_count(
    rng: np.random.Generator,
    *,
    phase_number: int,
    has_nosepoke: bool,
    is_correct_corner: bool,
    profile: GroupProfile) -> int:
    """Sample the visit-level lick count."""

    if not has_nosepoke:
        return 0

    if phase_number == 1:
        lick_probability = 0.38
        mean_licks = 5.0
    elif phase_number == 2:
        lick_probability = profile.phase2_lick_probability
        mean_licks = 11.0 if profile.group_name == "Group A" else 7.5
    elif is_correct_corner:
        lick_probability = profile.reward_probability
        mean_licks = 15.0 if profile.group_name == "Group A" else 11.0
    else:
        lick_probability = 0.12
        mean_licks = 3.0

    if rng.random() > lick_probability:
        return 0
    return int(max(1, rng.poisson(mean_licks)))

# %% ROW BUILDERS
def format_timestamp(timestamp: pd.Timestamp) -> str:
    """Format timestamps in the same style as IntelliCage text exports."""

    return timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def build_mice_table(
    profile: GroupProfile,
    *,
    mouse_count: int,
    rng: np.random.Generator) -> pd.DataFrame:
    """Create one synthetic `Mice.txt` table."""

    rows: list[dict[str, object]] = []
    for mouse_index in range(mouse_count):
        corner_phase3 = int((mouse_index % 4) + 1)
        corner_phase4 = int(((mouse_index + 2) % 4) + 1)
        rows.append(
            {
                "RFID": profile.rfid_start + mouse_index,
                "SEX": "female" if mouse_index % 2 else "male",
                "DOB": "01.09.25",
                "ET": f"{profile.group_name[-1]}{mouse_index + 1:02d}",
                "VIRUS": profile.group_name,
                "Corner Phase 3": corner_phase3,
                "Corner Phase 4": corner_phase4})
    return pd.DataFrame(rows)

def build_phase_tables(
    mice: pd.DataFrame,
    profile: GroupProfile,
    *,
    phase_number: int,
    experiment_start: pd.Timestamp,
    rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create visit and nose-poke rows for one group and phase."""

    phase_name = PHASE_NAMES[phase_number]
    phase_start_hours = PHASE_START_HOURS[phase_number]
    phase_duration_hours = PHASE_DURATIONS_HOURS[phase_number]
    phase_start = experiment_start + pd.to_timedelta(phase_start_hours, unit="h")
    phase_start_clock_hour = (experiment_start.hour + phase_start_hours) % 24.0
    base_rates = {
        1: 0.38,
        2: 0.48,
        3: 0.58,
        4: 0.54}
    visit_rows: list[dict[str, object]] = []
    nosepoke_rows: list[dict[str, object]] = []
    visit_id = 0

    for _, mouse in mice.iterrows():
        mouse_shift = float(rng.normal(0.0, 0.055))
        rate = base_rates[phase_number] * profile.visit_rate_scale * float(rng.lognormal(0.0, 0.12))
        target_visit_count = int(rng.poisson(rate * phase_duration_hours))
        elapsed_values = sorted(
            sample_elapsed_hours(
                rng,
                duration_hours=phase_duration_hours,
                phase_start_clock_hour=phase_start_clock_hour)
            for _ in range(target_visit_count))

        for elapsed_hours in elapsed_values:
            start = phase_start + pd.to_timedelta(elapsed_hours, unit="h")
            duration_seconds = float(np.clip(rng.gamma(shape=2.0, scale=2.2), 0.7, 24.0))
            end = start + pd.to_timedelta(duration_seconds, unit="s")
            corner_phase3 = int(mouse["Corner Phase 3"])
            corner_phase4 = int(mouse["Corner Phase 4"])
            corner = choose_phase_corner(
                rng,
                phase_number=phase_number,
                elapsed_hours=elapsed_hours,
                corner_phase3=corner_phase3,
                corner_phase4=corner_phase4,
                profile=profile,
                mouse_shift=mouse_shift)
            assigned_corner = corner_phase3 if phase_number == 3 else corner_phase4
            is_learning_phase = phase_number in {3, 4}
            is_correct_corner = bool(is_learning_phase and corner == assigned_corner)
            has_nosepoke = visit_has_nosepoke(
                rng,
                phase_number=phase_number,
                is_correct_corner=is_correct_corner,
                profile=profile)
            licks = visit_lick_count(
                rng,
                phase_number=phase_number,
                has_nosepoke=has_nosepoke,
                is_correct_corner=is_correct_corner,
                profile=profile)
            lick_contact_time = round(float(licks) * 0.018, 6)
            lick_duration = round(float(licks) * float(rng.uniform(0.14, 0.22)), 6)
            place_error = 0 if not is_learning_phase or is_correct_corner else 1
            corner_condition = 1 if is_correct_corner else (-1 if is_learning_phase else 0)
            visit_rows.append(
                {
                    "VisitID": visit_id,
                    "AnimalTag": int(mouse["RFID"]),
                    "Start": format_timestamp(start),
                    "End": format_timestamp(end),
                    "ModuleName": MODULE_NAMES[phase_number],
                    "Cage": 1,
                    "Corner": corner,
                    "CornerCondition": corner_condition,
                    "PlaceError": place_error,
                    "AntennaNumber": 1,
                    "AntennaDuration": round(duration_seconds, 6),
                    "PresenceNumber": 1,
                    "PresenceDuration": round(max(0.2, duration_seconds - 0.4), 6),
                    "VisitSolution": 0,
                    "LickNumber": licks,
                    "LickContactTime": lick_contact_time,
                    "LickDuration": lick_duration})

            if has_nosepoke:
                nosepoke_start = start + pd.to_timedelta(float(rng.uniform(0.05, 0.6)), unit="s")
                nosepoke_duration_seconds = float(np.clip(rng.gamma(shape=1.6, scale=1.2), 0.2, duration_seconds))
                nosepoke_end = nosepoke_start + pd.to_timedelta(nosepoke_duration_seconds, unit="s")
                side_condition = 1 if is_correct_corner or phase_number == 2 else (-1 if is_learning_phase else 0)
                side_error = 0 if side_condition == 1 else (1 if side_condition == -1 else 0)
                nosepoke_rows.append(
                    {
                        "VisitID": visit_id,
                        "Start": format_timestamp(nosepoke_start),
                        "End": format_timestamp(nosepoke_end),
                        "Side": CORNER_TO_SIDE[corner],
                        "SideCondition": side_condition,
                        "SideError": side_error,
                        "TimeError": 0,
                        "ConditionError": place_error if is_learning_phase else 0,
                        "LickNumber": licks,
                        "LickContactTime": lick_contact_time,
                        "LickDuration": lick_duration,
                        "AirState": 0,
                        "DoorState": 0,
                        "LED1State": 0,
                        "LED2State": 0,
                        "LED3State": 0,
                        "LickStartTime": format_timestamp(nosepoke_start) if licks > 0 else ""})
            visit_id += 1

    visits = pd.DataFrame(visit_rows, columns=VISITS_COLUMNS)
    nosepokes = pd.DataFrame(nosepoke_rows, columns=NOSEPOKE_COLUMNS)
    return visits, nosepokes

# %% FILE WRITING
def write_table(dataframe: pd.DataFrame, path: Path) -> None:
    """Write one tab-separated table with parent directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, sep="\t", index=False)

def write_dataset(
    output_root: Path,
    *,
    mouse_count_per_group: int,
    random_seed: int,
    overwrite: bool) -> None:
    """Generate all synthetic files below `output_root`."""

    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_root} already exists and is not empty. Re-run with --overwrite to replace files.")
    output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    experiment_start = pd.Timestamp("2026-01-05 06:00:00")
    profiles = [
        GroupProfile(
            run_group="GruppeA",
            group_name="Group A",
            rfid_start=910200000001000,
            visit_rate_scale=1.08,
            phase2_lick_probability=0.78,
            phase3_asymptote=0.88,
            phase3_tau_hours=14.0,
            phase4_asymptote=0.78,
            phase4_tau_hours=18.0,
            phase4_old_corner_floor=0.04,
            phase4_old_corner_tau_hours=14.0,
            reward_probability=0.92),
        GroupProfile(
            run_group="GruppeB",
            group_name="Group B",
            rfid_start=910200000002000,
            visit_rate_scale=0.82,
            phase2_lick_probability=0.48,
            phase3_asymptote=0.56,
            phase3_tau_hours=34.0,
            phase4_asymptote=0.48,
            phase4_tau_hours=46.0,
            phase4_old_corner_floor=0.22,
            phase4_old_corner_tau_hours=52.0,
            reward_probability=0.72)]

    manifest_rows: list[dict[str, object]] = []
    for profile in profiles:
        mice = build_mice_table(
            profile,
            mouse_count=mouse_count_per_group,
            rng=rng)
        run_root = output_root / profile.run_group
        write_table(mice, run_root / "Mice.txt")
        for phase_number in sorted(PHASE_NAMES):
            visits, nosepokes = build_phase_tables(
                mice,
                profile,
                phase_number=phase_number,
                experiment_start=experiment_start,
                rng=rng)
            intelli_root = run_root / PHASE_NAMES[phase_number] / "IntelliCage"
            write_table(visits, intelli_root / "Visits.txt")
            write_table(nosepokes, intelli_root / "Nosepokes.txt")
            manifest_rows.append(
                {
                    "RunGroup": profile.run_group,
                    "Group": profile.group_name,
                    "Phase": PHASE_NAMES[phase_number],
                    "MouseCount": mouse_count_per_group,
                    "VisitCount": len(visits),
                    "NosepokeCount": len(nosepokes),
                    "StartHour": PHASE_START_HOURS[phase_number],
                    "DurationHours": PHASE_DURATIONS_HOURS[phase_number]})

    manifest = pd.DataFrame(manifest_rows)
    write_table(manifest, output_root / "synthetic_dataset_manifest.tsv")
    (output_root / "README.md").write_text(
        "# Synthetic IntelliCage Place Learning Dataset\n\n"
        "Pseudo-data generated for documentation and demo analyses. Group A is "
        "simulated as a stronger learner, while Group B shows flatter learning "
        "and stronger phase-4 perseveration at the previous correct corner.\n",
        encoding="utf-8")

def parse_args(argv: list[str] | None = None) -> ArgumentParser:
    """Create the command-line parser."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination dataset root. Default: {DEFAULT_OUTPUT_ROOT}")
    parser.add_argument(
        "--mouse-count-per-group",
        type=int,
        default=10,
        help="Number of synthetic mice per group.")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2701,
        help="Random seed used for reproducible pseudo-data.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files below an existing non-empty output directory.")
    return parser

def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""

    parser = parse_args(argv)
    args = parser.parse_args(argv)
    write_dataset(
        args.output_root,
        mouse_count_per_group=args.mouse_count_per_group,
        random_seed=args.random_seed,
        overwrite=args.overwrite)
    print(f"Synthetic IntelliCage dataset written to: {args.output_root.resolve()}")

# %% ENTRY POINT
if __name__ == "__main__":
    main(sys.argv[1:])
