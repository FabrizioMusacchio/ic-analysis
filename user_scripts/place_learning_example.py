"""Analyze the public synthetic IntelliCage Group A/B PL/PR dataset.

The script demonstrates the simplified public API. It is intentionally written
as an interactive VS Code script: run one ``# %%`` cell after another, inspect
the intermediate object state, and adapt the method calls for your own data.

author: Fabrizio Musacchio
date: August 2026
"""
# %% IMPORTS
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

warnings.filterwarnings("ignore", message="scipy.stats.shapiro: Input data has range zero.*", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="The behavior of wald_test will change after 0.14.*", category=FutureWarning)

import ic_analysis as ic
# %% PATHS
DATASET_ROOT = PROJECT_ROOT / "example_data" / "synthetic_group_ab_place_learning"
RESULTS_ROOT = DATASET_ROOT / "results"
# %% EXPERIMENT AND PHASE METADATA
"""Define experiment-level settings and the four PL/PR protocol phases.

The ``PHASES`` dictionary is the protocol map. Each phase defines the readable
labels, display color, and scheduled protocol start relative to experiment
start. The synthetic export-block folders happen to be named like the protocol
phases, so ``folder_name`` is kept as a readable label here. Real phase time
windows are subject-specific and therefore defined below in ``SUBJECTS``.
"""
PHASES = {
    1: {
        "short_name": "Hab",
        "long_name": "Habituation",
        "folder_name": "Phase1",
        "color": "#bfe4f7",
        "scheduled_start_hour": 0.0},
    2: {
        "short_name": "NPA",
        "long_name": "Nose-poke adaptation",
        "folder_name": "Phase2",
        "color": "#6fb2e5",
        "scheduled_start_hour": 74.0},
    3: {
        "short_name": "PL",
        "long_name": "Place learning",
        "folder_name": "Phase3",
        "color": "#3d80b8",
        "scheduled_start_hour": 122.0},
    4: {
        "short_name": "PR",
        "long_name": "Place reversal",
        "folder_name": "Phase4",
        "color": "#1f2a78",
        "scheduled_start_hour": 194.0}}

EXPERIMENT = {
    "name": "Place Learning and Place Reversal, synthetic Group A/B example",
    "root_data_path": DATASET_ROOT,
    "results_data_path": RESULTS_ROOT,
    "group_names": ["Group A", "Group B"],
    "group_colors": {
        "Group A": "#267d8f",
        "Group B": "#c7523f"},
    "mouse_day": {
        "start": "06:00",
        "end": "18:00"}}

# %% SUBJECT METADATA
"""Define which raw IntelliCage animal IDs are analyzed.

Only animals declared in ``SUBJECTS`` are included after loading. Per-animal
metadata attach group, sex, public true-ID labels, date of birth, and PL/PR
corner assignments. The ``phases`` field stores the real begin/end timestamps
for each phase and each subject, which is necessary when run-group or cage
folders started at different real clock times.
"""
SUBJECTS = {
    "910200000001000": {
        "group": "Group A",
        "sex": "male",
        "true_id": "A01",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 1,
            4: 3},
        "phases": {
            1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
            2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")},
            3: {"time_window": ("2026-01-10 08:00:00", "2026-01-13 08:00:00")},
            4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}},
    "910200000001001": {
        "group": "Group A",
        "sex": "female",
        "true_id": "A02",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 2,
            4: 4},
        "phases": {
            1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
            2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")},
            3: {"time_window": ("2026-01-10 08:00:00", "2026-01-13 08:00:00")},
            4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}},
    "910200000001002": {
        "group": "Group A",
        "sex": "male",
        "true_id": "A03",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 3,
            4: 1},
        "phases": {
            1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
            2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")},
            3: {"time_window": ("2026-01-10 08:00:00", "2026-01-13 08:00:00")},
            4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}},
    "910200000001003": {
        "group": "Group A",
        "sex": "female",
        "true_id": "A04",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 4,
            4: 2},
        "phases": {
            1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
            2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")},
            3: {"time_window": ("2026-01-10 08:00:00", "2026-01-13 08:00:00")},
            4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}},
    "910200000001004": {
        "group": "Group A",
        "sex": "male",
        "true_id": "A05",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 1,
            4: 3},
        "phases": {
            1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
            2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")},
            3: {"time_window": ("2026-01-10 08:00:00", "2026-01-13 08:00:00")},
            4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}},
    "910200000001005": {
        "group": "Group A",
        "sex": "female",
        "true_id": "A06",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 2,
            4: 4},
        "phases": {
            1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
            2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")},
            3: {"time_window": ("2026-01-10 08:00:00", "2026-01-13 08:00:00")},
            4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}},
    "910200000001006": {
        "group": "Group A",
        "sex": "male",
        "true_id": "A07",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 3,
            4: 1},
        "phases": {
            1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
            2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")},
            3: {"time_window": ("2026-01-10 08:00:00", "2026-01-13 08:00:00")},
            4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}},
    "910200000001007": {
        "group": "Group A",
        "sex": "female",
        "true_id": "A08",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 4,
            4: 2},
        "phases": {
            1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
            2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")},
            3: {"time_window": ("2026-01-10 08:00:00", "2026-01-13 08:00:00")},
            4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}},
    "910200000001008": {
        "group": "Group A",
        "sex": "male",
        "true_id": "A09",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 1,
            4: 3},
        "phases": {
            1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
            2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")},
            3: {"time_window": ("2026-01-10 08:00:00", "2026-01-13 08:00:00")},
            4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}},
    "910200000001009": {
        "group": "Group A",
        "sex": "female",
        "true_id": "A10",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 2,
            4: 4},
        "phases": {
            1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
            2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")},
            3: {"time_window": ("2026-01-10 08:00:00", "2026-01-13 08:00:00")},
            4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}},
    "910200000002000": {
        "group": "Group B",
        "sex": "male",
        "true_id": "B01",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 1,
            4: 3},
        "phases": {
            1: {"time_window": ("2026-01-05 13:30:00", "2026-01-08 15:30:00")},
            2: {"time_window": ("2026-01-08 15:30:00", "2026-01-10 15:30:00")},
            3: {"time_window": ("2026-01-10 15:30:00", "2026-01-13 15:30:00")},
            4: {"time_window": ("2026-01-13 15:30:00", "2026-01-16 15:30:00")}}},
    "910200000002001": {
        "group": "Group B",
        "sex": "female",
        "true_id": "B02",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 2,
            4: 4},
        "phases": {
            1: {"time_window": ("2026-01-05 13:30:00", "2026-01-08 15:30:00")},
            2: {"time_window": ("2026-01-08 15:30:00", "2026-01-10 15:30:00")},
            3: {"time_window": ("2026-01-10 15:30:00", "2026-01-13 15:30:00")},
            4: {"time_window": ("2026-01-13 15:30:00", "2026-01-16 15:30:00")}}},
    "910200000002002": {
        "group": "Group B",
        "sex": "male",
        "true_id": "B03",
        "date_of_birth": "2025-08-15",
        "corner_assignments": {
            3: 3,
            4: 1},
        "phases": {
            1: {"time_window": ("2026-01-05 13:30:00", "2026-01-08 15:30:00")},
            2: {"time_window": ("2026-01-08 15:30:00", "2026-01-10 15:30:00")},
            3: {"time_window": ("2026-01-10 15:30:00", "2026-01-13 15:30:00")},
            4: {"time_window": ("2026-01-13 15:30:00", "2026-01-16 15:30:00")}}},
    "910200000002003": {
        "group": "Group B",
        "sex": "female",
        "true_id": "B04",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 4,
            4: 2},
        "phases": {
            1: {"time_window": ("2026-01-05 13:30:00", "2026-01-08 15:30:00")},
            2: {"time_window": ("2026-01-08 15:30:00", "2026-01-10 15:30:00")},
            3: {"time_window": ("2026-01-10 15:30:00", "2026-01-13 15:30:00")},
            4: {"time_window": ("2026-01-13 15:30:00", "2026-01-16 15:30:00")}}},
    "910200000002004": {
        "group": "Group B",
        "sex": "male",
        "true_id": "B05",
        "date_of_birth": "2025-09-10",
        "corner_assignments": {
            3: 1,
            4: 3},
        "phases": {
            1: {"time_window": ("2026-01-05 13:30:00", "2026-01-08 15:30:00")},
            2: {"time_window": ("2026-01-08 15:30:00", "2026-01-10 15:30:00")},
            3: {"time_window": ("2026-01-10 15:30:00", "2026-01-13 15:30:00")},
            4: {"time_window": ("2026-01-13 15:30:00", "2026-01-16 15:30:00")}}},
    "910200000002005": {
        "group": "Group B",
        "sex": "female",
        "true_id": "B06",
        "date_of_birth": "2025-10-01",
        "corner_assignments": {
            3: 2,
            4: 4},
        "phases": {
            1: {"time_window": ("2026-01-05 13:30:00", "2026-01-08 15:30:00")},
            2: {"time_window": ("2026-01-08 15:30:00", "2026-01-10 15:30:00")},
            3: {"time_window": ("2026-01-10 15:30:00", "2026-01-13 15:30:00")},
            4: {"time_window": ("2026-01-13 15:30:00", "2026-01-16 15:30:00")}}},
    "910200000002006": {
        "group": "Group B",
        "sex": "male",
        "true_id": "B07",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 3,
            4: 1},
        "phases": {
            1: {"time_window": ("2026-01-05 13:30:00", "2026-01-08 15:30:00")},
            2: {"time_window": ("2026-01-08 15:30:00", "2026-01-10 15:30:00")},
            3: {"time_window": ("2026-01-10 15:30:00", "2026-01-13 15:30:00")},
            4: {"time_window": ("2026-01-13 15:30:00", "2026-01-16 15:30:00")}}},
    "910200000002007": {
        "group": "Group B",
        "sex": "female",
        "true_id": "B08",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 4,
            4: 2},
        "phases": {
            1: {"time_window": ("2026-01-05 13:30:00", "2026-01-08 15:30:00")},
            2: {"time_window": ("2026-01-08 15:30:00", "2026-01-10 15:30:00")},
            3: {"time_window": ("2026-01-10 15:30:00", "2026-01-13 15:30:00")},
            4: {"time_window": ("2026-01-13 15:30:00", "2026-01-16 15:30:00")}}},
    "910200000002008": {
        "group": "Group B",
        "sex": "male",
        "true_id": "B09",
        "date_of_birth": "2025-07-01",
        "corner_assignments": {
            3: 1,
            4: 3},
        "phases": {
            1: {"time_window": ("2026-01-05 13:30:00", "2026-01-08 15:30:00")},
            2: {"time_window": ("2026-01-08 15:30:00", "2026-01-10 15:30:00")},
            3: {"time_window": ("2026-01-10 15:30:00", "2026-01-13 15:30:00")},
            4: {"time_window": ("2026-01-13 15:30:00", "2026-01-16 15:30:00")}}},
    "910200000002009": {
        "group": "Group B",
        "sex": "female",
        "true_id": "B10",
        "date_of_birth": "2025-09-01",
        "corner_assignments": {
            3: 2,
            4: 4},
        "phases": {
            1: {"time_window": ("2026-01-05 13:30:00", "2026-01-08 15:30:00")},
            2: {"time_window": ("2026-01-08 15:30:00", "2026-01-10 15:30:00")},
            3: {"time_window": ("2026-01-10 15:30:00", "2026-01-13 15:30:00")},
            4: {"time_window": ("2026-01-13 15:30:00", "2026-01-16 15:30:00")}}}}

"""Optional YAML subject workflow.

# ic.create_subjects_yaml_template(EXPERIMENT=EXPERIMENT, PHASES=PHASES)
# SUBJECTS = ic.load_subjects_yaml(DATASET_ROOT / "subjects.yaml")
"""
# %% PIPELINE SETTINGS
PHASE_MAX_HOURS = {3: 72.0,
                   4: 72.0}
BASE_FONT_SIZE   = 8.0
LEGEND_FONT_SIZE = 6

TIMELINE_FIGSIZE_CM             = (13.0, 5.5)
TIMELINE_FIGSIZE_CM_NO_LEGEND   = (10.0, 5.35)

PHASE_FIGSIZE_CM                = (8.8, 5.5)
PHASE_FIGSIZE_CM_W_LEGEND       = (10.8, 5.5)

SEGMENT_FIGSIZE_CM_NO_LEGEND    = (6.8, 6.0)

VIOLIN_FIGSIZE_CM               = (3.5, 5.0)

CUMULATIVE_FIGSIZE_CM           = (10.0, 5.5)

RATE_THRESHOLD_PCTS      = [50.0, 60.0, 70.0, 80.0]

PLR_PHASES = (3, 4)
PLR_METRICS = ("correct_corner_visit", 
               "correct_np_visit",
               "rewarded_correct_corner_visit")
# %% STEP 1: CREATE AND LOAD THE EXPERIMENT OBJECT
"""Create the generic IntelliCage experiment object and load raw data.

``ic.experiment`` validates the experiment/phase/subject dictionaries and
creates the results folder. ``load`` reads the IntelliCage exports, restricts
the dataset to the declared subjects, aligns analysis time columns, and writes
the loaded data plus experiment, phase, and subject metadata to the results
folder for reproducibility.
"""
my_pl_exp = ic.experiment(EXPERIMENT=EXPERIMENT, PHASES=PHASES, SUBJECTS=SUBJECTS)
my_pl_exp.load()
# %% STEP 2: PREPARE ANALYSIS TABLES
"""Prepare the analysis layer and render the first audit figure.

This step applies the 72 h PL/PR analysis window used for the public synthetic
dataset and writes analysis audit tables. The age plot reports mouse age at the
first protocol phase start, which is a useful cohort-level sanity check before
interpreting behavioral outputs.
"""
my_pl_exp.prepare_analysis(phase_max_hours=PHASE_MAX_HOURS)
# %% THE FIRST PLOT: AGE DISTRIBUTION
my_pl_exp.plot_ages(
    time_unit="months", # months, days, years
    show_N=True,
    base_font_size=BASE_FONT_SIZE,
    figsize_cm=VIOLIN_FIGSIZE_CM,
    plot_layout={"ylim": (0.0, 8),})
# %% INSPECT GENERAL ACTIVITY 
""" 
Let's first inspect the overall activity of the mice in the experiment. 
The following code will generate a plot showing the activity of the mice over 
time, binned by hours. You can adjust the bin size to suit your analysis needs. 
The plot will include all phases (phases="all") and day phases (dayphase="all"), 
and it will use the specified spread metric and plot style. 

In case you want to limit the plot to one specific phase, you can set the phases 
parameter to the desired phase number (e.g., phases=2 for phase 2).
"""
my_pl_exp.plot_mice_activity(
    bin_hours=1, # 1, or 12, or 24...whatever suits your analysis. 
    phases="all", # all, or select specific phases (2,3), (3,4), etc.
    dayphase="all", # all, day, night
    phase_max_hours=PHASE_MAX_HOURS,
    spread_metric="sem",
    plot_style="line",
    day_night_indicator=("aw", "sl"),
    base_font_size=BASE_FONT_SIZE,
    figsize_cm=TIMELINE_FIGSIZE_CM,
    plot_layout={"legend_font_size":LEGEND_FONT_SIZE, 
                    "legend":True})

""" 
Another way to summarize the activity is to plot the phase 
activity summary, which shows the average activity of the mice during 
each phase. This can be useful to compare the activity levels between 
different phases and groups. The statistics shown in the plot indicate
significant differences in activity compared to the phase 1 (=baseline) 
activity.
"""
my_pl_exp.plot_phase_activity_summary(
    dayphase="all", # "all", "day", "night"
    phase_max_hours=PHASE_MAX_HOURS,
    base_font_size=BASE_FONT_SIZE,
    figsize_cm=(7, 8),
    show_N=True,
    xtick_rotation=35,
    plot_layout={
        "legend": True,
        "legend_loc": "upper right",
        "legend_font_size": LEGEND_FONT_SIZE,
        "ylim": (0.0, 20),})
# %% INSPECT DRINKING BEHAVIOR
""" 
Next we would inspect the drinking behavior of the mice. This is important,
because drinking is a vital for the subjects' survival and can be a confounding factor
in the analysis of learning behavior. Thus, let's inspect, whether the mice adapted to 
the nosepoke (NPA) and show continuous drinking behavior during all phases. 
"""

my_pl_exp.plot_NP_adaptation(
    phases="all", # "all", (2,4)
    bin_hours=1, # 1, or 12, or 24...whatever suits your analysis.
    dayphase="all", # "all", "day", "night"
    phase_max_hours=PHASE_MAX_HOURS,
    spread_metric="sem",
    plot_style="line",
    base_font_size=BASE_FONT_SIZE,
    figsize_cm=TIMELINE_FIGSIZE_CM_NO_LEGEND,
    day_night_indicator=("aw", "sl"),
    plot_layout={"legend_font_size":LEGEND_FONT_SIZE, 
                    "legend":True})
my_pl_exp.plot_NP_counts(
    phases="all", #"all" (2,4)
    bin_hours=1, # 1, or 12, or 24...whatever suits your analysis.
    dayphase="all", # "all", "day", "night"
    phase_max_hours=PHASE_MAX_HOURS,
    spread_metric="sem",
    plot_style="line",
    base_font_size=BASE_FONT_SIZE,
    figsize_cm=TIMELINE_FIGSIZE_CM_NO_LEGEND,
    day_night_indicator=("aw", "sl"),
    plot_layout={"legend_font_size":LEGEND_FONT_SIZE, 
                    "legend":True})
my_pl_exp.plot_licking_counts(
    phases="all",
    bin_hours=1, # 1, or 12, or 24...whatever suits your analysis.
    dayphase="all", # "all", "day", "night"
    phase_max_hours=PHASE_MAX_HOURS,
    spread_metric="sem",
    plot_style="line",
    base_font_size=BASE_FONT_SIZE,
    figsize_cm=TIMELINE_FIGSIZE_CM_NO_LEGEND,
    day_night_indicator=("aw", "sl"),
    plot_layout={"legend_font_size":LEGEND_FONT_SIZE, 
                    "legend":True})

""" HEDONIC/ANHEDONIC BEHAVIOR: BOTTLE PREFERENCE
In case you have a two-bottle choice setup, with one bottle containing plain water 
and the other containing a sweet solution (e.g., saccharin) in each corner,
you can assess the hedonic or anhedonic behavior of the mice via the bottle preference.
The following code will generate a plot showing the bottle preference of the mice over
time, binned by hours. You can adjust the bin size to suit your analysis needs:
"""
BOTTLE_BIN_HOURS = [24, 2 * 24]
for current_bin_hours in BOTTLE_BIN_HOURS:
    # current_bin_hours=BOTTLE_BIN_HOURS[0]
    my_pl_exp.plot_bottle_preference(
        phases="all",
        dayphase="day", # "all", "day", "night" - here, it makes most sense to use the day phase, 
                        # because the mice are awake and active during the day. During night, they 
                        # are mostly asleep and not drinking, which would confound the analysis.
                        # You can inspect the night phase drinking behavior with the plots you
                        # have just generated above.
        left_bottle="plain water", # define which bottle is the left and which is the right, based on your experimental setup
        right_bottle="saccharin",
        calc="right_bottle/left_bottle", # "right_bottle/left_bottle" or "left_bottle/right_bottle" or "left_bottle" or "right_bottle"
        bin_h=current_bin_hours,
        phase_max_hours=PHASE_MAX_HOURS,
        spread_metric="sem",
        plot_style="line",
        base_font_size=BASE_FONT_SIZE,
        x_unit="days", # days, hours, weeks - in case your bin_h is e.g. 1 week, choose "weeks" as x_unit. 
        indicate_dots=True,
        figsize_cm=TIMELINE_FIGSIZE_CM_NO_LEGEND,
        plot_layout={
            "xticks": np.arange(0,13,1),
            "legend_loc": "best",
            "legend_font_size": 11,
            "legend_font_size":LEGEND_FONT_SIZE})

    # Alternative bottle-preference modes:
    my_pl_exp.plot_bottle_preference(
        phases="all",
        dayphase="day",
        left_bottle="plain water",
        right_bottle="saccharin",
        calc="left_bottle",
        bin_h=current_bin_hours,
        phase_max_hours=PHASE_MAX_HOURS,
        spread_metric="sem",
        plot_style="line",
        base_font_size=BASE_FONT_SIZE,
        x_unit="days",
        indicate_dots=True,
        figsize_cm=TIMELINE_FIGSIZE_CM_NO_LEGEND,
        plot_layout={
            "xticks": np.arange(0,13,1),
            "legend_loc": "best",
            "legend_font_size": 11,
            "legend_font_size":LEGEND_FONT_SIZE})
    my_pl_exp.plot_bottle_preference(
        phases="all",
        dayphase="day",
        left_bottle="plain water",
        right_bottle="saccharin",
        calc="right_bottle",
        bin_h=current_bin_hours,
        phase_max_hours=PHASE_MAX_HOURS,
        spread_metric="sem",
        plot_style="line",
        base_font_size=BASE_FONT_SIZE,
        x_unit="days",
        indicate_dots=True,
        figsize_cm=TIMELINE_FIGSIZE_CM_NO_LEGEND,
        plot_layout={
            "xticks": np.arange(0,13,1),
            "legend_loc": "best",
            "legend_font_size": 11,
            "legend_font_size":LEGEND_FONT_SIZE})
    my_pl_exp.plot_bottle_preference(
        phases="all",
        dayphase="day",
        left_bottle="plain water",
        right_bottle="saccharin",
        calc="left_bottle/right_bottle",
        bin_h=current_bin_hours,
        phase_max_hours=PHASE_MAX_HOURS,
        spread_metric="sem",
        plot_style="line",
        base_font_size=BASE_FONT_SIZE,
        x_unit="days",
        indicate_dots=True,
        figsize_cm=TIMELINE_FIGSIZE_CM_NO_LEGEND,
        plot_layout={
            "xticks": np.arange(0,13,1),
            "legend_loc": "best",
            "legend_font_size": 11,
            "legend_font_size":LEGEND_FONT_SIZE})
# %% EVALUATE LEARNING AND REVERSAL BEHAVIOR
""" 
A first step to evaluate learning and reversal behavior is to inspect the timecourse of the
learning curves. The following code will generate plots showing the learning curves of the 
mice over time, binned by day-parts (awake/sleep). We plot the learning curves (i.e., 
rate of correct corner visits and error rate) for three different metrics: 

1. correct corner visits: All visits to the correct corner, regardless of whether they were rewarded or not, 
2. correct nosepoke visits: All visits to the correct corner that were followed by a nosepoke, regardless of whether they were rewarded or not, 
3. rewarded correct corner visits: All visits to the correct corner that were rewarded.

"""
for current_phase in PLR_PHASES:
    # current_phase = PLR_PHASES[0]
    for current_metric in PLR_METRICS:
        # current_metric = PLR_METRICS[0]
        my_pl_exp.plot_plr_phase_segment_rate(
            phase_number=current_phase,
            metric=current_metric,
            dayphase="all", # all, day, night
            phase_max_hours=PHASE_MAX_HOURS,
            spread_metric="sem",
            base_font_size=BASE_FONT_SIZE,
            figsize_cm=SEGMENT_FIGSIZE_CM_NO_LEGEND,
            plot_layout={"legend": True,
                         "legend_loc": "lower right",
                         "legend_font_size": LEGEND_FONT_SIZE})
        my_pl_exp.plot_plr_phase_segment_error_rate(
            phase_number=current_phase,
            metric=current_metric,
            error_against="selected_success",
            dayphase="all", # all, day, night
            phase_max_hours=PHASE_MAX_HOURS,
            spread_metric="sem",
            base_font_size=BASE_FONT_SIZE,
            figsize_cm=SEGMENT_FIGSIZE_CM_NO_LEGEND,
            plot_layout={"legend": True,
                         "legend_loc": "lower right",
                         "legend_font_size": LEGEND_FONT_SIZE})

""" 
To assess day-by-day performance statistically, we can plot the endpoint-results
(both correct corner visit rate and error rate) for each day of the learning and 
reversal phases. Again we do so for the three different metrics described above:
"""
for current_phase in PLR_PHASES:
    # current_phase = PLR_PHASES[0]
    for current_metric in PLR_METRICS:
        # current_metric = PLR_METRICS[0]
        my_pl_exp.plot_plr_awake_day_rate(
            phase_number=current_phase,
            metric=current_metric,
            phase_day=(1, 2, 3),
            dayphase="day", # all, day, night
            phase_max_hours=PHASE_MAX_HOURS,
            base_font_size=BASE_FONT_SIZE,
            figsize_cm=VIOLIN_FIGSIZE_CM,
            plot_layout={"legend": True,
                         "legend_loc": "lower right",
                         "legend_font_size": LEGEND_FONT_SIZE})
        my_pl_exp.plot_plr_awake_day_error_rate(
            phase_number=current_phase,
            metric=current_metric,
            error_against="selected_success",
            phase_day=(1, 2, 3),
            dayphase="day", # all, day, night
            phase_max_hours=PHASE_MAX_HOURS,
            base_font_size=BASE_FONT_SIZE,
            figsize_cm=VIOLIN_FIGSIZE_CM,
            plot_layout={"legend": True,
                         "legend_loc": "lower right",
                         "legend_font_size": LEGEND_FONT_SIZE})

""" 
While plot_plr_phase_segment_rate() gives you the learning curves for a
fixed bin size (12 h or day/night phases), you may want to inspect the learning 
curves for different bin sizes. The toolbox allows you to plot the learning curves 
for different bin sizes of your choice (e.g., 1 h, 2 h, 4 h, etc.). The functions 
used below work on phases, thus you need to specify the phase number and the metric 
you want to plot:
"""
for current_phase in PLR_PHASES:
    # current_phase=3
    for current_metric in PLR_METRICS:
        # current_metric=PLR_METRICS[0]
        my_pl_exp.plot_plr_learning_rate(
            phase_number=current_phase,
            metric=current_metric,
            bin_hours=1,
            dayphase="all", # all, day, night
            phase_max_hours=PHASE_MAX_HOURS,
            spread_metric="sem",
            plot_style="line",
            base_font_size=BASE_FONT_SIZE,
            figsize_cm=PHASE_FIGSIZE_CM,
            plot_layout={"legend_font_size":LEGEND_FONT_SIZE, 
                            "legend":True})
        my_pl_exp.plot_plr_learning_counts(
            phase_number=current_phase,
            metric=current_metric,
            bin_hours=1,
            dayphase="all", # all, day, night
            phase_max_hours=PHASE_MAX_HOURS,
            spread_metric="sem",
            plot_style="line",
            base_font_size=BASE_FONT_SIZE,
            figsize_cm=PHASE_FIGSIZE_CM,
            plot_layout={"legend_font_size":LEGEND_FONT_SIZE, 
                         "legend":True})
        
""" 
In case you place learning experiment (PL) included a reversal phase (PR), you can also inspect 
the learning curves for the reversal phase. plot_plr_reversal_components() will plot the
corner visit rates for the as correct assigned corner during PR, the previously correct corner in PL, 
and the other two corners:
"""
my_pl_exp.plot_plr_reversal_components(
    phase_number=4,
    bin_hours=1,
    dayphase="all", # all, day, night
    phase_max_hours=PHASE_MAX_HOURS,
    spread_metric="sem",
    plot_style="line",
    base_font_size=BASE_FONT_SIZE,
    figsize_cm=PHASE_FIGSIZE_CM_W_LEGEND,
    plot_layout={"legend_font_size":LEGEND_FONT_SIZE, 
                    "legend":True})
# %% DEEPER ANALYSIS: ASSESS LEARNING ONSETS AND EXPERIENCE
"""
Next you may be interested in assessing the learning onsets and experience of 
the mice. The following code will generate plots showing the learning onsets 
and experience for each phase and metric. You can also specify different thresholds 
for the learning rates to assess the learning onsets at different levels of performance.

Please refer to the documentation of the toolbox for more details on how to interpret
these plots and the underlying data.
"""
for current_phase in PLR_PHASES:
    # current_phase = PLR_PHASES[0]
    for current_metric in PLR_METRICS:
        # current_metric = PLR_METRICS[0]
        my_pl_exp.plot_plr_experience_learning_curve(
            phase_number=current_phase,
            metric=current_metric,
            dayphase="day",
            phase_max_hours=PHASE_MAX_HOURS,
            spread_metric="sem",
            base_font_size=BASE_FONT_SIZE,
            figsize_cm=TIMELINE_FIGSIZE_CM)
        my_pl_exp.plot_plr_experience_learning_onset(
            phase_number=current_phase,
            metric=current_metric,
            dayphase="day",
            phase_max_hours=PHASE_MAX_HOURS,
            base_font_size=BASE_FONT_SIZE,
            figsize_cm=VIOLIN_FIGSIZE_CM,
            show_N=True,
            plot_layout={"ylim": (0.0, None)})
        for current_threshold in RATE_THRESHOLD_PCTS:
            # current_threshold=RATE_THRESHOLD_PCTS[0]
            my_pl_exp.plot_plr_threshold_onset(
                phase_number=current_phase,
                metric=current_metric,
                threshold_pct=current_threshold,
                bin_hours=1,
                dayphase="day",
                phase_max_hours=PHASE_MAX_HOURS,
                base_font_size=BASE_FONT_SIZE,
                figsize_cm=VIOLIN_FIGSIZE_CM)
    my_pl_exp.plot_plr_derived_ratio(
        phase_number=current_phase,
        numerator_col="rewarded_correct_corner_visit",
        denominator_col="correct_np_visit",
        metric_name="completion_efficiency",
        title="completion efficiency",
        ylabel="Rewarded correct / correct NP [%]",
        phase_day=(1, 2, 3),
        dayphase="day",
        phase_max_hours=PHASE_MAX_HOURS,
        base_font_size=BASE_FONT_SIZE,
        figsize_cm=VIOLIN_FIGSIZE_CM)

my_pl_exp.plot_plr_derived_ratio(
    phase_number=4,
    numerator_col="correct_corner_visit",
    denominator_col="new_or_previous_correct_corner_visit",
    metric_name="reversal_preference_index", 
    title="reversal preference index",
    ylabel="New / (new + previous)",
    phase_day=(1, 2, 3),
    dayphase="day",
    phase_max_hours=PHASE_MAX_HOURS,
    value_scale=1.0,
    format_as_percent=False,
    reference_line=0.5,
    base_font_size=BASE_FONT_SIZE,
    figsize_cm=VIOLIN_FIGSIZE_CM,
    plot_layout={"ylim": (0.0, 1.5)})

"""
The cumulative role curves show how PL and PR corner preferences accumulate
over time across the selected protocol phases:
"""
my_pl_exp.plot_plr_cumulative_preferences(
    phases=(2, 3, 4),
    dayphase="day",
    phase_max_hours=PHASE_MAX_HOURS,
    spread_metric="sem",
    plot_style="line",
    day_night_indicator=("aw", "sl"),
    output_dir=RESULTS_ROOT / "plr_cumulative",
    base_font_size=BASE_FONT_SIZE,
    figsize_cm=CUMULATIVE_FIGSIZE_CM,
    plot_layout={"legend_font_size":LEGEND_FONT_SIZE, 
                 "legend":True})
# %% END
print(f"Done. All synthetic analysis outputs were written to: {RESULTS_ROOT}")
