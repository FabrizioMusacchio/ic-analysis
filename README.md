# IntelliCage Analysis Toolkit

![GitHub Release](https://img.shields.io/github/v/release/FabrizioMusacchio/ic-analysis) [![PyPI version](https://img.shields.io/pypi/v/ic-analysis.svg)](https://pypi.org/project/ic-analysis/) [![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-green.svg)](https://github.com/FabrizioMusacchio/ic-analysis?tab=GPL-3.0-1-ov-file) ![Tests](https://github.com/FabrizioMusacchio/ic-analysis/actions/workflows/ic_analysis_tests.yml/badge.svg) [![GitHub last commit](https://img.shields.io/github/last-commit/FabrizioMusacchio/ic-analysis)](https://github.com/FabrizioMusacchio/ic-analysis/commits/main/)  [![codecov](https://img.shields.io/codecov/c/github/FabrizioMusacchio/ic-analysis?logo=codecov)](https://codecov.io/gh/fabriziomusacchio/ic-analysis)  [![GitHub Issues Open](https://img.shields.io/github/issues/FabrizioMusacchio/ic-analysis)](https://github.com/FabrizioMusacchio/ic-analysis/issues) [![GitHub Issues Closed](https://img.shields.io/github/issues-closed/FabrizioMusacchio/ic-analysis?color=53c92e)](https://github.com/FabrizioMusacchio/ic-analysis/issues?q=is%3Aissue%20state%3Aclosed) [![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-pr/FabrizioMusacchio/ic-analysis)](https://github.com/FabrizioMusacchio/ic-analysis/pulls)   ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/fabriziomusacchio/ic-analysis) [![PyPI - Downloads](https://img.shields.io/pypi/dm/ic-analysis?logo=pypy&label=PiPY%20downloads&color=blue)](https://pypistats.org/packages/ic-analysis) [![PyPI Total Downloads](https://static.pepy.tech/personalized-badge/ic-analysis?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=PiPY+total+downloads)](https://pepy.tech/projects/ic-analysis)   [![Zenodo Archive](https://img.shields.io/badge/Zenodo%20Archive-10.5281%2Fzenodo.22181525-blue)](https://doi.org/10.5281/zenodo.22181525)  

<!-- [![Documentation Status](https://readthedocs.org/projects/ic-analysis/badge/?version=latest)](https://ic-analysis.readthedocs.io/en/latest/?badge=latest) 
[![Example Datasets on Zenodo](https://img.shields.io/badge/Example%20Datasets-10.5281%2Fzenodo.21603005-blue)](https://doi.org/10.5281/zenodo.21603005) 
[![Read the docs](https://badgen.net/badge/rtd/Documentation)](https://ic-analysis.readthedocs.io)-->

A Python toolkit for standardizing the analysis of *IntelliCage* experiments.

The package provides reusable tools to load *IntelliCage* text exports, define experiment and subject metadata directly in Python, merge those metadata with visit and nose-poke records, compute behavioral metrics, and create publication-oriented summary plots. Raw export folders are treated as technical export blocks; biological analysis phases are defined by subject-specific time windows, ensuring standardized analysis across experiments.


## Installation
The toolkit requires Python 3.12 or newer.

Create a clean environment and install the package in editable mode:

```bash
conda create -n ic_analysis python=3.12 -y
conda activate ic_analysis
pip install -e .
```

For development and documentation work:

```bash
pip install -e ".[dev]"
```

## Package structure
The import package is `ic_analysis`:

- `ic_analysis.loader` Reads *IntelliCage*-style `Visits.txt` and `Nosepokes.txt` files from one or more export blocks, merges script-defined subject metadata, and adds experiment-relative timing and event annotations.
- `ic_analysis.metrics` Computes activity, bottle-preference, place-learning, reversal-learning, responder, onset, and group-comparison summary tables for internal workflow use and advanced custom analyses.
- `ic_analysis.plotting` Creates group-level and mouse-level figures from the computed metric tables, including raw or relative left/right bottle-consumption trajectories.
- `ic_analysis.metadata` Defines experiment phases and per-subject metadata in Python user scripts.
- `ic_analysis.experiment` Provides the generic object-oriented experiment workflow. Generic methods use names such as `plot_ages`, `plot_mice_activity`, and `plot_bottle_preference`; place-learning and reversal methods use the `plot_plr_*` prefix.

Example import:

```python
from pathlib import Path

import ic_analysis as ic

dataset_root = Path("example_data/synthetic_group_ab_place_learning")
phases = {
    1: {
        "short_name": "Hab",
        "long_name": "Habituation",
        "scheduled_start_hour": 0.0},
    2: {
        "short_name": "NPA",
        "long_name": "Nose-poke adaptation",
        "scheduled_start_hour": 74.0},
    3: {
        "short_name": "PL",
        "long_name": "Place learning",
        "scheduled_start_hour": 122.0},
    4: {
        "short_name": "PR",
        "long_name": "Place reversal",
        "scheduled_start_hour": 194.0}}

experiment = {
    "name": "Synthetic PL/PR example",
    "root_data_path": dataset_root,
    "results_data_path": dataset_root / "results",
    "group_names": ["Group A", "Group B"],
    "mouse_day": {
        "start": "06:00",
        "end": "18:00"}}

subjects = {
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
            4: {"time_window": ("2026-01-13 08:00:00", "2026-01-16 08:00:00")}}}}

my_pl_exp = ic.experiment(EXPERIMENT=experiment, PHASES=phases, SUBJECTS=subjects)
my_pl_exp.load()
my_pl_exp.prepare_analysis(phase_max_hours={3: 72.0, 4: 72.0})
my_pl_exp.plot_ages(time_unit="months", show_N=True, figsize_cm=(5.8, 10.0))
my_pl_exp.plot_mice_activity(bin_hours=1, phases="all", dayphase="all", figsize_cm=(24.0, 10.0))
my_pl_exp.plot_phase_activity_summary(dayphase="all", figsize_cm=(7.0, 8.0))
my_pl_exp.plot_NP_adaptation(phases=2, bin_hours=1, dayphase="day")
my_pl_exp.plot_NP_counts(phases="all", bin_hours=1, dayphase="all")
my_pl_exp.plot_licking_counts(phases="all", bin_hours=1, dayphase="all")
my_pl_exp.plot_bottle_preference(
    phases="all",
    dayphase="day",
    left_bottle="plain water",
    right_bottle="saccharin",
    calc="right_bottle/left_bottle",
    bin_h=24,
    x_unit="days",
    indicate_dots=True,
    figsize_cm=(24.0, 10.0))
my_pl_exp.plot_plr_learning_rate(
    phase_number=3,
    metric="rewarded_correct_corner_visit",
    bin_hours=1,
    dayphase="day",
    figsize_cm=(12.8, 8.0))
my_pl_exp.plot_plr_learning_counts(
    phase_number=3,
    metric="rewarded_correct_corner_visit",
    bin_hours=1,
    dayphase="day",
    figsize_cm=(12.8, 8.0))
my_pl_exp.plot_plr_experience_learning_curve(phase_number=3)
my_pl_exp.plot_plr_experience_learning_onset(phase_number=3, figsize_cm=(5.8, 10.0))
```

Most behavioral analysis methods accept `dayphase="day"`, `"night"`, or
`"all"`. The default is `"day"` so sparse inactive-phase visits do not bias
learning or preference estimates. The activity overview defaults to `"all"`
because it is meant to show the full day/night rhythm.

For larger cohorts, a YAML template can be generated from detected raw animal
IDs and edited before loading:

```python
ic.create_subjects_yaml_template(EXPERIMENT=experiment, PHASES=phases)
subjects = ic.load_subjects_yaml(dataset_root / "subjects.yaml")
```

## Synthetic example data
The repository includes a small synthetic place learning *IntelliCage*-style dataset at:

```text
example_data/synthetic_group_ab_place_learning
```

It contains two groups with ten pseudo-mice each:

- `Group A`: simulated stronger place learning, better reversal adaptation, and clear saccharin preference.
- `Group B`: simulated weaker place learning, stronger phase-4 perseveration at the previous correct corner, and a plain-water preference as an anhedonia-like phenotype.

It follows a four-phase *IntelliCage* place learning and place reversal protocol on an aligned 0-266 h analysis timeline:

![IntelliCage place-learning protocol](figures/intellicage_place_learning_protocol.jpg)

- Phase 1, Free Hab: 0-74 h, free habituation and exploration of all corners. This may vary from 48-74 h depending on the experiment design.
- Phase 2, NPA: 74-122 h, nose-poke adaptation and licking behavior.
- Phase 3, PL: 122-194 h, place learning with an assigned rewarded correct
  corner.
- Phase 4, PR: 194-266 h, place reversal with a new rewarded corner and
  tracking of visits to the previous correct corner.

The run-group folders intentionally start at different real clock times: Group
A begins on 2026-01-05 at 06:00, while Group B begins 7.5 h later at 13:30.
This demonstrates why real phase `time_window` values are stored per subject.

Real and synthetic datasets use the same cage-run plus export-block layout. In
the public synthetic PL/PR dataset, the export blocks happen to be named
`Phase1`, `Phase2`, etc. because the generated export pieces match the protocol
phases. For real data, these folder names can be generic technical names such
as `Export_Block_1` or dates:

```text
data_root/
|-- CageRun_A/
|   |-- Export_Block_1/
|   |   `-- IntelliCage/
|   |       |-- Visits.txt
|   |       `-- Nosepokes.txt
|   `-- Export_Block_2/
|       `-- IntelliCage/
|           |-- Visits.txt
|           `-- Nosepokes.txt
`-- CageRun_B/
    |-- Export_Block_1/
    |   `-- IntelliCage/
    |       |-- Visits.txt
    |       `-- Nosepokes.txt
    `-- Export_Block_2/
        `-- IntelliCage/
            |-- Visits.txt
            `-- Nosepokes.txt
```

The loader concatenates all detected export blocks per cage run. The actual
analysis phases come from each subject's `time_window` entries, so one long
export, phase-matching exports, and interrupted mid-phase exports can be
analyzed with the same downstream code.

To regenerate the example data:

```bash
conda run -n ic_analysis python additional_scripts/generate_synthetic_group_ab_data.py --overwrite
```

## Demo analysis
Run the public synthetic-data workflow with:

```bash
conda run -n ic_analysis python user_scripts/place_learning_example.py
```

The script writes compact result tables and figures to:

```text
example_data/synthetic_group_ab_place_learning/results
```


## Metric definitions
The workflow keeps several place-learning metrics in parallel:

- `correct_corner_visit_rate`  
  Visits in the assigned correct corner divided by all visits.
- `correct_np_visit_rate`  
  Correct-corner visits with at least one nose-poke divided by all visits.
- `rewarded_correct_corner_visit_rate`  
  Correct-corner visits with nose-poke and licking divided by all visits.
- `bottle_preference`  
  Left/right nose-poke-side licking summarized as raw bottle consumption or as a bounded preference fraction, e.g. `right_bottle / (left_bottle + right_bottle)`.

For a place reversal phase in a place learning experiment, the toolkit also separates:

- visits to the new correct corner
- visits to the previous correct corner
- visits to the neutral incorrect corners

## Time alignment
The analysis distinguishes raw *IntelliCage* export blocks from biological analysis phases:

- export-block columns preserve the observed *IntelliCage* export structure
- analysis windows are assigned from subject-specific `time_window` values
- mouse-day and awake/sleep windows can be configured for plotting and daily summaries

This makes runs with different start times, uninterrupted long exports, and
interrupted recordings comparable on a common experiment timeline.

## Which experiments are already supported?
The toolkit is designed to be extensible to any *IntelliCage* experiment. The current release already supports the analysis of the following experiments:

* General activity and locomotor cage engagement
* Liquid-intake surveillance / drinking behavior
* Nosepoke adaptation
* Bottle preference / two-bottle choice
* Place learning
* Place learning and place reversal
* General readouts from any phase-wise structured experiment

## Where to start
We recommend to start with usage examples on the documentation website (will come soon). The folder `user_scripts/` contains interactive scripts that are described in the documentation and can be run cell by cell in VS Code's interactive window or in a notebook-like environment. They are designed to be run with provided example datasets (download from [Zenodo](https://doi.org/10.5281/zenodo.22181525) or from this repository, here stored in `example_data/`) or with your own *IntelliCage* data.

## Citation
If you use the *IntelliCage Analysis Toolkit* in scientific work, please cite it as follows:

> Musacchio, F. (2026). *IntelliCage Analysis Toolkit: A Python toolkit for standardizing the analysis of IntelliCage experiments.*. Zenodo. https://doi.org/10.5281/zenodo.22181525

<!-- Please also cite the archived IntelliCage Analysis Toolkit software version used in your analysis:

> Musacchio, F. (2026). *IntelliCage Analysis Toolkit: A Python package for analyzing place learning experiments in the IntelliCage*. Zenodo. https://doi.org/10.5281/zenodo.22181525

Zenodo software archive:
[https://doi.org/10.5281/zenodo.22181525](https://doi.org/10.5281/zenodo.22181525) -->
