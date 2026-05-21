# IntelliCage Place Learning experiment analysis

This repository contains the original MATLAB scripts and a new Python workflow
for IntelliCage place-learning data. The first Python target is the dataset in
`Data IntelliCage/BioMedX_4MonthCohort_2019`.

## Environment

For reproducibility:

```bash
conda create -n ic_placelearning python=3.12 -y
conda activate ic_placelearning
conda install -y ipykernel matplotlib pandas numpy scipy scikit-learn statsmodels pingouin
```

## Python workflow

The new Python code lives in `python_scripts/` and is split into:

- `python_scripts/intellicage_place_learning/loader.py`
  Reads `Mice.txt`, `Visits.txt`, and `Nosepokes.txt`, merges them, and adds
  analysis-friendly columns.
- `python_scripts/intellicage_place_learning/metrics.py`
  Computes binned activity and place-learning summaries on the mouse and group
  level.
- `python_scripts/intellicage_place_learning/plotting.py`
  Creates group-wise summary figures for poster preparation.
- `python_scripts/analyze_4month_cohort.py`
  Main entry point for the BioMedX 4-month cohort.

Run the complete analysis with:

```bash
conda run -n ic_placelearning python python_scripts/analyze_4month_cohort.py
```

Useful options:

```bash
conda run -n ic_placelearning python python_scripts/analyze_4month_cohort.py \
  --bin-hours 1 2 \
  --phase2-secondary-metric lick_positive_visits
```

## Outputs

By default, the script writes results to:

`python_scripts/results/BioMedX_4MonthCohort_2019/`

The output folder contains:

- merged visit and nose-poke tables
- mouse metadata and a phase manifest
- binned summary tables for each requested bin size
- plots for:
  - total visits across the full experiment
  - phase-2 adaptation
  - phase-3 place learning
  - phase-4 reversal learning

## Metric definitions

Two place-learning metrics are kept in parallel:

- `strict_rewarded`
  Poster-oriented definition: correct corner visit plus at least one linked
  nose-poke plus at least one lick.
- `matlab_placeerror_only`
  Legacy MATLAB-compatible definition: `PlaceError == 0` only.

For phase 2, the default secondary metric is `lick_positive_visits`, because it
is usually easier to interpret for learning/adaptation than the raw lick count.
