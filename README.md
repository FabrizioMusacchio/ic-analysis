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

The Python package now lives at the project root and is split into:

- `intellicage_place_learning/loader.py`
  Reads `Mice.txt`, `Visits.txt`, and `Nosepokes.txt`, merges them, and adds
  analysis-friendly columns.
- `intellicage_place_learning/metrics.py`
  Computes binned activity and place-learning summaries on the mouse and group
  level.
- `intellicage_place_learning/plotting.py`
  Creates group-wise summary figures for poster preparation.
- `user_scripts/analyze_4month_cohort.py`
  Main entry point for the BioMedX 4-month cohort.

Run the complete analysis with:

```bash
conda run -n ic_placelearning python user_scripts/analyze_4month_cohort.py
```

Useful options:

```bash
conda run -n ic_placelearning python user_scripts/analyze_4month_cohort.py \
  --bin-hours 1 2 \
  --phase2-secondary-metric lick_positive_visits \
  --spread-metric sem
```

Optional phase-time limits can be passed explicitly, for example:

```bash
conda run -n ic_placelearning python user_scripts/analyze_4month_cohort.py \
  --bin-hours 2 \
  --phase-max-hours 4=73.8913
```

## Outputs

The results directory is always created relative to the selected dataset root.
For the 4-month cohort, the default output folder is:

`Data IntelliCage/BioMedX_4MonthCohort_2019/results/`

The output folder contains:

- merged visit and nose-poke tables
- mouse metadata and a phase manifest
- suggested and per-run phase-duration limit tables
- phase-wise median activity tables and plots
- binned summary tables for each requested bin size
- plots for:
  - total visits across the full experiment
  - phase-2 adaptation
  - phase-3 and phase-4 rewarded correct visit counts
  - phase-3 and phase-4 correct visit rates
  - all-group correct visit rate overlays

Plot filenames now include both the phase prefix and the plotted metric, for
example `phase3_correct_rewarded_visit_rate_*`.

## Metric definitions

Two place-learning metrics are kept in parallel:

- `strict_rewarded`
  Poster-oriented definition: correct corner visit plus at least one linked
  nose-poke plus at least one lick.
- `matlab_placeerror_only`
  Legacy MATLAB-compatible definition: `PlaceError == 0` only.

For phase 2, the default secondary metric is `lick_positive_visits`, because it
is usually easier to interpret for learning/adaptation than the raw lick count.

## Phase naming

Plot titles use the short phase labels:

- `Phase 1 -> Free Hab`
- `Phase 2 -> NPA`
- `Phase 3 -> PL`
- `Phase 4 -> PR`

The filenames keep the explicit `phase1` to `phase4` prefixes.
