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
  --spread-metric sem \
  --plot-style line \
  --phase2-plot-style line
```

Optional phase-time limits can be passed explicitly, for example:

```bash
conda run -n ic_placelearning python user_scripts/analyze_4month_cohort.py \
  --bin-hours 2 \
  --phase-max-hours 4=73.8913
```

The user script currently defaults to:

- excluding `WT` from the poster-oriented analysis
- keeping the original group names unchanged
- aligning the experiment to mouse day 0 at `06:00`
- using a `12 h` awake / `12 h` sleep cycle
- using the protocol phase starts `0 h`, `74 h`, `122 h`, `194 h`, `266 h`

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
  - phase-2 adaptation and full-experiment phase-2 control plots
  - phase-3 and phase-4 rewarded correct-corner visit counts
  - phase-3 and phase-4 correct-corner visit rates
  - phase-3 and phase-4 correct NP visit rates
  - phase-3 and phase-4 rewarded correct-corner visit rates
  - phase-4 reversal corner-component plots
  - all-group overlays for visit counts and place-learning rates

Plot filenames now include both the phase prefix and the plotted metric, for
example `phase3_rewarded_correct_corner_visit_rate_*`.

## Metric definitions

The Python workflow keeps four place-learning metrics in parallel:

- `correct_corner_visit_rate`
  `visits in assigned correct corner / all visits`
- `correct_np_visit_rate`
  `visits in assigned correct corner with nose-poke / all visits`
- `rewarded_correct_corner_visit_rate`
  `visits in assigned correct corner with nose-poke and licking / all visits`
- `matlab_placeerror_only`
  Legacy MATLAB-compatible definition: `PlaceError == 0` only.

For phase 2, the default secondary metric is `lick_positive_visits`, because it
is usually easier to interpret for learning/adaptation than the raw lick count.

For phase 4, additional reversal summaries separate:

- visits to the new correct corner
- visits to the previous correct corner
- visits to the neutral incorrect corners

## Phase naming

Plot titles use the short phase labels:

- `Phase 1 -> Free Hab`
- `Phase 2 -> NPA`
- `Phase 3 -> PL`
- `Phase 4 -> PR`

The filenames keep the explicit `phase1` to `phase4` prefixes.

## Time alignment

The analysis distinguishes between the raw IntelliCage file phases and a second
poster-oriented aligned timeline:

- day 0 starts at the configured mouse-day onset, default `06:00`
- the aligned phase windows follow the protocol schedule rather than the exact
  file boundary
- awake periods remain unshaded and sleep periods are shaded light grey

This makes datasets with slightly different placement times or slightly delayed
phase switches directly comparable on a common experimental timeline.
