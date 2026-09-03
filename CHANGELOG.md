## IntelliCage Analysis Toolkit Changelog

See here for a detailed list of changes made in each release of the
*IntelliCage Analysis Toolkit*. Please also refer to the repository
[Releases page](https://github.com/FabrizioMusacchio/ic-analysis/releases).

Each release can be archived on Zenodo for long-term preservation and citation
purposes.

<!-- ---

## 🔜 IntelliCage Analysis Toolkit v0.1.1 UPCOMING RELEASE

### 📚 Documentation

- Updated the public documentation to describe the toolkit as a general
  IntelliCage analysis package with PL/PR as the first supported workflow.
- Documented the new object-oriented synthetic-data example and subject
  metadata pattern.
- Documented the cage-run/export-block input layout and clarified that
  export-block folders are technical raw-data pieces rather than biological
  phases.
- Expanded the PL/PR usage example with modular analysis sections, plot
  interpretation, and mathematical definitions for rates, ratios, error rates,
  threshold onsets, experience-learning onsets, and cumulative preference
  onsets.
-->

---

## 🚀 IntelliCage Analysis Toolkit v0.1.0

September 3, 2026

This is a major architectural update that generalizes the project from a place-learning-only package into the *IntelliCage Analysis Toolkit*. We renamed the project from `ic-placelearning` to `ic-analysis`, moved the public import package to `ic_analysis`, and converted the workflow to a generic object-oriented experiment API. The release also separates technical IntelliCage export blocks from biological analysis phases: raw export folders are concatenated during loading, while phase assignment comes from subject-level `time_window` definitions.

We highly recommend to upgrade to this release, as it will be the basis for future development and new experiment modules. The API has changed, so please check the updated documentation and example scripts for guidance on how to adapt your own analysis scripts.

### ✨ Features

- Renamed the package to `ic-analysis` and the import package to `ic_analysis`.
- Added script-defined `ExperimentMetadata`, `PhaseMetadata`, `SubjectMetadata`, and `SubjectRegistry` classes.
- Added the public `ic.experiment(EXPERIMENT=..., PHASES=..., SUBJECTS=...)` entry point, returning a generic `IntelliCageExperiment` object.
- Added modular experiment methods such as `.load()`, `.prepare_analysis()`, `.plot_ages()`, `.plot_mice_activity()`, `.plot_phase_activity_summary()`, `.plot_NP_adaptation()`, `.plot_NP_counts()`, `.plot_licking_counts()`, and `.plot_bottle_preference()`.
- Added modular PL/PR-specific methods with the `plot_plr_*` prefix for learning rates, learning counts, error rates, day-wise endpoints, reversal components, threshold onsets, experience-learning curves, derived ratios, and cumulative corner preferences.
- Added subject-driven inclusion policy: only raw IntelliCage animal IDs with a matching subject metadata entry are analyzed when subject metadata is passed.
- Added support for subject-level phase windows and automatic schedule inference from those windows.
- Added export-block-agnostic loading: each cage-run folder may contain one long export block, several interrupted export blocks, or export blocks that happen to match protocol phases.
- Added subject YAML helper functions to generate editable templates from detected raw animal IDs and load edited subject metadata back into scripts.
- Added a generic bottle-preference analysis for left/right bottle consumption and relative preference ratios.

### 🧩 Changes

- Removed the all-in-one public analysis workflow in favor of explicit modular plot and analysis calls controlled by the user script.
- Replaced the old `user_scripts/analyze_synthetic_group_ab.py` script with `user_scripts/place_learning_example.py`, written as a VS Code interactive script that can be run cell by cell.
- Removed any reliance on `Mice.txt`; subject metadata now comes from the user script or an edited subject YAML file.
- Renamed the generic phase-activity endpoint from `plot_plr_phase_activity_summary()` to `plot_phase_activity_summary()`.
- Updated package metadata, repository links, CI import checks, README, and RTD references from `ic-placelearning`/`ic_placelearning` to `ic-analysis`/`ic_analysis`.

### 🧪 Tests

- Added tests for metadata validation, subject-registry schedule inference,
  subject-based loader filtering, and experiment-object loading.
- Added tests for export-block-agnostic loading, including a single long export
  block that is split into biological phases by subject-level time windows.
- Kept the full public synthetic workflow test and maintained coverage above
  the 75% threshold.

---

## 🚀 IntelliCage Analysis Toolkit v0.0.4

August 31, 2026

Just a dummy release to correct a wrong version number on PyPI. The release does not change any code or documentation.

---

## 🚀 IntelliCage Analysis Toolkit v0.0.3

August 31, 2026

Just a minor release to rename the synthetic-data groups to avoid naming confusion.

### 🧩 Changes
* renamed the synthetic-data groups to "GroupA" and "GroupB" to avoid confusion

---

## 🚀 IntelliCage Analysis Toolkit v0.0.2

August 31, 2026

Just a minor release to request Python version 3.12 as minimum requirement for the package.

### 🧩 Changes
* We set the minimum Python version to 3.12 for the package and GitHub Actions workflow.

---

## 🚀 IntelliCage Analysis Toolkit v0.0.1

August 30, 2026

This is just a dummy release to add a Zenodo archive for the package. The release does not change any code or documentation.

### 🗄️ Archiving
- Archived the public `ic_analysis` package on Zenodo for long-term preservation and citation.

---

## 🚀 IntelliCage Analysis Toolkit v0.0.0

August 30, 2026

This is the first public release of the *IntelliCage Analysis Toolkit*, then focused on place learning experiments conducted in the *IntelliCage*. The release establishes the public package structure, synthetic example dataset, documentation entry points, and automated tests for the core loader, metric, and plotting workflow.

### ✨ Features

- Added the public `ic_analysis` Python package with loader, metric, and plotting modules.
- Added IntelliCage export loading for cohort folders containing `Visits.txt` and `Nosepokes.txt` files across protocol phases.
- Added metadata harmonization for mouse group, sex, RFID, experiment label, date of birth, and assigned phase-3/phase-4 corners.
- Added aligned analysis-time columns for protocol-level phase windows, experiment elapsed time, phase elapsed time, mouse days, and awake/sleep summaries.
- Added place-learning metrics for correct-corner visits, correct-corner nose-poke visits, rewarded correct-corner visits, and MATLAB-compatible `PlaceError == 0` scoring.
- Added phase-4 reversal metrics separating visits to the new correct corner, previous correct corner, and neutral incorrect corners.
- Added binned activity, learning-rate, learning-count, responder, onset, phase-activity, cumulative-corner-role, and group-comparison summary functions.
- Added plotting helpers for experiment overviews, phase-wise learning curves, reversal components, day-wise violins, onset distributions, cumulative role curves, and phase-activity summaries.

### 🗂 Example Data

- Added a synthetic Group A/B IntelliCage-style dataset with 20 pseudo-mice and four protocol phases.
- Added a reproducible synthetic data generator in `additional_scripts/`.
- Added a public synthetic-data analysis script in `user_scripts/`.
- Added generated protocol schematic assets in `figures/` for README and documentation use.

### 📚 Documentation

- Added public README documentation focused on installation, package structure, synthetic example data, demo analysis, metric definitions, time alignment, testing, and citation.
- Added an IntelliCage place-learning protocol schematic for the four-phase workflow: Free Hab, NPA, PL, and PR.
- Added `CITATION.cff` metadata for citation-aware repositories.
- Added contribution guidelines tailored to IntelliCage place-learning analysis and synthetic-data-based testing.

### 🧪 Tests and packaging

- Added `pyproject.toml` for editable installs and future PyPI packaging under the distribution name `ic-analysis`.
- Added development dependencies for testing and documentation via `pip install -e ".[dev]"`.
- Added automated tests for loader behavior, aligned timing, core metrics, statistical summaries, plotting smoke tests, synthetic data generation, and the public demo script.
- Established a test coverage threshold of 75% for the public `ic_analysis` package.

---
