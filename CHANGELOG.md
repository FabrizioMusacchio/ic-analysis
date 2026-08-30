## IntelliCage Place Learning Toolkit Changelog

See here for a detailed list of changes made in each release of the
*IntelliCage Place Learning Toolkit*. Please also refer to the repository
[Releases page](https://github.com/FabrizioMusacchio/ic-placelearning/releases).

Each release can be archived on Zenodo for long-term preservation and citation
purposes.

---

## 🚀 IntelliCage Place Learning Toolkit v0.0.2

August 31, 2026

Just a minor release to request Python version 3.12 as minimum requirement for the package.

### 🧩 Changes
* We set the minimum Python version to 3.12 for the package and GitHub Actions workflow.

---

## 🚀 IntelliCage Place Learning Toolkit v0.0.1

August 30, 2026

This is just a dummy release to add a Zenodo archive for the package. The release does not change any code or documentation.

### 🗄️ Archiving
- Archived the public `ic_placelearning` package on Zenodo for long-term preservation and citation.

---

## 🚀 IntelliCage Place Learning Toolkit v0.0.0

August 30, 2026

This is the first public release of the *IntelliCage Place Learning Toolkit*, a Python toolkit for analyzing place learning experiments conducted in the *IntelliCage*. The release establishes the public package structure, synthetic example dataset, documentation entry points, and automated tests for the core loader, metric, and plotting workflow.

### ✨ Features

- Added the public `ic_placelearning` Python package with loader, metric, and plotting modules.
- Added IntelliCage export loading for cohort folders containing `Mice.txt`, `Visits.txt`, and `Nosepokes.txt` files across protocol phases.
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

- Added `pyproject.toml` for editable installs and future PyPI packaging under the distribution name `ic-placelearning`.
- Added development dependencies for testing and documentation via `pip install -e ".[dev]"`.
- Added automated tests for loader behavior, aligned timing, core metrics, statistical summaries, plotting smoke tests, synthetic data generation, and the public demo script.
- Established a test coverage threshold of 75% for the public `ic_placelearning` package.

---
