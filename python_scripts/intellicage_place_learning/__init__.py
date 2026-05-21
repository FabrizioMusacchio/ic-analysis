"""Top-level package for the IntelliCage place-learning analysis workflow.

The package contains reusable building blocks for reading IntelliCage text
exports, harmonizing them with mouse metadata, computing behavior metrics, and
producing poster-ready summary plots. The first implementation target is the
BioMedX 4-month cohort, but the modules are intentionally written in a generic
way so that additional cohorts can be plugged in later with minimal code
changes.
"""

from .loader import CohortData, load_cohort_data

__all__ = ["CohortData", "load_cohort_data"]
