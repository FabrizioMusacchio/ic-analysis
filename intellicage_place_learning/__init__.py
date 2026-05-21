"""Python toolkit for IntelliCage place-learning analyses.

The package provides reusable utilities to read IntelliCage exports, harmonize
mouse metadata with visits and nose-pokes, compute behavior metrics, and create
poster-ready plots. The current workflow targets the BioMedX 4-month cohort but
is structured so the same code can be reused for additional cohorts later on.
"""

from .loader import CohortData, load_cohort_data

__all__ = ["CohortData", "load_cohort_data"]
