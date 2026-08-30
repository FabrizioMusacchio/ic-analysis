"""IntelliCage Place Learning Toolkit.

The package provides reusable utilities to read IntelliCage exports, harmonize
mouse metadata with visits and nose-pokes, compute behavior metrics, and create
publication-oriented plots for place-learning experiments.
"""

from .loader import CohortData, load_cohort_data

__all__ = ["CohortData", "load_cohort_data"]
__version__ = "0.0.0"
