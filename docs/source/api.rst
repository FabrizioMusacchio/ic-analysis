API reference
=============

This page documents the public functions and experiment objects used by the
tutorial workflow. New user scripts should normally start with
``import ic_analysis as ic`` and build an experiment object via
:func:`ic_analysis.experiment`.

Experiment creation
-------------------

.. currentmodule:: ic_analysis

.. autofunction:: experiment
.. autofunction:: create_subjects_yaml_template
.. autofunction:: load_subjects_yaml

Experiment definitions
----------------------

.. currentmodule:: ic_analysis.metadata

.. autoclass:: PhaseMetadata
   :members:

.. autoclass:: ExperimentMetadata
   :members:

.. autoclass:: SubjectMetadata
   :members:

.. autoclass:: SubjectRegistry
   :members:

The experiment object
---------------------

.. currentmodule:: ic_analysis.experiment


.. autoclass:: IntelliCageExperiment
   :members:

Analysis and plotting functions
--------------------------------

.. currentmodule:: ic_analysis.plotting

.. autofunction:: configure_plot_style
.. autofunction:: set_group_colors
.. autofunction:: sanitize_filename_part
.. autofunction:: plot_experiment_overview
.. autofunction:: plot_experiment_overview_groups
.. autofunction:: plot_bottle_preference_groups
.. autofunction:: plot_bottle_preference_day_violin
.. autofunction:: plot_phase2_adaptation
.. autofunction:: plot_phase2_adaptation_groups
.. autofunction:: plot_phase_learning_counts
.. autofunction:: plot_phase_learning_rate
.. autofunction:: plot_phase_learning_rate_groups
.. autofunction:: plot_phase_learning_counts_groups
.. autofunction:: plot_experiment_dual_metric_bars
.. autofunction:: plot_experiment_dual_metric_groups
.. autofunction:: plot_phase4_reversal_components
.. autofunction:: plot_phase_segment_rate_groups
.. autofunction:: plot_group_day_violin
.. autofunction:: plot_cumulative_role_curves
.. autofunction:: plot_visit_learning_curve_groups
.. autofunction:: plot_onset_violin
.. autofunction:: plot_phase_activity_boxplot


The PlotLayout object
---------------------

.. currentmodule:: ic_analysis.experiment

.. autoclass:: PlotLayout
   :members:

Data loading and time alignment
-------------------------------

.. currentmodule:: ic_analysis.loader

.. autoclass:: CohortData
   :members:

.. autofunction:: load_cohort_data
.. autofunction:: attach_analysis_time_columns
.. autofunction:: read_visits_file
.. autofunction:: read_nosepokes_file
.. autofunction:: summarize_nosepokes_by_visit

Metric tables
-------------

.. currentmodule:: ic_analysis.metrics

.. autofunction:: flag_iqr_outliers
.. autofunction:: infer_phase_boundaries
.. autofunction:: build_phase_time_limit_table
.. autofunction:: suggest_common_phase_limits
.. autofunction:: filter_visits_by_phase_limits
.. autofunction:: filter_by_dayphase
.. autofunction:: build_analysis_phase_window_table
.. autofunction:: compute_experiment_visit_bins
.. autofunction:: compute_experiment_drinking_visit_bins
.. autofunction:: compute_experiment_nosepoke_count_bins
.. autofunction:: compute_experiment_lick_count_bins
.. autofunction:: compute_bottle_preference_bins
.. autofunction:: compute_binned_group_statistics
.. autofunction:: compute_phase2_adaptation_bins
.. autofunction:: compute_place_learning_count_bins
.. autofunction:: compute_phase_visit_count_bins
.. autofunction:: compute_place_learning_rate_bins
.. autofunction:: compute_phase4_reversal_rate_bins
.. autofunction:: compute_phase4_reversal_count_bins
.. autofunction:: compute_phase_segment_rate_tables
.. autofunction:: compute_phase_segment_error_rate_tables
.. autofunction:: compute_awake_day_rate_tables
.. autofunction:: compute_awake_day_error_rate_tables
.. autofunction:: compute_awake_day_ratio_tables
.. autofunction:: compute_first_hours_rate_table
.. autofunction:: compute_threshold_responder_table
.. autofunction:: compute_responder_group_statistics
.. autofunction:: compute_binomial_glm_group_statistics
.. autofunction:: compute_binomial_gee_group_statistics
.. autofunction:: compute_group_day_violin_statistics
.. autofunction:: compute_role_cumulative_curves
.. autofunction:: compute_time_window_learning_curves
.. autofunction:: compute_visit_window_learning_curves
.. autofunction:: compute_onset_group_statistics
.. autofunction:: compute_phase_activity_medians
.. autofunction:: compute_phase_activity_statistics
