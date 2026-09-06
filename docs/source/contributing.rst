Contributing and community guidelines
=====================================

The IntelliCage Analysis Toolkit is an open source project for scriptable,
reproducible analysis of IntelliCage experiments. Contributions range from bug
reports and documentation improvements to tests, synthetic example datasets,
metadata models, metric helpers, experiment modules, and plotting improvements.

The goal of the project is to make IntelliCage workflows explicit and
reproducible. Place learning and reversal are the first supported workflow;
future modules can add other paradigms on top of the same metadata, loading,
metrics, and plotting foundation.

How to contribute
-----------------

Recommended entry points are:

* reporting bugs or unexpected behavior
* suggesting improvements to documentation or tutorials
* requesting support for additional IntelliCage export variants
* improving synthetic example data
* adding focused metric or figure helpers
* submitting pull requests with code changes

Bug reports and feature requests should be submitted via the
`GitHub issue tracker <https://github.com/FabrizioMusacchio/ic-analysis/issues>`_.
For code changes and larger contributions, please open a pull request against
the main repository.

Contribution guide
------------------

The repository contains a dedicated contribution guide in
`CONTRIBUTING.md <https://github.com/FabrizioMusacchio/ic-analysis?tab=contributing-ov-file>`_.
It describes in more detail:

* how to set up a local development environment
* the preferred workflow for branching and pull requests
* conventions for commit messages and code style
* expectations regarding tests and documentation
* expectations for synthetic or public example data
* how to avoid committing private or unpublished experimental datasets

Before opening a pull request, please make sure that:

* existing tests pass locally
* new functionality is covered by focused tests where applicable
* public functions and modules have useful docstrings
* user-facing changes are reflected in the documentation pages
* changes that affect output tables or figures are reflected in the synthetic
  example workflow

Development setup
-----------------

Install the package in editable mode:

.. code-block:: bash

   git clone https://github.com/FabrizioMusacchio/ic-analysis.git
   cd ic-analysis

   conda create -n ic_analysis python=3.12 -y
   conda activate ic_analysis

   pip install -e ".[dev]"

Run tests with:

.. code-block:: bash

   pytest

Build the documentation locally with:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

Input data expectations
-----------------------

Contributions that affect loading should preserve the central input contract:

* a dataset root contains one or more direct run-group folders
* phase folders contain ``IntelliCage/Visits.txt`` and
  ``IntelliCage/Nosepokes.txt``
* subject metadata, group names, and colors are configured by user scripts, not hard-coded in the
  package core
* public examples must use synthetic data or publicly shareable data

Output expectations
-------------------

Contributions that affect metrics or plots should keep outputs reproducible:

* mouse-level and group-level summary tables should be saved separately when
  both are useful
* output column names should be explicit and stable
* plots should be generated from summary tables rather than hidden intermediate
  state
* phase timing, bin size, group names, and metric definitions should remain
  visible in user scripts

Reporting issues
----------------

When reporting an analysis issue, please include:

* package version
* Python version and operating system
* the relevant user-script settings
* input folder structure
* a small synthetic or anonymized reproducer when possible
* the error message or unexpected output table/figure

Large raw datasets should not be committed to the repository. Use public
archives, minimal synthetic examples, or temporary private links when needed.

Code of conduct
---------------

All interactions in the project are governed by a
`Code of Conduct <https://github.com/FabrizioMusacchio/ic-analysis?tab=coc-ov-file>`_
based on the `Contributor Covenant <https://www.contributor-covenant.org>`_.
By participating in the project, you agree to abide by these guidelines.

Where to start
--------------

If you are looking for a first contribution, start with documentation
improvements, focused tests, clearer error messages for unusual IntelliCage
exports, or small additions to the synthetic example workflow.
