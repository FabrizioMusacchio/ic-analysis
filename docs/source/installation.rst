Installation
============

The IntelliCage Analysis Toolkit is intended for scientific Python
environments and requires Python 3.12 or newer. The package is tested with
Python 3.12.

Create an environment
---------------------

We recommend creating a fresh conda environment first:

.. code-block:: bash

   conda create -n ic_analysis python=3.12 -y
   conda activate ic_analysis

Install from PyPI
-----------------

After the package has been released, it can be installed from PyPI:

.. code-block:: bash

   pip install ic-analysis

Development installation
------------------------

From a local checkout:

.. code-block:: bash

   git clone https://github.com/FabrizioMusacchio/ic-analysis.git
   cd ic-analysis
   pip install -e .

For development and documentation work:

.. code-block:: bash

   pip install -e ".[dev]"

Verify installation
-------------------

After installation, verify that the package imports correctly and reports its
version:

.. code-block:: bash

   python -c "import ic_analysis; print(f'ic_analysis {ic_analysis.__version__} imported successfully')"

This one-liner should print the installed version. If it fails, check that the
active terminal session uses the environment in which the package was
installed.

Run the tests
-------------

From the repository root:

.. code-block:: bash

   pytest

The test suite builds and analyzes a synthetic IntelliCage dataset and checks
the loader, metric, and plotting functions. The project currently enforces a
minimum test coverage of 75%.

Build the documentation locally
-------------------------------

Install the development dependencies, then build the HTML documentation:

.. code-block:: bash

   pip install -e ".[dev]"
   sphinx-build -b html docs/source docs/build/html

Open ``docs/build/html/index.html`` in a browser to inspect the rendered pages.

Core dependencies
-----------------

The toolkit depends on:

- ``pandas`` for tabular data handling,
- ``numpy`` for numerical operations,
- ``scipy`` and ``statsmodels`` for statistical summaries,
- ``matplotlib`` for figures,
- ``pytest`` and ``pytest-cov`` for the test suite when installed with the
  ``dev`` extra.
