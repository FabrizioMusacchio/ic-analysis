Preparing IntelliCage data
==========================

The toolkit expects IntelliCage export tables to be extracted from the original
export zip archives and grouped by cage run. A cage run is one physical cage/group
recording session with its own start time. Multiple cage runs can belong to the
same experiment, and they do not have to start at the same clock time.

Recommended folder layout
-------------------------

Put each cage run into one folder below the experiment data root. Inside each
cage-run folder, keep one folder per IntelliCage export block (= the unpacked
exported IntelliCage zip archive). From each export block/extracted zip archive, 
the toolbox only needs the ``IntelliCage/`` subfolder, which contains the 
``Visits.txt`` and ``Nosepokes.txt`` tables. Just keep the ``IntelliCage/`` 
subfolder and its two text tables, and place it into a clearly named export-block:

.. code-block:: text

   data_root/
   |-- CageRun_A/
   |   |-- Export_Block_1/
   |   |   `-- IntelliCage/
   |   |       |-- Visits.txt
   |   |       `-- Nosepokes.txt
   |   |-- Export_Block_2/
   |   |   `-- IntelliCage/
   |   |       |-- Visits.txt
   |   |       `-- Nosepokes.txt
   |   |-- Export_Block_3/
   |   |   `-- IntelliCage/
   |   |       |-- Visits.txt
   |   |       `-- Nosepokes.txt
   |   `-- Export_Block_4/
   |       `-- IntelliCage/
   |           |-- Visits.txt
   |           `-- Nosepokes.txt
   `-- CageRun_B/
       |-- Export_Block_1/
       |   `-- IntelliCage/
       |       |-- Visits.txt
       |       `-- Nosepokes.txt
       |-- Export_Block_2/
       |   `-- IntelliCage/
       |       |-- Visits.txt
       |       `-- Nosepokes.txt
       |-- Export_Block_3/
       |   `-- IntelliCage/
       |       |-- Visits.txt
       |       `-- Nosepokes.txt
       `-- Export_Block_4/
           `-- IntelliCage/
               |-- Visits.txt
               `-- Nosepokes.txt

Only ``Visits.txt`` and ``Nosepokes.txt`` are currently required by the toolbox.
However, you can keep all exported IntelliCage text tables from the ``IntelliCage/`` 
folder, they may become useful for future analysis features, and they will not 
interfere with the current workflow.

"clearly named export-block" means that these folder names should be readable, 
unique, and establish an alphanumerical order that matches the chronological 
order of the IntelliCage exports. The toolbox will read the export-block folders 
in alphanumerical order and concatenate them into one dataset, thus, any
incorrect order will result in a wrong concatenation order.

Mouse identity, group, sex, date of birth, true ID, 
phase windows, and corner assignments are defined explicitly in the user 
script or loaded from a subject YAML file.

.. note::

   Export-block folders are not biological phases. They are simply the pieces
   in which the IntelliCage data were exported. In a place-learning experiment
   with cage stops at phase transitions, export blocks may happen to match the
   protocol phases and may be named ``Phase1``, ``Phase2``, and so on. In a
   long uninterrupted experiment, one export block can contain the entire
   protocol. In an interrupted experiment, several export blocks can cut across
   protocol phases. The toolkit concatenates all detected export blocks and
   assigns the actual analysis phases from the subject-level ``time_window``
   definitions.

.. note::

   Do the experimental groups need to be in separate cage runs? No, the toolkit 
   can handle multiple groups in the same cage run. The only requirement is that 
   each cage run has its own folder and one or more export-block subfolders, as
   shown above.

Phase windows
-------------

In each analysis script, you define experimental phase windows per subject, 
not globally. This is important because cage runs can start at different 
times. Each subject entry should provide one ``time_window`` tuple per phase,
e.g.:

.. code-block:: python

   SUBJECTS = {
       "910200000001000": {
           "group": "Group A",
           "sex": "male",
           "true_id": "A01",
           "date_of_birth": "2025-09-01",
           "corner_assignments": {
               3: 1,
               4: 3},
           "phases": {
               1: {"time_window": ("2026-01-05 06:00:00", "2026-01-08 08:00:00")},
               2: {"time_window": ("2026-01-08 08:00:00", "2026-01-10 08:00:00")}}}}

During ``load()``, the toolkit keeps only animal IDs that are defined in
``SUBJECTS``. This makes the analysis reproducible and prevents accidental
analysis of animals that were exported but should not belong to the study.

Practical checklist
-------------------

Before running an analysis script:

1. Extract all IntelliCage export ZIP files.
2. Create one folder per cage run below the experiment data root.
3. Place each export block's ``IntelliCage`` folder containing 
   ``Visits.txt`` and ``Nosepokes.txt`` into the corresponding cage-run 
   folder, and name the export-block folders clearly, e.g., 
   ``Export_Block_1``, ``Export_Block_2``, ... or ``Phase1``, ``Phase2``, ...,
   or any other readable and unique names that sort alphanumerically in the
   same order as the chronological order of the IntelliCage exports.
4. Define all included animals in ``SUBJECTS`` or create/load a subject YAML
   template.
5. Check that every included subject has a ``time_window`` for every required
   phase.
