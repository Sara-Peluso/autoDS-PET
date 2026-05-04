CLI Reference
=============

autods-pet provides a command-line interface built with `Typer <https://typer.tiangolo.com/>`_.
All commands share the following global options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-c, --config PATH``
     - Path to the INI configuration file (**required**).
   * - ``-p, --patients TEXT``
     - Patient(s) to process: a single ID, comma-separated IDs, or path to a ``.txt`` list (overrides config).
   * - ``--log-file PATH``
     - Write detailed logs to a file.
   * - ``-v, --verbose``
     - Enable DEBUG-level logging.
   * - ``--force``
     - Re-run stages even if outputs already exist (applies to ``convert``,
       ``normalize``, ``register``, ``segment``, ``extract``, ``score``,
       ``run``).  By default, completed stages are skipped.
   * - ``-V, --version``
     - Show version and exit.


``autods-pet create-config``
----------------------------

Generate a configuration file from a named profile.

.. code-block:: bash

   # Default profile (standard)
   autods-pet create-config -o my_config.ini

   # Choose a specific profile
   autods-pet create-config -p quick -o my_config.ini

Extra options:

.. list-table::
   :widths: 30 70

   * - ``-p, --profile TEXT``
     - Config profile: ``quick``, ``standard`` (default), ``advanced``,
       ``full``, ``brain``.
   * - ``-o, --output PATH``
     - Output path (default: ``config.ini``).
   * - ``--force``
     - Overwrite an existing file.

Creates a commented INI file with profile-specific defaults.
See also the example configs in the ``configs/`` folder.

.. list-table:: Available profiles
   :header-rows: 1
   :widths: 15 15 10 12 15 33

   * - Profile
     - TotalSeg
     - License
     - Targets
     - Masks saved
     - Use case
   * - ``quick``
     - Fast
     - No
     - None
     - None
     - Rapid screening, QC, testing
   * - ``standard``
     - High-res
     - No
     - None
     - Refined only
     - General-purpose research (default)
   * - ``advanced``
     - High-res
     - Yes
     - None
     - Refined only
     - Research with bone marrow DS
   * - ``full``
     - High-res
     - Yes
     - FL, PM, EM
     - Raw + refined
     - Complete clinical analysis
   * - ``brain``
     - High-res
     - No
     - None
     - Refined only
     - Brain-to-Liver Ratio studies


``autods-pet validate-config``
------------------------------

Validate an existing configuration file.

.. code-block:: bash

   autods-pet validate-config my_config.ini

Checks that all required keys are present and paths are valid.

Pass ``--patients <id1,id2>`` to additionally print a **mask discovery
preview** for each named patient -- a dry-run of which manual lesion
masks (NIfTI/NRRD or DICOM SEG) would be picked up for each configured
target ROI section, before launching a full ``extract`` / ``score`` run.

.. code-block:: bash

   autods-pet validate-config my_config.ini --patients PATIENT_001,PATIENT_002


``autods-pet convert``
----------------------

Convert DICOM, NIfTI, or NRRD images to the standard NIfTI layout.

.. code-block:: bash

   autods-pet convert -c config.ini -p PATIENT_001,PATIENT_002

Detects the input format automatically.  For DICOM inputs, also extracts
PET metadata (radiopharmaceutical tags, patient weight) and writes a JSON
sidecar file.


``autods-pet normalize``
------------------------

Compute SUV body-weight from raw PET images.

.. code-block:: bash

   autods-pet normalize -c config.ini

Reads PET metadata from the JSON sidecar (written by ``convert``) or from
the CSV specified via ``metadata_csv`` in config.  Writes
``PET_SUV.nii.gz`` to the patient results folder
(``output_dir/{patient_id}_results/images/``).

If patients have incomplete metadata, an auto-generated template
``metadata.csv`` is written to ``output_dir/``.


``autods-pet register``
-----------------------

Rigidly register the PET SUV image onto the CT grid.

.. code-block:: bash

   autods-pet register -c config.ini

Uses SimpleElastix for rigid registration.  Writes
``PET_SUV_reg.nii.gz`` to ``output_dir/{patient_id}_results/images/`` and
saves the Elastix transform (``elastix_transform.txt``, native Elastix
format) to ``output_dir/{patient_id}_results/metadata/``.


``autods-pet segment``
----------------------

Run TotalSegmentator on CT images.

.. code-block:: bash

   autods-pet segment -c config.ini --fast

Extra options:

.. list-table::
   :widths: 30 70

   * - ``--fast``
     - Use TotalSegmentator fast mode (lower accuracy, faster).
       Note: the ``vertebrae_body`` task always runs in high-res and does
       not support fast mode.

Produces a single multilabel output
``output_dir/{patient_id}_results/segmentations/TotSeg_multilabel.nii.gz``
(no individual per-label files are created).


``autods-pet extract``
----------------------

Extract ROI statistics from registered PET images.

.. code-block:: bash

   autods-pet extract -c config.ini

Processes all configured ROIs (aorta MBP, liver, lumbar VB, long bones,
brain, and target ROIs) and prints status for each.

Extra options:

.. list-table::
   :widths: 30 70

   * - ``--save-masks``
     - Save both raw and refined masks as NIfTI.
   * - ``--save-raw-masks``
     - Save individual raw label masks as NIfTI.
   * - ``--save-refined-masks``
     - Save refined ROI masks as NIfTI.
   * - ``--subtract-lesions``
     - Subtract target lesion masks (FL, PM, EM, custom) from marrow ROIs
       (BM, LB) before computing statistics.
   * - ``--explain-masks``
     - Print a per-patient mask discovery report (every DICOM SEG and
       NIfTI/NRRD file found, every segment, which targets matched,
       which came up empty) before running the extraction.

The same ``--explain-masks`` and ``--subtract-lesions`` flags are
available on ``score`` and ``run``.


``autods-pet score``
--------------------

Assign Deauville Scores from pre-extracted ROI statistics.

.. code-block:: bash

   autods-pet score -c config.ini -p PATIENT_001,PATIENT_002

Runs extraction then scoring, printing a results table.


``autods-pet run``
------------------

Run the full Deauville Score pipeline end-to-end.

.. code-block:: bash

   autods-pet run -c config.ini --format xlsx

This command chains all stages (convert, normalize, register, segment,
extract, score) and writes results per patient and in batch.

Per-patient CSVs are written to the patient results folder:

- ``output_dir/{patient_id}_results/DeauvilleScores/deauville_scores.csv``
- ``output_dir/{patient_id}_results/SUV/SUV_values.csv``

For multi-patient runs, batch summaries are written to ``output_dir/``:

- ``batch_results_DS.csv`` -- Deauville Score columns
- ``batch_results_SUV.csv`` -- SUV statistics
- ``batch_errors.csv`` -- errors (only created if any patients fail)

Extra options:

.. list-table::
   :widths: 30 70

   * - ``--fast``
     - Use TotalSegmentator fast mode.
   * - ``-f, --format TEXT``
     - Output format: ``csv`` (default) or ``xlsx``.
   * - ``--save-masks``
     - Save both raw and refined masks as NIfTI.
   * - ``--save-raw-masks``
     - Save individual raw label masks as NIfTI.
   * - ``--save-refined-masks``
     - Save refined ROI masks as NIfTI.
   * - ``--subtract-lesions``
     - Subtract target lesion masks (FL, PM, EM, custom) from marrow ROIs
       (BM, LB) before computing statistics.
   * - ``--explain-masks``
     - Print a per-patient mask discovery report before running.


``autods-pet list-segments``
----------------------------

List the segments contained in a DICOM SEG (``.dcm``) file.

.. code-block:: bash

   autods-pet list-segments path/to/segmentation.dcm

Takes a single positional ``PATH`` argument pointing to an existing,
readable DICOM SEG file and prints a table of segment numbers, labels,
and descriptions.  Useful for choosing the correct ``segment_label``
value to set in the ``[focal_lesion]`` / ``[paramedullary]`` /
``[extramedullary]`` / ``[targets.<name>]`` config sections when working
with multi-segment DICOM SEG masks.

Requires the ``dicom-seg`` extra
(``pip install autods-pet[dicom-seg]``).
