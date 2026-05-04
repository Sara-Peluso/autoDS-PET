Getting Started
===============

Prerequisites
-------------

* Python >= 3.10
* `TotalSegmentator <https://github.com/wasserth/TotalSegmentator>`_ for CT
  segmentation (``pip install TotalSegmentator``)
* **Elastix / SimpleElastix** for PET-to-CT registration
  (``pip install SimpleITK-SimpleElastix`` -- replaces the plain ``SimpleITK`` package)

Installation
------------

.. code-block:: bash

   pip install autods-pet

Optional extras
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Extra
     - Command
     - What it adds
   * - ``dicom-seg``
     - ``pip install "autods-pet[dicom-seg]"``
     - `highdicom <https://github.com/ImagingDataCommons/highdicom>`_ --
       read DICOM SEG (``.dcm``) segmentation masks produced by 3D Slicer,
       OHIF, dcmqi, Kaapana, MONAILabel, etc.
   * - ``dev``
     - ``pip install "autods-pet[dev]"``
     - pytest, pytest-cov, hypothesis (for running tests with coverage)
   * - ``docs``
     - ``pip install "autods-pet[docs]"``
     - Sphinx, pydata-sphinx-theme (for building documentation)

Install everything at once:

.. code-block:: bash

   pip install "autods-pet[all]"

.. note::

   Without the ``dicom-seg`` extra, NIfTI (``.nii``, ``.nii.gz``) and NRRD
   (``.nrrd``) segmentation masks are fully supported. The extra is only
   needed when working with DICOM SEG files.

Quick Start
-----------

1. **Prepare your data** following the expected layout::

      basepath/
        PATIENT_001/
          CT.nii.gz
          PET.nii.gz
        PATIENT_002/
          ...

2. **Create a configuration file** using the built-in template generator, or
   copy one of the example configs from the ``configs/`` folder
   (``standard.ini``, ``quick.ini``, ``full.ini``, ``advanced.ini``,
   ``brain.ini``):

   .. code-block:: bash

      # Generate a template config (defaults to "standard" profile)
      autods-pet create-config -o my_config.ini

      # Or choose a specific profile
      autods-pet create-config -p quick -o my_config.ini

   Then edit the ``[paths]`` section:

   .. code-block:: ini

      [paths]
      basepath = /path/to/your/nifti/data
      patient_list = /path/to/patients.txt
      output_dir = /path/to/output

3. **Run the full pipeline**:

   .. code-block:: bash

      # Single patient
      autods-pet run -c my_config.ini -p PATIENT_001

      # Comma-separated patients
      autods-pet run -c my_config.ini -p PATIENT_001,PATIENT_002

      # Patients from a text file
      autods-pet run -c my_config.ini -p patients.txt

      # Entire cohort (from config patient_list or auto-discover)
      autods-pet run -c my_config.ini

      # Re-run even if outputs already exist
      autods-pet run -c my_config.ini --force

   By default, completed stages are skipped. Use ``--force`` to re-run all
   stages regardless of existing outputs.

4. **Output structure**: Results are written to
   ``output_dir/{patient_id}_results/`` with the following subfolders::

      output_dir/
        PATIENT_001_results/
          images/
          segmentations/
          metadata/
          DeauvilleScores/
            deauville_scores.csv
          SUV/
            SUV_values.csv

   Source data in ``basepath/{patient_id}/`` is never modified.

   For batch runs, two summary CSVs are written to ``output_dir/``:

   - ``batch_results_DS.csv`` -- Deauville Score columns
   - ``batch_results_SUV.csv`` -- SUV statistics

   If any patients fail, errors are collected in ``batch_errors.csv``
   (only created when errors occur).

Python API Quick Start
----------------------

.. code-block:: python

   from autods_pet import DeauvillePipeline, load_config

   cfg = load_config("my_config.ini")
   pipeline = DeauvillePipeline(cfg)

   # Single patient
   result = pipeline.run("PATIENT_001")
   print(result.scores)

   # Re-run even if outputs already exist
   pipeline = DeauvillePipeline(cfg, force=True)
   result = pipeline.run("PATIENT_001")

   # Batch
   results = pipeline.run_batch(["PATIENT_001", "PATIENT_002"])
   df = DeauvillePipeline.to_dataframe(results)
