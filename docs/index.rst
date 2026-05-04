.. image:: _static/logo.png
   :alt: autoDS-PET logo
   :align: center
   :width: 350px

autoDS-PET
==========

**Automated Deauville Score computation from PET/CT images.**

autoDS-PET extracts reference uptake values from anatomical ROIs (bone marrow,
mediastinal blood pool, liver, long bones, brain, focal lesions) on
SUV-normalised PET and assigns Deauville Scores by comparing target uptake
against standard thresholds.

.. code-block:: bash

   pip install autods-pet
   autods-pet run -c config.ini -p PATIENT_001

----

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting-started
      :link-type: doc

      Installation, prerequisites, and your first pipeline run.

   .. grid-item-card:: CLI Reference
      :link: cli
      :link-type: doc

      All ``autods-pet`` commands, options, and usage examples.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc

      INI file parameters for paths, ROIs, erosion, and statistics.

   .. grid-item-card:: Methodology
      :link: methods
      :link-type: doc

      Scientific methods: SUV normalization, registration, ROI extraction, and Deauville scoring.

----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   getting-started
   cli
   configuration
   methods

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/index
