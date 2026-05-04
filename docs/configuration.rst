Configuration
=============

autods-pet reads parameters from an INI file.  Generate a template with
``autods-pet create-config`` or copy one of the example configs from the
``configs/`` folder (``standard.ini``, ``quick.ini``, ``full.ini``,
``advanced.ini``, ``brain.ini``).  The file is parsed by
:func:`autods_pet.config.load_config`.

``[paths]`` Section
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``basepath``
     - *(required)*
     - Root directory containing one subfolder per patient.
   * - ``patient_list``
     - *(empty)*
     - Path to a text file listing patient IDs (one per line, ``#`` for comments).
   * - ``metadata_csv``
     - *(empty)*
     - Path to a CSV with PET metadata (radiopharmaceutical tags, patient weight).
       When patients have incomplete metadata, an auto-generated template is
       written to ``output_dir/metadata.csv``.
   * - ``output_dir``
     - *(empty)*
     - Root directory for all outputs.  Per-patient results are written to
       ``output_dir/{patient_id}_results/`` with subfolders ``images/``,
       ``segmentations/``, ``metadata/``, ``DeauvilleScores/``, and ``SUV/``.
       Batch summaries (``batch_results_DS.csv``, ``batch_results_SUV.csv``)
       are written directly to ``output_dir/``.

``[lumbar_vb]`` Section
-----------------------

Lumbar vertebral body bone marrow ROI.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``labels``
     - ``29, 28, 27``
     - TotalSegmentator label IDs for L3, L4, L5.  Add ``31, 30`` for L1, L2.
   * - ``erosion_mm``
     - ``3.0``
     - Physical erosion radius in mm.
   * - ``stats``
     - ``p95``
     - Statistics to compute.

``[aorta_mbp]`` Section
-----------------------

Mediastinal blood pool reference ROI.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``vertebra_labels``
     - ``36, 37, 38, 39, 40``
     - T4--T8 labels for slice-gating slab.
   * - ``slab_axis``
     - ``2``
     - SimpleITK axis for slice-gating (0=x, 1=y, 2=z).
   * - ``heart_exclusion_mode``
     - ``dilate_intersection``
     - ``dilate_intersection`` or ``distance``.
   * - ``heart_dilation_mm``
     - ``6.0``
     - Heart buffer dilation (mode: dilate_intersection).
   * - ``heart_distance_mm``
     - ``12.0``
     - Heart distance threshold (mode: distance).
   * - ``aorta_erosion_mm``
     - ``4.0``
     - Intraluminal erosion radius.
   * - ``stats``
     - ``median``
     - Statistics to compute.

``[liver]`` Section
-------------------

Liver reference ROI.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``erosion_mm``
     - ``10.0``
     - Physical erosion radius in mm.
   * - ``max_hole_volume_mm3``
     - ``none``
     - Fill only holes below this volume (mm\ :sup:`3`).  Default ``none`` fills all holes.
   * - ``stats``
     - ``median``
     - Statistics to compute.

``[long_bones]`` Section
------------------------

Femoral and humeral diaphysis ROIs.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``diaphysis_keep_pct``
     - ``60``
     - Central percentage of the bone axis to keep.
   * - ``stats``
     - ``p95``
     - Statistics to compute.

Per-bone erosion is set via sub-sections ``[long_bones.femur_L]``,
``[long_bones.femur_R]``, ``[long_bones.humerus_L]``,
``[long_bones.humerus_R]``, each with an ``erosion_mm`` key.

``[brain]`` Section
-------------------

Brain cortical grey matter ROI (for Brain-to-Liver Ratio).

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``label``
     - ``90``
     - TotalSegmentator label ID for brain.
   * - ``grey_matter_only``
     - ``true``
     - If true, extract cortical grey matter shell (original minus eroded).
       If false, use the full brain mask.
   * - ``cortical_thickness_mm``
     - ``5.0``
     - Erosion radius in mm for the shell extraction (only when
       ``grey_matter_only`` is true).
   * - ``stats``
     - ``median``
     - Statistics to compute.

``[output]`` Section
--------------------

Controls mask saving during ROI extraction.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``save_raw_masks``
     - ``false``
     - Save individual raw TotalSegmentator label masks as NIfTI in ``{seg_dir}/raw/``.
   * - ``save_refined_masks``
     - ``true``
     - Save refined ROI masks (after erosion, slabbing, etc.) as NIfTI in ``{seg_dir}/refined/``.
   * - ``subtract_lesions_from_marrow``
     - ``false``
     - Subtract target lesion masks (FL, PM, EM, custom) from marrow ROIs
       (Lumbar VB, Long bones) before computing statistics.  See the note
       below.

.. note:: **Lesion subtraction and input consistency**

   When ``subtract_lesions_from_marrow`` is enabled, the accuracy of the
   corrected marrow uptake depends entirely on the completeness and
   consistency of the input lesion masks.  If lesion segmentation criteria
   vary across patients (e.g. only the hottest lesion is segmented in
   some patients while multiple lesions are segmented in others), the
   subtraction will be applied unevenly, potentially introducing a
   systematic bias.  To ensure valid cohort-level comparisons, users
   should apply uniform lesion segmentation criteria across all patients.

``[deauville]`` Section
-----------------------

Deauville Score scoring parameters.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``liver_multiplier``
     - ``2.0``
     - Multiplier applied to the liver reference for the DS 4 / DS 5
       cutoff (DS 5 = target uptake > ``liver_multiplier × liver``).
       The default of ``2.0`` follows the Lugano 2014 consensus; set to
       ``3.0`` for protocols using a 3× liver threshold.

``[totalsegmentator]`` Section
------------------------------

TotalSegmentator runtime options.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``license``
     - *(empty)*
     - TotalSegmentator license key (set automatically via ``totalseg_set_license``).
   * - ``fast``
     - ``false``
     - Use TotalSegmentator fast mode (lower accuracy, faster).  Note: the
       ``vertebrae_body`` task always runs in high-res regardless of this
       setting.  Output is a single multilabel file
       (``TotSeg_multilabel.nii.gz``); no individual per-label files are
       created.

``[dicom]`` Section
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``size_threshold_kb``
     - ``100``
     - Minimum file size (KB) to consider as DICOM.

Target ROIs
-----------

Opt-in sections for user-provided manual lesion masks.  Both NIfTI/NRRD
and DICOM SEG inputs are supported and the discovery is **recursive**:
the mask file may live anywhere under the patient input directory,
including nested study/series sub-folders exported by DICOM viewers.

Each section may set:

- ``mask_filename`` -- one stem (or comma-separated list of stems) for
  ``.nii.gz`` / ``.nii`` / ``.nrrd`` files.  Searched recursively under
  ``input_dir``.  Existing NIfTI workflows keep working unchanged --
  the only difference from previous releases is that the search now
  descends into sub-folders as well as the patient root and
  ``segmentations/``.

- ``segment_label`` -- one DICOM SEG ``SegmentLabel`` (or comma list,
  case-insensitive).  The SEG file is identified by matching its
  ``ReferencedSeriesSequence`` to the patient's PET
  ``SeriesInstanceUID`` (read from ``PET_metadata.json``); no filename
  or location is required.  A single multi-segment SEG file may supply
  several target sections at once.  Only consulted for ``.dcm`` files.

Set either, both, or neither.  When both formats resolve the same
target for the same patient, **DICOM SEG wins** and a note is logged.
A target with neither key set is treated as disabled.

A configured target whose mask cannot be found anywhere produces a
loud warning naming the patient, the patterns searched for, and the
locations checked -- it never silently disappears from the output.

Sections:

- ``[focal_lesion]`` -- Focal lesion (FL)
- ``[paramedullary]`` -- Paramedullary disease (PM)
- ``[extramedullary]`` -- Extramedullary disease (EM)
- ``[targets.<name>]`` -- Custom targets

Use ``autods-pet validate-config <config> --patients <id1,id2>`` to
preview which masks would be discovered for each patient before
launching a full ``extract`` / ``score`` run, or pass
``--explain-masks`` to ``extract`` / ``score`` / ``run`` to print the
discovery report alongside the actual computation.

Available Statistics
--------------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Name
     - Description
   * - ``mean``
     - Arithmetic mean of voxel values in the mask.
   * - ``median``
     - Voxelwise median.
   * - ``min``
     - Minimum value.
   * - ``max``
     - Maximum value.
   * - ``p<N>``
     - *N*-th percentile (e.g. ``p90``, ``p95``, ``p99``).
