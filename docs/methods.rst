Methodology
===========

This page describes the scientific methods used by autoDS-PET for automated
Deauville Score computation from PET/CT images.

Image Preprocessing
-------------------

SUV Normalization
~~~~~~~~~~~~~~~~~

Raw PET images, acquired in units of activity concentration (Bq/mL), are
converted to standardized uptake values normalized by body weight (SUVbw):

.. math::

   \text{SUV}_\text{bw} = \frac{C \times W \times 1000}{D}

where:

- :math:`C` is the voxel activity concentration in Bq/mL,
- :math:`W` is the patient body weight in kg,
- :math:`D` is the effective injected dose in Bq.

When the DICOM ``DecayCorrection`` field indicates that no decay correction was
applied at acquisition (value ``"NONE"`` or empty), the injected dose is
corrected for radioactive decay between administration and acquisition time
using the radionuclide half-life.  Otherwise, the reported total dose is used
directly.

See :func:`autods_pet.imaging.normalization.compute_suvbw`.


Intra-patient Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~

PET SUVbw images are rigidly registered to the corresponding CT images using
the Elastix framework (SimpleElastix).  Registration parameters:

- **Similarity metric:** Advanced Mattes Mutual Information
- **Resolution levels:** 3 (multi-resolution scheme)
- **Max iterations per level:** 512
- **Interpolation:** Linear (final resampling)

The output PET image is resampled onto the CT grid, sharing identical spatial
geometry (voxel spacing, origin, orientation, and matrix size).

See :func:`autods_pet.imaging.registration.rigid_register_pet_to_ct`.


Anatomical Segmentation
-----------------------

CT images are segmented using `TotalSegmentator
<https://github.com/wasserth/TotalSegmentator>`_ to obtain a multilabel
segmentation map providing anatomical structures including vertebrae, aorta,
heart, liver, femora, and humeri.

See :func:`autods_pet.imaging.segmentation.run_totalsegmentator`.


Region of Interest Extraction
-----------------------------

Six ROIs are derived from the TotalSegmentator segmentation and, where
applicable, further refined using image processing operations.  All
morphological operations (erosion, dilation) are performed in **physical space
(mm)** using signed Euclidean distance transforms, making them robust to
anisotropic voxel spacing.

Bone Marrow (Lumbar Vertebral Bodies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The L3, L4, and L5 vertebral masks are extracted from the multilabel
segmentation and combined via logical union.  This mask is intersected with a
binary vertebral body segmentation to isolate the vertebral body compartment,
excluding posterior elements.  The resulting mask is eroded by **3 mm** to
reduce partial-volume effects at cortical boundaries.

**Metric:** 95th percentile of SUVbw within the eroded mask.

See :class:`autods_pet.roi.lumbar_vb.LumbarVB`.


Mediastinal Blood Pool (Thoracic Aorta)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The aorta mask is restricted to the axial extent of the **T4-T8** vertebral
slab through slice-wise gating.  Cardiac spill-in is mitigated by dilating the
heart segmentation by **6 mm** and subtracting the dilated mask from the aorta.
The resulting mask is eroded by **4 mm** to obtain an intraluminal ROI.  When
multiple disconnected components remain, only the largest connected component
is retained.

**Metric:** Voxelwise median SUVbw.

See :class:`autods_pet.roi.aorta_mbp.AortaMBP`.


Liver
~~~~~

The liver mask's largest connected component is retained and holes are filled
using binary flood-fill with full connectivity.  The filled mask is eroded by
**10 mm** to avoid partial-volume effects.

**Metric:** Voxelwise median SUVbw.

See :class:`autods_pet.roi.liver.LiverROI`.


Long Bones (Femoral and Humeral Diaphyses)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each of the four long bones (left/right femur, left/right humerus), the
mask is cropped to its central **60%** of axial extent to approximate the
diaphysis, excluding the epiphyses.  Each cropped mask is then eroded (**5 mm**
for femora, **4 mm** for humeri) to exclude cortical bone.  The four diaphyseal
masks are combined via logical union.

**Metric:** 95th percentile of SUVbw within the combined mask.

See :class:`autods_pet.roi.long_bones.LongBonesROI`.


Brain (Cortical Grey Matter)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The brain mask (TotalSegmentator label 90) is extracted from the multilabel
segmentation and its largest connected component is retained.  When
``grey_matter_only`` is enabled (the default), the mask is eroded by the
configured ``cortical_thickness_mm`` (default **5 mm**) and the eroded
white-matter core is subtracted from the original to yield a cortical grey
matter shell.

**Metric:** Voxelwise median SUVbw.

The brain median is divided by the liver median to produce the
**Brain-to-Liver Ratio (BLR)**, a continuous metric (not a Deauville Score)
reported alongside the standard DS columns.

*Reference:* Aide N et al. (2026). Brain-to-liver ratio in FDG-PET/CT for
myeloma response assessment. *Eur J Nucl Med Mol Imaging*.
`doi:10.1007/s00259-026-07844-z <https://doi.org/10.1007/s00259-026-07844-z>`_

See :class:`autods_pet.roi.brain.BrainROI`.


Focal Lesion
~~~~~~~~~~~~

Focal lesion masks are provided as binary segmentations (semi-automatic
approach).  Patients without a focal lesion mask are assigned **DS 1** for the
focal lesion component.

**Metric:** Maximum SUVbw within the lesion mask.

See :class:`autods_pet.roi.target_roi.TargetROI`.


Deauville Score Assignment
--------------------------

For each target ROI the Deauville Score is assigned by comparing target
uptake against the MBP and liver reference values.  Five component scores
and one continuous metric are reported:

- **BM_DS** -- Bone marrow (vertebral bodies)
- **LB_DS** -- Long bones (femoral/humeral diaphyses)
- **FL_DS** -- Focal lesion
- **PM_DS** -- Paramedullary disease
- **EM_DS** -- Extramedullary disease
- **BLR** -- Brain-to-Liver Ratio (continuous value, not a Deauville Score)

Scoring criteria:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Score
     - Criterion
   * - **DS 1**
     - No measurable uptake (focal lesion only; absence of lesion)
   * - **DS 2**
     - Target uptake ≤ MBP
   * - **DS 3**
     - MBP < target uptake ≤ liver
   * - **DS 4**
     - Liver < target uptake ≤ 2 × liver
   * - **DS 5**
     - Target uptake > 2 × liver

This yields five component Deauville Scores per patient (``BM_DS``,
``LB_DS``, ``FL_DS``, ``PM_DS``, ``EM_DS``) plus the continuous
``BLR`` (Brain-to-Liver Ratio), each independently derived from the
automated pipeline.  Per-patient scores are saved to
``output_dir/{patient_id}_results/DeauvilleScores/deauville_scores.csv``.

See :func:`autods_pet.deauville.assign_ds`.


Implementation
--------------

The pipeline is implemented in Python using SimpleITK-SimpleElastix for image
processing and registration, NumPy for array operations, and TotalSegmentator
for anatomical segmentation.  The software is available as an installable
Python package (``autods-pet``).
