"""Brain reference ROI: cortical gray matter extraction and PET statistics."""

from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk

from .. import labels
from ..imaging.geometry import check_same_geometry
from ..ops.masks import keep_largest_component, label_mask
from ..ops.morphology import erode_mask_mm
from ..ops.stats import compute_stats, shrinkage_report
from ..results import ROIResult

log = logging.getLogger(__name__)


class BrainROI:
    """Brain reference ROI extraction and refinement.

    When ``grey_matter_only`` is True (default), the brain mask is refined
    to isolate cortical gray matter using a shell extraction: the mask is
    eroded by ``cortical_thickness_mm`` to obtain the white-matter core,
    then subtracted from the original to yield the cortical shell.

    Parameters
    ----------
    brain_label : int
        TotalSegmentator label for brain (default 90).
    grey_matter_only : bool
        If True, extract cortical gray matter shell. If False, use the
        full brain mask.
    cortical_thickness_mm : float
        Erosion radius used for the shell extraction (only when
        ``grey_matter_only`` is True).
    stats : list[str] or None
        Statistics to compute (e.g. ``["median"]``).
        Defaults to ``["median"]``.
    """

    def __init__(
        self,
        brain_label: int = labels.BRAIN,
        grey_matter_only: bool = True,
        cortical_thickness_mm: float = 5.0,
        stats: list[str] | None = None,
    ) -> None:
        self.brain_label = brain_label
        self.grey_matter_only = grey_matter_only
        self.cortical_thickness_mm = cortical_thickness_mm
        self.stats = stats if stats is not None else ["median"]

    def refine(self, whole_seg: sitk.Image) -> ROIResult | None:
        """Refine TotalSegmentator brain label.

        If ``grey_matter_only`` is True, produces a cortical shell mask
        (original minus eroded).  Otherwise returns the cleaned brain mask.

        Parameters
        ----------
        whole_seg : sitk.Image
            TotalSegmentator multilabel segmentation.

        Returns
        -------
        ROIResult or None
            Result with ``refined_mask`` and ``shrinkage`` populated, or
            ``None`` if the brain label is absent from the segmentation.
        """
        brain_raw = label_mask(whole_seg, self.brain_label)
        if np.count_nonzero(sitk.GetArrayViewFromImage(brain_raw)) == 0:
            log.warning(
                "Brain label %d not found in segmentation (FOV may not include brain).",
                self.brain_label,
            )
            return None
        brain_clean = keep_largest_component(brain_raw)

        if self.grey_matter_only and self.cortical_thickness_mm > 0:
            white_matter = erode_mask_mm(brain_clean, self.cortical_thickness_mm)
            cortex = sitk.Cast(brain_clean, sitk.sitkUInt8) - sitk.Cast(
                white_matter, sitk.sitkUInt8
            )
            refined = sitk.Cast(cortex > 0, sitk.sitkUInt8)
        else:
            refined = brain_clean

        return ROIResult(
            refined_mask=refined,
            shrinkage=shrinkage_report(brain_raw, refined),
        )

    def extract(self, whole_seg: sitk.Image, pet: sitk.Image) -> ROIResult | None:
        """Refine mask *and* compute PET statistics in one call.

        Parameters
        ----------
        whole_seg : sitk.Image
            TotalSegmentator multilabel segmentation.
        pet : sitk.Image
            PET SUV image (must share geometry with *whole_seg*).

        Returns
        -------
        ROIResult or None
            Result with ``refined_mask``, ``shrinkage``, and ``stats``
            populated, or ``None`` if the brain label is absent from the
            segmentation.

        Raises
        ------
        ValueError
            If *whole_seg* and *pet* have mismatched geometry.
        """
        if not check_same_geometry(whole_seg, pet):
            raise ValueError("Geometry mismatch between whole_seg and PET")

        result = self.refine(whole_seg)
        if result is None:
            return None
        result.stats = compute_stats(self.stats, pet, result.refined_mask)
        return result
