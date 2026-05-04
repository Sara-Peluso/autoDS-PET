"""Liver reference ROI: mask refinement and PET statistics."""

from __future__ import annotations

import SimpleITK as sitk

from .. import labels
from ..imaging.geometry import check_same_geometry
from ..ops.masks import fill_holes, keep_largest_component, label_mask
from ..ops.morphology import erode_mask_mm
from ..ops.stats import compute_stats, shrinkage_report
from ..results import ROIResult


class LiverROI:
    """Liver reference ROI extraction and refinement.

    Parameters
    ----------
    liver_label : int
        TotalSegmentator label for liver (default 5).
    erosion_mm : float
        Erosion radius in mm.
    max_hole_volume_mm3 : float or None
        If set, only fill holes smaller than this volume (mm³).
        Large holes (e.g. portal vein) are preserved.
    stats : list[str] or None
        Statistics to compute (e.g. ``["median"]``).
        Defaults to ``["median"]``.
    """

    def __init__(
        self,
        liver_label: int = labels.LIVER,
        erosion_mm: float = 10.0,
        max_hole_volume_mm3: float | None = None,
        stats: list[str] | None = None,
    ) -> None:
        self.liver_label = liver_label
        self.erosion_mm = erosion_mm
        self.max_hole_volume_mm3 = max_hole_volume_mm3
        self.stats = stats if stats is not None else ["median"]

    def refine(self, whole_seg: sitk.Image) -> ROIResult:
        """Refine TotalSegmentator liver label into a core-parenchyma mask.

        Pipeline: extract label -> keep largest component -> fill holes ->
        erode to avoid partial-volume at boundaries.

        Parameters
        ----------
        whole_seg : sitk.Image
            TotalSegmentator multilabel segmentation.

        Returns
        -------
        ROIResult
            Result with ``refined_mask`` and ``shrinkage`` populated.
        """
        liver_raw = label_mask(whole_seg, self.liver_label)
        liver_clean = keep_largest_component(liver_raw)
        liver_filled = fill_holes(
            liver_clean, max_hole_volume_mm3=self.max_hole_volume_mm3
        )
        liver_eroded = erode_mask_mm(liver_filled, self.erosion_mm)

        return ROIResult(
            refined_mask=liver_eroded,
            shrinkage=shrinkage_report(liver_raw, liver_eroded),
        )

    def extract(self, whole_seg: sitk.Image, pet: sitk.Image) -> ROIResult:
        """Refine mask *and* compute PET statistics in one call.

        Parameters
        ----------
        whole_seg : sitk.Image
            TotalSegmentator multilabel segmentation.
        pet : sitk.Image
            PET SUV image (must share geometry with *whole_seg*).

        Returns
        -------
        ROIResult
            Result with ``refined_mask``, ``shrinkage``, and ``stats``
            populated.

        Raises
        ------
        ValueError
            If *whole_seg* and *pet* have mismatched geometry.
        """
        if not check_same_geometry(whole_seg, pet):
            raise ValueError("Geometry mismatch between whole_seg and PET")

        result = self.refine(whole_seg)
        result.stats = compute_stats(self.stats, pet, result.refined_mask)
        return result
