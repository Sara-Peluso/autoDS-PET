"""Lumbar vertebral body: mask refinement and PET statistics."""

from __future__ import annotations

import SimpleITK as sitk

from .. import labels
from ..imaging.geometry import check_same_geometry
from ..ops.masks import label_union
from ..ops.morphology import erode_mask_mm
from ..ops.stats import compute_stats, shrinkage_report
from ..results import ROIResult


class LumbarVB:
    """Lumbar vertebral body ROI extraction and refinement.

    Parameters
    ----------
    lumbar_labels : list[int] or None
        TotalSegmentator labels for the target vertebrae.
        Defaults to L3, L4, L5.
    erosion_mm : float
        Physical erosion radius in mm.
    stats : list[str] or None
        Statistics to compute (e.g. ``["p95"]``).
        Defaults to ``["p95"]``.
    """

    def __init__(
        self,
        lumbar_labels: list[int] | None = None,
        erosion_mm: float = 3.0,
        stats: list[str] | None = None,
    ) -> None:
        self.lumbar_labels = (
            lumbar_labels if lumbar_labels is not None else labels.LUMBAR_L3_L5
        )
        self.erosion_mm = erosion_mm
        self.stats = stats if stats is not None else ["p95"]

    def refine(self, whole_seg: sitk.Image, vert_body_seg: sitk.Image) -> ROIResult:
        """Refine TotalSegmentator vertebra labels into a lumbar VB mask.

        Pipeline: union of lumbar labels -> intersection with vertebral body
        binary mask -> physical erosion.

        Parameters
        ----------
        whole_seg : sitk.Image
            TotalSegmentator multilabel segmentation.
        vert_body_seg : sitk.Image
            Binary vertebral body segmentation.

        Returns
        -------
        ROIResult
            Result with ``refined_mask`` and ``shrinkage`` populated.
        """
        vertebra_mask = label_union(whole_seg, self.lumbar_labels)

        vb_bin = sitk.Cast(vert_body_seg != 0, sitk.sitkUInt8)
        vb_mask = sitk.Cast(sitk.And(vertebra_mask, vb_bin), sitk.sitkUInt8)

        vb_eroded = erode_mask_mm(vb_mask, self.erosion_mm)

        return ROIResult(
            refined_mask=vb_eroded,
            shrinkage=shrinkage_report(vertebra_mask, vb_eroded),
        )

    def extract(
        self,
        whole_seg: sitk.Image,
        vert_body_seg: sitk.Image,
        pet: sitk.Image,
    ) -> ROIResult:
        """Refine mask *and* compute PET statistics in one call.

        Parameters
        ----------
        whole_seg : sitk.Image
            TotalSegmentator multilabel segmentation.
        vert_body_seg : sitk.Image
            Binary vertebral body segmentation.
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

        result = self.refine(whole_seg, vert_body_seg)
        result.stats = compute_stats(self.stats, pet, result.refined_mask)
        return result
