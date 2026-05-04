"""Aorta metabolic blood pool (MBP): mask refinement and PET statistics.

Extracts the aorta from TotalSegmentator, restricts it to a vertebral
slab via slice gating, excludes heart spill-in, erodes intraluminally.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import SimpleITK as sitk

from .. import labels
from ..imaging.geometry import check_same_geometry
from ..ops.masks import keep_largest_component, label_mask, label_union
from ..ops.morphology import dilate_mask_mm, erode_mask_mm, signed_distance_mm
from ..ops.stats import compute_stats, shrinkage_report
from ..results import ROIResult


def _slicegate_by_slab(
    aorta: sitk.Image,
    slab: sitk.Image,
    axis_xyz: int,
) -> sitk.Image:
    """Keep aorta voxels only in slices where the vertebral slab is present.

    Parameters
    ----------
    aorta : sitk.Image
        Binary aorta mask.
    slab : sitk.Image
        Binary vertebral slab mask.
    axis_xyz : int
        SimpleITK index axis (0=x, 1=y, 2=z).
    """
    a_arr = sitk.GetArrayFromImage(aorta)
    s_arr = sitk.GetArrayFromImage(slab)

    # SimpleITK (x,y,z) -> numpy (z,y,x): x->2, y->1, z->0
    axis_map = {0: 2, 1: 1, 2: 0}
    ax = axis_map[axis_xyz]

    # Vectorized: per-slice slab presence → broadcast mask
    other_axes = tuple(i for i in range(3) if i != ax)
    has_slab = np.any(s_arr != 0, axis=other_axes)  # (n_slices,)

    shape = [1, 1, 1]
    shape[ax] = a_arr.shape[ax]
    gate = has_slab.reshape(shape)

    out = np.where(gate, a_arr, np.uint8(0)).astype(np.uint8)

    img = sitk.GetImageFromArray(out)
    img.CopyInformation(aorta)
    return sitk.Cast(img, sitk.sitkUInt8)


def _exclude_heart_dilate(
    aorta_slab: sitk.Image,
    heart: sitk.Image,
    params: dict[str, float],
) -> sitk.Image:
    """Exclude heart by dilating it and subtracting from aorta."""
    heart_buf = dilate_mask_mm(heart, params["heart_dilation_mm"])
    return sitk.Cast(
        sitk.And(aorta_slab, sitk.Cast(sitk.Not(heart_buf), sitk.sitkUInt8)),
        sitk.sitkUInt8,
    )


def _exclude_heart_distance(
    aorta_slab: sitk.Image,
    heart: sitk.Image,
    params: dict[str, float],
) -> sitk.Image:
    """Exclude aorta voxels within a distance threshold of heart."""
    dist_heart = signed_distance_mm(heart)
    far = sitk.LessEqual(dist_heart, float(-params["heart_distance_mm"]))
    return sitk.Cast(
        sitk.And(aorta_slab, sitk.Cast(far, sitk.sitkUInt8)),
        sitk.sitkUInt8,
    )


_HeartExclusionFn = Callable[[sitk.Image, sitk.Image, dict[str, float]], sitk.Image]

HEART_EXCLUSION_STRATEGIES: dict[str, _HeartExclusionFn] = {
    "dilate_intersection": _exclude_heart_dilate,
    "distance": _exclude_heart_distance,
}


class AortaMBP:
    """Aorta metabolic blood pool (MBP) extraction and refinement.

    Parameters
    ----------
    vertebra_labels : list[int] or None
        Labels for the vertebral slab. Defaults to T4-T8.
    slab_axis : int
        SimpleITK index axis for slice gating (0=x, 1=y, 2=z).
    heart_exclusion_mode : str
        ``"dilate_intersection"`` or ``"distance"``.
    heart_dilation_mm : float
        Dilation radius for heart buffer (mode ``"dilate_intersection"``).
    heart_distance_mm : float
        Distance threshold (mode ``"distance"``).
    aorta_erosion_mm : float
        Intraluminal erosion radius in mm.
    stats : list[str] or None
        Statistics to compute. Defaults to ``["median"]``.
    """

    def __init__(
        self,
        vertebra_labels: list[int] | None = None,
        slab_axis: int = 2,
        heart_exclusion_mode: str = "dilate_intersection",
        heart_dilation_mm: float = 6.0,
        heart_distance_mm: float = 12.0,
        aorta_erosion_mm: float = 4.0,
        stats: list[str] | None = None,
    ) -> None:
        self.vertebra_labels = (
            vertebra_labels if vertebra_labels is not None else labels.THORACIC_T4_T8
        )
        self.slab_axis = slab_axis
        self.heart_exclusion_mode = heart_exclusion_mode
        self.heart_dilation_mm = heart_dilation_mm
        self.heart_distance_mm = heart_distance_mm
        self.aorta_erosion_mm = aorta_erosion_mm
        self.stats = stats if stats is not None else ["median"]

    def refine(self, whole_seg: sitk.Image) -> ROIResult:
        """Refine TotalSegmentator aorta into a thoracic intraluminal MBP mask.

        Pipeline: extract aorta label -> slice-gate by T4-T8 vertebral slab ->
        exclude heart buffer -> erode intraluminally -> keep largest component.

        Parameters
        ----------
        whole_seg : sitk.Image
            TotalSegmentator multilabel segmentation (must include aorta,
            heart, and vertebra labels).

        Returns
        -------
        ROIResult
            Result with ``refined_mask`` and ``shrinkage`` populated.
        """
        strategy = HEART_EXCLUSION_STRATEGIES.get(self.heart_exclusion_mode)
        if strategy is None:
            raise ValueError(
                f"heart_exclusion_mode must be one of "
                f"{list(HEART_EXCLUSION_STRATEGIES)}, "
                f"got {self.heart_exclusion_mode!r}"
            )

        aorta_raw = label_mask(whole_seg, labels.AORTA)
        heart = label_mask(whole_seg, labels.HEART)
        slab = label_union(whole_seg, self.vertebra_labels)

        # Slice gating
        aorta_slab = _slicegate_by_slab(aorta_raw, slab, self.slab_axis)

        # Heart exclusion via strategy dispatch
        aorta_noheart = strategy(
            aorta_slab,
            heart,
            {
                "heart_dilation_mm": self.heart_dilation_mm,
                "heart_distance_mm": self.heart_distance_mm,
            },
        )

        # Intraluminal erosion + largest component
        aorta_final = erode_mask_mm(aorta_noheart, self.aorta_erosion_mm)
        aorta_final = keep_largest_component(aorta_final)

        return ROIResult(
            refined_mask=aorta_final,
            shrinkage=shrinkage_report(aorta_raw, aorta_final),
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
