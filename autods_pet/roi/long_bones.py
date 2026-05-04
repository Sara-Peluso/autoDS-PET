"""Long bones: diaphysis mask refinement and PET statistics."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from .. import labels
from ..imaging.geometry import check_same_geometry
from ..ops.masks import label_mask
from ..ops.morphology import erode_mask_mm
from ..ops.stats import (
    compute_stats,
    count_voxels,
    mask_volume_mm3,
    shrinkage_report,
)
from ..results import ROIResult

# Default bone definitions: (name, TotalSegmentator label, erosion in mm)
DEFAULT_BONES: list[tuple[str, int, float]] = [
    ("femur_L", labels.FEMUR_L, 5.0),
    ("femur_R", labels.FEMUR_R, 5.0),
    ("humerus_L", labels.HUMERUS_L, 4.0),
    ("humerus_R", labels.HUMERUS_R, 4.0),
]


def _crop_to_diaphysis_z(
    mask: sitk.Image,
    keep_pct: int,
) -> tuple[sitk.Image, dict]:
    """Crop a single bone mask to its central diaphysis along the Z axis.

    NOTE: assumes the bone's long axis aligns with array axis 0 (Z in standard
    axial CT orientation). TotalSegmentator outputs are always in standard
    orientation, so this is safe for the intended workflow.

    Parameters
    ----------
    mask : sitk.Image
        Binary mask for a single bone.
    keep_pct : int
        Percentage (1-100) of the bone's axial extent to keep (central portion).

    Returns
    -------
    tuple[sitk.Image, dict]
        Cropped mask and info dict with slice window details.
    """
    if keep_pct <= 0 or keep_pct > 100:
        raise ValueError(f"keep_pct must be in 1..100, got {keep_pct}")

    arr = sitk.GetArrayFromImage(mask).astype(np.uint8)  # (z, y, x)

    z_has = np.any(arr > 0, axis=(1, 2))
    idxs = np.where(z_has)[0]

    if idxs.size == 0:
        out = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
        out.CopyInformation(mask)
        return out, {
            "start_z": None,
            "end_z": None,
            "n_slices": 0,
            "keep": 0,
            "mid_start": None,
            "mid_end": None,
        }

    start_z = int(idxs[0])
    end_z = int(idxs[-1])
    n_slices = int(idxs.size)

    keep = max(1, int(round(n_slices * (keep_pct / 100.0))))

    # Extent midpoint approximates diaphysis center; assumes symmetric
    # TotalSegmentator segmentation (both epiphyses fully in FOV).
    middle = (start_z + end_z) // 2
    half = keep // 2

    mid_start = max(start_z, middle - half)
    mid_end = min(end_z, mid_start + keep - 1)
    mid_start = max(start_z, mid_end - keep + 1)

    out_arr = np.zeros_like(arr, dtype=np.uint8)
    out_arr[mid_start : mid_end + 1, :, :] = arr[mid_start : mid_end + 1, :, :]

    out = sitk.GetImageFromArray(out_arr)
    out.CopyInformation(mask)

    info = {
        "start_z": start_z,
        "end_z": end_z,
        "n_slices": n_slices,
        "keep": keep,
        "mid_start": int(mid_start),
        "mid_end": int(mid_end),
    }
    return sitk.Cast(out, sitk.sitkUInt8), info


class LongBonesROI:
    """Long bones diaphysis ROI extraction and refinement.

    Parameters
    ----------
    bones : list[tuple[str, int, float]] or None
        ``(name, label, erosion_mm)`` tuples.  Defaults to both femurs
        (5 mm) and both humeri (4 mm).
    diaphysis_keep_pct : int
        Central percentage of each bone's axial extent to keep.
    stats : list[str] or None
        Statistics to compute. Defaults to ``["p95"]``.
    """

    def __init__(
        self,
        bones: list[tuple[str, int, float]] | None = None,
        diaphysis_keep_pct: int = 60,
        stats: list[str] | None = None,
    ) -> None:
        self.bones = bones if bones is not None else DEFAULT_BONES
        self.diaphysis_keep_pct = diaphysis_keep_pct
        self.stats = stats if stats is not None else ["p95"]

    def refine(self, whole_seg: sitk.Image) -> ROIResult:
        """Refine TotalSegmentator bone labels into diaphyseal marrow masks.

        Pipeline per bone: extract label -> crop to central diaphysis ->
        erode cortex.  The four masks are combined via logical union.

        Parameters
        ----------
        whole_seg : sitk.Image
            TotalSegmentator multilabel segmentation.

        Returns
        -------
        ROIResult
            Result with ``refined_mask`` and ``shrinkage`` populated.
            The ``shrinkage`` dict includes a ``per_bone`` sub-dict with
            per-bone volume details.
        """
        combined_raw = None
        combined_refined = None
        per_bone: dict[str, dict] = {}

        for bone_name, bone_label, bone_erosion_mm in self.bones:
            raw = label_mask(whole_seg, bone_label)
            crop, z_info = _crop_to_diaphysis_z(raw, self.diaphysis_keep_pct)
            eroded = erode_mask_mm(crop, bone_erosion_mm)

            per_bone[bone_name] = {
                "raw_voxels": count_voxels(raw),
                "raw_volume_mm3": mask_volume_mm3(raw),
                "refined_voxels": count_voxels(eroded),
                "refined_volume_mm3": mask_volume_mm3(eroded),
                "z_window": z_info,
            }

            if combined_raw is None:
                combined_raw = raw
                combined_refined = eroded
            else:
                combined_raw = sitk.Or(combined_raw, raw)
                combined_refined = sitk.Or(combined_refined, eroded)

        if combined_raw is None or combined_refined is None:
            empty = sitk.Image(whole_seg.GetSize(), sitk.sitkUInt8)
            empty.CopyInformation(whole_seg)
            return ROIResult(
                refined_mask=empty,
                shrinkage={
                    "original_voxels": 0,
                    "refined_voxels": 0,
                    "original_volume_mm3": 0.0,
                    "refined_volume_mm3": 0.0,
                    "delta_voxels": 0,
                    "delta_volume_mm3": 0.0,
                    "shrinkage_pct": 0.0,
                    "per_bone": per_bone,
                },
            )

        combined_raw = sitk.Cast(combined_raw, sitk.sitkUInt8)
        combined_refined = sitk.Cast(combined_refined, sitk.sitkUInt8)

        return ROIResult(
            refined_mask=combined_refined,
            shrinkage={
                **shrinkage_report(combined_raw, combined_refined),
                "per_bone": per_bone,
            },
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
