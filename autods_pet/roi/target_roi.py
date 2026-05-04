"""Target ROI: compute PET statistics from a user-provided binary mask.

Works for any target ROI (focal lesion, paramedullary, extramedullary,
or custom targets). No refinement step - the mask is provided externally.
"""

from __future__ import annotations

import SimpleITK as sitk

from ..imaging.geometry import check_same_geometry
from ..ops.stats import compute_stats
from ..results import ROIResult


class TargetROI:
    """Target ROI statistics from a user-provided binary mask.

    Parameters
    ----------
    stats : list[str] or None
        Statistics to compute (e.g. ``["max", "p90"]``).
        Defaults to ``["max"]``.
    """

    def __init__(self, stats: list[str] | None = None) -> None:
        self.stats = stats if stats is not None else ["max"]

    def extract(self, mask: sitk.Image, pet: sitk.Image) -> ROIResult:
        """Extract statistics from PET within a user-provided binary mask.

        Parameters
        ----------
        mask : sitk.Image
            Binary mask selecting the target voxels.
        pet : sitk.Image
            PET SUV image (must share geometry with *mask*).

        Returns
        -------
        ROIResult
            Result with ``stats`` and ``refined_mask`` populated.

        Raises
        ------
        ValueError
            If *mask* and *pet* have mismatched geometry.
        """
        if not check_same_geometry(mask, pet):
            raise ValueError("Geometry mismatch between target mask and PET image")

        mask_bin = sitk.Cast(mask != 0, sitk.sitkUInt8)

        return ROIResult(
            stats=compute_stats(self.stats, pet, mask_bin),
            refined_mask=mask_bin,
        )
