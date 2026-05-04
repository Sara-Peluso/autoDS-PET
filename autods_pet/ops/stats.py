"""Statistics helpers for voxel-level image analysis.

For computing multiple statistics on the same image/mask pair, prefer
:func:`compute_stats` over calling individual functions (e.g.
:func:`mean_in_mask`, :func:`max_in_mask`).  ``compute_stats`` extracts
the masked voxel array once and reuses it for all requested statistics,
avoiding redundant O(n) array copies per call.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk


def percentile_in_mask(
    img: sitk.Image,
    mask: sitk.Image,
    pct: float,
) -> float | None:
    """Return the *pct*-th percentile of voxel values inside *mask*.

    Parameters
    ----------
    img : sitk.Image
        Scalar image (e.g. PET SUVbw).
    mask : sitk.Image
        Binary mask selecting the voxels of interest.
    pct : float
        Percentile in the range ``[0, 100]``.

    Returns
    -------
    float or None
        The requested percentile, or ``None`` if the mask is empty.

    Raises
    ------
    ValueError
        If *pct* is outside ``[0, 100]``.
    """
    if not (0 <= pct <= 100):
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    vals = sitk.GetArrayFromImage(img)[sitk.GetArrayViewFromImage(mask).astype(bool)]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, pct, method="linear"))


def mean_in_mask(img: sitk.Image, mask: sitk.Image) -> float | None:
    """Return the arithmetic mean of voxel values inside *mask*.

    Parameters
    ----------
    img : sitk.Image
        Scalar image.
    mask : sitk.Image
        Binary mask selecting the voxels of interest.

    Returns
    -------
    float or None
        The mean value, or ``None`` if the mask is empty.
    """
    vals = sitk.GetArrayFromImage(img)[sitk.GetArrayViewFromImage(mask).astype(bool)]
    if vals.size == 0:
        return None
    return float(np.mean(vals))


def max_in_mask(img: sitk.Image, mask: sitk.Image) -> float | None:
    """Return the maximum voxel value inside *mask*.

    Parameters
    ----------
    img : sitk.Image
        Scalar image.
    mask : sitk.Image
        Binary mask selecting the voxels of interest.

    Returns
    -------
    float or None
        The maximum value, or ``None`` if the mask is empty.
    """
    vals = sitk.GetArrayFromImage(img)[sitk.GetArrayViewFromImage(mask).astype(bool)]
    if vals.size == 0:
        return None
    return float(np.max(vals))


def min_in_mask(img: sitk.Image, mask: sitk.Image) -> float | None:
    """Return the minimum voxel value inside *mask*.

    Parameters
    ----------
    img : sitk.Image
        Scalar image.
    mask : sitk.Image
        Binary mask selecting the voxels of interest.

    Returns
    -------
    float or None
        The minimum value, or ``None`` if the mask is empty.
    """
    vals = sitk.GetArrayFromImage(img)[sitk.GetArrayViewFromImage(mask).astype(bool)]
    if vals.size == 0:
        return None
    return float(np.min(vals))


def voxelwise_median(img: sitk.Image, mask: sitk.Image) -> float | None:
    """Return the median of voxel values inside *mask*.

    Parameters
    ----------
    img : sitk.Image
        Scalar image.
    mask : sitk.Image
        Binary mask selecting the voxels of interest.

    Returns
    -------
    float or None
        The median value, or ``None`` if the mask is empty.
    """
    vals = sitk.GetArrayFromImage(img)[sitk.GetArrayViewFromImage(mask).astype(bool)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def _compute_on_vals(
    vals: npt.NDArray[np.floating[Any]], kind: str, param: float | None
) -> float | None:
    """Compute a single stat on pre-extracted masked values."""
    if vals.size == 0:
        return None
    if kind == "mean":
        return float(np.mean(vals))
    if kind == "median":
        return float(np.median(vals))
    if kind == "min":
        return float(np.min(vals))
    if kind == "max":
        return float(np.max(vals))
    if kind == "percentile":
        if param is None:
            raise ValueError("percentile stat requires a numeric parameter")
        return float(np.percentile(vals, param, method="linear"))
    raise ValueError(f"Unsupported stat kind: {kind!r}")


def compute_stats(
    stat_names: list[str],
    img: sitk.Image,
    mask: sitk.Image,
) -> dict[str, float | None]:
    """Compute multiple named statistics for *img* inside *mask*.

    Parameters
    ----------
    stat_names : list[str]
        Names like ``"mean"``, ``"median"``, ``"min"``, ``"max"``,
        ``"p90"``, ``"p95"``.
    img : sitk.Image
        Scalar image (e.g. PET SUVbw).
    mask : sitk.Image
        Binary mask selecting voxels of interest.

    Returns
    -------
    dict[str, float | None]
        ``{stat_name: value}`` for each requested stat.
    """
    from autods_pet.config import parse_stat  # local import to avoid circular dep

    vals = sitk.GetArrayFromImage(img)[sitk.GetArrayViewFromImage(mask).astype(bool)]
    results: dict[str, float | None] = {}
    for name in stat_names:
        kind, param = parse_stat(name)
        results[name] = _compute_on_vals(vals, kind, param)
    return results


def count_voxels(mask: sitk.Image) -> int:
    """Count non-zero voxels in a binary mask.

    Parameters
    ----------
    mask : sitk.Image
        Binary mask image.

    Returns
    -------
    int
        Number of non-zero voxels.
    """
    return int(sitk.GetArrayFromImage(mask).astype(bool).sum())


def mask_volume_mm3(mask: sitk.Image) -> float:
    r"""Compute the physical volume of non-zero voxels in mm\ :sup:`3`.

    Parameters
    ----------
    mask : sitk.Image
        Binary mask image with valid spacing metadata.

    Returns
    -------
    float
        Volume in mm\ :sup:`3`.
    """
    spacing = mask.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    return count_voxels(mask) * voxel_vol


def shrinkage_report(original: sitk.Image, refined: sitk.Image) -> dict[str, Any]:
    """Compute voxel-count and physical-volume deltas between two masks.

    Parameters
    ----------
    original : sitk.Image
        Binary mask before refinement (e.g. raw segmentation).
    refined : sitk.Image
        Binary mask after refinement (e.g. eroded version).

    Returns
    -------
    dict
        Keys: ``original_voxels``, ``refined_voxels``, ``original_volume_mm3``,
        ``refined_volume_mm3``, ``delta_voxels``, ``delta_volume_mm3``,
        ``shrinkage_pct``.
    """
    orig_vox = count_voxels(original)
    ref_vox = count_voxels(refined)

    spacing = original.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]

    orig_vol = orig_vox * voxel_vol
    ref_vol = ref_vox * voxel_vol

    return {
        "original_voxels": orig_vox,
        "refined_voxels": ref_vox,
        "original_volume_mm3": orig_vol,
        "refined_volume_mm3": ref_vol,
        "delta_voxels": ref_vox - orig_vox,
        "delta_volume_mm3": ref_vol - orig_vol,
        "shrinkage_pct": (1.0 - ref_vol / orig_vol) * 100.0 if orig_vol > 0 else 0.0,
    }
