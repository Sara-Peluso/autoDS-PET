"""Distance-transform based morphology, robust to anisotropic spacing."""

from __future__ import annotations

import SimpleITK as sitk


def signed_distance_mm(mask: sitk.Image) -> sitk.Image:
    """Compute a signed distance map in mm from a binary mask.

    Parameters
    ----------
    mask : sitk.Image
        Binary mask (non-zero values are foreground).

    Returns
    -------
    sitk.Image
        Float image where positive values are *inside* the mask and negative
        values are *outside*, in physical mm units (respects voxel spacing).
    """
    mask = sitk.Cast(mask != 0, sitk.sitkUInt8)
    return sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )


def erode_mask_mm(
    mask: sitk.Image,
    radius_mm: float,
    *,
    dist: sitk.Image | None = None,
) -> sitk.Image:
    """Erode a binary mask by *radius_mm* using a distance transform.

    Pass a pre-computed *dist* (from :func:`signed_distance_mm`) to avoid
    recomputing it when eroding and dilating the same mask.

    Parameters
    ----------
    mask : sitk.Image
        Binary mask (non-zero values are foreground).
    radius_mm : float
        Erosion radius in physical mm units.
    dist : sitk.Image or None
        Pre-computed signed distance map.  When ``None`` (default), it is
        computed internally via :func:`signed_distance_mm`.

    Returns
    -------
    sitk.Image
        Eroded binary mask (uint8).
    """
    mask = sitk.Cast(mask != 0, sitk.sitkUInt8)
    if radius_mm <= 0:
        return mask
    if dist is None:
        dist = signed_distance_mm(mask)
    eroded = sitk.GreaterEqual(dist, float(radius_mm))
    return sitk.Cast(eroded, sitk.sitkUInt8)


def dilate_mask_mm(
    mask: sitk.Image,
    radius_mm: float,
    *,
    dist: sitk.Image | None = None,
) -> sitk.Image:
    """Dilate a binary mask by *radius_mm* using a distance transform.

    Pass a pre-computed *dist* (from :func:`signed_distance_mm`) to avoid
    recomputing it when eroding and dilating the same mask.

    Parameters
    ----------
    mask : sitk.Image
        Binary mask (non-zero values are foreground).
    radius_mm : float
        Dilation radius in physical mm units.
    dist : sitk.Image or None
        Pre-computed signed distance map.  When ``None`` (default), it is
        computed internally via :func:`signed_distance_mm`.

    Returns
    -------
    sitk.Image
        Dilated binary mask (uint8).
    """
    mask = sitk.Cast(mask != 0, sitk.sitkUInt8)
    if radius_mm <= 0:
        return mask
    if dist is None:
        dist = signed_distance_mm(mask)
    dilated = sitk.GreaterEqual(dist, float(-radius_mm))
    return sitk.Cast(dilated, sitk.sitkUInt8)
