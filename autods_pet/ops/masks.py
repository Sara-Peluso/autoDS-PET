"""Mask operations for multilabel segmentations."""

from __future__ import annotations

import SimpleITK as sitk


def label_mask(seg: sitk.Image, label: int) -> sitk.Image:
    """Extract a binary mask for a single label from a multilabel segmentation.

    Parameters
    ----------
    seg : sitk.Image
        Multilabel segmentation image (integer-valued).
    label : int
        The label value to extract.

    Returns
    -------
    sitk.Image
        Binary ``uint8`` mask where the selected label is 1 and all else is 0.
    """
    return sitk.Cast(sitk.Equal(seg, int(label)), sitk.sitkUInt8)


def label_union(seg: sitk.Image, labels: list[int]) -> sitk.Image:
    """Binary mask that is the union of several labels.

    Parameters
    ----------
    seg : sitk.Image
        Multilabel segmentation image (integer-valued).
    labels : list[int]
        Label values to include.  An empty list returns an all-zero mask.

    Returns
    -------
    sitk.Image
        Binary ``uint8`` mask where any of the selected labels is 1.
    """
    if not labels:
        empty = sitk.Image(seg.GetSize(), sitk.sitkUInt8)
        empty.CopyInformation(seg)
        return empty
    mask = None
    for lab in labels:
        m = label_mask(seg, lab)
        mask = m if mask is None else sitk.Or(mask, m)
    return sitk.Cast(mask, sitk.sitkUInt8)


def keep_largest_component(mask: sitk.Image) -> sitk.Image:
    """Keep only the largest connected component of a binary mask.

    Parameters
    ----------
    mask : sitk.Image
        Binary mask (non-zero values are foreground).

    Returns
    -------
    sitk.Image
        Binary ``uint8`` mask containing only the largest connected component.
        If the mask is empty, returns the input unchanged.
    """
    mask = sitk.Cast(mask != 0, sitk.sitkUInt8)
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    if stats.GetNumberOfLabels() == 0:
        return mask

    best = max(stats.GetLabels(), key=lambda L: stats.GetNumberOfPixels(L))
    return sitk.Cast(sitk.Equal(cc, best), sitk.sitkUInt8)


def fill_holes(
    mask: sitk.Image,
    max_hole_volume_mm3: float | None = None,
) -> sitk.Image:
    """Fill holes in a binary mask.

    Parameters
    ----------
    mask : sitk.Image
        Binary mask.
    max_hole_volume_mm3 : float or None
        If *None*, fills all holes (original behavior).
        If set, only fills holes whose volume is below this threshold (mm³).
        Large holes (e.g. portal vein in liver) are preserved.

    Returns
    -------
    sitk.Image
        Binary ``uint8`` mask with holes filled.
    """
    mask = sitk.Cast(mask != 0, sitk.sitkUInt8)

    if max_hole_volume_mm3 is None:
        filled = sitk.BinaryFillhole(mask, fullyConnected=True)
        return sitk.Cast(filled, sitk.sitkUInt8)

    # Identify holes: invert mask, then label connected components.
    # The background is the component touching the image border; everything
    # else is a hole inside the mask.
    inverted = sitk.Cast(mask == 0, sitk.sitkUInt8)
    cc = sitk.ConnectedComponent(inverted, True)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    if stats.GetNumberOfLabels() == 0:
        return mask

    # Find the background label: the component whose bounding box touches
    # the image border. If none touches the border, there is no exterior
    # (mask doesn't reach the image edge) - all components are holes.
    cc_arr = sitk.GetArrayFromImage(cc)
    border_labels: set[int] = set()
    for slc in (
        cc_arr[0, :, :],
        cc_arr[-1, :, :],
        cc_arr[:, 0, :],
        cc_arr[:, -1, :],
        cc_arr[:, :, 0],
        cc_arr[:, :, -1],
    ):
        border_labels.update(slc.ravel())
    border_labels.discard(0)  # 0 is the mask itself, not a hole

    spacing = mask.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]

    # Build a mask of small holes to fill
    fill_mask = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
    fill_mask.CopyInformation(mask)

    for label in stats.GetLabels():
        if label in border_labels:
            continue
        hole_vol = stats.GetNumberOfPixels(label) * voxel_vol
        if hole_vol < max_hole_volume_mm3:
            fill_mask = sitk.Or(
                fill_mask, sitk.Cast(sitk.Equal(cc, label), sitk.sitkUInt8)
            )

    result = sitk.Or(mask, fill_mask)
    return sitk.Cast(result, sitk.sitkUInt8)


def subtract_mask(base: sitk.Image, subtract: sitk.Image) -> sitk.Image:
    """Set difference of two binary masks (*base* minus *subtract*).

    Parameters
    ----------
    base : sitk.Image
        Binary mask to subtract from.
    subtract : sitk.Image
        Binary mask whose foreground voxels are removed from *base*.

    Returns
    -------
    sitk.Image
        Binary ``uint8`` mask containing voxels in *base* that are not
        in *subtract*.
    """
    return sitk.Cast(
        sitk.And(
            sitk.Cast(base != 0, sitk.sitkUInt8),
            sitk.Not(sitk.Cast(subtract != 0, sitk.sitkUInt8)),
        ),
        sitk.sitkUInt8,
    )
