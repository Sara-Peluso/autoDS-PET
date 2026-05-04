"""Geometry checks and spatial alignment for SimpleITK images."""

from __future__ import annotations

import SimpleITK as sitk


def check_same_geometry(
    a: sitk.Image,
    b: sitk.Image,
    tol: float = 1e-4,
) -> bool:
    """Check whether two images share the same spatial geometry.

    Compares size, spacing, origin, and direction cosine matrix.

    Parameters
    ----------
    a : sitk.Image
        First image.
    b : sitk.Image
        Second image.
    tol : float
        Absolute tolerance for floating-point comparisons of spacing,
        origin, and direction elements.

    Returns
    -------
    bool
        ``True`` if the images match in size, spacing, origin, and direction
        within *tol*.
    """
    if a.GetSize() != b.GetSize():
        return False
    if any(abs(a.GetSpacing()[i] - b.GetSpacing()[i]) > tol for i in range(3)):
        return False
    if any(abs(a.GetOrigin()[i] - b.GetOrigin()[i]) > tol for i in range(3)):
        return False
    if any(abs(a.GetDirection()[i] - b.GetDirection()[i]) > tol for i in range(9)):
        return False
    return True


def check_sub_geometry(
    a: sitk.Image,
    b: sitk.Image,
    tol: float = 1e-4,
) -> bool:
    """Check whether *a* lives on a compatible voxel grid as *b* (sub-volume).

    Returns ``True`` when spacing matches within *tol* and direction
    cosine axes are parallel (signs may differ - a Z-flip between LPS
    and RAS conventions is allowed).  Size and origin may differ freely.

    This identifies the common DICOM SEG case where the mask only covers
    the slices that contain a contour (fewer slices, different origin)
    and may have been exported with a flipped axis convention.

    Parameters
    ----------
    a : sitk.Image
        Candidate sub-volume (e.g. a partial-FOV mask).
    b : sitk.Image
        Reference volume (e.g. full PET or CT).
    tol : float
        Absolute tolerance for floating-point comparisons.

    Returns
    -------
    bool
        ``True`` if spacing matches and direction axes are parallel.
    """
    if any(abs(a.GetSpacing()[i] - b.GetSpacing()[i]) > tol for i in range(3)):
        return False
    # Compare direction cosines allowing sign flips (LPS ↔ RAS).
    if any(
        abs(abs(a.GetDirection()[i]) - abs(b.GetDirection()[i])) > tol for i in range(9)
    ):
        return False
    return True


def resample_to_reference(
    mask: sitk.Image,
    reference: sitk.Image,
) -> sitk.Image:
    """Resample *mask* onto the *reference* grid (zero-padded, nearest-neighbour).

    Used to embed a partial-FOV binary mask (e.g. a DICOM SEG covering
    only a few slices) into the full reference volume.  Voxels outside
    the mask's original extent are set to 0.

    Parameters
    ----------
    mask : sitk.Image
        Binary mask (sub-volume).
    reference : sitk.Image
        Full-size reference image whose grid defines the output.

    Returns
    -------
    sitk.Image
        Resampled mask with the same size, origin, spacing, and
        direction as *reference*.
    """
    return sitk.Resample(
        mask,
        reference,
        sitk.Transform(),  # identity - same physical coordinate system
        sitk.sitkNearestNeighbor,
        0,  # default pixel value for out-of-bounds regions
        mask.GetPixelID(),
    )
