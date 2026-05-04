"""Read DICOM SEG segmentation objects as SimpleITK images.

This module lazily imports ``highdicom`` and ``pydicom`` so that the
optional dependency is only required when a ``.dcm`` mask file is
actually encountered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# DICOM SEG SOP Class UID (Segmentation Storage).
_SEG_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66.4"

_INSTALL_HINT = (
    "DICOM SEG support requires the 'highdicom' package. "
    "Install it with:  pip install autods-pet[dicom-seg]"
)


def _import_highdicom() -> Any:
    """Lazy-import highdicom, raising a helpful error if missing."""
    try:
        import highdicom  # noqa: F811
    except ModuleNotFoundError:
        raise ModuleNotFoundError(_INSTALL_HINT) from None
    return highdicom


def is_dicom_seg(path: Path) -> bool:
    """Check whether *path* is a DICOM SEG file (header-only read).

    Parameters
    ----------
    path : Path
        Path to a ``.dcm`` file.

    Returns
    -------
    bool
        ``True`` if the file's SOPClassUID matches Segmentation Storage.
    """
    import pydicom

    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return getattr(ds, "SOPClassUID", None) == _SEG_SOP_CLASS_UID
    except Exception:
        log.debug("Cannot read %s as DICOM SEG", path, exc_info=True)
        return False


def read_referenced_series_uids(path: Path) -> list[str]:
    """Read the ``ReferencedSeriesSequence`` SeriesInstanceUIDs from a DICOM SEG.

    Header-only read; no ``highdicom`` dependency.  Used by mask discovery
    to match a SEG file against the patient's PET series.

    Parameters
    ----------
    path : Path
        Path to a DICOM SEG (``.dcm``) file.

    Returns
    -------
    list[str]
        ``SeriesInstanceUID`` values referenced by the SEG.  Empty list if
        the sequence is missing or the file cannot be parsed.
    """
    import pydicom

    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        log.debug("Cannot read DICOM headers from %s", path, exc_info=True)
        return []

    seq = getattr(ds, "ReferencedSeriesSequence", None)
    if seq is None:
        return []

    uids: list[str] = []
    for item in seq:
        uid = getattr(item, "SeriesInstanceUID", None)
        if uid is not None:
            uids.append(str(uid))
    return uids


def list_segments(path: Path) -> list[dict[str, Any]]:
    """List all segments in a DICOM SEG file.

    Parameters
    ----------
    path : Path
        Path to a DICOM SEG (``.dcm``) file.

    Returns
    -------
    list[dict[str, Any]]
        Each dict has keys ``number``, ``label``, and ``description``.
    """
    hd = _import_highdicom()
    seg = hd.seg.segread(str(path))
    result: list[dict[str, Any]] = []
    for seg_num in seg.segment_numbers:
        desc = seg.get_segment_description(seg_num)
        result.append(
            {
                "number": int(seg_num),
                "label": str(desc.segment_label),
                "description": str(getattr(desc, "SegmentDescription", "")),
            }
        )
    return result


def _select_segment_number(
    seg: Any,
    segment_label: str | None,
) -> int:
    """Resolve the segment number to extract.

    Parameters
    ----------
    seg : highdicom.seg.Segmentation
        Loaded DICOM SEG object.
    segment_label : str or None
        User-requested label (case-insensitive match).

    Returns
    -------
    int
        The matching segment number.

    Raises
    ------
    ValueError
        If the label is not found or is ambiguous.
    """
    seg_nums = list(seg.segment_numbers)
    labels_map: dict[str, int] = {}
    for num in seg_nums:
        desc = seg.get_segment_description(num)
        labels_map[str(desc.segment_label)] = int(num)

    if segment_label is not None:
        # Case-insensitive lookup.
        lower_map = {k.lower(): (k, v) for k, v in labels_map.items()}
        key = segment_label.lower()
        if key in lower_map:
            return lower_map[key][1]
        available = ", ".join(f"'{lab}' (#{num})" for lab, num in labels_map.items())
        raise ValueError(
            f"Segment label '{segment_label}' not found in DICOM SEG. "
            f"Available segments: {available}"
        )

    # No label specified.
    if len(seg_nums) == 1:
        return int(seg_nums[0])

    available = ", ".join(f"'{lab}' (#{num})" for lab, num in labels_map.items())
    raise ValueError(
        f"DICOM SEG contains {len(seg_nums)} segments: {available}. "
        "Specify 'segment_label' in the config to select one."
    )


def read_dicom_seg(
    path: Path,
    segment_label: str | None = None,
) -> Any:
    """Read a DICOM SEG file and return the selected segment as a SimpleITK image.

    Parameters
    ----------
    path : Path
        Path to a DICOM SEG (``.dcm``) file.
    segment_label : str or None
        Label of the segment to extract.  Required when the file contains
        more than one segment.  Matched case-insensitively against each
        segment's ``SegmentLabel`` attribute.

    Returns
    -------
    SimpleITK.Image
        Binary ``uint8`` mask in LPS orientation (native DICOM / SimpleITK
        coordinate system).

    Raises
    ------
    ModuleNotFoundError
        If ``highdicom`` is not installed.
    ValueError
        If the file is not a valid DICOM SEG, the requested label is not
        found, or a multi-segment file is loaded without specifying a label.
    """
    import SimpleITK as sitk

    hd = _import_highdicom()

    seg = hd.seg.segread(str(path))
    seg_num = _select_segment_number(seg, segment_label)

    # Handle FRACTIONAL segmentation type.
    is_fractional = seg.segmentation_type == hd.seg.SegmentationTypeValues.FRACTIONAL
    if is_fractional:
        log.warning(
            "DICOM SEG '%s' uses FRACTIONAL segmentation type; "
            "thresholding at 0.5 to produce a binary mask.",
            path.name,
        )

    volume = seg.get_volume(
        segment_numbers=[seg_num],
        combine_segments=True,
        rescale_fractional=is_fractional,
    )

    arr = volume.array
    # get_volume returns shape (slices, rows, cols) or with extra dim.
    if arr.ndim > 3:
        arr = arr.squeeze()
    if arr.ndim != 3:
        raise ValueError(
            f"Unexpected array shape {arr.shape} from DICOM SEG '{path.name}'."
        )

    # Binarize: combine_segments=True may assign segment numbers as values;
    # fractional types need thresholding.
    import numpy as np

    if is_fractional:
        arr = (arr >= 0.5).astype(np.uint8)
    else:
        arr = (arr > 0).astype(np.uint8)

    # Build SimpleITK image with spatial metadata from the Volume's affine.
    #
    # highdicom Volume axes are (slices, rows, cols).
    # SimpleITK (via GetImageFromArray) maps numpy axes as:
    #   numpy axis 0 → k (z), axis 1 → j (y), axis 2 → i (x)
    # So SimpleITK axes (i, j, k) correspond to (cols, rows, slices).
    #
    # The affine maps (slice, row, col) indices → (x, y, z) patient coords.
    # We reorder columns to (col, row, slice) for SimpleITK's (i, j, k).
    affine = np.asarray(volume.affine, dtype=np.float64)
    origin = affine[:3, 3]
    # Reorder spatial columns: (slice=0, row=1, col=2) → (col, row, slice)
    spacing_vecs = affine[:3, [2, 1, 0]]
    spacing = np.linalg.norm(spacing_vecs, axis=0)
    if np.any(spacing < 1e-10):
        raise ValueError(
            f"Degenerate affine in DICOM SEG '{path.name}': "
            f"zero-length spacing vector (spacing={spacing.tolist()})."
        )
    direction = spacing_vecs / spacing

    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(origin.tolist())
    img.SetSpacing(spacing.tolist())
    img.SetDirection(direction.T.flatten().tolist())
    img = sitk.Cast(img, sitk.sitkUInt8)

    desc = seg.get_segment_description(seg_num)
    log.info(
        "Loaded DICOM SEG segment '%s' (#%d) from %s  [%s].",
        desc.segment_label,
        seg_num,
        path.name,
        "FRACTIONAL→binary" if is_fractional else "BINARY",
    )
    return img
