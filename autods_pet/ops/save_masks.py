"""Save raw and refined segmentation masks to disk as NIfTI files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

RAW_LABEL_MAP: dict[int, str] = {
    52: "aorta",
    51: "heart",
    5: "liver",
    29: "L3",
    28: "L4",
    27: "L5",
    40: "T4",
    39: "T5",
    38: "T6",
    37: "T7",
    36: "T8",
    75: "femur_L",
    76: "femur_R",
    69: "humerus_L",
    70: "humerus_R",
    90: "brain",
}

_REFINED_NAME_MAP: dict[str, str] = {
    "Aorta MBP": "aorta_mbp",
    "Liver": "liver",
    "Lumbar VB": "lumbar_vb",
    "Long bones": "long_bones",
    "Brain": "grey_matter",
    "Focal lesion": "focal_lesion",
    "Paramedullary": "paramedullary",
    "Extramedullary": "extramedullary",
}


def save_raw_masks(whole_seg: Any, seg_dir: Path) -> list[Path]:
    """Extract individual binary masks from *whole_seg* and save as NIfTI.

    Parameters
    ----------
    whole_seg : sitk.Image
        The TotalSegmentator multilabel segmentation.
    seg_dir : Path
        Patient's segmentation directory.

    Returns
    -------
    list[Path]
        Paths to the saved mask files.
    """
    import SimpleITK as sitk

    from autods_pet.ops.masks import label_mask

    raw_dir = seg_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for label_id, name in RAW_LABEL_MAP.items():
        mask = label_mask(whole_seg, label_id)
        out_path = raw_dir / f"{name}.nii.gz"
        sitk.WriteImage(mask, str(out_path))
        log.debug("Saved raw mask: %s", out_path)
        saved.append(out_path)

    log.info("Saved %d raw masks to %s", len(saved), raw_dir)
    return saved


def save_refined_masks(results: dict[str, Any], seg_dir: Path) -> list[Path]:
    """Save refined ROI masks from extraction results.

    Parameters
    ----------
    results : dict[str, Any]
        The dict returned by :func:`~autods_pet.pipeline.extract_rois`,
        mapping ROI display names to :class:`~autods_pet.results.ROIResult`.
    seg_dir : Path
        Patient's segmentation directory.

    Returns
    -------
    list[Path]
        Paths to the saved mask files.
    """
    import SimpleITK as sitk

    from autods_pet.results import ROIResult

    refined_dir = seg_dir / "refined"
    refined_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for roi_name, roi_data in results.items():
        if roi_name.startswith("_"):
            continue
        if not isinstance(roi_data, ROIResult):
            continue
        if roi_data.refined_mask is None:
            continue

        filename = _REFINED_NAME_MAP.get(roi_name, roi_name.lower().replace(" ", "_"))
        out_path = refined_dir / f"{filename}.nii.gz"
        sitk.WriteImage(roi_data.refined_mask, str(out_path))
        log.debug("Saved refined mask: %s", out_path)
        saved.append(out_path)

    log.info("Saved %d refined masks to %s", len(saved), refined_dir)
    return saved
