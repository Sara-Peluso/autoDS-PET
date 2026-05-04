"""autods_pet -- Deauville Score computation from PET/CT images."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

__version__ = "0.1.0"


def setup_logging(
    level: int = logging.INFO, rich_console: Console | None = None
) -> None:
    """Configure root logger for autods_pet scripts.

    Parameters
    ----------
    level : int
        Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).
    rich_console : rich.console.Console or None
        If provided, use Rich's ``RichHandler`` for coloured, formatted
        log output.  When *None*, fall back to plain ``basicConfig``.
    """
    if rich_console is not None:
        from rich.logging import RichHandler

        logging.basicConfig(
            format="%(message)s",
            datefmt="[%X]",
            level=level,
            handlers=[RichHandler(console=rich_console, rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            format="%(levelname)s %(name)s: %(message)s",
            level=level,
        )


__all__ = [
    # Classes
    "DeauvillePipeline",
    "DeauvilleResult",
    "ROIResult",
    "PatientCase",
    # ROI classes
    "AortaMBP",
    "LiverROI",
    "LongBonesROI",
    "LumbarVB",
    "TargetROI",
    "BrainROI",
    # Config
    "ConfigValidator",
    # Core functions
    "assign_ds",
    "load_config",
    # Imaging functions
    "compute_suvbw",
    "rigid_register_pet_to_ct",
    "run_totalsegmentator",
    "find_series_by_modality",
    "dicom_series_to_nifti",
    "extract_pet_tags",
    "extract_ct_tags",
    # Stats + Masks utilities
    "mean_in_mask",
    "voxelwise_median",
    "max_in_mask",
    "min_in_mask",
    "percentile_in_mask",
    "compute_stats",
    "label_mask",
    "label_union",
    "keep_largest_component",
    "fill_holes",
    "subtract_mask",
    # DICOM SEG
    "read_dicom_seg",
    "list_dicom_seg_segments",
    "read_referenced_series_uids",
    # Mask discovery
    "DiscoveredMask",
    "discover_all_masks",
    "discover_dicom_seg_masks",
    "discover_file_masks",
]

# Maps public name -> (module_path, attribute_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Classes
    "DeauvillePipeline": ("autods_pet.pipeline", "DeauvillePipeline"),
    "DeauvilleResult": ("autods_pet.results", "DeauvilleResult"),
    "ROIResult": ("autods_pet.results", "ROIResult"),
    "PatientCase": ("autods_pet.patient", "PatientCase"),
    # ROI classes
    "AortaMBP": ("autods_pet.roi.aorta_mbp", "AortaMBP"),
    "LiverROI": ("autods_pet.roi.liver", "LiverROI"),
    "LongBonesROI": ("autods_pet.roi.long_bones", "LongBonesROI"),
    "LumbarVB": ("autods_pet.roi.lumbar_vb", "LumbarVB"),
    "TargetROI": ("autods_pet.roi.target_roi", "TargetROI"),
    "BrainROI": ("autods_pet.roi.brain", "BrainROI"),
    # Config
    "ConfigValidator": ("autods_pet.config", "ConfigValidator"),
    # Core functions
    "assign_ds": ("autods_pet.deauville", "assign_ds"),
    "load_config": ("autods_pet.config", "load_config"),
    # Imaging
    "compute_suvbw": ("autods_pet.imaging.normalization", "compute_suvbw"),
    "rigid_register_pet_to_ct": (
        "autods_pet.imaging.registration",
        "rigid_register_pet_to_ct",
    ),
    "run_totalsegmentator": ("autods_pet.imaging.segmentation", "run_totalsegmentator"),
    "find_series_by_modality": ("autods_pet.imaging.dicom", "find_series_by_modality"),
    "dicom_series_to_nifti": ("autods_pet.imaging.dicom", "dicom_series_to_nifti"),
    "extract_pet_tags": ("autods_pet.imaging.dicom", "extract_pet_tags"),
    "extract_ct_tags": ("autods_pet.imaging.dicom", "extract_ct_tags"),
    # Stats + Masks
    "mean_in_mask": ("autods_pet.ops.stats", "mean_in_mask"),
    "voxelwise_median": ("autods_pet.ops.stats", "voxelwise_median"),
    "max_in_mask": ("autods_pet.ops.stats", "max_in_mask"),
    "min_in_mask": ("autods_pet.ops.stats", "min_in_mask"),
    "percentile_in_mask": ("autods_pet.ops.stats", "percentile_in_mask"),
    "compute_stats": ("autods_pet.ops.stats", "compute_stats"),
    "label_mask": ("autods_pet.ops.masks", "label_mask"),
    "label_union": ("autods_pet.ops.masks", "label_union"),
    "keep_largest_component": ("autods_pet.ops.masks", "keep_largest_component"),
    "fill_holes": ("autods_pet.ops.masks", "fill_holes"),
    "subtract_mask": ("autods_pet.ops.masks", "subtract_mask"),
    # DICOM SEG
    "read_dicom_seg": ("autods_pet.ops.dicom_seg", "read_dicom_seg"),
    "list_dicom_seg_segments": ("autods_pet.ops.dicom_seg", "list_segments"),
    "read_referenced_series_uids": (
        "autods_pet.ops.dicom_seg",
        "read_referenced_series_uids",
    ),
    # Mask discovery
    "DiscoveredMask": ("autods_pet.ops.mask_discovery", "DiscoveredMask"),
    "discover_all_masks": ("autods_pet.ops.mask_discovery", "discover_all_masks"),
    "discover_dicom_seg_masks": (
        "autods_pet.ops.mask_discovery",
        "discover_dicom_seg_masks",
    ),
    "discover_file_masks": ("autods_pet.ops.mask_discovery", "discover_file_masks"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module 'autods_pet' has no attribute {name!r}")
