"""Operations subpackage: mask manipulation and statistics."""

from autods_pet.ops.masks import (
    fill_holes,
    keep_largest_component,
    label_mask,
    label_union,
)
from autods_pet.ops.stats import (
    compute_stats,
    count_voxels,
    mask_volume_mm3,
    max_in_mask,
    mean_in_mask,
    min_in_mask,
    percentile_in_mask,
    shrinkage_report,
    voxelwise_median,
)

__all__ = [
    "compute_stats",
    "count_voxels",
    "fill_holes",
    "keep_largest_component",
    "label_mask",
    "label_union",
    "mask_volume_mm3",
    "max_in_mask",
    "mean_in_mask",
    "min_in_mask",
    "percentile_in_mask",
    "shrinkage_report",
    "voxelwise_median",
]
