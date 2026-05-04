"""Tests for autods_pet.roi.long_bones - long bone diaphysis refinement and stats."""

import numpy as np
import pytest
import SimpleITK as sitk

from autods_pet.ops.stats import count_voxels
from autods_pet.roi.long_bones import (
    DEFAULT_BONES,
    LongBonesROI,
    _crop_to_diaphysis_z,
)


def _bone_column(height=30, width=3, label=75, spacing=(1.0, 1.0, 1.0)):
    """Create a segmentation with a single bone column of given height."""
    arr = np.zeros((height, 10, 10), dtype=np.uint8)
    c = 5  # center
    hw = width // 2
    arr[:, c - hw : c + hw + 1, c - hw : c + hw + 1] = label
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def test_crop_diaphysis_keep_100_preserves_all():
    seg = _bone_column(height=20, label=75)
    mask = sitk.Cast(sitk.Equal(seg, 75), sitk.sitkUInt8)
    cropped, info = _crop_to_diaphysis_z(mask, 100)
    assert count_voxels(cropped) == count_voxels(mask)


def test_crop_diaphysis_keep_50_halves_extent():
    seg = _bone_column(height=20, label=75)
    mask = sitk.Cast(sitk.Equal(seg, 75), sitk.sitkUInt8)
    cropped, info = _crop_to_diaphysis_z(mask, 50)
    assert info["keep"] == 10  # 50% of 20
    assert count_voxels(cropped) < count_voxels(mask)


def test_crop_diaphysis_keep_1_minimum():
    seg = _bone_column(height=20, label=75)
    mask = sitk.Cast(sitk.Equal(seg, 75), sitk.sitkUInt8)
    cropped, info = _crop_to_diaphysis_z(mask, 1)
    assert info["keep"] >= 1
    assert count_voxels(cropped) > 0


@pytest.mark.parametrize("keep_pct", [0, -1, 101, 200])
def test_crop_diaphysis_invalid_keep_pct_raises(keep_pct):
    seg = _bone_column(height=10, label=75)
    mask = sitk.Cast(sitk.Equal(seg, 75), sitk.sitkUInt8)
    with pytest.raises(ValueError, match="keep_pct"):
        _crop_to_diaphysis_z(mask, keep_pct)


def test_crop_diaphysis_empty_mask_returns_empty():
    mask = sitk.Image([10, 10, 10], sitk.sitkUInt8)
    cropped, info = _crop_to_diaphysis_z(mask, 60)
    assert count_voxels(cropped) == 0
    assert info["start_z"] is None


def test_crop_diaphysis_single_slice():
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[5, 4:7, 4:7] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing((1.0, 1.0, 1.0))
    cropped, info = _crop_to_diaphysis_z(mask, 60)
    assert info["n_slices"] == 1
    assert count_voxels(cropped) > 0


def test_crop_diaphysis_z_window_centered():
    seg = _bone_column(height=20, label=75)
    mask = sitk.Cast(sitk.Equal(seg, 75), sitk.sitkUInt8)
    _, info = _crop_to_diaphysis_z(mask, 50)
    mid = (info["mid_start"] + info["mid_end"]) / 2
    center = (info["start_z"] + info["end_z"]) / 2
    assert abs(mid - center) <= 1  # approximately centered


def test_refine_long_bones_default_bones(seg_phantom):
    roi = LongBonesROI(diaphysis_keep_pct=60)
    result = roi.refine(seg_phantom)
    assert len(result.shrinkage["per_bone"]) == len(DEFAULT_BONES)


def test_refine_long_bones_returns_result_with_attributes(seg_phantom):
    roi = LongBonesROI(diaphysis_keep_pct=80)
    result = roi.refine(seg_phantom)
    assert result.refined_mask is not None
    assert result.shrinkage is not None


def test_refine_long_bones_per_bone_has_all_fields(seg_phantom):
    roi = LongBonesROI(diaphysis_keep_pct=80)
    result = roi.refine(seg_phantom)
    for _name, info in result.shrinkage["per_bone"].items():
        assert "raw_voxels" in info
        assert "refined_voxels" in info
        assert "raw_volume_mm3" in info
        assert "refined_volume_mm3" in info
        assert "z_window" in info


def test_refine_long_bones_custom_single_bone():
    seg = _bone_column(height=30, label=75)
    roi = LongBonesROI(bones=[("femur_L", 75, 0.0)], diaphysis_keep_pct=80)
    result = roi.refine(seg)
    assert "femur_L" in result.shrinkage["per_bone"]
    assert len(result.shrinkage["per_bone"]) == 1
    assert count_voxels(result.refined_mask) > 0


def test_refine_long_bones_missing_label_produces_empty_bone():
    seg = _bone_column(height=20, label=75)
    roi = LongBonesROI(bones=[("phantom_bone", 99, 0.0)], diaphysis_keep_pct=60)
    result = roi.refine(seg)
    assert result.shrinkage["per_bone"]["phantom_bone"]["raw_voxels"] == 0


def test_extract_long_bones_geometry_mismatch_raises():
    seg = sitk.Image([10, 10, 10], sitk.sitkUInt8)
    pet = sitk.Image([10, 10, 10], sitk.sitkFloat64)
    pet.SetSpacing((2.0, 2.0, 2.0))
    roi = LongBonesROI()
    with pytest.raises(ValueError, match="Geometry mismatch"):
        roi.extract(seg, pet)


def test_extract_long_bones_default_stats_p95(seg_phantom, pet_phantom):
    roi = LongBonesROI(diaphysis_keep_pct=80)
    result = roi.extract(seg_phantom, pet_phantom)
    assert "p95" in result.stats


def test_extract_long_bones_custom_stats(seg_phantom, pet_phantom):
    roi = LongBonesROI(diaphysis_keep_pct=80, stats=["mean", "median"])
    result = roi.extract(seg_phantom, pet_phantom)
    assert "mean" in result.stats
    assert "median" in result.stats
