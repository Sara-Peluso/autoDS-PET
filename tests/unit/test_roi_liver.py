"""Tests for autods_pet.roi.liver - liver mask refinement and PET statistics."""

import numpy as np
import pytest
import SimpleITK as sitk

from autods_pet.ops.stats import count_voxels
from autods_pet.roi.liver import LiverROI


def test_refine_liver_returns_result_with_attributes(seg_phantom):
    roi = LiverROI(erosion_mm=1.0)
    result = roi.refine(seg_phantom)
    assert result.refined_mask is not None
    assert result.shrinkage is not None


def test_refine_liver_default_label_is_5(seg_phantom):
    roi = LiverROI(erosion_mm=1.0)
    result = roi.refine(seg_phantom)
    assert count_voxels(result.refined_mask) > 0


def test_refine_liver_largest_component_kept(make_image):
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    # Large blob
    arr[2:18, 2:18, 2:18] = 5
    # Small disconnected blob
    arr[0, 0, 0] = 5
    seg = make_image(arr)
    roi = LiverROI(liver_label=5, erosion_mm=0.0)
    result = roi.refine(seg)
    # The single-voxel component should be removed
    refined_vox = count_voxels(result.refined_mask)
    raw_with_big = 16 * 16 * 16
    assert refined_vox == raw_with_big


def test_refine_liver_holes_filled_no_max(make_image):
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[2:18, 2:18, 2:18] = 5
    arr[8:12, 8:12, 8:12] = 0  # hole inside
    seg = make_image(arr)
    roi = LiverROI(liver_label=5, erosion_mm=0.0, max_hole_volume_mm3=None)
    result = roi.refine(seg)
    # Hole should be filled
    assert count_voxels(result.refined_mask) == 16 * 16 * 16


def test_refine_liver_large_holes_preserved(make_image):
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[2:18, 2:18, 2:18] = 5
    arr[8:12, 8:12, 8:12] = 0  # 64-voxel hole (64 mm³ at 1mm spacing)
    seg = make_image(arr)
    roi = LiverROI(liver_label=5, erosion_mm=0.0, max_hole_volume_mm3=10.0)
    result = roi.refine(seg)
    # Hole too large to fill - result should not contain the hole voxels
    assert count_voxels(result.refined_mask) < 16 * 16 * 16


def test_refine_liver_erosion_shrinks_mask(seg_phantom):
    roi = LiverROI(erosion_mm=2.0)
    result = roi.refine(seg_phantom)
    shrinkage = result.shrinkage
    assert shrinkage["refined_voxels"] < shrinkage["original_voxels"]


def test_refine_liver_zero_erosion(seg_phantom):
    roi = LiverROI(erosion_mm=0.0)
    result = roi.refine(seg_phantom)
    assert count_voxels(result.refined_mask) > 0
    # With zero erosion, refined should be close to raw (after largest component + fill)
    assert result.shrinkage["shrinkage_pct"] >= 0


def test_refine_liver_no_liver_label_returns_empty(make_image):
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:8, 2:8, 2:8] = 99  # not liver label
    seg = make_image(arr)
    roi = LiverROI(liver_label=5, erosion_mm=0.0)
    result = roi.refine(seg)
    assert count_voxels(result.refined_mask) == 0


def test_refine_liver_custom_label(make_image):
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[2:18, 2:18, 2:18] = 99
    seg = make_image(arr)
    roi = LiverROI(liver_label=99, erosion_mm=0.0)
    result = roi.refine(seg)
    assert count_voxels(result.refined_mask) > 0


def test_extract_liver_geometry_mismatch_raises():
    seg = sitk.Image([10, 10, 10], sitk.sitkUInt8)
    pet = sitk.Image([10, 10, 10], sitk.sitkFloat64)
    pet.SetSpacing((2.0, 2.0, 2.0))
    roi = LiverROI()
    with pytest.raises(ValueError, match="Geometry mismatch"):
        roi.extract(seg, pet)


def test_extract_liver_default_stats_median(seg_phantom, pet_phantom):
    roi = LiverROI(erosion_mm=1.0)
    result = roi.extract(seg_phantom, pet_phantom)
    assert "median" in result.stats


def test_extract_liver_custom_stats(seg_phantom, pet_phantom):
    roi = LiverROI(erosion_mm=1.0, stats=["mean", "max"])
    result = roi.extract(seg_phantom, pet_phantom)
    assert "mean" in result.stats
    assert "max" in result.stats


def test_extract_liver_known_value(make_image):
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[2:18, 2:18, 2:18] = 5
    seg = make_image(arr)
    pet_arr = np.full((20, 20, 20), 3.0, dtype=np.float64)
    pet = make_image(pet_arr)
    roi = LiverROI(erosion_mm=0.0, stats=["median"])
    result = roi.extract(seg, pet)
    assert result.stats["median"] == pytest.approx(3.0)
