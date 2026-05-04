"""Tests for autods_pet.roi.target_roi - target ROI statistics extraction."""

import numpy as np
import pytest
import SimpleITK as sitk

from autods_pet.roi.target_roi import TargetROI


def test_extract_target_roi_returns_result_with_attributes(seg_phantom, pet_phantom):
    mask = sitk.Cast(sitk.Equal(seg_phantom, 5), sitk.sitkUInt8)
    roi = TargetROI()
    result = roi.extract(mask, pet_phantom)
    assert result.stats is not None


def test_extract_target_roi_geometry_mismatch_raises(make_image):
    mask = sitk.Image([10, 10, 10], sitk.sitkUInt8)
    pet = sitk.Image([10, 10, 10], sitk.sitkFloat64)
    pet.SetSpacing((2.0, 2.0, 2.0))
    roi = TargetROI()
    with pytest.raises(ValueError, match="Geometry mismatch"):
        roi.extract(mask, pet)


def test_extract_target_roi_default_stats_max(seg_phantom, pet_phantom):
    mask = sitk.Cast(sitk.Equal(seg_phantom, 5), sitk.sitkUInt8)
    roi = TargetROI()
    result = roi.extract(mask, pet_phantom)
    assert "max" in result.stats


def test_extract_target_roi_custom_stats(seg_phantom, pet_phantom):
    mask = sitk.Cast(sitk.Equal(seg_phantom, 5), sitk.sitkUInt8)
    roi = TargetROI(stats=["mean", "p90"])
    result = roi.extract(mask, pet_phantom)
    assert "mean" in result.stats
    assert "p90" in result.stats


def test_extract_target_roi_known_max_value(make_image):
    pet_arr = np.array([[[1.0, 2.0, 7.0]]], dtype=np.float64)
    mask_arr = np.array([[[1, 1, 1]]], dtype=np.uint8)
    pet = make_image(pet_arr)
    mask = make_image(mask_arr)
    roi = TargetROI()
    result = roi.extract(mask, pet)
    assert result.stats["max"] == pytest.approx(7.0)


def test_extract_target_roi_empty_mask_returns_none_stats(make_image):
    pet = make_image(np.ones((3, 3, 3), dtype=np.float64))
    mask = make_image(np.zeros((3, 3, 3), dtype=np.uint8))
    roi = TargetROI()
    result = roi.extract(mask, pet)
    assert result.stats["max"] is None


def test_extract_target_roi_single_voxel(make_image):
    pet_arr = np.zeros((3, 3, 3), dtype=np.float64)
    pet_arr[1, 1, 1] = 42.0
    mask_arr = np.zeros((3, 3, 3), dtype=np.uint8)
    mask_arr[1, 1, 1] = 1
    roi = TargetROI()
    result = roi.extract(make_image(mask_arr), make_image(pet_arr))
    assert result.stats["max"] == pytest.approx(42.0)


def test_extract_target_roi_mask_binarized(make_image):
    pet_arr = np.ones((3, 3, 3), dtype=np.float64) * 5.0
    mask_arr = np.zeros((3, 3, 3), dtype=np.uint8)
    mask_arr[0, 0, 0] = 2  # non-zero but not 1
    mask_arr[1, 1, 1] = 5
    roi = TargetROI()
    result = roi.extract(make_image(mask_arr), make_image(pet_arr))
    assert result.stats is not None
