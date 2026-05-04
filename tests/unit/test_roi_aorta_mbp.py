"""Tests for autods_pet.roi.aorta_mbp - aorta MBP mask refinement and PET stats."""

import numpy as np
import pytest
import SimpleITK as sitk
from hypothesis import given, settings
from hypothesis import strategies as st

from autods_pet.ops.stats import count_voxels
from autods_pet.roi.aorta_mbp import (
    AortaMBP,
    _slicegate_by_slab,
)


def test_slicegate_keeps_only_slab_slices():
    arr_aorta = np.zeros((10, 10, 10), dtype=np.uint8)
    arr_aorta[:, 4:7, 4:7] = 1  # aorta spans all z

    arr_slab = np.zeros((10, 10, 10), dtype=np.uint8)
    arr_slab[3:7, 4:7, 4:7] = 1  # slab at z=3..6

    aorta = sitk.GetImageFromArray(arr_aorta)
    slab = sitk.GetImageFromArray(arr_slab)
    aorta.SetSpacing((1.0, 1.0, 1.0))
    slab.SetSpacing((1.0, 1.0, 1.0))

    result = _slicegate_by_slab(aorta, slab, axis_xyz=2)
    result_arr = sitk.GetArrayFromImage(result)
    # Only slices 3-6 should have non-zero voxels
    assert np.any(result_arr[3:7] > 0)
    assert np.all(result_arr[:3] == 0)
    assert np.all(result_arr[7:] == 0)


def test_slicegate_empty_slab_empties_aorta():
    arr_aorta = np.ones((10, 10, 10), dtype=np.uint8)
    arr_slab = np.zeros((10, 10, 10), dtype=np.uint8)
    aorta = sitk.GetImageFromArray(arr_aorta)
    slab = sitk.GetImageFromArray(arr_slab)
    aorta.SetSpacing((1.0, 1.0, 1.0))
    slab.SetSpacing((1.0, 1.0, 1.0))
    result = _slicegate_by_slab(aorta, slab, axis_xyz=2)
    assert count_voxels(result) == 0


def test_slicegate_full_slab_preserves_all():
    arr_aorta = np.zeros((10, 10, 10), dtype=np.uint8)
    arr_aorta[:, 4:7, 4:7] = 1
    arr_slab = np.ones((10, 10, 10), dtype=np.uint8)
    aorta = sitk.GetImageFromArray(arr_aorta)
    slab = sitk.GetImageFromArray(arr_slab)
    aorta.SetSpacing((1.0, 1.0, 1.0))
    slab.SetSpacing((1.0, 1.0, 1.0))
    result = _slicegate_by_slab(aorta, slab, axis_xyz=2)
    assert count_voxels(result) == count_voxels(aorta)


def test_refine_aorta_invalid_heart_mode_raises(seg_phantom):
    roi = AortaMBP(heart_exclusion_mode="foo")
    with pytest.raises(ValueError, match="heart_exclusion_mode"):
        roi.refine(seg_phantom)


def test_refine_aorta_returns_result_with_attributes(seg_phantom):
    roi = AortaMBP(aorta_erosion_mm=0.0)
    result = roi.refine(seg_phantom)
    assert result.refined_mask is not None
    assert result.shrinkage is not None


def test_refine_aorta_default_labels_T4_T8(seg_phantom):
    roi = AortaMBP(aorta_erosion_mm=0.0)
    result = roi.refine(seg_phantom)
    assert count_voxels(result.refined_mask) > 0


@pytest.mark.parametrize("mode", ["dilate_intersection", "distance"])
def test_refine_aorta_heart_exclusion_modes(seg_phantom, mode):
    roi = AortaMBP(heart_exclusion_mode=mode, aorta_erosion_mm=0.0)
    result = roi.refine(seg_phantom)
    assert isinstance(result.refined_mask, sitk.Image)


def test_refine_aorta_zero_erosion(seg_phantom):
    roi = AortaMBP(aorta_erosion_mm=0.0)
    result = roi.refine(seg_phantom)
    assert count_voxels(result.refined_mask) > 0


def test_refine_aorta_large_erosion_empties(seg_phantom):
    roi = AortaMBP(aorta_erosion_mm=100.0)
    result = roi.refine(seg_phantom)
    assert count_voxels(result.refined_mask) == 0


def test_extract_aorta_geometry_mismatch_raises():
    seg = sitk.Image([10, 10, 10], sitk.sitkUInt8)
    pet = sitk.Image([10, 10, 10], sitk.sitkFloat64)
    pet.SetSpacing((2.0, 2.0, 2.0))
    roi = AortaMBP()
    with pytest.raises(ValueError, match="Geometry mismatch"):
        roi.extract(seg, pet)


def test_extract_aorta_default_stats_median(seg_phantom, pet_phantom):
    roi = AortaMBP(aorta_erosion_mm=0.0)
    result = roi.extract(seg_phantom, pet_phantom)
    assert "median" in result.stats


def test_extract_aorta_custom_stats(seg_phantom, pet_phantom):
    roi = AortaMBP(aorta_erosion_mm=0.0, stats=["mean", "p95"])
    result = roi.extract(seg_phantom, pet_phantom)
    assert "mean" in result.stats
    assert "p95" in result.stats


def test_slicegate_axis_x():
    """Slice gating along axis_xyz=0 (x-axis) keeps only slab slices."""
    from autods_pet.roi.aorta_mbp import _slicegate_by_slab

    # Create a 10x10x10 aorta (all ones)
    arr = np.ones((10, 10, 10), dtype=np.uint8)
    aorta = sitk.GetImageFromArray(arr)
    aorta.SetSpacing((1.0, 1.0, 1.0))

    # Slab present only in x=3..6 (numpy axis 2 for sitk axis 0)
    slab_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    slab_arr[:, :, 3:7] = 1
    slab = sitk.GetImageFromArray(slab_arr)
    slab.CopyInformation(aorta)

    result = _slicegate_by_slab(aorta, slab, axis_xyz=0)
    out = sitk.GetArrayFromImage(result)
    # Only x-slices 3..6 should have non-zero voxels
    assert out[:, :, :3].sum() == 0
    assert out[:, :, 7:].sum() == 0
    assert out[:, :, 3:7].sum() > 0


def test_slicegate_axis_y():
    """Slice gating along axis_xyz=1 (y-axis) keeps only slab slices."""
    from autods_pet.roi.aorta_mbp import _slicegate_by_slab

    arr = np.ones((10, 10, 10), dtype=np.uint8)
    aorta = sitk.GetImageFromArray(arr)
    aorta.SetSpacing((1.0, 1.0, 1.0))

    # Slab present only in y=2..5 (numpy axis 1 for sitk axis 1)
    slab_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    slab_arr[:, 2:6, :] = 1
    slab = sitk.GetImageFromArray(slab_arr)
    slab.CopyInformation(aorta)

    result = _slicegate_by_slab(aorta, slab, axis_xyz=1)
    out = sitk.GetArrayFromImage(result)
    assert out[:, :2, :].sum() == 0
    assert out[:, 6:, :].sum() == 0
    assert out[:, 2:6, :].sum() > 0


def test_exclude_heart_dilate_removes_overlap():
    """Aorta voxels overlapping with dilated heart are removed."""
    from autods_pet.ops.stats import count_voxels
    from autods_pet.roi.aorta_mbp import _exclude_heart_dilate

    # 20x20x20 images, 1mm spacing
    aorta_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    aorta_arr[5:15, 5:15, 5:15] = 1  # 10x10x10 block

    heart_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    heart_arr[8:12, 8:12, 8:12] = 1  # 4x4x4 block inside aorta

    aorta = sitk.GetImageFromArray(aorta_arr)
    aorta.SetSpacing((1.0, 1.0, 1.0))
    heart = sitk.GetImageFromArray(heart_arr)
    heart.CopyInformation(aorta)

    result = _exclude_heart_dilate(aorta, heart, {"heart_dilation_mm": 2.0})
    assert count_voxels(result) < count_voxels(aorta)


def test_exclude_heart_dilate_no_overlap_preserves():
    """When heart is far from aorta, all aorta voxels preserved."""
    from autods_pet.ops.stats import count_voxels
    from autods_pet.roi.aorta_mbp import _exclude_heart_dilate

    aorta_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    aorta_arr[0:3, 0:3, 0:3] = 1  # corner

    heart_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    heart_arr[17:20, 17:20, 17:20] = 1  # far corner

    aorta = sitk.GetImageFromArray(aorta_arr)
    aorta.SetSpacing((1.0, 1.0, 1.0))
    heart = sitk.GetImageFromArray(heart_arr)
    heart.CopyInformation(aorta)

    result = _exclude_heart_dilate(aorta, heart, {"heart_dilation_mm": 1.0})
    assert count_voxels(result) == count_voxels(aorta)


def test_exclude_heart_dilate_zero_dilation():
    """With zero dilation, only direct overlap is removed."""
    from autods_pet.ops.stats import count_voxels
    from autods_pet.roi.aorta_mbp import _exclude_heart_dilate

    aorta_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    aorta_arr[5:15, 5:15, 5:15] = 1

    heart_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    heart_arr[10:15, 10:15, 10:15] = 1  # overlaps half

    aorta = sitk.GetImageFromArray(aorta_arr)
    aorta.SetSpacing((1.0, 1.0, 1.0))
    heart = sitk.GetImageFromArray(heart_arr)
    heart.CopyInformation(aorta)

    orig_count = count_voxels(aorta)
    result = _exclude_heart_dilate(aorta, heart, {"heart_dilation_mm": 0.0})
    # With zero dilation, only direct heart voxels that overlap aorta are removed
    result_count = count_voxels(result)
    assert result_count < orig_count
    assert result_count > 0


def test_exclude_heart_distance_removes_near():
    """Aorta voxels within distance threshold of heart are removed."""
    from autods_pet.ops.stats import count_voxels
    from autods_pet.roi.aorta_mbp import _exclude_heart_distance

    aorta_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    aorta_arr[5:15, 5:15, 5:15] = 1

    heart_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    heart_arr[8:12, 8:12, 8:12] = 1  # inside aorta

    aorta = sitk.GetImageFromArray(aorta_arr)
    aorta.SetSpacing((1.0, 1.0, 1.0))
    heart = sitk.GetImageFromArray(heart_arr)
    heart.CopyInformation(aorta)

    result = _exclude_heart_distance(aorta, heart, {"heart_distance_mm": 3.0})
    assert count_voxels(result) < count_voxels(aorta)


def test_exclude_heart_distance_far_preserved():
    """Aorta voxels far from heart are preserved."""
    from autods_pet.ops.stats import count_voxels
    from autods_pet.roi.aorta_mbp import _exclude_heart_distance

    aorta_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    aorta_arr[0:3, 0:3, 0:3] = 1

    heart_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    heart_arr[17:20, 17:20, 17:20] = 1

    aorta = sitk.GetImageFromArray(aorta_arr)
    aorta.SetSpacing((1.0, 1.0, 1.0))
    heart = sitk.GetImageFromArray(heart_arr)
    heart.CopyInformation(aorta)

    result = _exclude_heart_distance(aorta, heart, {"heart_distance_mm": 3.0})
    assert count_voxels(result) == count_voxels(aorta)


def test_exclude_heart_distance_large_threshold_empties():
    """Large distance threshold removes all aorta voxels near heart."""
    from autods_pet.ops.stats import count_voxels
    from autods_pet.roi.aorta_mbp import _exclude_heart_distance

    # Small image where all aorta is near the heart
    aorta_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    aorta_arr[3:7, 3:7, 3:7] = 1

    heart_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    heart_arr[4:6, 4:6, 4:6] = 1  # inside aorta

    aorta = sitk.GetImageFromArray(aorta_arr)
    aorta.SetSpacing((1.0, 1.0, 1.0))
    heart = sitk.GetImageFromArray(heart_arr)
    heart.CopyInformation(aorta)

    result = _exclude_heart_distance(aorta, heart, {"heart_distance_mm": 50.0})
    assert count_voxels(result) == 0


@given(radius=st.floats(min_value=0.0, max_value=10.0))
@settings(max_examples=20)
def test_exclude_heart_dilate_always_subset(radius):
    """Output of _exclude_heart_dilate is always a subset of input aorta."""
    from autods_pet.ops.stats import count_voxels
    from autods_pet.roi.aorta_mbp import _exclude_heart_dilate

    aorta_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    aorta_arr[2:8, 2:8, 2:8] = 1
    heart_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    heart_arr[4:6, 4:6, 4:6] = 1

    aorta = sitk.GetImageFromArray(aorta_arr)
    aorta.SetSpacing((1.0, 1.0, 1.0))
    heart = sitk.GetImageFromArray(heart_arr)
    heart.CopyInformation(aorta)

    result = _exclude_heart_dilate(aorta, heart, {"heart_dilation_mm": radius})
    assert count_voxels(result) <= count_voxels(aorta)
