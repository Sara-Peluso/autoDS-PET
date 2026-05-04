"""Tests for autods_pet.roi.brain - BrainROI refinement and extraction."""

import numpy as np
import pytest
import SimpleITK as sitk

from autods_pet import labels
from autods_pet.ops.stats import count_voxels
from autods_pet.roi.brain import BrainROI


def _brain_seg(radius=8, size=20):
    """Build a segmentation with a solid sphere of brain label."""
    arr = np.zeros((size, size, size), dtype=np.uint8)
    center = size // 2
    zz, yy, xx = np.ogrid[:size, :size, :size]
    sphere = ((zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2) <= radius**2
    arr[sphere] = labels.BRAIN
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def _uniform_pet(value, size=20):
    """Build a uniform-value PET image matching _brain_seg geometry."""
    arr = np.full((size, size, size), value, dtype=np.float64)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


# -- Constructor defaults ---------------------------------------------


def test_brain_defaults():
    roi = BrainROI()
    assert roi.brain_label == labels.BRAIN
    assert roi.grey_matter_only is True
    assert roi.cortical_thickness_mm == 5.0
    assert roi.stats == ["median"]


# -- refine() ---------------------------------------------------------


def test_refine_grey_matter_cortical_shell():
    seg = _brain_seg(radius=8)
    roi = BrainROI(cortical_thickness_mm=2.0)
    result = roi.refine(seg)

    assert result.refined_mask is not None
    refined_count = count_voxels(result.refined_mask)
    assert refined_count > 0
    # Shell must be smaller than the full brain sphere
    from autods_pet.ops.masks import label_mask

    raw_count = count_voxels(label_mask(seg, labels.BRAIN))
    assert refined_count < raw_count
    # Shrinkage report present
    assert result.shrinkage is not None
    assert "shrinkage_pct" in result.shrinkage


def test_refine_full_mask_grey_matter_disabled():
    seg = _brain_seg(radius=8)
    roi = BrainROI(grey_matter_only=False)
    result = roi.refine(seg)

    from autods_pet.ops.masks import keep_largest_component, label_mask

    expected = keep_largest_component(label_mask(seg, labels.BRAIN))
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(result.refined_mask),
        sitk.GetArrayFromImage(expected),
    )


def test_refine_zero_cortical_thickness():
    seg = _brain_seg(radius=8)
    roi = BrainROI(grey_matter_only=True, cortical_thickness_mm=0.0)
    result = roi.refine(seg)

    from autods_pet.ops.masks import keep_largest_component, label_mask

    expected = keep_largest_component(label_mask(seg, labels.BRAIN))
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(result.refined_mask),
        sitk.GetArrayFromImage(expected),
    )


def test_refine_no_brain_label_returns_none():
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:8, 2:8, 2:8] = 5  # liver label, not brain
    seg = sitk.GetImageFromArray(arr)
    seg.SetSpacing((1.0, 1.0, 1.0))

    roi = BrainROI()
    result = roi.refine(seg)
    assert result is None


# -- extract() --------------------------------------------------------


def test_extract_computes_stats():
    seg = _brain_seg(radius=8)
    pet = _uniform_pet(6.0)
    roi = BrainROI(grey_matter_only=False, stats=["median"])
    result = roi.extract(seg, pet)

    assert result.stats is not None
    assert "median" in result.stats
    assert result.stats["median"] == pytest.approx(6.0)


def test_extract_geometry_mismatch_raises():
    seg = _brain_seg(radius=8, size=20)
    pet = _uniform_pet(6.0, size=20)
    pet.SetSpacing((2.0, 2.0, 2.0))  # different spacing

    roi = BrainROI()
    with pytest.raises(ValueError, match="Geometry mismatch"):
        roi.extract(seg, pet)


def test_extract_custom_stats():
    seg = _brain_seg(radius=8)
    pet = _uniform_pet(6.0)
    roi = BrainROI(grey_matter_only=False, stats=["mean", "max"])
    result = roi.extract(seg, pet)

    assert "mean" in result.stats
    assert "max" in result.stats
    assert result.stats["mean"] == pytest.approx(6.0)
    assert result.stats["max"] == pytest.approx(6.0)
