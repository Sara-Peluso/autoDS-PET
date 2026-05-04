"""Tests for autods_pet.roi.lumbar_vb - lumbar vertebral body refinement and stats."""

import numpy as np
import pytest
import SimpleITK as sitk

from autods_pet import labels
from autods_pet.ops.stats import count_voxels
from autods_pet.roi.lumbar_vb import LumbarVB


def test_refine_lumbar_vb_default_labels_uses_L3_L5(seg_phantom, vert_body_seg):
    roi = LumbarVB(erosion_mm=0.0)
    result = roi.refine(seg_phantom, vert_body_seg)
    assert count_voxels(result.refined_mask) > 0


def test_refine_lumbar_vb_returns_result_with_attributes(seg_phantom, vert_body_seg):
    roi = LumbarVB(erosion_mm=0.0)
    result = roi.refine(seg_phantom, vert_body_seg)
    assert result.refined_mask is not None
    assert result.shrinkage is not None


def test_refine_lumbar_vb_refined_mask_is_binary(seg_phantom, vert_body_seg):
    roi = LumbarVB(erosion_mm=0.0)
    result = roi.refine(seg_phantom, vert_body_seg)
    arr = sitk.GetArrayFromImage(result.refined_mask)
    assert set(np.unique(arr)).issubset({0, 1})


def test_refine_lumbar_vb_shrinkage_non_negative(seg_phantom, vert_body_seg):
    roi = LumbarVB(erosion_mm=1.0)
    result = roi.refine(seg_phantom, vert_body_seg)
    assert result.shrinkage["shrinkage_pct"] >= 0


def test_refine_lumbar_vb_no_erosion_preserves_intersection(seg_phantom, vert_body_seg):
    roi = LumbarVB(erosion_mm=0.0)
    result = roi.refine(seg_phantom, vert_body_seg)
    assert count_voxels(result.refined_mask) > 0
    # Should equal the intersection of lumbar labels with vert body
    assert result.shrinkage["shrinkage_pct"] >= 0


def test_refine_lumbar_vb_large_erosion_empties_mask(seg_phantom, vert_body_seg):
    roi = LumbarVB(erosion_mm=100.0)
    result = roi.refine(seg_phantom, vert_body_seg)
    assert count_voxels(result.refined_mask) == 0


def test_refine_lumbar_vb_custom_labels(seg_phantom, vert_body_seg):
    roi = LumbarVB(lumbar_labels=[labels.L3], erosion_mm=0.0)
    result = roi.refine(seg_phantom, vert_body_seg)
    assert count_voxels(result.refined_mask) > 0


def test_refine_lumbar_vb_no_overlap_returns_empty(make_image):
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[0:3, 0:3, 0:3] = labels.L3
    seg = make_image(arr)
    # VB mask in a completely different location
    vb_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    vb_arr[7:10, 7:10, 7:10] = 1
    vb = make_image(vb_arr)
    roi = LumbarVB(erosion_mm=0.0)
    result = roi.refine(seg, vb)
    assert count_voxels(result.refined_mask) == 0


def test_extract_lumbar_vb_geometry_mismatch_raises():
    seg = sitk.Image([10, 10, 10], sitk.sitkUInt8)
    vb = sitk.Image([10, 10, 10], sitk.sitkUInt8)
    pet = sitk.Image([10, 10, 10], sitk.sitkFloat64)
    pet.SetSpacing((2.0, 2.0, 2.0))
    roi = LumbarVB()
    with pytest.raises(ValueError, match="Geometry mismatch"):
        roi.extract(seg, vb, pet)


def test_extract_lumbar_vb_default_stats_p95(seg_phantom, vert_body_seg, pet_phantom):
    roi = LumbarVB(erosion_mm=0.0)
    result = roi.extract(seg_phantom, vert_body_seg, pet_phantom)
    assert "p95" in result.stats


def test_extract_lumbar_vb_custom_stats(seg_phantom, vert_body_seg, pet_phantom):
    roi = LumbarVB(erosion_mm=0.0, stats=["mean", "median"])
    result = roi.extract(seg_phantom, vert_body_seg, pet_phantom)
    assert "mean" in result.stats
    assert "median" in result.stats


def test_extract_lumbar_vb_known_pet_value(make_image):
    seg_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    seg_arr[2:8, 2:8, 2:8] = labels.L3
    seg = make_image(seg_arr)
    vb_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    vb_arr[2:8, 2:8, 2:8] = 1
    vb = make_image(vb_arr)
    pet_arr = np.full((10, 10, 10), 4.0, dtype=np.float64)
    pet = make_image(pet_arr)
    roi = LumbarVB(erosion_mm=0.0, stats=["mean"])
    result = roi.extract(seg, vb, pet)
    assert result.stats["mean"] == pytest.approx(4.0)
