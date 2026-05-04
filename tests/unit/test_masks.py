"""Tests for autods_pet.masks - mask operations."""

import numpy as np
import SimpleITK as sitk

from autods_pet.ops.masks import (
    fill_holes,
    keep_largest_component,
    label_mask,
    label_union,
    subtract_mask,
)
from autods_pet.ops.stats import count_voxels


def _make_image(arr, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def test_label_union_empty_returns_image():
    seg = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    result = label_union(seg, [])
    assert isinstance(result, sitk.Image)
    assert result.GetSize() == (3, 3, 3)
    assert count_voxels(result) == 0


def test_label_union_single_label():
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[1, 1, 1] = 5
    seg = _make_image(arr)
    result = label_union(seg, [5])
    assert count_voxels(result) == 1


def test_label_union_multiple_labels():
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[0, 0, 0] = 1
    arr[1, 1, 1] = 2
    arr[2, 2, 2] = 3
    seg = _make_image(arr)
    result = label_union(seg, [1, 3])
    assert count_voxels(result) == 2


def test_fill_holes_no_max():
    # 5x5x5 shell with a 3x3x3 hole in the center
    arr = np.ones((5, 5, 5), dtype=np.uint8)
    arr[1:4, 1:4, 1:4] = 0  # 27-voxel hole
    mask = _make_image(arr)
    filled = fill_holes(mask, max_hole_volume_mm3=None)
    assert count_voxels(filled) == 125  # all filled


def test_fill_holes_preserves_large_holes():
    # 7x7x7 shell with a 3x3x3 hole in the center (27 mm³ at 1mm spacing)
    arr = np.ones((7, 7, 7), dtype=np.uint8)
    arr[2:5, 2:5, 2:5] = 0  # 27-voxel hole
    mask = _make_image(arr, spacing=(1.0, 1.0, 1.0))
    total = 7**3
    # Threshold below hole volume → hole preserved
    filled = fill_holes(mask, max_hole_volume_mm3=10.0)
    assert count_voxels(filled) < total


def test_fill_holes_fills_small_holes():
    # 7x7x7 shell with a 3x3x3 hole in the center (27 mm³ at 1mm spacing)
    # Thick enough shell (2 voxels) so the hole is fully enclosed
    arr = np.ones((7, 7, 7), dtype=np.uint8)
    arr[2:5, 2:5, 2:5] = 0  # 27-voxel hole
    mask = _make_image(arr, spacing=(1.0, 1.0, 1.0))
    total = 7**3
    # Threshold above hole volume → hole filled
    filled = fill_holes(mask, max_hole_volume_mm3=50.0)
    assert count_voxels(filled) == total


def test_label_mask_extracts_single_label():
    arr = np.zeros((5, 5, 5), dtype=np.uint8)
    arr[1, 1, 1] = 5
    arr[2, 2, 2] = 5
    arr[3, 3, 3] = 10
    seg = _make_image(arr)
    result = label_mask(seg, 5)
    assert count_voxels(result) == 2


def test_label_mask_missing_label_returns_empty():
    arr = np.zeros((5, 5, 5), dtype=np.uint8)
    arr[1, 1, 1] = 5
    seg = _make_image(arr)
    result = label_mask(seg, 99)
    assert count_voxels(result) == 0


def test_keep_largest_component_single_component():
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:5, 2:5, 2:5] = 1  # single 27-voxel blob
    mask = _make_image(arr)
    result = keep_largest_component(mask)
    assert count_voxels(result) == 27


def test_keep_largest_component_removes_smaller():
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[0:3, 0:3, 0:3] = 1  # 27-voxel blob
    arr[7:9, 7:9, 7:9] = 1  # 8-voxel blob (smaller, disconnected)
    mask = _make_image(arr)
    result = keep_largest_component(mask)
    assert count_voxels(result) == 27


def test_keep_largest_component_empty_mask():
    mask = _make_image(np.zeros((5, 5, 5), dtype=np.uint8))
    result = keep_largest_component(mask)
    assert count_voxels(result) == 0


def test_fill_holes_all_ones_returns_same():
    """fill_holes on a fully-filled mask returns it unchanged (no inverted components)."""
    arr = np.ones((5, 5, 5), dtype=np.uint8)
    mask = _make_image(arr)
    result = fill_holes(mask)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(result), arr)


def test_subtract_mask_basic():
    """Subtracting overlapping region removes those voxels from base."""
    base_arr = np.zeros((5, 5, 5), dtype=np.uint8)
    base_arr[1:4, 1:4, 1:4] = 1  # 27 voxels
    sub_arr = np.zeros((5, 5, 5), dtype=np.uint8)
    sub_arr[2:4, 2:4, 2:4] = 1  # 8 voxels overlap

    base = _make_image(base_arr)
    sub = _make_image(sub_arr)
    result = subtract_mask(base, sub)

    assert count_voxels(result) == 27 - 8


def test_subtract_mask_no_overlap():
    """Subtracting a disjoint mask returns base unchanged."""
    base_arr = np.zeros((5, 5, 5), dtype=np.uint8)
    base_arr[0, 0, 0] = 1
    sub_arr = np.zeros((5, 5, 5), dtype=np.uint8)
    sub_arr[4, 4, 4] = 1

    result = subtract_mask(_make_image(base_arr), _make_image(sub_arr))
    assert count_voxels(result) == 1


def test_subtract_mask_full_overlap():
    """Subtracting the same mask yields an empty result."""
    arr = np.zeros((5, 5, 5), dtype=np.uint8)
    arr[1:3, 1:3, 1:3] = 1
    mask = _make_image(arr)
    result = subtract_mask(mask, mask)
    assert count_voxels(result) == 0


def test_subtract_mask_empty_subtract():
    """Subtracting an empty mask returns the base unchanged."""
    base_arr = np.ones((3, 3, 3), dtype=np.uint8)
    empty_arr = np.zeros((3, 3, 3), dtype=np.uint8)
    result = subtract_mask(_make_image(base_arr), _make_image(empty_arr))
    assert count_voxels(result) == 27


def test_subtract_mask_preserves_metadata():
    """Output image preserves spacing, origin, and direction from base."""
    base_arr = np.ones((3, 3, 3), dtype=np.uint8)
    sub_arr = np.zeros((3, 3, 3), dtype=np.uint8)
    spacing = (0.5, 0.75, 1.25)
    origin = (10.0, 20.0, 30.0)

    base = _make_image(base_arr, spacing=spacing)
    base.SetOrigin(origin)
    sub = _make_image(sub_arr, spacing=spacing)
    sub.SetOrigin(origin)

    result = subtract_mask(base, sub)
    assert result.GetSpacing() == spacing
    assert result.GetOrigin() == origin
