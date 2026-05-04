"""Tests for autods_pet.stats - statistics helpers."""

import numpy as np
import pytest
import SimpleITK as sitk
from hypothesis import given, settings
from hypothesis import strategies as st

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


def _make_image(arr, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def _make_mask(shape, ones_slices=None, spacing=(1.0, 1.0, 1.0)):
    """Binary mask. If ones_slices is None, all ones."""
    arr = (
        np.ones(shape, dtype=np.uint8)
        if ones_slices is None
        else np.zeros(shape, dtype=np.uint8)
    )
    if ones_slices is not None:
        for s in ones_slices:
            arr[s] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def test_percentile_known_values():
    arr = np.array([[[1, 2, 3, 4, 5]]], dtype=np.float64)
    img = _make_image(arr)
    mask = _make_image(np.ones_like(arr, dtype=np.uint8))
    assert percentile_in_mask(img, mask, 50) == pytest.approx(3.0)


def test_percentile_p0_is_min():
    arr = np.array([[[10, 20, 30]]], dtype=np.float64)
    img = _make_image(arr)
    mask = _make_image(np.ones_like(arr, dtype=np.uint8))
    assert percentile_in_mask(img, mask, 0) == pytest.approx(10.0)


def test_percentile_p100_is_max():
    arr = np.array([[[10, 20, 30]]], dtype=np.float64)
    img = _make_image(arr)
    mask = _make_image(np.ones_like(arr, dtype=np.uint8))
    assert percentile_in_mask(img, mask, 100) == pytest.approx(30.0)


def test_percentile_empty_mask_returns_none():
    img = _make_image(np.ones((2, 2, 2), dtype=np.float64))
    mask = _make_image(np.zeros((2, 2, 2), dtype=np.uint8))
    assert percentile_in_mask(img, mask, 50) is None


def test_percentile_out_of_range_raises():
    img = _make_image(np.ones((2, 2, 2), dtype=np.float64))
    mask = _make_image(np.ones((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="pct"):
        percentile_in_mask(img, mask, 200)


def test_mean_in_mask_known():
    arr = np.array([[[1, 2, 3]]], dtype=np.float64)
    img = _make_image(arr)
    mask = _make_image(np.ones_like(arr, dtype=np.uint8))
    assert mean_in_mask(img, mask) == pytest.approx(2.0)


def test_max_in_mask_known():
    arr = np.array([[[1, 5, 3]]], dtype=np.float64)
    img = _make_image(arr)
    mask = _make_image(np.ones_like(arr, dtype=np.uint8))
    assert max_in_mask(img, mask) == pytest.approx(5.0)


def test_min_in_mask_known():
    arr = np.array([[[1, 5, 3]]], dtype=np.float64)
    img = _make_image(arr)
    mask = _make_image(np.ones_like(arr, dtype=np.uint8))
    assert min_in_mask(img, mask) == pytest.approx(1.0)


def test_median_known():
    arr = np.array([[[1, 2, 3, 4, 5]]], dtype=np.float64)
    img = _make_image(arr)
    mask = _make_image(np.ones_like(arr, dtype=np.uint8))
    assert voxelwise_median(img, mask) == pytest.approx(3.0)


def test_stats_empty_mask_returns_none():
    img = _make_image(np.ones((2, 2, 2), dtype=np.float64))
    mask = _make_image(np.zeros((2, 2, 2), dtype=np.uint8))
    assert mean_in_mask(img, mask) is None
    assert max_in_mask(img, mask) is None
    assert min_in_mask(img, mask) is None
    assert voxelwise_median(img, mask) is None


def test_count_voxels_full():
    mask = _make_mask((3, 3, 3))
    assert count_voxels(mask) == 27


def test_count_voxels_empty():
    mask = _make_image(np.zeros((3, 3, 3), dtype=np.uint8))
    assert count_voxels(mask) == 0


def test_mask_volume_mm3():
    # 8 voxels, spacing 2x2x2 → voxel volume = 8 mm³ → total = 64 mm³
    mask = _make_mask((2, 2, 2), spacing=(2.0, 2.0, 2.0))
    assert mask_volume_mm3(mask) == pytest.approx(64.0)


def test_compute_stats_multiple():
    arr = np.array([[[1, 2, 3, 4, 5]]], dtype=np.float64)
    img = _make_image(arr)
    mask = _make_image(np.ones_like(arr, dtype=np.uint8))
    results = compute_stats(["mean", "median", "p90", "max"], img, mask)
    assert results["mean"] == pytest.approx(3.0)
    assert results["median"] == pytest.approx(3.0)
    assert results["max"] == pytest.approx(5.0)
    assert "p90" in results


def test_shrinkage_report_identical_masks():
    mask = _make_mask((3, 3, 3))
    report = shrinkage_report(mask, mask)
    assert report["delta_voxels"] == 0
    assert report["delta_volume_mm3"] == pytest.approx(0.0)
    assert report["shrinkage_pct"] == pytest.approx(0.0)


def test_shrinkage_report_known_shrinkage():
    orig = _make_mask((2, 2, 2))  # 8 voxels
    arr_ref = np.zeros((2, 2, 2), dtype=np.uint8)
    arr_ref[0, :, :] = 1  # 4 voxels
    refined = _make_image(arr_ref)
    report = shrinkage_report(orig, refined)
    assert report["original_voxels"] == 8
    assert report["refined_voxels"] == 4
    assert report["shrinkage_pct"] == pytest.approx(50.0)


def test_shrinkage_report_empty_original():
    empty = _make_image(np.zeros((2, 2, 2), dtype=np.uint8))
    report = shrinkage_report(empty, empty)
    assert report["shrinkage_pct"] == pytest.approx(0.0)


def test_shrinkage_report_respects_spacing():
    mask = _make_mask((2, 2, 2), spacing=(2.0, 2.0, 2.0))
    report = shrinkage_report(mask, mask)
    assert report["original_volume_mm3"] == pytest.approx(8 * 8.0)  # 8 voxels * 8 mm³


def test_shrinkage_report_all_keys_present():
    mask = _make_mask((2, 2, 2))
    report = shrinkage_report(mask, mask)
    expected_keys = {
        "original_voxels",
        "refined_voxels",
        "original_volume_mm3",
        "refined_volume_mm3",
        "delta_voxels",
        "delta_volume_mm3",
        "shrinkage_pct",
    }
    assert set(report.keys()) == expected_keys


def test_count_voxels_partial_mask():
    """Partial mask: only slice 0 filled → exactly 9 voxels in a 3x3x3."""
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[0, :, :] = 1
    mask = _make_image(arr)
    assert count_voxels(mask) == 9


def test_count_voxels_all_nonzero_labels_counted():
    """Multilabel mask: all non-zero values treated as True."""
    arr = np.zeros((2, 2, 2), dtype=np.uint8)
    arr[0, 0, 0] = 5
    arr[1, 1, 1] = 200
    mask = _make_image(arr)
    assert count_voxels(mask) == 2


def test_mask_volume_mm3_anisotropic_spacing():
    """Anisotropic spacing (1,2,3) with 8 voxels → 8 * 6 = 48 mm³."""
    mask = _make_mask((2, 2, 2), spacing=(1.0, 2.0, 3.0))
    assert mask_volume_mm3(mask) == pytest.approx(48.0)


def test_mask_volume_mm3_empty_mask():
    """Empty mask returns 0.0 volume."""
    mask = _make_image(np.zeros((2, 2, 2), dtype=np.uint8))
    assert mask_volume_mm3(mask) == pytest.approx(0.0)


def test_mask_volume_mm3_single_voxel():
    """Single voxel with spacing (0.5, 0.5, 0.5) → 0.125 mm³."""
    arr = np.zeros((2, 2, 2), dtype=np.uint8)
    arr[0, 0, 0] = 1
    mask = _make_image(arr, spacing=(0.5, 0.5, 0.5))
    assert mask_volume_mm3(mask) == pytest.approx(0.125)


def test_compute_stats_empty_mask_all_none():
    """compute_stats returns None for every stat when mask is empty."""
    img = _make_image(np.ones((2, 2, 2), dtype=np.float64))
    mask = _make_image(np.zeros((2, 2, 2), dtype=np.uint8))
    results = compute_stats(["mean", "median", "min", "max", "p95"], img, mask)
    assert all(v is None for v in results.values())


def test_shrinkage_report_refined_larger_than_original():
    """Negative shrinkage when refined has more voxels than original."""
    arr_orig = np.zeros((2, 2, 2), dtype=np.uint8)
    arr_orig[0, 0, 0] = 1  # 1 voxel
    orig = _make_image(arr_orig)
    refined = _make_mask((2, 2, 2))  # 8 voxels
    report = shrinkage_report(orig, refined)
    assert report["shrinkage_pct"] < 0


def test_percentile_in_mask_boundary_p0_and_p100():
    """p0 equals min, p100 equals max of values in mask."""
    arr = np.array([[[2.0, 7.0, 4.0, 1.0, 9.0]]], dtype=np.float64)
    img = _make_image(arr)
    mask = _make_image(np.ones_like(arr, dtype=np.uint8))
    assert percentile_in_mask(img, mask, 0) == pytest.approx(1.0)
    assert percentile_in_mask(img, mask, 100) == pytest.approx(9.0)


@given(
    spacing=st.tuples(
        st.floats(min_value=0.1, max_value=10.0),
        st.floats(min_value=0.1, max_value=10.0),
        st.floats(min_value=0.1, max_value=10.0),
    )
)
@settings(max_examples=30)
def test_mask_volume_equals_count_times_voxel_vol(spacing):
    """Invariant: volume = count * product(spacing)."""
    mask = _make_mask((2, 2, 2), spacing=spacing)
    expected = count_voxels(mask) * spacing[0] * spacing[1] * spacing[2]
    assert mask_volume_mm3(mask) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# _compute_on_vals edge cases
# ---------------------------------------------------------------------------


def test_compute_on_vals_min():
    """_compute_on_vals handles 'min' kind."""
    from autods_pet.ops.stats import _compute_on_vals

    vals = np.array([3.0, 1.0, 2.0])
    assert _compute_on_vals(vals, "min", None) == pytest.approx(1.0)


def test_compute_on_vals_percentile_no_param_raises():
    """_compute_on_vals raises when percentile is called without a parameter."""
    from autods_pet.ops.stats import _compute_on_vals

    vals = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="percentile stat requires"):
        _compute_on_vals(vals, "percentile", None)


def test_compute_on_vals_unsupported_kind_raises():
    """_compute_on_vals raises on unknown stat kind."""
    from autods_pet.ops.stats import _compute_on_vals

    vals = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Unsupported stat kind"):
        _compute_on_vals(vals, "unknown_stat", None)
