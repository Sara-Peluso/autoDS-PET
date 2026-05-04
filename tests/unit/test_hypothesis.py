"""Property-based tests using Hypothesis for autods_pet invariants."""

import numpy as np
import SimpleITK as sitk
from hypothesis import given, settings
from hypothesis import strategies as st

from autods_pet.deauville import assign_ds
from autods_pet.ops.morphology import dilate_mask_mm, erode_mask_mm
from autods_pet.ops.stats import count_voxels, shrinkage_report
from autods_pet.roi.long_bones import _crop_to_diaphysis_z


@given(
    target=st.floats(min_value=0.01, max_value=100.0),
    mbp=st.floats(min_value=0.01, max_value=100.0),
    liver=st.floats(min_value=0.01, max_value=100.0),
)
def test_assign_ds_always_returns_valid_score(target, mbp, liver):
    ds = assign_ds(target, mbp, liver)
    assert ds in {2, 3, 4, 5}


@given(
    mbp=st.floats(min_value=0.01, max_value=100.0),
    liver=st.floats(min_value=0.01, max_value=100.0),
)
def test_assign_ds_none_target_returns_0_or_1(mbp, liver):
    assert assign_ds(None, mbp, liver, allow_ds1=False) == 0
    assert assign_ds(None, mbp, liver, allow_ds1=True) == 1


def _make_bone_mask(height=20):
    """Create a simple bone mask column."""
    arr = np.zeros((height, 10, 10), dtype=np.uint8)
    arr[:, 4:7, 4:7] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


@given(keep_pct=st.integers(min_value=1, max_value=100))
@settings(max_examples=30)
def test_crop_diaphysis_output_leq_input(keep_pct):
    mask = _make_bone_mask(20)
    cropped, info = _crop_to_diaphysis_z(mask, keep_pct)
    assert count_voxels(cropped) <= count_voxels(mask)


@given(keep_pct=st.integers(min_value=1, max_value=100))
@settings(max_examples=30)
def test_crop_diaphysis_always_nonempty_for_nonempty_input(keep_pct):
    mask = _make_bone_mask(20)
    cropped, info = _crop_to_diaphysis_z(mask, keep_pct)
    assert count_voxels(cropped) > 0


def _make_cube_mask():
    """6×6×6 cube inside 10×10×10 image."""
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:8, 2:8, 2:8] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


@given(radius=st.floats(min_value=0.1, max_value=5.0))
@settings(max_examples=20)
def test_erode_always_subset_of_original(radius):
    mask = _make_cube_mask()
    eroded = erode_mask_mm(mask, radius)
    assert count_voxels(eroded) <= count_voxels(mask)


@given(radius=st.floats(min_value=0.0, max_value=5.0))
@settings(max_examples=20)
def test_dilate_always_superset_of_original(radius):
    mask = _make_cube_mask()
    dilated = dilate_mask_mm(mask, radius)
    assert count_voxels(dilated) >= count_voxels(mask)


@given(
    orig_size=st.integers(min_value=2, max_value=8),
    ref_size=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=20)
def test_shrinkage_pct_bounded(orig_size, ref_size):
    ref_size = min(ref_size, orig_size)  # ensure refined <= original
    orig_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    orig_arr[:orig_size, :orig_size, :orig_size] = 1
    ref_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    ref_arr[:ref_size, :ref_size, :ref_size] = 1

    orig = sitk.GetImageFromArray(orig_arr)
    orig.SetSpacing((1.0, 1.0, 1.0))
    ref = sitk.GetImageFromArray(ref_arr)
    ref.SetSpacing((1.0, 1.0, 1.0))

    report = shrinkage_report(orig, ref)
    assert 0.0 <= report["shrinkage_pct"] <= 100.0


@given(
    target_low=st.floats(min_value=0.01, max_value=50.0),
    delta=st.floats(min_value=0.0, max_value=50.0),
    mbp=st.floats(min_value=0.01, max_value=50.0),
    liver=st.floats(min_value=0.01, max_value=50.0),
)
def test_score_deauville_monotonic_in_target(target_low, delta, mbp, liver):
    target_high = target_low + delta
    ds_low = assign_ds(target_low, mbp, liver)
    ds_high = assign_ds(target_high, mbp, liver)
    assert ds_high >= ds_low
