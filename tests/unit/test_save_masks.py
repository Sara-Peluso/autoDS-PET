"""Tests for autods_pet.ops.save_masks - saving raw and refined masks."""

import numpy as np
import SimpleITK as sitk

from autods_pet.ops.save_masks import (
    _REFINED_NAME_MAP,
    RAW_LABEL_MAP,
    save_raw_masks,
    save_refined_masks,
)
from autods_pet.results import ROIResult


def _make_seg(labels: list[int], size: tuple[int, int, int] = (10, 10, 10)):
    """Create a synthetic multilabel segmentation with given labels."""
    arr = np.zeros(size, dtype=np.uint8)
    for i, lbl in enumerate(labels):
        arr[i, :, :] = lbl
    return sitk.GetImageFromArray(arr)


def _make_binary_mask(size: tuple[int, int, int] = (10, 10, 10)):
    """Create a simple binary mask."""
    arr = np.zeros(size, dtype=np.uint8)
    arr[3:7, 3:7, 3:7] = 1
    return sitk.GetImageFromArray(arr)


def test_save_raw_masks_creates_files(tmp_path):
    seg = _make_seg([52, 5])
    saved = save_raw_masks(seg, tmp_path)
    assert len(saved) == len(RAW_LABEL_MAP)
    raw_dir = tmp_path / "raw"
    assert raw_dir.is_dir()
    for path in saved:
        assert path.exists()
        assert path.suffix == ".gz"


def test_save_raw_masks_binary_values(tmp_path):
    seg = _make_seg([52, 5, 29])
    save_raw_masks(seg, tmp_path)
    for path in (tmp_path / "raw").iterdir():
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)
        assert set(np.unique(arr)).issubset({0, 1})


def test_save_raw_masks_correct_filenames(tmp_path):
    seg = _make_seg([52])
    save_raw_masks(seg, tmp_path)
    raw_dir = tmp_path / "raw"
    expected_names = {f"{name}.nii.gz" for name in RAW_LABEL_MAP.values()}
    actual_names = {f.name for f in raw_dir.iterdir()}
    assert actual_names == expected_names


def test_save_refined_masks_creates_files(tmp_path):
    results = {
        "Aorta MBP": ROIResult(stats={"median": 2.5}, refined_mask=_make_binary_mask()),
        "Liver": ROIResult(stats={"median": 3.0}, refined_mask=_make_binary_mask()),
        "_roi_statuses": [],
    }
    saved = save_refined_masks(results, tmp_path)
    assert len(saved) == 2
    refined_dir = tmp_path / "refined"
    assert refined_dir.is_dir()
    assert (refined_dir / "aorta_mbp.nii.gz").exists()
    assert (refined_dir / "liver.nii.gz").exists()


def test_save_refined_masks_skips_none_mask(tmp_path):
    results = {
        "Aorta MBP": ROIResult(stats={"median": 2.5}, refined_mask=_make_binary_mask()),
        "Lumbar VB": ROIResult(stats={"p95": 1.0}, refined_mask=None),
        "_roi_statuses": [],
    }
    saved = save_refined_masks(results, tmp_path)
    assert len(saved) == 1
    assert saved[0].name == "aorta_mbp.nii.gz"


def test_save_refined_masks_skips_internal_keys(tmp_path):
    results = {
        "_roi_statuses": [("Aorta MBP", "ok", "")],
        "Liver": ROIResult(stats={"median": 3.0}, refined_mask=_make_binary_mask()),
    }
    saved = save_refined_masks(results, tmp_path)
    assert len(saved) == 1
    assert saved[0].name == "liver.nii.gz"


def test_save_refined_masks_custom_target_name(tmp_path):
    results = {
        "my custom roi": ROIResult(
            stats={"max": 5.0}, refined_mask=_make_binary_mask()
        ),
        "_roi_statuses": [],
    }
    saved = save_refined_masks(results, tmp_path)
    assert len(saved) == 1
    assert saved[0].name == "my_custom_roi.nii.gz"


def test_save_refined_masks_readback(tmp_path):
    mask = _make_binary_mask()
    results = {
        "Liver": ROIResult(stats={"median": 3.0}, refined_mask=mask),
        "_roi_statuses": [],
    }
    saved = save_refined_masks(results, tmp_path)
    reloaded = sitk.ReadImage(str(saved[0]))
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(reloaded),
        sitk.GetArrayFromImage(mask),
    )


def test_save_refined_masks_empty_results(tmp_path):
    results = {"_roi_statuses": []}
    saved = save_refined_masks(results, tmp_path)
    assert saved == []


def test_save_refined_masks_all_known_names(tmp_path):
    results = {"_roi_statuses": []}
    for display_name in _REFINED_NAME_MAP:
        results[display_name] = ROIResult(
            stats={"median": 1.0}, refined_mask=_make_binary_mask()
        )
    saved = save_refined_masks(results, tmp_path)
    assert len(saved) == len(_REFINED_NAME_MAP)
    actual_names = {p.stem.replace(".nii", "") for p in saved}
    expected_names = set(_REFINED_NAME_MAP.values())
    assert actual_names == expected_names
