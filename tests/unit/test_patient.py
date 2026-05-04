"""Tests for autods_pet.patient - path resolution and lazy image loading."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autods_pet.patient import PatientCase, resolve_paths


def test_resolve_paths_substitutes_patient_id(minimal_cfg):
    paths = resolve_paths(minimal_cfg, "P001")
    assert "P001" in str(paths["ct_nifti"])
    assert "{patient_id}" not in str(paths["ct_nifti"])


def test_resolve_paths_all_keys_present(minimal_cfg):
    paths = resolve_paths(minimal_cfg, "P001")
    expected_keys = {
        "basepath",
        "input_dir",
        "input_seg_dir",
        "ct_nifti",
        "pet_nifti",
        "pet_suv",
        "pet_registered",
        "seg_dir",
        "vert_body_seg",
        "pet_metadata",
        "output_dir",
        "elastix_report",
        "deauville_csv",
        "suv_csv",
    }
    assert expected_keys == set(paths.keys())


def test_resolve_paths_empty_template_falls_back(tmp_path):
    cfg = {
        "paths": {
            "basepath": str(tmp_path),
            "output_dir": str(tmp_path / "results"),
            "ct_nifti": "",
            "pet_nifti": "{patient_id}/PET.nii.gz",
        }
    }
    paths = resolve_paths(cfg, "P001")
    # Empty template falls back to output_dir / P001_results / ct_nifti.nii.gz
    assert (
        paths["ct_nifti"] == tmp_path / "results" / "P001_results" / "ct_nifti.nii.gz"
    )


def test_resolve_paths_absolute_output_dir(tmp_path):
    cfg = {
        "paths": {
            "basepath": str(tmp_path),
            "output_dir": "/absolute/output",
        }
    }
    paths = resolve_paths(cfg, "P001")
    assert paths["output_dir"] == Path("/absolute/output")


def test_resolve_paths_relative_output_dir(tmp_path):
    from pathlib import Path

    cfg = {
        "paths": {
            "basepath": str(tmp_path),
            "output_dir": "my_results",
        }
    }
    paths = resolve_paths(cfg, "P001")
    assert paths["output_dir"] == Path.cwd() / "my_results"


def test_resolve_paths_no_output_dir_defaults(tmp_path):
    from pathlib import Path

    cfg = {"paths": {"basepath": str(tmp_path)}}
    paths = resolve_paths(cfg, "P001")
    assert paths["output_dir"] == Path.cwd() / "results"


def test_patient_case_init_sets_attributes(minimal_cfg):
    pc = PatientCase(minimal_cfg, "P001")
    assert pc.cfg is minimal_cfg
    assert pc.patient_id == "P001"
    assert isinstance(pc.paths, dict)
    assert pc._cache == {}


def test_patient_case_path_properties(minimal_cfg):
    pc = PatientCase(minimal_cfg, "P001")
    assert pc.ct_path == pc.paths["ct_nifti"]
    assert pc.pet_path == pc.paths["pet_nifti"]
    assert pc.pet_suv_path == pc.paths["pet_suv"]
    assert pc.pet_registered_path == pc.paths["pet_registered"]
    assert pc.seg_dir == pc.paths["seg_dir"]
    assert pc.vert_body_seg_path == pc.paths["vert_body_seg"]
    assert pc.metadata_path == pc.paths["pet_metadata"]
    assert pc.output_dir == pc.paths["output_dir"]


@pytest.mark.parametrize(
    "method_name,cache_key",
    [
        ("load_ct", "ct"),
        ("load_pet_suv", "pet_suv"),
        ("load_pet_registered", "pet_registered"),
    ],
)
def test_load_cached_reads_and_caches(minimal_cfg, method_name, cache_key):
    pc = PatientCase(minimal_cfg, "P001")
    sentinel = MagicMock(name="fake_image")
    with patch("SimpleITK.ReadImage", return_value=sentinel) as mock_read:
        result1 = getattr(pc, method_name)()
        result2 = getattr(pc, method_name)()
    assert result1 is sentinel
    assert result2 is sentinel
    mock_read.assert_called_once()  # only one read, second from cache


def test_load_segmentation_finds_nii(minimal_cfg, tmp_path):
    pc = PatientCase(minimal_cfg, "P001")
    seg_dir = pc.seg_dir
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_file = seg_dir / "whole_seg.nii"
    seg_file.touch()  # create empty file

    sentinel = MagicMock(name="seg_image")
    with patch("SimpleITK.ReadImage", return_value=sentinel):
        result = pc.load_segmentation()
    assert result is sentinel


def test_load_segmentation_finds_nii_gz(minimal_cfg, tmp_path):
    pc = PatientCase(minimal_cfg, "P001")
    seg_dir = pc.seg_dir
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_file = seg_dir / "whole_seg.nii.gz"
    seg_file.touch()

    sentinel = MagicMock(name="seg_image_gz")
    with patch("SimpleITK.ReadImage", return_value=sentinel):
        result = pc.load_segmentation()
    assert result is sentinel


def test_load_segmentation_raises_when_missing(minimal_cfg, tmp_path):
    pc = PatientCase(minimal_cfg, "P001")
    seg_dir = pc.seg_dir
    seg_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="Segmentation not found"):
        pc.load_segmentation()


def test_load_segmentation_caches(minimal_cfg, tmp_path):
    pc = PatientCase(minimal_cfg, "P001")
    seg_dir = pc.seg_dir
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "whole_seg.nii").touch()

    sentinel = MagicMock(name="seg_cached")
    with patch("SimpleITK.ReadImage", return_value=sentinel) as mock_read:
        pc.load_segmentation()
        pc.load_segmentation()
    mock_read.assert_called_once()


def test_load_vert_body_seg_returns_none_when_missing(minimal_cfg, tmp_path):
    pc = PatientCase(minimal_cfg, "P001")
    assert pc.load_vert_body_seg() is None


def test_load_vert_body_seg_loads_and_caches(minimal_cfg, tmp_path):
    pc = PatientCase(minimal_cfg, "P001")
    vb_path = pc.vert_body_seg_path
    vb_path.parent.mkdir(parents=True, exist_ok=True)
    vb_path.touch()

    sentinel = MagicMock(name="vb_image")
    with patch("SimpleITK.ReadImage", return_value=sentinel) as mock_read:
        r1 = pc.load_vert_body_seg()
        r2 = pc.load_vert_body_seg()
    assert r1 is sentinel
    assert r2 is sentinel
    mock_read.assert_called_once()


def test_clear_cache_empties_all(minimal_cfg):
    pc = PatientCase(minimal_cfg, "P001")
    pc._cache["test_key"] = "test_value"
    pc.clear_cache()
    assert pc._cache == {}


def test_load_segmentation_prefers_nii_over_nii_gz(minimal_cfg, tmp_path):
    """When both .nii and .nii.gz exist, .nii is tried first."""
    pc = PatientCase(minimal_cfg, "P001")
    seg_dir = pc.seg_dir
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "whole_seg.nii").touch()
    (seg_dir / "whole_seg.nii.gz").touch()

    sentinel = MagicMock(name="nii_image")
    with patch("SimpleITK.ReadImage", return_value=sentinel) as mock_read:
        result = pc.load_segmentation()
    assert result is sentinel
    # Should have read the .nii variant
    mock_read.assert_called_once_with(str(seg_dir / "whole_seg.nii"))


def test_load_vert_body_seg_caches_none_on_missing(minimal_cfg, tmp_path):
    """Second call for missing vert_body_seg returns None without re-checking filesystem."""
    pc = PatientCase(minimal_cfg, "P001")
    assert pc.load_vert_body_seg() is None
    # Manually verify it's cached
    assert "vb" in pc._cache


def test_patient_case_custom_path_templates(tmp_path):
    """Non-default path templates are correctly substituted."""
    cfg = {
        "paths": {
            "basepath": str(tmp_path),
            "ct_nifti": "images/{patient_id}/ct_scan.nii.gz",
        }
    }
    pc = PatientCase(cfg, "P001")
    assert str(pc.ct_path).endswith("images/P001/ct_scan.nii.gz")


def test_clear_cache_allows_reload(minimal_cfg, tmp_path):
    """After clear_cache(), loading triggers a fresh ReadImage call."""
    pc = PatientCase(minimal_cfg, "P001")
    sentinel1 = MagicMock(name="img1")
    sentinel2 = MagicMock(name="img2")
    with patch("SimpleITK.ReadImage", side_effect=[sentinel1, sentinel2]) as mock_read:
        r1 = pc.load_ct()
        pc.clear_cache()
        r2 = pc.load_ct()
    assert r1 is sentinel1
    assert r2 is sentinel2
    assert mock_read.call_count == 2
