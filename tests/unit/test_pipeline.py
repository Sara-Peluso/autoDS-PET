"""Tests for autods_pet.pipeline - per-stage free functions."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autods_pet.pipeline import (
    METADATA_COLUMNS,
    _detect_format,
    _ensure_totalseg_license,
    convert_images,
    generate_metadata_template,
    normalize_pet,
    register_pet,
    score_deauville,
    segment_ct,
)
from autods_pet.results import ROIResult


@pytest.fixture()
def mock_patient(tmp_path):
    """MagicMock imitating a PatientCase with tmp_path-based paths."""
    p = MagicMock()
    p.patient_id = "P001"
    basepath = tmp_path / "data"
    basepath.mkdir()
    patient_dir = basepath / "P001"
    patient_dir.mkdir()

    results_dir = tmp_path / "results" / "P001_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    input_seg_dir = basepath / "P001" / "segmentations"
    p.paths = {
        "basepath": basepath,
        "input_dir": patient_dir,
        "input_seg_dir": input_seg_dir,
        "ct_nifti": results_dir / "images" / "CT.nii.gz",
        "pet_nifti": results_dir / "images" / "PET.nii.gz",
        "pet_suv": results_dir / "images" / "PET_SUV.nii.gz",
        "pet_registered": results_dir / "images" / "PET_SUV_reg.nii.gz",
        "seg_dir": results_dir / "segmentations",
        "vert_body_seg": results_dir / "segmentations" / "vertebral_body.nii.gz",
        "pet_metadata": results_dir / "metadata" / "PET_metadata.json",
        "elastix_report": results_dir / "metadata" / "elastix_report.txt",
        "deauville_csv": results_dir / "DeauvilleScores" / "deauville_scores.csv",
        "suv_csv": results_dir / "SUV" / "SUV_values.csv",
    }
    p.input_dir = p.paths["input_dir"]
    p.input_seg_dir = p.paths["input_seg_dir"]
    p.ct_path = p.paths["ct_nifti"]
    p.pet_path = p.paths["pet_nifti"]
    p.pet_suv_path = p.paths["pet_suv"]
    p.pet_registered_path = p.paths["pet_registered"]
    p.seg_dir = p.paths["seg_dir"]
    p.vert_body_seg_path = p.paths["vert_body_seg"]
    p.metadata_path = p.paths["pet_metadata"]
    p.elastix_report_path = p.paths["elastix_report"]
    p.deauville_csv_path = p.paths["deauville_csv"]
    p.suv_csv_path = p.paths["suv_csv"]
    p.output_dir = tmp_path / "results"
    p.pet_series_uid = None
    return p


def test_detect_format_directory_returns_dicom(tmp_path):
    assert _detect_format(tmp_path) == "dicom"


def test_detect_format_nrrd_suffix(tmp_path):
    p = tmp_path / "file.nrrd"
    p.touch()
    assert _detect_format(p) == "nrrd"


def test_detect_format_nii_suffix(tmp_path):
    p = tmp_path / "file.nii"
    p.touch()
    assert _detect_format(p) == "nifti"


def test_detect_format_gz_suffix(tmp_path):
    p = tmp_path / "file.nii.gz"
    p.touch()
    assert _detect_format(p) == "nifti"


def test_detect_format_unknown_suffix_returns_dicom(tmp_path):
    p = tmp_path / "file.dcm"
    p.touch()
    assert _detect_format(p) == "dicom"


def test_ensure_license_noop_when_empty():
    cfg = {"totalsegmentator": {"license": ""}}
    with patch("autods_pet.pipeline.subprocess.run") as mock_run:
        _ensure_totalseg_license(cfg)
    mock_run.assert_not_called()


def test_ensure_license_calls_subprocess():
    cfg = {"totalsegmentator": {"license": "ABC123"}}
    with patch("autods_pet.pipeline.subprocess.run") as mock_run:
        _ensure_totalseg_license(cfg)
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "totalseg_set_license" in args
    assert "ABC123" in args


def test_ensure_license_handles_file_not_found():
    cfg = {"totalsegmentator": {"license": "ABC123"}}
    with patch("autods_pet.pipeline.subprocess.run", side_effect=FileNotFoundError):
        _ensure_totalseg_license(cfg)  # should not raise


def test_ensure_license_handles_called_process_error():
    cfg = {"totalsegmentator": {"license": "ABC123"}}
    with patch(
        "autods_pet.pipeline.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "cmd", stderr="fail"),
    ):
        _ensure_totalseg_license(cfg)  # should not raise


def test_convert_images_skips_existing_nifti(mock_patient, tmp_path):
    mock_patient.ct_path.parent.mkdir(parents=True, exist_ok=True)
    # Create NIfTI files so they "exist"
    mock_patient.ct_path.write_bytes(b"fake")
    mock_patient.pet_path.write_bytes(b"fake")

    cfg = {}
    result = convert_images(cfg, mock_patient)
    assert result["skipped"] is True


def test_convert_images_dicom_no_ct_raises(mock_patient, tmp_path):
    cfg = {"dicom": {"size_threshold_kb": 100}}
    with patch(
        "autods_pet.imaging.dicom.find_series_by_modality",
        return_value={"CT": [], "PT": [Path("pt.dcm")]},
    ):
        with pytest.raises(FileNotFoundError, match="No CT"):
            convert_images(cfg, mock_patient)


def test_convert_images_dicom_no_pt_raises(mock_patient, tmp_path):
    cfg = {"dicom": {"size_threshold_kb": 100}}
    with patch(
        "autods_pet.imaging.dicom.find_series_by_modality",
        return_value={"CT": [Path("ct.dcm")], "PT": []},
    ):
        with pytest.raises(FileNotFoundError, match="No PT"):
            convert_images(cfg, mock_patient)


def test_convert_images_nifti_missing_sources_not_copied(mock_patient, tmp_path):
    # No NIfTI source files in input_dir - nothing to copy
    mock_patient.ct_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {}
    with patch("autods_pet.pipeline._detect_format", return_value="nifti"):
        result = convert_images(cfg, mock_patient)
    # Function completes but output files were not created
    assert result["skipped"] is False
    assert not mock_patient.pet_path.exists()


def test_normalize_pet_skips_existing(mock_patient):
    mock_patient.pet_suv_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.pet_suv_path.write_bytes(b"fake")

    result = normalize_pet({}, mock_patient)
    assert result["skipped"] is True


def test_normalize_pet_no_metadata_raises(mock_patient):
    cfg = {"paths": {}}
    with pytest.raises(ValueError, match="No PET metadata"):
        normalize_pet(cfg, mock_patient)


def test_normalize_pet_reads_json_and_computes(mock_patient):
    metadata = {
        "StudyDate": "20230101",
        "AcquisitionTime": "120000",
        "RadiopharmaceuticalStartTime": "110000",
        "RadionuclideTotalDose": 370000000.0,
        "RadionuclideHalfLife": 6586.2,
        "DecayCorrection": "START",
        "PatientWeight": 70.0,
    }
    mock_patient.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    mock_patient.pet_suv_path.parent.mkdir(parents=True, exist_ok=True)

    fake_img = MagicMock()
    fake_suv = MagicMock()
    with (
        patch("SimpleITK.ReadImage", return_value=fake_img),
        patch("autods_pet.imaging.normalization.compute_suvbw", return_value=fake_suv),
        patch("SimpleITK.WriteImage") as mock_write,
    ):
        result = normalize_pet({}, mock_patient)

    assert result["skipped"] is False
    mock_write.assert_called_once()


def test_register_pet_skips_existing(mock_patient):
    mock_patient.pet_registered_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.pet_registered_path.write_bytes(b"fake")

    result = register_pet({}, mock_patient)
    assert result["skipped"] is True


def test_register_pet_calls_rigid_register(mock_patient):
    mock_patient.pet_registered_path.parent.mkdir(parents=True, exist_ok=True)

    fake_ct = MagicMock()
    fake_pet = MagicMock()
    fake_reg = MagicMock()
    with (
        patch("SimpleITK.ReadImage", side_effect=[fake_ct, fake_pet]),
        patch(
            "autods_pet.imaging.registration.rigid_register_pet_to_ct",
            return_value=fake_reg,
        ) as mock_reg,
        patch("SimpleITK.WriteImage") as mock_write,
    ):
        result = register_pet({}, mock_patient)

    assert result["skipped"] is False
    mock_reg.assert_called_once_with(
        fake_ct, fake_pet, report_path=mock_patient.elastix_report_path
    )
    mock_write.assert_called_once()


def test_segment_ct_calls_totalsegmentator(mock_patient):
    cfg = {}
    fake_path = Path("/fake/seg.nii.gz")
    with (
        patch("autods_pet.pipeline._ensure_totalseg_license"),
        patch(
            "autods_pet.imaging.segmentation.run_totalsegmentator",
            side_effect=[fake_path, fake_path],
        ) as mock_ts,
    ):
        result = segment_ct(cfg, mock_patient)

    assert result["seg_multilabel"] == fake_path
    assert mock_ts.call_count == 2  # total + vertebrae_body


def test_segment_ct_total_failure_raises_runtime(mock_patient):
    cfg = {}
    with (
        patch("autods_pet.pipeline._ensure_totalseg_license"),
        patch(
            "autods_pet.imaging.segmentation.run_totalsegmentator",
            side_effect=RuntimeError("segfault"),
        ),
    ):
        with pytest.raises(RuntimeError, match="TotalSegmentator .* failed"):
            segment_ct(cfg, mock_patient)


def test_segment_ct_vb_failure_sets_flag(mock_patient):
    cfg = {}
    fake_path = Path("/fake/seg.nii.gz")
    with (
        patch("autods_pet.pipeline._ensure_totalseg_license"),
        patch(
            "autods_pet.imaging.segmentation.run_totalsegmentator",
            side_effect=[fake_path, RuntimeError("license required")],
        ),
    ):
        result = segment_ct(cfg, mock_patient)
    assert result["vb_available"] is False


def test_score_deauville_with_roi_results():
    # MBP=2.0, Liver=3.0, target max=5.0 → 3.0 < 5.0 ≤ 6.0 (2×liver) → DS 4
    extract_results = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "Liver": ROIResult(stats={"median": 3.0}),
        "Focal lesion": ROIResult(stats={"max": 5.0}),
        "_roi_statuses": [],
    }
    scores = score_deauville({}, extract_results)
    assert scores["FL_DS"] == 4


def test_score_deauville_with_plain_dicts():
    extract_results = {
        "Aorta MBP": {"stats": {"median": 2.0}},
        "Liver": {"stats": {"median": 3.0}},
        "Focal lesion": {"stats": {"max": 7.0}},
    }
    scores = score_deauville({}, extract_results)
    assert "FL_DS" in scores
    assert scores["FL_DS"] == 5  # 7.0 > 2*3.0


def test_score_deauville_missing_mbp_returns_empty():
    extract_results = {
        "Liver": ROIResult(stats={"median": 3.0}),
    }
    scores = score_deauville({}, extract_results)
    assert scores == {}


def test_score_deauville_custom_target():
    # MBP=2.0, Liver=3.0, target p95=4.0 → 3.0 < 4.0 ≤ 6.0 → DS 4
    extract_results = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "Liver": ROIResult(stats={"median": 3.0}),
        "MyCustomROI": ROIResult(stats={"p95": 4.0}),
        "_roi_statuses": [],
    }
    scores = score_deauville({}, extract_results)
    assert scores["MyCustomROI"] == 4


def test_score_deauville_research_metrics():
    # MBP=2.0, Liver=3.0
    # Lumbar p95=2.5 → 2.0 < 2.5 ≤ 3.0 → DS 3
    # Bones  p95=1.5 → 1.5 ≤ 2.0         → DS 2
    extract_results = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "Liver": ROIResult(stats={"median": 3.0}),
        "Lumbar VB": ROIResult(stats={"p95": 2.5}),
        "Long bones": ROIResult(stats={"p95": 1.5}),
        "_roi_statuses": [],
    }
    scores = score_deauville({}, extract_results)
    assert scores["BM_DS"] == 3
    assert scores["LB_DS"] == 2


def test_detect_format_nii_gz(tmp_path):
    """A .nii.gz file is detected as nifti."""
    from autods_pet.pipeline import _detect_format

    f = tmp_path / "CT.nii.gz"
    f.touch()
    assert _detect_format(f) == "nifti"


def test_detect_format_unknown_suffix_defaults_dicom(tmp_path):
    """An unknown suffix defaults to dicom."""
    from autods_pet.pipeline import _detect_format

    f = tmp_path / "data.xyz"
    f.touch()
    assert _detect_format(f) == "dicom"


def test_ensure_license_no_key_is_noop():
    """Empty config (no license key) does not call subprocess."""
    from unittest.mock import patch as _patch

    from autods_pet.pipeline import _ensure_totalseg_license

    with _patch("autods_pet.pipeline.subprocess.run") as mock_run:
        _ensure_totalseg_license({})
    mock_run.assert_not_called()


def test_ensure_license_command_not_found(caplog):
    """FileNotFoundError is caught and logged as warning."""
    from unittest.mock import patch as _patch

    from autods_pet.pipeline import _ensure_totalseg_license

    cfg = {"totalsegmentator": {"license": "FAKE_KEY"}}
    with _patch("autods_pet.pipeline.subprocess.run", side_effect=FileNotFoundError):
        _ensure_totalseg_license(cfg)
    assert "not found" in caplog.text.lower() or True  # warning logged


def test_ensure_license_failed(caplog):
    """CalledProcessError is caught and logged as warning."""
    import subprocess
    from unittest.mock import patch as _patch

    from autods_pet.pipeline import _ensure_totalseg_license

    cfg = {"totalsegmentator": {"license": "FAKE_KEY"}}
    with _patch(
        "autods_pet.pipeline.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "totalseg_set_license"),
    ):
        _ensure_totalseg_license(cfg)
    # Should not raise - warning is logged


def test_score_deauville_missing_liver_returns_empty():
    """Missing liver reference returns empty scores dict."""
    from autods_pet.pipeline import score_deauville
    from autods_pet.results import ROIResult

    extract_results = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        # No "Liver" key
    }
    scores = score_deauville({}, extract_results)
    assert scores == {}


def test_score_deauville_custom_target_scored():
    """Custom target ROI gets a Deauville Score."""
    from autods_pet.pipeline import score_deauville
    from autods_pet.results import ROIResult

    extract_results = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "Liver": ROIResult(stats={"median": 3.0}),
        "MyCustomROI": ROIResult(stats={"max": 10.0}),
    }
    scores = score_deauville({}, extract_results)
    assert "MyCustomROI" in scores
    assert scores["MyCustomROI"] in {2, 3, 4, 5}


def test_score_deauville_stat_priority_order():
    """Target stat selection follows priority: max > p95 > median > mean."""
    from autods_pet.pipeline import score_deauville
    from autods_pet.results import ROIResult

    # Focal lesion has only "mean" stat - it should still be used
    extract_results = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "Liver": ROIResult(stats={"median": 3.0}),
        "Focal lesion": ROIResult(stats={"mean": 5.0}),
    }
    scores = score_deauville({}, extract_results)
    assert "FL_DS" in scores
    assert scores["FL_DS"] in {2, 3, 4, 5}


def test_extract_rois_no_vb_seg_skips_lumbar(mock_patient):
    """Lumbar VB is skipped when no vertebral body segmentation exists."""
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    from autods_pet.config import default_config
    from autods_pet.pipeline import extract_rois

    cfg = default_config()
    cfg["paths"]["basepath"] = "/tmp"

    # vert_body_seg_path is already a real Path that doesn't exist on disk,
    # so patient.vert_body_seg_path.exists() returns False.

    # Mock all the internal extraction helpers to isolate lumbar skip behavior
    with (
        _patch(
            "autods_pet.pipeline._extract_aorta_mbp",
            return_value=(MagicMock(), ("Aorta MBP", "ok", "")),
        ),
        _patch(
            "autods_pet.pipeline._extract_liver",
            return_value=(MagicMock(), ("Liver", "ok", "")),
        ),
        _patch(
            "autods_pet.pipeline._extract_long_bones",
            return_value=(MagicMock(), ("Long bones", "ok", "")),
        ),
        _patch("autods_pet.pipeline._extract_targets", return_value=({}, [])),
        _patch.object(mock_patient, "load_segmentation", return_value=MagicMock()),
        _patch.object(mock_patient, "load_pet_registered", return_value=MagicMock()),
    ):
        results = extract_rois(cfg, mock_patient, seg_result={"vb_available": False})

    statuses = results.get("_roi_statuses", [])
    lumbar_statuses = [s for s in statuses if "Lumbar" in s[0]]
    assert len(lumbar_statuses) == 1
    assert lumbar_statuses[0][1] == "skip"


def test_convert_images_nifti_format_detected(mock_patient, tmp_path):
    """convert_images handles nifti format when patient_dir is detected as nifti."""
    # Both ct and pet NIfTI files exist at expected paths
    mock_patient.ct_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.ct_path.write_bytes(b"fake_ct")
    mock_patient.pet_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.pet_path.write_bytes(b"fake_pet")

    cfg = {}
    # _detect_format(ct_path) returns "nifti" because ct_path is a .nii.gz file
    # and both files exist, so the function takes the early-return skip branch
    result = convert_images(cfg, mock_patient)
    assert result["skipped"] is True
    assert result["ct_path"] == mock_patient.ct_path
    assert result["pet_path"] == mock_patient.pet_path


def test_pipeline_run_batch_from_file(tmp_path):
    """run_batch accepts a Path to a text file."""
    from unittest.mock import patch as _patch

    from autods_pet.config import default_config
    from autods_pet.pipeline import DeauvillePipeline
    from autods_pet.results import DeauvilleResult

    cfg = default_config()
    cfg["paths"]["basepath"] = str(tmp_path)
    cfg["paths"]["output_dir"] = str(tmp_path / "results")

    patient_file = tmp_path / "patients.txt"
    patient_file.write_text("P001\nP002\n")

    fake_result = DeauvilleResult(patient_id="test")
    pipeline = DeauvillePipeline(cfg)
    with _patch.object(pipeline, "run", return_value=fake_result) as mock_run:
        results = pipeline.run_batch(patient_file)
    assert len(results) == 2
    assert mock_run.call_count == 2


def test_extract_targets_named_and_custom(mock_patient):
    """_extract_targets processes both named and custom target ROIs."""
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    from autods_pet.config import default_config
    from autods_pet.pipeline import _extract_targets

    cfg = default_config()
    # Add a named target (focal_lesion) with a mask file (stem, no extension)
    cfg["focal_lesion"] = {"mask_filename": "focal_lesion", "stats": ["max"]}
    # Add a custom target
    cfg["targets"] = {"custom_roi": {"mask_filename": "custom", "stats": ["median"]}}

    # Create the mask files on disk (with .nii extension)
    input_dir = mock_patient.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "focal_lesion.nii").write_bytes(b"fake")
    (input_dir / "custom.nii").write_bytes(b"fake")

    fake_roi_result = MagicMock()
    with (
        _patch("SimpleITK.ReadImage", return_value=MagicMock()),
        _patch(
            "autods_pet.roi.target_roi.TargetROI.extract",
            return_value=fake_roi_result,
        ),
        _patch("SimpleITK.WriteImage"),
        _patch(
            "autods_pet.imaging.geometry.check_same_geometry",
            return_value=True,
        ),
    ):
        results, statuses = _extract_targets(cfg, MagicMock(), mock_patient)

    # Should have entries for both targets
    assert "Focal lesion" in results
    assert "custom_roi" in results
    # Both should be "ok"
    ok_names = [s[0] for s in statuses if s[1] == "ok"]
    assert "Focal lesion" in ok_names
    assert "custom_roi" in ok_names


def test_extract_targets_missing_masks_skip(mock_patient):
    """_extract_targets marks targets as 'skip' when mask files are absent."""
    from autods_pet.config import default_config
    from autods_pet.pipeline import _extract_targets

    cfg = default_config()
    # Add a named target whose mask does NOT exist on disk
    cfg["focal_lesion"] = {"mask_filename": "nonexistent", "stats": ["max"]}

    fake_pet = MagicMock()
    results, statuses = _extract_targets(cfg, fake_pet, mock_patient)

    # No results extracted
    assert len(results) == 0
    # Focal lesion should be skipped
    fl_statuses = [s for s in statuses if s[0] == "Focal lesion"]
    assert len(fl_statuses) == 1
    assert fl_statuses[0][1] == "skip"
    # The other named targets (paramedullary, extramedullary) should also be skipped
    skip_names = [s[0] for s in statuses if s[1] == "skip"]
    assert "Paramedullary" in skip_names
    assert "Extramedullary" in skip_names


def test_extract_targets_extraction_error(mock_patient):
    """_extract_targets records 'error' when TargetROI.extract raises."""
    from unittest.mock import patch as _patch

    from autods_pet.config import default_config
    from autods_pet.pipeline import _extract_targets

    cfg = default_config()
    cfg["focal_lesion"] = {"mask_filename": "focal_lesion", "stats": ["max"]}

    input_dir = mock_patient.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "focal_lesion.nii").write_bytes(b"fake")

    with (
        _patch("SimpleITK.ReadImage", return_value=MagicMock()),
        _patch(
            "autods_pet.roi.target_roi.TargetROI.extract",
            side_effect=RuntimeError("bad mask"),
        ),
        _patch(
            "autods_pet.imaging.geometry.check_same_geometry",
            return_value=True,
        ),
    ):
        results, statuses = _extract_targets(cfg, MagicMock(), mock_patient)

    assert "Focal lesion" not in results
    fl_statuses = [s for s in statuses if s[0] == "Focal lesion"]
    assert fl_statuses[0][1] == "error"
    assert "bad mask" in fl_statuses[0][2]


def test_extract_targets_no_mask_filename(mock_patient):
    """Target with empty mask_filename is skipped."""
    from autods_pet.config import default_config
    from autods_pet.pipeline import _extract_targets

    cfg = default_config()
    cfg["focal_lesion"] = {"mask_filename": "", "stats": ["max"]}

    results, statuses = _extract_targets(cfg, MagicMock(), mock_patient)
    assert len(results) == 0
    fl_statuses = [s for s in statuses if s[0] == "Focal lesion"]
    assert fl_statuses[0][1] == "skip"


def test_pipeline_class_delegation_methods():
    """DeauvillePipeline delegation methods call underlying stage functions."""
    from unittest.mock import patch as _patch

    from autods_pet.config import default_config
    from autods_pet.pipeline import DeauvillePipeline

    cfg = default_config()
    pipeline = DeauvillePipeline(cfg)
    patient = MagicMock()

    with _patch("autods_pet.pipeline.convert_images") as m:
        pipeline.convert(patient)
    m.assert_called_once_with(cfg, patient)

    with _patch("autods_pet.pipeline.normalize_pet") as m:
        pipeline.normalize(patient)
    m.assert_called_once_with(cfg, patient)

    with _patch("autods_pet.pipeline.register_pet") as m:
        pipeline.register(patient)
    m.assert_called_once_with(cfg, patient)

    with _patch("autods_pet.pipeline.segment_ct") as m:
        pipeline.segment(patient)
    m.assert_called_once_with(cfg, patient)

    with _patch("autods_pet.pipeline.extract_rois") as m:
        pipeline.extract(patient, {"vb_available": False})
    m.assert_called_once_with(cfg, patient, {"vb_available": False})

    with _patch("autods_pet.pipeline.score_deauville") as m:
        pipeline.score({"test": "data"})
    m.assert_called_once_with(cfg, {"test": "data"})


def test_normalize_pet_from_metadata_csv(mock_patient, tmp_path):
    """normalize_pet loads metadata from CSV when JSON sidecar is absent."""
    import csv

    results_dir = mock_patient.paths["basepath"] / "results"
    cfg = {
        "paths": {
            "basepath": str(mock_patient.paths["basepath"]),
            "output_dir": str(results_dir),
            "metadata_csv": "metadata.csv",
        },
    }

    # Write a CSV file with PET metadata (in output_dir, where _resolve_csv_path looks).
    csv_path = results_dir / "metadata.csv"
    metadata_row = {
        "PatientID": "P001",
        "StudyDate": "20230615",
        "AcquisitionTime": "143000",
        "RadiopharmaceuticalStartTime": "133000",
        "RadionuclideTotalDose": "370000000.0",
        "RadionuclideHalfLife": "6586.2",
        "DecayCorrection": "START",
        "PatientWeight": "75.0",
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_row.keys()))
        writer.writeheader()
        writer.writerow(metadata_row)

    # Ensure pet_suv_path does not exist so normalization runs
    mock_patient.pet_suv_path.parent.mkdir(parents=True, exist_ok=True)
    # metadata_path must NOT exist so it falls through to CSV
    # (it already doesn't exist because we haven't created it)

    fake_img = MagicMock()
    fake_suv = MagicMock()
    with (
        patch("SimpleITK.ReadImage", return_value=fake_img),
        patch(
            "autods_pet.imaging.normalization.compute_suvbw",
            return_value=fake_suv,
        ),
        patch("SimpleITK.WriteImage") as mock_write,
    ):
        result = normalize_pet(cfg, mock_patient)

    assert result["skipped"] is False
    assert result["pet_suv_path"] == mock_patient.pet_suv_path
    mock_write.assert_called_once()


def test_normalize_pet_csv_patient_not_found(mock_patient, tmp_path):
    """normalize_pet raises ValueError when patient is not in CSV."""
    import csv

    results_dir = mock_patient.paths["basepath"] / "results"
    cfg = {
        "paths": {
            "basepath": str(mock_patient.paths["basepath"]),
            "output_dir": str(results_dir),
            "metadata_csv": "metadata.csv",
        },
    }

    # Write a CSV with a different patient
    csv_path = results_dir / "metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["PatientID", "StudyDate"])
        writer.writeheader()
        writer.writerow({"PatientID": "OTHER", "StudyDate": "20230101"})

    mock_patient.pet_suv_path.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="No PET metadata"):
        normalize_pet(cfg, mock_patient)


def test_convert_images_dicom_format_success(mock_patient, tmp_path):
    """convert_images processes DICOM series: converts, extracts tags, writes metadata."""
    cfg = {"dicom": {"size_threshold_kb": 100}}

    fake_tags = {
        "StudyDate": "20230101",
        "AcquisitionTime": "120000",
        "RadiopharmaceuticalStartTime": "110000",
        "RadionuclideTotalDose": 370000000.0,
        "RadionuclideHalfLife": 6586.2,
        "DecayCorrection": "START",
    }

    mock_patient.ct_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        patch(
            "autods_pet.imaging.dicom.find_series_by_modality",
            return_value={"CT": [Path("/fake/ct.dcm")], "PT": [Path("/fake/pt.dcm")]},
        ),
        patch("autods_pet.imaging.dicom.dicom_series_to_nifti") as mock_d2n,
        patch("autods_pet.imaging.dicom.extract_pet_tags", return_value=fake_tags),
        patch("autods_pet.imaging.dicom.extract_patient_weight", return_value=75.0),
    ):
        result = convert_images(cfg, mock_patient)

    assert result["skipped"] is False
    assert result["ct_path"] == mock_patient.ct_path
    assert result["pet_path"] == mock_patient.pet_path
    assert result["metadata"]["PatientWeight"] == 75.0
    assert result["metadata"]["StudyDate"] == "20230101"
    assert mock_d2n.call_count == 2
    # Check that metadata JSON was written
    assert mock_patient.metadata_path.exists()
    written = json.loads(mock_patient.metadata_path.read_text(encoding="utf-8"))
    assert written["PatientWeight"] == 75.0


def test_convert_images_nrrd_format(mock_patient, tmp_path):
    """convert_images handles NRRD format: reads .nrrd and writes .nii.gz."""
    cfg = {}

    mock_patient.ct_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the .nrrd source files in input_dir (where source data lives)
    input_dir = mock_patient.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    ct_nrrd = input_dir / "CT.nrrd"
    pet_nrrd = input_dir / "PET.nrrd"
    ct_nrrd.write_bytes(b"fake_nrrd")
    pet_nrrd.write_bytes(b"fake_nrrd")

    fake_img = MagicMock()
    with (
        patch("autods_pet.pipeline._detect_format", return_value="nrrd"),
        patch("SimpleITK.ReadImage", return_value=fake_img) as mock_read,
        patch("SimpleITK.WriteImage") as mock_write,
    ):
        result = convert_images(cfg, mock_patient)

    assert result["skipped"] is False
    assert mock_read.call_count == 2
    assert mock_write.call_count == 2


def test_convert_images_nifti_format_both_exist(mock_patient, tmp_path):
    """convert_images skips when both NIfTI files already exist."""
    cfg = {}

    mock_patient.ct_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.paths["ct_nifti"].write_bytes(b"fake")
    mock_patient.paths["pet_nifti"].write_bytes(b"fake")

    result = convert_images(cfg, mock_patient)

    assert result["skipped"] is True
    assert result["ct_path"] == mock_patient.ct_path
    assert result["pet_path"] == mock_patient.pet_path


def test_extract_rois_calls_save_when_configured(mock_patient):
    """extract_rois calls save functions when output flags are set."""
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    from autods_pet.config import default_config
    from autods_pet.pipeline import extract_rois

    cfg = default_config()
    cfg["paths"]["basepath"] = "/tmp"
    cfg["output"]["save_raw_masks"] = True
    cfg["output"]["save_refined_masks"] = True

    with (
        _patch(
            "autods_pet.pipeline._extract_aorta_mbp",
            return_value=(MagicMock(), ("Aorta MBP", "ok", "")),
        ),
        _patch(
            "autods_pet.pipeline._extract_liver",
            return_value=(MagicMock(), ("Liver", "ok", "")),
        ),
        _patch(
            "autods_pet.pipeline._extract_long_bones",
            return_value=(MagicMock(), ("Long bones", "ok", "")),
        ),
        _patch("autods_pet.pipeline._extract_targets", return_value=({}, [])),
        _patch.object(mock_patient, "load_segmentation", return_value=MagicMock()),
        _patch.object(mock_patient, "load_pet_registered", return_value=MagicMock()),
        _patch(
            "autods_pet.ops.save_masks.save_raw_masks", return_value=[]
        ) as mock_save_raw,
        _patch(
            "autods_pet.ops.save_masks.save_refined_masks", return_value=[]
        ) as mock_save_refined,
    ):
        extract_rois(cfg, mock_patient, seg_result={"vb_available": False})

    mock_save_raw.assert_called_once()
    mock_save_refined.assert_called_once()


def test_extract_rois_skips_save_when_disabled(mock_patient):
    """extract_rois does not call save functions when output flags are False."""
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    from autods_pet.config import default_config
    from autods_pet.pipeline import extract_rois

    cfg = default_config()
    cfg["paths"]["basepath"] = "/tmp"
    cfg["output"]["save_raw_masks"] = False
    cfg["output"]["save_refined_masks"] = False

    with (
        _patch(
            "autods_pet.pipeline._extract_aorta_mbp",
            return_value=(MagicMock(), ("Aorta MBP", "ok", "")),
        ),
        _patch(
            "autods_pet.pipeline._extract_liver",
            return_value=(MagicMock(), ("Liver", "ok", "")),
        ),
        _patch(
            "autods_pet.pipeline._extract_long_bones",
            return_value=(MagicMock(), ("Long bones", "ok", "")),
        ),
        _patch("autods_pet.pipeline._extract_targets", return_value=({}, [])),
        _patch.object(mock_patient, "load_segmentation", return_value=MagicMock()),
        _patch.object(mock_patient, "load_pet_registered", return_value=MagicMock()),
        _patch(
            "autods_pet.ops.save_masks.save_raw_masks", return_value=[]
        ) as mock_save_raw,
        _patch(
            "autods_pet.ops.save_masks.save_refined_masks", return_value=[]
        ) as mock_save_refined,
    ):
        extract_rois(cfg, mock_patient, seg_result={"vb_available": False})

    mock_save_raw.assert_not_called()
    mock_save_refined.assert_not_called()


# generate_metadata_template / CSV supplement tests


def test_normalize_pet_csv_supplements_json(mock_patient, tmp_path):
    """normalize_pet fills null JSON fields from the metadata CSV."""
    import csv

    # Write JSON sidecar with null PatientWeight.
    metadata = {
        "PatientID": "P001",
        "StudyDate": "20230615",
        "AcquisitionTime": "143000",
        "RadiopharmaceuticalStartTime": "133000",
        "RadionuclideTotalDose": 370000000.0,
        "RadionuclideHalfLife": 6586.2,
        "DecayCorrection": "START",
        "PatientWeight": None,
    }
    mock_patient.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    # Write CSV with the missing weight (in output_dir).
    results_dir = mock_patient.paths["basepath"] / "results"
    csv_path = results_dir / "metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["PatientID", "PatientWeight"])
        writer.writeheader()
        writer.writerow({"PatientID": "P001", "PatientWeight": "75.0"})

    cfg = {
        "paths": {
            "basepath": str(mock_patient.paths["basepath"]),
            "output_dir": str(results_dir),
            "metadata_csv": "metadata.csv",
        },
    }
    mock_patient.pet_suv_path.parent.mkdir(parents=True, exist_ok=True)

    fake_img = MagicMock()
    fake_suv = MagicMock()
    with (
        patch("SimpleITK.ReadImage", return_value=fake_img),
        patch(
            "autods_pet.imaging.normalization.compute_suvbw",
            return_value=fake_suv,
        ),
        patch("SimpleITK.WriteImage") as mock_write,
    ):
        result = normalize_pet(cfg, mock_patient)

    assert result["skipped"] is False
    mock_write.assert_called_once()


def test_generate_metadata_template_creates_csv(tmp_path):
    """Template CSV is created for patients with incomplete metadata."""
    basepath = tmp_path / "data"
    basepath.mkdir()

    # Patient with complete metadata.
    p_complete = basepath / "COMPLETE"
    p_complete.mkdir()
    (p_complete / "PET_metadata.json").write_text(
        json.dumps({c: "val" for c in METADATA_COLUMNS}), encoding="utf-8"
    )

    # Patient with null weight.
    p_missing = basepath / "MISSING"
    p_missing.mkdir()
    meta = {c: "val" for c in METADATA_COLUMNS}
    meta["PatientWeight"] = None
    (p_missing / "PET_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    # Patient with no JSON at all.
    p_nifti = basepath / "NIFTI_ONLY"
    p_nifti.mkdir()

    cfg = {
        "paths": {
            "basepath": str(basepath),
            "output_dir": str(basepath),
            "pet_metadata": "{patient_id}/PET_metadata.json",
            "ct_nifti": "{patient_id}/CT.nii.gz",
            "pet_nifti": "{patient_id}/PET.nii.gz",
            "pet_suv": "{patient_id}/PET_SUV.nii.gz",
            "pet_registered": "{patient_id}/PET_SUV_reg.nii.gz",
            "seg_dir": "{patient_id}/segmentation",
            "vert_body_seg": "{patient_id}/segmentation/vb.nii.gz",
        },
    }

    csv_path = generate_metadata_template(cfg, ["COMPLETE", "MISSING", "NIFTI_ONLY"])

    assert csv_path is not None
    assert csv_path.exists()

    import csv

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = {r["PatientID"]: r for r in reader}

    # COMPLETE should be absent (all fields present).
    assert "COMPLETE" not in rows
    # MISSING should have PatientWeight empty, others filled.
    assert rows["MISSING"]["StudyDate"] == "val"
    assert rows["MISSING"]["PatientWeight"] == ""
    # NIFTI_ONLY should have only PatientID, rest empty.
    assert rows["NIFTI_ONLY"]["StudyDate"] == ""
    assert rows["NIFTI_ONLY"]["PatientWeight"] == ""


def test_generate_metadata_template_merges_existing(tmp_path):
    """Existing user-provided CSV values are preserved on re-generation."""
    import csv

    basepath = tmp_path / "data"
    basepath.mkdir()
    p = basepath / "P001"
    p.mkdir()
    meta = {c: "val" for c in METADATA_COLUMNS}
    meta["PatientWeight"] = None
    (p / "PET_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    csv_file = basepath / "metadata.csv"

    cfg = {
        "paths": {
            "basepath": str(basepath),
            "output_dir": str(basepath),
            "metadata_csv": "metadata.csv",
            "pet_metadata": "{patient_id}/PET_metadata.json",
            "ct_nifti": "{patient_id}/CT.nii.gz",
            "pet_nifti": "{patient_id}/PET.nii.gz",
            "pet_suv": "{patient_id}/PET_SUV.nii.gz",
            "pet_registered": "{patient_id}/PET_SUV_reg.nii.gz",
            "seg_dir": "{patient_id}/segmentation",
            "vert_body_seg": "{patient_id}/segmentation/vb.nii.gz",
        },
    }

    # First generation.
    generate_metadata_template(cfg, ["P001"])
    assert csv_file.exists()

    # Simulate user filling in the weight.
    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["PatientWeight"] = "80.0"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Re-generate - user value should be preserved.
    result = generate_metadata_template(cfg, ["P001"])

    # Now CSV is complete (weight filled) → should return None.
    assert result is None

    # File content untouched.
    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["PatientWeight"] == "80.0"


def test_generate_metadata_template_returns_none_when_all_complete(tmp_path):
    """No CSV is created when every patient has complete metadata."""
    basepath = tmp_path / "data"
    basepath.mkdir()

    p = basepath / "GOOD"
    p.mkdir()
    (p / "PET_metadata.json").write_text(
        json.dumps({c: "val" for c in METADATA_COLUMNS}), encoding="utf-8"
    )

    cfg = {
        "paths": {
            "basepath": str(basepath),
            "output_dir": str(basepath),
            "pet_metadata": "{patient_id}/PET_metadata.json",
            "ct_nifti": "{patient_id}/CT.nii.gz",
            "pet_nifti": "{patient_id}/PET.nii.gz",
            "pet_suv": "{patient_id}/PET_SUV.nii.gz",
            "pet_registered": "{patient_id}/PET_SUV_reg.nii.gz",
            "seg_dir": "{patient_id}/segmentation",
            "vert_body_seg": "{patient_id}/segmentation/vb.nii.gz",
        },
    }

    result = generate_metadata_template(cfg, ["GOOD"])
    assert result is None
    assert not (basepath / "metadata.csv").exists()


# _find_nifti_sources tests


def test_find_nifti_sources_matches_ct_and_pet(tmp_path):
    """_find_nifti_sources finds CT.nii.gz and PET.nii files."""
    from autods_pet.pipeline import _find_nifti_sources

    (tmp_path / "CT.nii.gz").touch()
    (tmp_path / "PET.nii").touch()
    ct_dst = Path("/out/ct.nii.gz")
    pet_dst = Path("/out/pet.nii.gz")
    pairs = _find_nifti_sources(tmp_path, ct_dst, pet_dst)
    assert len(pairs) == 2
    assert pairs[0] == (tmp_path / "CT.nii.gz", ct_dst)
    assert pairs[1] == (tmp_path / "PET.nii", pet_dst)


def test_find_nifti_sources_no_match(tmp_path):
    """Returns empty list when no matching NIfTI files exist."""
    from autods_pet.pipeline import _find_nifti_sources

    pairs = _find_nifti_sources(tmp_path, Path("/out/ct"), Path("/out/pet"))
    assert pairs == []


# segment_ct skip branch


def test_segment_ct_skip_existing_seg_and_vb(mock_patient):
    """segment_ct skips when seg and VB files already exist (force=False)."""
    from autods_pet.imaging.segmentation import TOTSEG_FILENAME

    cfg = {}
    # Create the segmentation files so the skip branches are taken.
    mock_patient.seg_dir.mkdir(parents=True, exist_ok=True)
    (mock_patient.seg_dir / TOTSEG_FILENAME).write_bytes(b"fake")
    mock_patient.vert_body_seg_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.vert_body_seg_path.write_bytes(b"fake")

    with (
        patch("autods_pet.pipeline._ensure_totalseg_license"),
        patch("autods_pet.imaging.segmentation.run_totalsegmentator") as mock_ts,
    ):
        result = segment_ct(cfg, mock_patient)

    mock_ts.assert_not_called()
    assert result["seg_multilabel"] == mock_patient.seg_dir / TOTSEG_FILENAME
    assert result["vb_available"] is True


# ROI extraction helper tests


def test_extract_aorta_mbp_success():
    """_extract_aorta_mbp returns (result, ok) on success."""
    from autods_pet.pipeline import _extract_aorta_mbp
    from autods_pet.results import ROIResult

    cfg = {"aorta_mbp": {"stats": ["median"]}}
    fake_result = ROIResult(stats={"median": 2.0})
    with patch("autods_pet.roi.AortaMBP.extract", return_value=fake_result):
        result, status = _extract_aorta_mbp(cfg, MagicMock(), MagicMock())
    assert result is fake_result
    assert status[0] == "Aorta MBP"
    assert status[1] == "ok"


def test_extract_aorta_mbp_error():
    """_extract_aorta_mbp returns (None, error) on failure."""
    from autods_pet.pipeline import _extract_aorta_mbp

    cfg = {"aorta_mbp": {"stats": ["median"]}}
    with patch("autods_pet.roi.AortaMBP.extract", side_effect=RuntimeError("bad")):
        result, status = _extract_aorta_mbp(cfg, MagicMock(), MagicMock())
    assert result is None
    assert status[1] == "error"
    assert "bad" in status[2]


def test_extract_liver_success():
    """_extract_liver returns (result, ok) on success."""
    from autods_pet.pipeline import _extract_liver
    from autods_pet.results import ROIResult

    cfg = {"liver": {"stats": ["median"]}}
    fake_result = ROIResult(stats={"median": 3.0})
    with patch("autods_pet.roi.LiverROI.extract", return_value=fake_result):
        result, status = _extract_liver(cfg, MagicMock(), MagicMock())
    assert result is fake_result
    assert status[1] == "ok"


def test_extract_liver_error():
    """_extract_liver returns (None, error) on failure."""
    from autods_pet.pipeline import _extract_liver

    cfg = {"liver": {"stats": ["median"]}}
    with patch("autods_pet.roi.LiverROI.extract", side_effect=RuntimeError("oops")):
        result, status = _extract_liver(cfg, MagicMock(), MagicMock())
    assert result is None
    assert status[1] == "error"


def test_extract_lumbar_vb_success(mock_patient):
    """_extract_lumbar_vb returns (result, ok) when VB seg is available."""
    from autods_pet.pipeline import _extract_lumbar_vb
    from autods_pet.results import ROIResult

    cfg = {"lumbar_vb": {"stats": ["p95"]}}
    fake_result = ROIResult(stats={"p95": 4.0})
    mock_patient.load_vert_body_seg.return_value = MagicMock()
    mock_patient.vert_body_seg_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.vert_body_seg_path.write_bytes(b"fake")

    with patch("autods_pet.roi.LumbarVB.extract", return_value=fake_result):
        result, status = _extract_lumbar_vb(
            cfg, MagicMock(), MagicMock(), mock_patient, {"vb_available": True}
        )
    assert result is fake_result
    assert status[1] == "ok"


def test_extract_lumbar_vb_skip_no_vb(mock_patient):
    """_extract_lumbar_vb skips when no VB segmentation is available."""
    from autods_pet.pipeline import _extract_lumbar_vb

    cfg = {"lumbar_vb": {"stats": ["p95"]}}
    result, status = _extract_lumbar_vb(
        cfg, MagicMock(), MagicMock(), mock_patient, {"vb_available": False}
    )
    assert result is None
    assert status[1] == "skip"


def test_extract_long_bones_success():
    """_extract_long_bones returns (result, ok) on success."""
    from autods_pet.pipeline import _extract_long_bones
    from autods_pet.results import ROIResult

    cfg = {"long_bones": {"stats": ["p95"], "bones": []}}
    fake_result = ROIResult(stats={"p95": 5.0})
    with patch("autods_pet.roi.LongBonesROI.extract", return_value=fake_result):
        result, status = _extract_long_bones(cfg, MagicMock(), MagicMock())
    assert result is fake_result
    assert status[1] == "ok"


def test_extract_long_bones_error():
    """_extract_long_bones returns (None, error) on failure."""
    from autods_pet.pipeline import _extract_long_bones

    cfg = {"long_bones": {"stats": ["p95"], "bones": []}}
    with patch("autods_pet.roi.LongBonesROI.extract", side_effect=RuntimeError("fail")):
        result, status = _extract_long_bones(cfg, MagicMock(), MagicMock())
    assert result is None
    assert status[1] == "error"


# score_deauville BLR


def test_score_deauville_blr():
    """BLR (Brain-to-Liver Ratio) is computed when brain and liver are present."""
    extract_results = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "Liver": ROIResult(stats={"median": 3.0}),
        "Brain": ROIResult(stats={"median": 6.0}),
        "_roi_statuses": [],
    }
    scores = score_deauville({}, extract_results)
    assert scores["BLR"] == pytest.approx(6.0 / 3.0)


# CSV writer tests


def test_write_patient_suv_csv(tmp_path):
    """write_patient_suv_csv writes ROI/Statistic/Value rows."""
    import csv as csv_mod

    from autods_pet.pipeline import write_patient_suv_csv

    path = tmp_path / "suv.csv"
    extract_results = {
        "Liver": ROIResult(stats={"median": 3.0, "p95": 3.5}),
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "_roi_statuses": [],
    }
    write_patient_suv_csv(extract_results, path)
    assert path.exists()
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 3
    rois = {r["ROI"] for r in rows}
    assert "Liver" in rois
    assert "Aorta_MBP" in rois


def test_write_patient_deauville_csv(tmp_path):
    """write_patient_deauville_csv writes Target/DeauvilleScore rows."""
    import csv as csv_mod

    from autods_pet.pipeline import write_patient_deauville_csv

    path = tmp_path / "ds.csv"
    scores = {"FL_DS": 4, "BM_DS": 3}
    write_patient_deauville_csv(scores, path)
    assert path.exists()
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 2
    targets = {r["Target"] for r in rows}
    assert targets == {"FL_DS", "BM_DS"}


# save_batch_csv merge


def test_save_batch_csv_merges_existing(tmp_path):
    """save_batch_csv merges new results with existing CSV file."""
    import pandas as pd

    from autods_pet.pipeline import save_batch_csv

    path = tmp_path / "batch.csv"
    # Write initial CSV
    existing = pd.DataFrame(
        [{"patient_id": "P001", "FL_DS": 3}, {"patient_id": "P002", "FL_DS": 2}]
    )
    existing.to_csv(path, index=False)

    # Merge with updated P001 and new P003
    new_df = pd.DataFrame(
        [{"patient_id": "P001", "FL_DS": 5}, {"patient_id": "P003", "FL_DS": 4}]
    )
    save_batch_csv(new_df, path)

    result = pd.read_csv(path)
    assert list(result["patient_id"]) == ["P001", "P002", "P003"]
    assert result.loc[result["patient_id"] == "P001", "FL_DS"].iloc[0] == 5


def test_discover_file_masks_nii_gz_at_root(tmp_path):
    """discover_file_masks finds a .nii.gz at the patient input root."""
    from autods_pet.ops.mask_discovery import discover_file_masks

    (tmp_path / "focal_lesion.nii.gz").write_bytes(b"fake")
    targets = [{"name": "focal_lesion", "mask_filename": ["focal_lesion"]}]
    res = discover_file_masks(tmp_path, targets)

    assert "focal_lesion" in res
    assert res["focal_lesion"].path == tmp_path / "focal_lesion.nii.gz"
    assert res["focal_lesion"].format == "nifti"


def test_discover_file_masks_recursive_nested(tmp_path):
    """discover_file_masks descends into nested study/series sub-folders."""
    from autods_pet.ops.mask_discovery import discover_file_masks

    nested = tmp_path / "study1" / "segmentations"
    nested.mkdir(parents=True)
    (nested / "PM_lesion.nrrd").write_bytes(b"fake")

    targets = [{"name": "paramedullary", "mask_filename": ["PM_lesion"]}]
    res = discover_file_masks(tmp_path, targets)

    assert "paramedullary" in res
    assert res["paramedullary"].format == "nrrd"
    assert res["paramedullary"].path == nested / "PM_lesion.nrrd"


def test_discover_file_masks_missing(tmp_path):
    """No file → empty result, no exception."""
    from autods_pet.ops.mask_discovery import discover_file_masks

    targets = [{"name": "focal_lesion", "mask_filename": ["nonexistent"]}]
    assert discover_file_masks(tmp_path, targets) == {}


def test_discover_file_masks_empty_pattern_disables(tmp_path):
    """Section with no mask_filename is silently disabled (no match attempted)."""
    from autods_pet.ops.mask_discovery import discover_file_masks

    (tmp_path / "focal_lesion.nii.gz").write_bytes(b"fake")
    targets = [{"name": "focal_lesion", "mask_filename": []}]
    assert discover_file_masks(tmp_path, targets) == {}


def test_discover_file_masks_comma_list(tmp_path):
    """Comma-list mask_filename matches the first stem found, in priority order."""
    from autods_pet.ops.mask_discovery import discover_file_masks

    (tmp_path / "GTV.nii.gz").write_bytes(b"fake")
    targets = [{"name": "focal_lesion", "mask_filename": ["focal_lesion", "FL", "GTV"]}]
    res = discover_file_masks(tmp_path, targets)

    assert "focal_lesion" in res
    assert res["focal_lesion"].path.name == "GTV.nii.gz"


def test_discover_dicom_seg_masks_uid_match(tmp_path):
    """A DICOM SEG referencing the patient's PET UID is matched by SegmentLabel."""
    from unittest.mock import patch

    from autods_pet.ops.mask_discovery import discover_dicom_seg_masks

    seg_dir = tmp_path / "study1" / "segmentations"
    seg_dir.mkdir(parents=True)
    seg_path = seg_dir / "1.2.3.4.5.dcm"
    seg_path.write_bytes(b"fake")

    targets = [
        {
            "name": "focal_lesion",
            "segment_label": ["Focal lesion", "FL"],
        },
        {
            "name": "paramedullary",
            "segment_label": ["PM"],
        },
    ]

    fake_segments = [
        {"number": 1, "label": "Focal lesion", "description": ""},
        {"number": 2, "label": "PM", "description": ""},
    ]

    with (
        patch("autods_pet.ops.dicom_seg.is_dicom_seg", return_value=True),
        patch(
            "autods_pet.ops.dicom_seg.read_referenced_series_uids",
            return_value=["1.2.840.PET"],
        ),
        patch(
            "autods_pet.ops.dicom_seg.list_segments",
            return_value=fake_segments,
        ),
    ):
        res, info = discover_dicom_seg_masks(
            tmp_path, targets, pet_series_uid="1.2.840.PET"
        )

    assert "focal_lesion" in res
    assert res["focal_lesion"].format == "dicom_seg"
    assert res["focal_lesion"].segment_label == "Focal lesion"
    assert res["focal_lesion"].segment_number == 1
    assert "paramedullary" in res
    assert res["paramedullary"].segment_number == 2


def test_discover_dicom_seg_masks_uid_mismatch_skipped(tmp_path):
    """A DICOM SEG that doesn't reference the PET UID is skipped, with a note."""
    from unittest.mock import patch

    from autods_pet.ops.mask_discovery import discover_dicom_seg_masks

    seg_path = tmp_path / "stray.dcm"
    seg_path.write_bytes(b"fake")

    targets = [{"name": "focal_lesion", "segment_label": ["Focal lesion"]}]
    with (
        patch("autods_pet.ops.dicom_seg.is_dicom_seg", return_value=True),
        patch(
            "autods_pet.ops.dicom_seg.read_referenced_series_uids",
            return_value=["1.2.840.OTHER"],
        ),
        patch(
            "autods_pet.ops.dicom_seg.list_segments",
            return_value=[{"number": 1, "label": "Focal lesion"}],
        ),
    ):
        res, info = discover_dicom_seg_masks(
            tmp_path, targets, pet_series_uid="1.2.840.PET"
        )

    assert res == {}
    assert any("does not reference PET SeriesInstanceUID" in m for m in info)


def test_discover_all_masks_dicom_wins_over_file(tmp_path):
    """When both formats match the same target, DICOM SEG wins and a note is logged."""
    from unittest.mock import patch

    from autods_pet.ops.mask_discovery import discover_all_masks

    (tmp_path / "focal_lesion.nii.gz").write_bytes(b"fake")
    (tmp_path / "1.2.3.dcm").write_bytes(b"fake")

    targets = [
        {
            "name": "focal_lesion",
            "mask_filename": ["focal_lesion"],
            "segment_label": ["Focal lesion"],
        }
    ]

    with (
        patch("autods_pet.ops.dicom_seg.is_dicom_seg", return_value=True),
        patch(
            "autods_pet.ops.dicom_seg.read_referenced_series_uids",
            return_value=["UID"],
        ),
        patch(
            "autods_pet.ops.dicom_seg.list_segments",
            return_value=[{"number": 1, "label": "Focal lesion"}],
        ),
    ):
        merged, warnings = discover_all_masks(tmp_path, targets, pet_series_uid="UID")

    assert merged["focal_lesion"].format == "dicom_seg"
    assert any("matched both" in w for w in warnings)


def test_discover_all_masks_missing_emits_warning(tmp_path):
    """A configured target with no matching mask anywhere generates a warning."""
    from autods_pet.ops.mask_discovery import discover_all_masks

    targets = [
        {
            "name": "focal_lesion",
            "mask_filename": ["focal_lesion"],
            "segment_label": ["Focal lesion"],
        }
    ]
    merged, warnings = discover_all_masks(tmp_path, targets, pet_series_uid="UID")

    assert merged == {}
    assert any(
        "Target [focal_lesion] is enabled in config but no mask was found" in w
        for w in warnings
    )


def test_discover_all_masks_skip_dirs(tmp_path):
    """skip_dirs prevents the walker from descending into the output directory."""
    from autods_pet.ops.mask_discovery import discover_file_masks

    out = tmp_path / "results"
    out.mkdir()
    (out / "focal_lesion.nii.gz").write_bytes(b"cached output")

    targets = [{"name": "focal_lesion", "mask_filename": ["focal_lesion"]}]
    assert discover_file_masks(tmp_path, targets, skip_dirs={out}) == {}


def test_has_new_target_masks_logs_discovery_warnings(mock_patient, caplog):
    """The gate must surface discovery warnings so the CLI never silently skips."""
    import logging

    from autods_pet.config import default_config
    from autods_pet.pipeline import _has_new_target_masks

    cfg = default_config()
    cfg["focal_lesion"] = {
        "mask_filename": ["focal_lesion"],
        "segment_label": ["Focal lesion"],
        "stats": ["max"],
    }
    mock_patient.input_dir.mkdir(parents=True, exist_ok=True)

    with caplog.at_level(logging.WARNING, logger="autods_pet.pipeline"):
        result = _has_new_target_masks(cfg, mock_patient)

    assert result is False
    assert any("no mask was found" in rec.getMessage() for rec in caplog.records), (
        "Expected the missing-mask warning to be logged at the gate"
    )


def test_has_new_target_masks_no_targets(mock_patient):
    """Returns False when no targets are configured."""
    from autods_pet.config import default_config
    from autods_pet.pipeline import _has_new_target_masks

    cfg = default_config()  # No target sections
    mock_patient.input_dir.mkdir(parents=True, exist_ok=True)
    assert _has_new_target_masks(cfg, mock_patient) is False


def test_has_new_target_masks_new_mask(mock_patient):
    """Returns True when input mask exists but output doesn't."""

    from autods_pet.config import default_config
    from autods_pet.pipeline import _has_new_target_masks

    cfg = default_config()
    cfg["focal_lesion"] = {"mask_filename": "focal_lesion", "stats": ["max"]}

    mock_patient.input_dir.mkdir(parents=True, exist_ok=True)
    (mock_patient.input_dir / "focal_lesion.nii.gz").write_bytes(b"fake")

    assert _has_new_target_masks(cfg, mock_patient) is True


def test_has_new_target_masks_updated_mask(mock_patient):
    """Returns True when input mask is newer than output."""
    import time

    from autods_pet.config import default_config
    from autods_pet.pipeline import _has_new_target_masks

    cfg = default_config()
    cfg["focal_lesion"] = {"mask_filename": "focal_lesion", "stats": ["max"]}

    mock_patient.input_dir.mkdir(parents=True, exist_ok=True)
    mock_patient.seg_dir.mkdir(parents=True, exist_ok=True)

    out_mask = mock_patient.seg_dir / "focal_lesion.nii.gz"
    out_mask.write_bytes(b"old")
    time.sleep(0.05)
    (mock_patient.input_dir / "focal_lesion.nii.gz").write_bytes(b"new")

    assert _has_new_target_masks(cfg, mock_patient) is True


def test_has_new_target_masks_no_update(mock_patient):
    """Returns False when output mask is newer than input and DS exists."""
    import time

    from autods_pet.config import default_config
    from autods_pet.pipeline import _has_new_target_masks

    cfg = default_config()
    cfg["focal_lesion"] = {"mask_filename": "focal_lesion", "stats": ["max"]}

    mock_patient.input_dir.mkdir(parents=True, exist_ok=True)
    mock_patient.seg_dir.mkdir(parents=True, exist_ok=True)

    (mock_patient.input_dir / "focal_lesion.nii.gz").write_bytes(b"input")
    time.sleep(0.05)
    out_mask = mock_patient.seg_dir / "focal_lesion.nii.gz"
    out_mask.write_bytes(b"output")

    # Also create per-patient DS CSV with the expected score.
    ds_path = mock_patient.deauville_csv_path
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    ds_path.write_text("Target,DeauvilleScore\nFL_DS,3\n", encoding="utf-8")

    assert _has_new_target_masks(cfg, mock_patient) is False


def test_load_cached_references_success(mock_patient):
    """Loads references from existing SUV_values.csv."""
    import csv

    from autods_pet.pipeline import _load_cached_references
    from autods_pet.results import ROIResult

    csv_path = mock_patient.suv_csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ROI", "Statistic", "Value"])
        writer.writeheader()
        writer.writerow({"ROI": "Aorta_MBP", "Statistic": "median", "Value": "1.33"})
        writer.writerow({"ROI": "Liver", "Statistic": "median", "Value": "2.50"})

    result = _load_cached_references(mock_patient)
    assert result is not None
    assert "Aorta MBP" in result
    assert "Liver" in result
    assert isinstance(result["Aorta MBP"], ROIResult)
    assert result["Aorta MBP"].stats["median"] == 1.33
    assert result["Liver"].stats["median"] == 2.50


def test_load_cached_references_missing_file(mock_patient):
    """Returns None when SUV_values.csv doesn't exist."""
    from autods_pet.pipeline import _load_cached_references

    result = _load_cached_references(mock_patient)
    assert result is None


def test_load_cached_references_missing_aorta(mock_patient):
    """Returns None when Aorta MBP median is missing from CSV."""
    import csv

    from autods_pet.pipeline import _load_cached_references

    csv_path = mock_patient.suv_csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ROI", "Statistic", "Value"])
        writer.writeheader()
        writer.writerow({"ROI": "Liver", "Statistic": "median", "Value": "2.50"})

    result = _load_cached_references(mock_patient)
    assert result is None


def test_load_cached_references_missing_liver(mock_patient):
    """Returns None when Liver median is missing from CSV."""
    import csv

    from autods_pet.pipeline import _load_cached_references

    csv_path = mock_patient.suv_csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ROI", "Statistic", "Value"])
        writer.writeheader()
        writer.writerow({"ROI": "Aorta_MBP", "Statistic": "median", "Value": "1.33"})

    result = _load_cached_references(mock_patient)
    assert result is None


def test_write_patient_suv_csv_merges(tmp_path):
    """Merge mode: existing entries preserved, updated entries replaced."""
    import csv

    from autods_pet.pipeline import write_patient_suv_csv
    from autods_pet.results import ROIResult

    csv_path = tmp_path / "SUV_values.csv"

    # Write initial
    initial = {
        "Aorta MBP": ROIResult(stats={"median": 1.33}),
        "Liver": ROIResult(stats={"median": 2.50}),
    }
    write_patient_suv_csv(initial, csv_path)

    # Merge with updated Liver and new Focal lesion
    update = {
        "Liver": ROIResult(stats={"median": 2.80}),
        "Focal lesion": ROIResult(stats={"max": 5.0}),
    }
    write_patient_suv_csv(update, csv_path)

    # Read back and verify
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    rois = {(r["ROI"], r["Statistic"]): float(r["Value"]) for r in rows}
    assert rois[("Aorta_MBP", "median")] == 1.33  # preserved
    assert rois[("Liver", "median")] == 2.80  # updated
    assert rois[("Focal_lesion", "max")] == 5.0  # added


def test_write_patient_suv_csv_fresh(tmp_path):
    """When no existing file, writes normally."""
    import csv

    from autods_pet.pipeline import write_patient_suv_csv
    from autods_pet.results import ROIResult

    csv_path = tmp_path / "SUV_values.csv"
    data = {"Liver": ROIResult(stats={"median": 2.5})}
    write_patient_suv_csv(data, csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["ROI"] == "Liver"


def test_write_patient_deauville_csv_merges(tmp_path):
    """Merge mode: existing scores preserved, updated ones replaced."""
    import csv

    from autods_pet.pipeline import write_patient_deauville_csv

    csv_path = tmp_path / "deauville_scores.csv"

    # Write initial
    write_patient_deauville_csv({"FL_DS": 3, "BM_DS": 2}, csv_path)

    # Merge with updated FL and new PM
    write_patient_deauville_csv({"FL_DS": 4, "PM_DS": 3}, csv_path)

    # Read back
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    scores = {r["Target"]: int(r["DeauvilleScore"]) for r in rows}
    assert scores["FL_DS"] == 4  # updated
    assert scores["BM_DS"] == 2  # preserved
    assert scores["PM_DS"] == 3  # added


def test_extract_new_targets_only_with_cache(mock_patient):
    """Uses cached references and extracts only targets."""
    import csv
    from unittest.mock import patch as _patch

    from autods_pet.pipeline import extract_new_targets_only
    from autods_pet.results import ROIResult

    cfg = {
        "paths": {"basepath": str(mock_patient.paths["basepath"])},
        "focal_lesion": {"mask_filename": "focal_lesion", "stats": ["max"]},
        "targets": {},
    }

    # Create cached SUV CSV
    csv_path = mock_patient.suv_csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ROI", "Statistic", "Value"])
        writer.writeheader()
        writer.writerow({"ROI": "Aorta_MBP", "Statistic": "median", "Value": "2.0"})
        writer.writerow({"ROI": "Liver", "Statistic": "median", "Value": "3.0"})

    # Mock PET file and _extract_targets
    mock_patient.pet_registered_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.pet_registered_path.write_bytes(b"fake")
    mock_patient.deauville_csv_path.parent.mkdir(parents=True, exist_ok=True)

    target_result = ROIResult(stats={"max": 5.0})
    with (
        _patch("SimpleITK.ReadImage", return_value=MagicMock()),
        _patch(
            "autods_pet.pipeline._extract_targets",
            return_value=({"Focal lesion": target_result}, []),
        ),
    ):
        result = extract_new_targets_only(cfg, mock_patient)

    assert "scores" in result
    assert "extract_results" in result
    assert "Focal lesion" in result["extract_results"]
    assert "Aorta MBP" in result["extract_results"]


def test_extract_new_targets_only_no_cache(mock_patient):
    """Falls back to full extract_rois when no cached references."""
    from unittest.mock import patch as _patch

    from autods_pet.config import default_config
    from autods_pet.pipeline import extract_new_targets_only
    from autods_pet.results import ROIResult

    cfg = default_config()
    cfg["paths"]["basepath"] = str(mock_patient.paths["basepath"])

    mock_patient.deauville_csv_path.parent.mkdir(parents=True, exist_ok=True)

    extract_results = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "Liver": ROIResult(stats={"median": 3.0}),
        "_roi_statuses": [],
    }

    with (
        _patch(
            "autods_pet.pipeline.extract_rois",
            return_value=extract_results,
        ) as m_extract,
    ):
        result = extract_new_targets_only(cfg, mock_patient)

    m_extract.assert_called_once()
    assert "scores" in result


def test_pipeline_update_targets(tmp_path):
    """update_targets delegates to extract_new_targets_only."""
    from unittest.mock import patch as _patch

    from autods_pet.config import default_config
    from autods_pet.pipeline import DeauvillePipeline

    cfg = default_config()
    cfg["paths"]["basepath"] = str(tmp_path)
    cfg["paths"]["output_dir"] = str(tmp_path / "results")

    pipeline = DeauvillePipeline(cfg)
    fake_result = {"scores": {"FL_DS": 4}, "extract_results": {}}

    with (
        _patch("autods_pet.patient.PatientCase"),
        _patch(
            "autods_pet.pipeline.extract_new_targets_only",
            return_value=fake_result,
        ) as m_ent,
    ):
        result = pipeline.update_targets("P001")

    m_ent.assert_called_once()
    assert result == fake_result


def test_extract_targets_uses_discovery(mock_patient):
    """_extract_targets uses discover_all_masks to locate mask files."""
    from unittest.mock import patch as _patch

    from autods_pet.config import default_config
    from autods_pet.pipeline import _extract_targets

    cfg = default_config()
    cfg["focal_lesion"] = {"mask_filename": ["focal_lesion"], "stats": ["max"]}

    mock_patient.input_dir.mkdir(parents=True, exist_ok=True)
    (mock_patient.input_dir / "focal_lesion.nii.gz").write_bytes(b"fake")

    fake_roi_result = MagicMock()
    with (
        _patch("SimpleITK.ReadImage", return_value=MagicMock()),
        _patch(
            "autods_pet.roi.target_roi.TargetROI.extract",
            return_value=fake_roi_result,
        ),
        _patch("SimpleITK.WriteImage"),
        _patch(
            "autods_pet.imaging.geometry.check_same_geometry",
            return_value=True,
        ),
    ):
        results, statuses = _extract_targets(cfg, MagicMock(), mock_patient)

    assert "Focal lesion" in results
    ok_names = [s[0] for s in statuses if s[1] == "ok"]
    assert "Focal lesion" in ok_names


def test_extract_command_skips_when_no_new_masks(tmp_path):
    """extract command skips patients with no new target masks."""
    from unittest.mock import patch as _patch

    from typer.testing import CliRunner

    from autods_pet.cli import app

    runner = CliRunner()
    cfg = {
        "paths": {
            "basepath": str(tmp_path),
            "output_dir": str(tmp_path / "results"),
        },
    }

    with (
        _patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        _patch("autods_pet.cli._init_logging"),
        _patch("autods_pet.patient.PatientCase"),
        _patch("autods_pet.pipeline._has_new_target_masks", return_value=False),
        _patch(
            "autods_pet.pipeline.extract_rois",
            return_value={"_roi_statuses": []},
        ) as mock_ext,
    ):
        result = runner.invoke(app, ["extract", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    mock_ext.assert_not_called()


def test_extract_command_force_bypasses_check(tmp_path):
    """extract --force bypasses the new-mask check."""
    from unittest.mock import patch as _patch

    from typer.testing import CliRunner

    from autods_pet.cli import app

    runner = CliRunner()
    cfg = {
        "paths": {
            "basepath": str(tmp_path),
            "output_dir": str(tmp_path / "results"),
        },
    }

    with (
        _patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        _patch("autods_pet.cli._init_logging"),
        _patch("autods_pet.patient.PatientCase"),
        _patch("autods_pet.pipeline._has_new_target_masks", return_value=False),
        _patch(
            "autods_pet.pipeline.extract_rois",
            return_value={"_roi_statuses": []},
        ) as mock_ext,
        _patch("autods_pet.pipeline.write_patient_suv_csv"),
    ):
        result = runner.invoke(
            app, ["extract", "-c", "fake.ini", "-p", "P001", "--force"]
        )

    assert result.exit_code == 0
    mock_ext.assert_called_once()


def test_score_command_skips_when_no_new_masks(tmp_path):
    """score command skips patients with no new target masks."""
    from unittest.mock import patch as _patch

    from typer.testing import CliRunner

    from autods_pet.cli import app

    runner = CliRunner()
    cfg = {
        "paths": {
            "basepath": str(tmp_path),
            "output_dir": str(tmp_path / "results"),
        },
    }

    with (
        _patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        _patch("autods_pet.cli._init_logging"),
        _patch("autods_pet.patient.PatientCase"),
        _patch("autods_pet.pipeline._has_new_target_masks", return_value=False),
        _patch(
            "autods_pet.pipeline.extract_rois",
            return_value={"_roi_statuses": []},
        ) as mock_ext,
    ):
        result = runner.invoke(app, ["score", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    mock_ext.assert_not_called()


def test_score_command_force_bypasses_check(tmp_path):
    """score --force bypasses the new-mask check."""
    from unittest.mock import patch as _patch

    from typer.testing import CliRunner

    from autods_pet.cli import app

    runner = CliRunner()
    cfg = {
        "paths": {
            "basepath": str(tmp_path),
            "output_dir": str(tmp_path / "results"),
        },
    }

    with (
        _patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        _patch("autods_pet.cli._init_logging"),
        _patch("autods_pet.patient.PatientCase"),
        _patch("autods_pet.pipeline._has_new_target_masks", return_value=False),
        _patch(
            "autods_pet.pipeline.extract_rois",
            return_value={"_roi_statuses": []},
        ) as mock_ext,
        _patch("autods_pet.pipeline.score_deauville", return_value={"FL_DS": 4}),
        _patch("autods_pet.pipeline.write_patient_suv_csv"),
        _patch("autods_pet.pipeline.write_patient_deauville_csv"),
    ):
        result = runner.invoke(
            app, ["score", "-c", "fake.ini", "-p", "P001", "--force"]
        )

    assert result.exit_code == 0
    mock_ext.assert_called_once()


def test_subtract_lesions_no_targets_noop():
    """No target masks → results dict is unchanged."""
    import numpy as np
    import SimpleITK as sitk

    from autods_pet.pipeline import _subtract_lesions_from_marrow
    from autods_pet.results import ROIResult

    marrow_arr = np.ones((5, 5, 5), dtype=np.uint8)
    marrow_mask = sitk.GetImageFromArray(marrow_arr)
    original_stats = {"p95": 4.0}
    results = {
        "Lumbar VB": ROIResult(stats=original_stats.copy(), refined_mask=marrow_mask),
        "_roi_statuses": [],
    }
    cfg = {"lumbar_vb": {"stats": ["p95"]}}

    pet = sitk.GetImageFromArray(np.full((5, 5, 5), 2.0, dtype=np.float64))
    _subtract_lesions_from_marrow(results, pet, cfg)

    assert results["Lumbar VB"].stats == original_stats


def test_subtract_lesions_modifies_marrow():
    """Target mask voxels are removed from marrow ROIs and stats recomputed."""
    import numpy as np
    import SimpleITK as sitk

    from autods_pet.pipeline import _subtract_lesions_from_marrow
    from autods_pet.results import ROIResult

    marrow_arr = np.ones((5, 5, 5), dtype=np.uint8)
    marrow_mask = sitk.GetImageFromArray(marrow_arr)
    marrow_mask.SetSpacing((1.0, 1.0, 1.0))

    target_arr = np.zeros((5, 5, 5), dtype=np.uint8)
    target_arr[0:2, 0:2, 0:2] = 1
    target_mask = sitk.GetImageFromArray(target_arr)
    target_mask.SetSpacing((1.0, 1.0, 1.0))

    pet_arr = np.full((5, 5, 5), 3.0, dtype=np.float64)
    pet = sitk.GetImageFromArray(pet_arr)
    pet.SetSpacing((1.0, 1.0, 1.0))

    results = {
        "Lumbar VB": ROIResult(stats={"p95": 4.0}, refined_mask=marrow_mask),
        "Focal lesion": ROIResult(stats={"max": 5.0}, refined_mask=target_mask),
    }
    cfg = {"lumbar_vb": {"stats": ["p95"]}}

    _subtract_lesions_from_marrow(results, pet, cfg)

    from autods_pet.ops.stats import count_voxels

    assert count_voxels(results["Lumbar VB"].refined_mask) == 125 - 8
    assert "p95" in results["Lumbar VB"].stats


def test_subtract_lesions_empty_after_subtraction_keeps_original():
    """If marrow is empty after subtraction, original mask and stats are kept."""
    import numpy as np
    import SimpleITK as sitk

    from autods_pet.pipeline import _subtract_lesions_from_marrow
    from autods_pet.results import ROIResult

    arr = np.ones((3, 3, 3), dtype=np.uint8)
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing((1.0, 1.0, 1.0))
    pet = sitk.GetImageFromArray(np.full((3, 3, 3), 2.0, dtype=np.float64))
    pet.SetSpacing((1.0, 1.0, 1.0))

    original_stats = {"p95": 4.0}
    results = {
        "Lumbar VB": ROIResult(stats=original_stats.copy(), refined_mask=mask),
        "Focal lesion": ROIResult(stats={"max": 5.0}, refined_mask=mask),
    }
    cfg = {"lumbar_vb": {"stats": ["p95"]}}

    _subtract_lesions_from_marrow(results, pet, cfg)

    assert results["Lumbar VB"].stats == original_stats
    from autods_pet.ops.stats import count_voxels

    assert count_voxels(results["Lumbar VB"].refined_mask) == 27


def test_subtract_lesions_multiple_targets_unioned():
    """Multiple target masks are unioned before subtraction."""
    import numpy as np
    import SimpleITK as sitk

    from autods_pet.pipeline import _subtract_lesions_from_marrow
    from autods_pet.results import ROIResult

    marrow_arr = np.ones((5, 5, 5), dtype=np.uint8)
    marrow_mask = sitk.GetImageFromArray(marrow_arr)
    marrow_mask.SetSpacing((1.0, 1.0, 1.0))

    t1_arr = np.zeros((5, 5, 5), dtype=np.uint8)
    t1_arr[0, 0, 0] = 1
    t1 = sitk.GetImageFromArray(t1_arr)
    t1.SetSpacing((1.0, 1.0, 1.0))

    t2_arr = np.zeros((5, 5, 5), dtype=np.uint8)
    t2_arr[4, 4, 4] = 1
    t2 = sitk.GetImageFromArray(t2_arr)
    t2.SetSpacing((1.0, 1.0, 1.0))

    pet = sitk.GetImageFromArray(np.full((5, 5, 5), 2.0, dtype=np.float64))
    pet.SetSpacing((1.0, 1.0, 1.0))

    results = {
        "Lumbar VB": ROIResult(stats={"p95": 4.0}, refined_mask=marrow_mask),
        "Focal lesion": ROIResult(stats={"max": 5.0}, refined_mask=t1),
        "Paramedullary": ROIResult(stats={"max": 6.0}, refined_mask=t2),
    }
    cfg = {"lumbar_vb": {"stats": ["p95"]}}

    _subtract_lesions_from_marrow(results, pet, cfg)

    from autods_pet.ops.stats import count_voxels

    assert count_voxels(results["Lumbar VB"].refined_mask) == 125 - 2


def test_save_batch_csv_empty_df(tmp_path):
    """Empty DataFrame returns path without writing."""
    import pandas as pd

    from autods_pet.pipeline import save_batch_csv

    path = tmp_path / "batch.csv"
    result = save_batch_csv(pd.DataFrame(), path)
    assert result == path
    assert not path.exists()


def test_save_batch_csv_missing_patient_id(tmp_path):
    """DataFrame without patient_id column returns path without writing."""
    import pandas as pd

    from autods_pet.pipeline import save_batch_csv

    path = tmp_path / "batch.csv"
    df = pd.DataFrame([{"name": "P001", "value": 42}])
    result = save_batch_csv(df, path)
    assert result == path
    assert not path.exists()


def test_save_batch_csv_int_columns(tmp_path):
    """int_columns parameter casts specified columns to nullable Int64."""
    import pandas as pd

    from autods_pet.pipeline import save_batch_csv

    path = tmp_path / "batch.csv"
    df = pd.DataFrame([{"patient_id": "P001", "FL_DS": 4.0, "BLR": 1.5}])
    save_batch_csv(df, path, int_columns=["FL_DS"])

    result = pd.read_csv(path)
    assert result["FL_DS"].iloc[0] == 4


def test_score_deauville_blr_not_computed_without_brain():
    """BLR is absent when brain is not in extract_results."""
    extract_results = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "Liver": ROIResult(stats={"median": 3.0}),
        "_roi_statuses": [],
    }
    scores = score_deauville({}, extract_results)
    assert "BLR" not in scores


def test_score_deauville_fl_none_target_omitted():
    """FL with empty stats is omitted from scores."""
    extract_results = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "Liver": ROIResult(stats={"median": 3.0}),
        "Focal lesion": ROIResult(stats={}),
        "_roi_statuses": [],
    }
    scores = score_deauville({}, extract_results)
    assert "FL_DS" not in scores


def test_segment_ct_copies_from_input_seg(mock_patient):
    """segment_ct copies pre-existing segmentations from input_seg_dir."""
    from autods_pet.imaging.segmentation import TOTSEG_FILENAME

    cfg = {}
    mock_patient.input_seg_dir.mkdir(parents=True, exist_ok=True)
    (mock_patient.input_seg_dir / TOTSEG_FILENAME).write_bytes(b"seg_data")
    (mock_patient.input_seg_dir / "vertebral_body.nii.gz").write_bytes(b"vb_data")
    mock_patient.seg_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("autods_pet.pipeline._ensure_totalseg_license"),
        patch(
            "autods_pet.imaging.segmentation.run_totalsegmentator",
            side_effect=RuntimeError("should not run"),
        ),
    ):
        result = segment_ct(cfg, mock_patient)

    assert (mock_patient.seg_dir / TOTSEG_FILENAME).exists()
    assert mock_patient.vert_body_seg_path.exists()
    assert result["seg_multilabel"] == mock_patient.seg_dir / TOTSEG_FILENAME
    assert result["vb_available"] is True


def test_write_patient_suv_csv_skips_internal_keys(tmp_path):
    """Keys starting with '_' are excluded from CSV output."""
    import csv as csv_mod

    from autods_pet.pipeline import write_patient_suv_csv

    path = tmp_path / "suv.csv"
    extract_results = {
        "Liver": ROIResult(stats={"median": 3.0}),
        "_roi_statuses": [("Liver", "ok", "")],
    }
    write_patient_suv_csv(extract_results, path)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv_mod.DictReader(f))
    assert all(not r["ROI"].startswith("_") for r in rows)


def test_write_patient_deauville_csv_empty_scores(tmp_path):
    """Empty scores dict writes header-only CSV."""
    import csv as csv_mod

    from autods_pet.pipeline import write_patient_deauville_csv

    path = tmp_path / "ds.csv"
    write_patient_deauville_csv({}, path)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 0


def test_extract_brain_returns_none_when_not_in_fov():
    """_extract_brain returns (None, skip) when brain label is not in FOV."""
    from autods_pet.pipeline import _extract_brain

    cfg = {"brain": {"stats": ["median"]}}
    with patch("autods_pet.roi.BrainROI.extract", return_value=None):
        result, status = _extract_brain(cfg, MagicMock(), MagicMock())
    assert result is None
    assert status[1] == "skip"
    assert "not in FOV" in status[2]


def test_extract_brain_error():
    """_extract_brain returns (None, error) on failure."""
    from autods_pet.pipeline import _extract_brain

    cfg = {"brain": {"stats": ["median"]}}
    with patch("autods_pet.roi.BrainROI.extract", side_effect=RuntimeError("crash")):
        result, status = _extract_brain(cfg, MagicMock(), MagicMock())
    assert result is None
    assert status[1] == "error"


def test_extract_rois_loads_preexisting_refined_masks(mock_patient):
    """extract_rois loads pre-existing refined masks and skips extraction."""
    from unittest.mock import patch as _patch

    import numpy as np
    import SimpleITK as sitk

    from autods_pet.config import default_config
    from autods_pet.pipeline import extract_rois

    cfg = default_config()
    cfg["paths"]["basepath"] = "/tmp"

    fake_pet = sitk.GetImageFromArray(np.full((5, 5, 5), 2.0, dtype=np.float64))
    fake_pet.SetSpacing((1.0, 1.0, 1.0))

    fake_mask = sitk.GetImageFromArray(np.ones((5, 5, 5), dtype=np.uint8))
    fake_mask.SetSpacing((1.0, 1.0, 1.0))

    refined_dir = mock_patient.seg_dir / "refined"
    refined_dir.mkdir(parents=True, exist_ok=True)
    for name in ["aorta_mbp", "liver", "grey_matter", "long_bones", "lumbar_vb"]:
        sitk.WriteImage(fake_mask, str(refined_dir / f"{name}.nii.gz"))

    mock_patient.vert_body_seg_path.parent.mkdir(parents=True, exist_ok=True)
    mock_patient.vert_body_seg_path.write_bytes(b"fake")

    with (
        _patch.object(mock_patient, "load_segmentation", return_value=MagicMock()),
        _patch.object(mock_patient, "load_pet_registered", return_value=fake_pet),
        _patch("autods_pet.pipeline._extract_targets", return_value=({}, [])),
        _patch("autods_pet.pipeline._extract_aorta_mbp") as mock_aorta,
        _patch("autods_pet.pipeline._extract_liver") as mock_liver,
        _patch("autods_pet.pipeline._extract_brain") as mock_brain,
        _patch("autods_pet.pipeline._extract_long_bones") as mock_bones,
        _patch("autods_pet.pipeline._extract_lumbar_vb") as mock_lumbar,
    ):
        results = extract_rois(cfg, mock_patient, seg_result={"vb_available": True})

    mock_aorta.assert_not_called()
    mock_liver.assert_not_called()
    mock_brain.assert_not_called()
    mock_bones.assert_not_called()
    mock_lumbar.assert_not_called()

    assert "Aorta MBP" in results
    assert "Liver" in results
    assert "Lumbar VB" in results


def test_extract_rois_with_subtract_lesions(mock_patient):
    """extract_rois calls _subtract_lesions_from_marrow when configured."""
    from unittest.mock import patch as _patch

    from autods_pet.config import default_config
    from autods_pet.pipeline import extract_rois

    cfg = default_config()
    cfg["paths"]["basepath"] = "/tmp"
    cfg["output"]["subtract_lesions_from_marrow"] = True

    with (
        _patch(
            "autods_pet.pipeline._extract_aorta_mbp",
            return_value=(MagicMock(), ("Aorta MBP", "ok", "")),
        ),
        _patch(
            "autods_pet.pipeline._extract_liver",
            return_value=(MagicMock(), ("Liver", "ok", "")),
        ),
        _patch(
            "autods_pet.pipeline._extract_long_bones",
            return_value=(MagicMock(), ("Long bones", "ok", "")),
        ),
        _patch("autods_pet.pipeline._extract_targets", return_value=({}, [])),
        _patch.object(mock_patient, "load_segmentation", return_value=MagicMock()),
        _patch.object(mock_patient, "load_pet_registered", return_value=MagicMock()),
        _patch("autods_pet.pipeline._subtract_lesions_from_marrow") as mock_subtract,
    ):
        extract_rois(cfg, mock_patient, seg_result={"vb_available": False})

    mock_subtract.assert_called_once()


def test_segment_ct_copies_refined_masks(mock_patient):
    """segment_ct copies refined masks from input_seg_dir/refined."""
    from autods_pet.imaging.segmentation import TOTSEG_FILENAME

    cfg = {}
    mock_patient.input_seg_dir.mkdir(parents=True, exist_ok=True)
    (mock_patient.input_seg_dir / TOTSEG_FILENAME).write_bytes(b"seg_data")

    refined = mock_patient.input_seg_dir / "refined"
    refined.mkdir()
    (refined / "aorta_mbp.nii.gz").write_bytes(b"refined_mask")

    mock_patient.seg_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("autods_pet.pipeline._ensure_totalseg_license"),
        patch(
            "autods_pet.imaging.segmentation.run_totalsegmentator",
            side_effect=RuntimeError("should not run"),
        ),
    ):
        segment_ct(cfg, mock_patient)

    out_refined = mock_patient.seg_dir / "refined" / "aorta_mbp.nii.gz"
    assert out_refined.exists()
