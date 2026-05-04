"""Tests for autods_pet.pipeline.DeauvillePipeline - orchestration class."""

from unittest.mock import patch

import pytest

from autods_pet.pipeline import DeauvillePipeline
from autods_pet.results import DeauvilleResult, ROIResult


@pytest.fixture()
def dummy_cfg(tmp_path):
    """Minimal config for DeauvillePipeline."""
    return {
        "paths": {
            "basepath": str(tmp_path),
            "output_dir": str(tmp_path / "results"),
        },
    }


def _make_extract_results():
    """Return a fake extract_results dict with ROI data."""
    return {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "Liver": ROIResult(stats={"median": 3.0}),
        "Focal lesion": ROIResult(stats={"max": 5.0}),
        "_roi_statuses": [("Aorta MBP", "ok", "")],
    }


def test_pipeline_run_calls_all_stages(dummy_cfg):
    pipeline = DeauvillePipeline(dummy_cfg)

    extract_results = _make_extract_results()

    with (
        patch.object(pipeline, "convert", return_value={}) as m_convert,
        patch.object(pipeline, "normalize", return_value={}) as m_normalize,
        patch.object(pipeline, "register", return_value={}) as m_register,
        patch.object(
            pipeline, "segment", return_value={"vb_available": False}
        ) as m_segment,
        patch.object(pipeline, "extract", return_value=extract_results) as m_extract,
        patch.object(pipeline, "score", return_value={"FL_DS": 4}) as m_score,
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.write_patient_suv_csv"),
        patch("autods_pet.pipeline.write_patient_deauville_csv"),
    ):
        pipeline.run("P001")

    m_convert.assert_called_once()
    m_normalize.assert_called_once()
    m_register.assert_called_once()
    m_segment.assert_called_once()
    m_extract.assert_called_once()
    m_score.assert_called_once()


def test_pipeline_run_returns_deauville_result(dummy_cfg):
    pipeline = DeauvillePipeline(dummy_cfg)
    extract_results = _make_extract_results()

    with (
        patch.object(pipeline, "convert", return_value={}),
        patch.object(pipeline, "normalize", return_value={}),
        patch.object(pipeline, "register", return_value={}),
        patch.object(pipeline, "segment", return_value={}),
        patch.object(pipeline, "extract", return_value=extract_results),
        patch.object(pipeline, "score", return_value={"FL_DS": 4}),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.write_patient_suv_csv"),
        patch("autods_pet.pipeline.write_patient_deauville_csv"),
    ):
        result = pipeline.run("P001")

    assert isinstance(result, DeauvilleResult)
    assert result.patient_id == "P001"
    assert result.error is None
    assert result.scores == {"FL_DS": 4}


def test_pipeline_run_captures_error(dummy_cfg):
    pipeline = DeauvillePipeline(dummy_cfg)

    with (
        patch.object(pipeline, "convert", side_effect=RuntimeError("Convert boom")),
        patch("autods_pet.patient.PatientCase"),
    ):
        result = pipeline.run("P001")

    assert isinstance(result, DeauvilleResult)
    assert result.error is not None
    assert "Convert boom" in result.error


def test_pipeline_run_batch_list(dummy_cfg):
    pipeline = DeauvillePipeline(dummy_cfg)
    fake_result = DeauvilleResult(patient_id="P001", scores={"FL_DS": 3})

    with patch.object(pipeline, "run", return_value=fake_result) as m_run:
        results = pipeline.run_batch(["P001", "P002"])

    assert len(results) == 2
    assert m_run.call_count == 2


def test_pipeline_run_batch_from_file(dummy_cfg, tmp_path):
    patient_file = tmp_path / "patients.txt"
    patient_file.write_text("P001\nP002\n# comment\nP003\n")

    pipeline = DeauvillePipeline(dummy_cfg)
    fake_result = DeauvilleResult(patient_id="test")

    with patch.object(pipeline, "run", return_value=fake_result) as m_run:
        pipeline.run_batch(patient_file)

    assert m_run.call_count == 3  # P001, P002, P003


def test_pipeline_run_batch_per_patient_errors(dummy_cfg):
    pipeline = DeauvillePipeline(dummy_cfg)
    ok = DeauvilleResult(patient_id="P001", scores={"FL_DS": 3})
    err = DeauvilleResult(patient_id="P002", error="failed")

    with patch.object(pipeline, "run", side_effect=[ok, err]):
        results = pipeline.run_batch(["P001", "P002"])

    assert results[0].error is None
    assert results[1].error == "failed"


def test_pipeline_to_dataframe_success():
    roi = ROIResult(stats={"median": 2.5})
    results = [
        DeauvilleResult(
            patient_id="P001",
            scores={"FL_DS": 4},
            rois={"Liver": roi},
        ),
    ]
    df = DeauvillePipeline.to_dataframe(results)
    assert "patient_id" in df.columns
    assert "Liver_median" in df.columns
    assert "DS_FL_DS" in df.columns
    assert len(df) == 1


def test_pipeline_to_dataframe_with_error():
    results = [
        DeauvilleResult(patient_id="P002", error="Pipeline failed"),
    ]
    df = DeauvillePipeline.to_dataframe(results)
    assert len(df) == 1
    assert df.iloc[0]["error"] == "Pipeline failed"


def test_pipeline_to_dataframe_empty():
    df = DeauvillePipeline.to_dataframe([])
    assert len(df) == 0
