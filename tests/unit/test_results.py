"""Tests for autods_pet.results - ROIResult and DeauvilleResult dataclasses."""

from autods_pet.results import DeauvilleResult, ROIResult


def test_roi_result_defaults():
    r = ROIResult()
    assert r.stats == {}
    assert r.refined_mask is None
    assert r.shrinkage is None


def test_roi_result_with_values():
    r = ROIResult(
        stats={"median": 2.5, "p95": 3.1},
        refined_mask="sentinel",
        shrinkage={"delta_voxels": -10},
    )
    assert r.stats == {"median": 2.5, "p95": 3.1}
    assert r.refined_mask == "sentinel"
    assert r.shrinkage == {"delta_voxels": -10}


def test_roi_result_stats_dict_isolation():
    r1 = ROIResult()
    r2 = ROIResult()
    r1.stats["mean"] = 1.0
    assert "mean" not in r2.stats


def test_deauville_result_defaults():
    r = DeauvilleResult(patient_id="P001")
    assert r.patient_id == "P001"
    assert r.scores == {}
    assert r.rois == {}
    assert r.error is None


def test_deauville_result_with_values():
    roi = ROIResult(stats={"median": 2.0})
    r = DeauvilleResult(
        patient_id="P002",
        scores={"FL": 3, "PM": 4},
        rois={"Liver": roi},
        error=None,
    )
    assert r.patient_id == "P002"
    assert r.scores == {"FL": 3, "PM": 4}
    assert r.rois["Liver"].stats["median"] == 2.0
    assert r.error is None


def test_deauville_result_error_state():
    r = DeauvilleResult(patient_id="P003", error="Pipeline failed")
    assert r.error == "Pipeline failed"
    assert r.scores == {}
    assert r.rois == {}
