"""Tests for autods_pet.cli - Typer CLI commands."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from autods_pet.cli import app

runner = CliRunner()


def _default_cfg(tmp_path):
    """Return a minimal cfg dict for mocking _load_and_resolve."""
    return {
        "paths": {
            "basepath": str(tmp_path),
            "output_dir": str(tmp_path / "results"),
        },
        "totalsegmentator": {"fast": False},
    }


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0


def test_no_subcommand_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code == 0


def test_convert_command(tmp_path):
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.convert_images", return_value={}) as mock_convert,
    ):
        result = runner.invoke(app, ["convert", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    mock_convert.assert_called_once()


def test_convert_command_handles_error(tmp_path):
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.convert_images", side_effect=RuntimeError("boom")),
    ):
        result = runner.invoke(app, ["convert", "-c", "fake.ini", "-p", "P001"])

    # Should not crash - error is printed to stderr via Rich
    assert result.exit_code == 0
    assert "boom" in result.output


def test_normalize_command(tmp_path):
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.normalize_pet", return_value={}) as mock_norm,
    ):
        result = runner.invoke(app, ["normalize", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    mock_norm.assert_called_once()


def test_register_command(tmp_path):
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.register_pet", return_value={}) as mock_reg,
    ):
        result = runner.invoke(app, ["register", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    mock_reg.assert_called_once()


def test_segment_command(tmp_path):
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.segment_ct", return_value={}) as mock_seg,
    ):
        result = runner.invoke(app, ["segment", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    mock_seg.assert_called_once()


def test_segment_fast_flag(tmp_path):
    cfg = _default_cfg(tmp_path)
    captured_cfg = {}

    def _capture_segment_ct(c, p):
        captured_cfg.update(c)
        return {}

    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.segment_ct", side_effect=_capture_segment_ct),
    ):
        result = runner.invoke(
            app, ["segment", "-c", "fake.ini", "-p", "P001", "--fast"]
        )

    assert result.exit_code == 0
    assert captured_cfg.get("totalsegmentator", {}).get("fast") is True


def test_extract_command(tmp_path):
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline._has_new_target_masks", return_value=True),
        patch(
            "autods_pet.pipeline.extract_rois",
            return_value={"_roi_statuses": []},
        ) as mock_ext,
        patch("autods_pet.pipeline.write_patient_suv_csv"),
    ):
        result = runner.invoke(app, ["extract", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    mock_ext.assert_called_once()


def test_score_command(tmp_path):
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline._has_new_target_masks", return_value=True),
        patch(
            "autods_pet.pipeline.extract_rois",
            return_value={"_roi_statuses": []},
        ) as mock_ext,
        patch(
            "autods_pet.pipeline.score_deauville", return_value={"FL_DS": 4}
        ) as mock_score,
        patch("autods_pet.pipeline.write_patient_suv_csv"),
        patch("autods_pet.pipeline.write_patient_deauville_csv"),
    ):
        result = runner.invoke(app, ["score", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    mock_ext.assert_called_once()
    mock_score.assert_called_once()


def test_run_single_patient(tmp_path):
    cfg = _default_cfg(tmp_path)
    output_dir = tmp_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch(
            "autods_pet.cli._run_pipeline_for_patient",
            return_value={
                "patient_id": "P001",
                "scores": {},
                "extract_results": {},
            },
        ) as mock_run,
        patch(
            "autods_pet.patient.resolve_paths",
            return_value={"output_dir": output_dir},
        ),
        patch(
            "autods_pet.io.save_dataframe",
            return_value=output_dir / "results.csv",
        ),
    ):
        result = runner.invoke(app, ["run", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_run_patient_error_captured(tmp_path):
    cfg = _default_cfg(tmp_path)
    cfg["paths"]["output_dir"] = str(tmp_path / "results")
    output_dir = tmp_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch(
            "autods_pet.cli._run_pipeline_for_patient",
            side_effect=RuntimeError("boom"),
        ),
        patch(
            "autods_pet.pipeline.save_batch_csv",
        ),
    ):
        result = runner.invoke(app, ["run", "-c", "fake.ini", "-p", "P001"])

    # Should not crash - error is captured per-patient
    assert result.exit_code == 0


def test_parse_patients_single_id():
    """Plain string returns single-element list."""
    from autods_pet.cli import _parse_patients

    assert _parse_patients("P001") == ["P001"]


def test_parse_patients_comma_separated():
    """Comma-separated IDs are split and stripped."""
    from autods_pet.cli import _parse_patients

    assert _parse_patients("P001, P002, P003") == ["P001", "P002", "P003"]


def test_parse_patients_trailing_comma_filtered():
    """Trailing comma does not produce an empty string entry."""
    from autods_pet.cli import _parse_patients

    assert _parse_patients("P001,P002,") == ["P001", "P002"]


def test_parse_patients_txt_file(tmp_path):
    """A .txt file path is read as a patient list."""
    from autods_pet.cli import _parse_patients

    f = tmp_path / "patients.txt"
    f.write_text("P001\nP002\n# comment\nP003\n")
    result = _parse_patients(str(f))
    assert result == ["P001", "P002", "P003"]


def test_parse_patients_txt_missing_file_literal():
    """A .txt path that does not exist is treated as a literal ID."""
    from autods_pet.cli import _parse_patients

    result = _parse_patients("/nonexistent/missing.txt")
    assert result == ["/nonexistent/missing.txt"]


def test_parse_patients_strips_whitespace():
    """Leading/trailing whitespace is stripped."""
    from autods_pet.cli import _parse_patients

    assert _parse_patients("  P001  ") == ["P001"]


def test_load_and_resolve_with_patients_arg(tmp_path):
    """When patients arg is provided, it overrides config."""
    from autods_pet.cli import _load_and_resolve

    ini = tmp_path / "config.ini"
    ini.write_text(f"[paths]\nbasepath = {tmp_path}\n")
    cfg, ids = _load_and_resolve(ini, "P001,P002")
    assert ids == ["P001", "P002"]


def test_load_and_resolve_auto_discover_subdirs(tmp_path):
    """Without patients arg or patient_list, discover subdirectories."""
    from autods_pet.cli import _load_and_resolve

    (tmp_path / "B_patient").mkdir()
    (tmp_path / "A_patient").mkdir()
    ini = tmp_path / "config.ini"
    ini.write_text(f"[paths]\nbasepath = {tmp_path}\n")
    cfg, ids = _load_and_resolve(ini)
    assert ids == ["A_patient", "B_patient"]  # sorted


def test_load_and_resolve_ignores_files_in_basepath(tmp_path):
    """Files in basepath are not treated as patient directories."""
    from autods_pet.cli import _load_and_resolve

    (tmp_path / "P001").mkdir()
    (tmp_path / "not_a_patient.txt").write_text("junk")
    ini = tmp_path / "config.ini"
    ini.write_text(f"[paths]\nbasepath = {tmp_path}\n")
    cfg, ids = _load_and_resolve(ini)
    assert ids == ["P001"]


def test_load_and_resolve_no_patients_raises_exit(tmp_path):
    """Empty basepath with no patient_list raises typer.Exit."""
    from click.exceptions import Exit

    from autods_pet.cli import _load_and_resolve

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ini = tmp_path / "config.ini"
    ini.write_text(f"[paths]\nbasepath = {empty_dir}\n")
    with pytest.raises(Exit):
        _load_and_resolve(ini)


def test_build_ds_dataframe_success_row():
    """Successful result row produces correct DS DataFrame columns."""
    from autods_pet.cli import _build_ds_dataframe

    results = [
        {
            "patient_id": "P001",
            "extract_results": {"Liver": {"stats": {"median": 2.5}}},
            "scores": {"FL_DS": 4},
        }
    ]
    df = _build_ds_dataframe(results)
    assert "FL_DS" in df.columns
    assert df.iloc[0]["FL_DS"] == 4


def test_build_suv_dataframe_success_row():
    """Successful result row produces correct SUV DataFrame columns."""
    from autods_pet.cli import _build_suv_dataframe

    results = [
        {
            "patient_id": "P001",
            "extract_results": {"Liver": {"stats": {"median": 2.5}}},
            "scores": {"FL_DS": 4},
        }
    ]
    df = _build_suv_dataframe(results)
    assert "Liver_median" in df.columns
    assert df.iloc[0]["Liver_median"] == 2.5


def test_build_errors_dataframe_error_row():
    """Error result row has patient_id and error."""
    from autods_pet.cli import _build_errors_dataframe

    results = [{"patient_id": "P002", "error": "boom"}]
    df = _build_errors_dataframe(results)
    assert df.iloc[0]["error"] == "boom"
    assert df.iloc[0]["patient_id"] == "P002"


def test_build_errors_dataframe_no_errors():
    """No errors returns None."""
    from autods_pet.cli import _build_errors_dataframe

    results = [{"patient_id": "P001", "scores": {"FL_DS": 4}}]
    df = _build_errors_dataframe(results)
    assert df is None


def test_build_ds_dataframe_empty_input():
    """Empty input produces empty DataFrame."""
    from autods_pet.cli import _build_ds_dataframe

    df = _build_ds_dataframe([])
    assert len(df) == 0


def test_build_suv_dataframe_empty_input():
    """Empty input produces empty DataFrame."""
    from autods_pet.cli import _build_suv_dataframe

    df = _build_suv_dataframe([])
    assert len(df) == 0


@pytest.mark.parametrize("status", ["ok", "skip", "error", "running"])
def test_print_stage_all_status_variants(status):
    """_print_stage handles all status variants without error."""
    from autods_pet.cli import _print_stage

    _print_stage(1, 6, "Test stage", status)  # should not raise


@pytest.mark.parametrize("status", ["ok", "skip", "error"])
def test_print_roi_status_all_variants(status):
    """_print_roi_status and _print_roi_status_last handle all variants."""
    from autods_pet.cli import _print_roi_status, _print_roi_status_last

    _print_roi_status("TestROI", status, "detail")
    _print_roi_status_last("TestROI", status, "detail")


def test_print_results_table_with_roi_result():
    """_print_results_table handles ROIResult objects."""
    from autods_pet.cli import _print_results_table
    from autods_pet.results import ROIResult

    rois = {"Liver": ROIResult(stats={"median": 3.5, "mean": 3.2})}
    scores = {"FL": 4}
    _print_results_table(rois, scores)  # should not raise


def test_print_results_table_with_dict():
    """_print_results_table handles plain dict ROI data."""
    from autods_pet.cli import _print_results_table

    rois = {"Liver": {"stats": {"median": 3.5}}}
    scores = {"FL": 0}  # 0 → "--"
    _print_results_table(rois, scores)


def test_print_results_table_none_value():
    """_print_results_table handles None stat values (→ '--')."""
    from autods_pet.cli import _print_results_table
    from autods_pet.results import ROIResult

    rois = {"Liver": ROIResult(stats={"median": None})}
    scores = {}
    _print_results_table(rois, scores)


def test_print_batch_summary_ok_and_fail():
    """_print_batch_summary handles both ok and failed rows."""
    from autods_pet.cli import _print_batch_summary

    results = [
        {
            "patient_id": "P001",
            "mbp_median": 2.0,
            "liver_median": 3.0,
            "fl_value": 5.0,
            "ds_fl": 4,
        },
        {"patient_id": "P002", "error": "boom"},
    ]
    _print_batch_summary(results)


def test_print_batch_summary_none_values():
    """_print_batch_summary handles None numeric values (→ '--')."""
    from autods_pet.cli import _print_batch_summary

    results = [
        {
            "patient_id": "P001",
            "mbp_median": None,
            "liver_median": None,
            "fl_value": None,
            "ds_fl": None,
        }
    ]
    _print_batch_summary(results)


def test_print_extract_statuses_multiple():
    """_print_extract_statuses prints tree for multiple ROIs."""
    from autods_pet.cli import _print_extract_statuses

    statuses = [
        ("Aorta MBP", "ok", ""),
        ("Liver", "ok", ""),
        ("Lumbar VB", "skip", "no VB seg"),
    ]
    _print_extract_statuses(statuses)


def test_print_extract_statuses_single():
    """_print_extract_statuses with a single entry uses last marker."""
    from autods_pet.cli import _print_extract_statuses

    _print_extract_statuses([("Liver", "ok", "")])


def test_init_logging_with_log_file(tmp_path):
    """_init_logging creates a file handler when log_file is provided."""
    import logging

    from autods_pet.cli import _init_logging

    log_file = tmp_path / "logs" / "test.log"
    _init_logging(verbose=True, log_file=log_file)
    assert log_file.parent.exists()
    # Clean up: remove the file handler to avoid interfering with other tests
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.FileHandler)]


def test_load_and_resolve_patient_list_from_config(tmp_path):
    """patient_list from config INI is read when no --patients arg."""
    from autods_pet.cli import _load_and_resolve

    patient_file = tmp_path / "patients.txt"
    patient_file.write_text("P001\nP002\n")
    ini = tmp_path / "config.ini"
    ini.write_text(f"[paths]\nbasepath = {tmp_path}\npatient_list = {patient_file}\n")
    cfg, ids = _load_and_resolve(ini)
    assert ids == ["P001", "P002"]


def test_run_pipeline_for_patient_success(tmp_path):
    """_run_pipeline_for_patient runs all stages and returns result row."""
    from autods_pet.cli import _run_pipeline_for_patient
    from autods_pet.results import ROIResult

    cfg = _default_cfg(tmp_path)

    mock_extract = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "Liver": ROIResult(stats={"median": 3.0}),
        "Focal lesion": ROIResult(stats={"max": 5.0}),
        "_roi_statuses": [("Aorta MBP", "ok", ""), ("Liver", "ok", "")],
    }

    with (
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.convert_images"),
        patch("autods_pet.pipeline.normalize_pet"),
        patch("autods_pet.pipeline.register_pet"),
        patch(
            "autods_pet.pipeline.segment_ct",
            return_value={"vb_available": False},
        ),
        patch(
            "autods_pet.pipeline.extract_rois",
            return_value=mock_extract,
        ),
        patch(
            "autods_pet.pipeline.score_deauville",
            return_value={"FL_DS": 4},
        ),
        patch("autods_pet.pipeline.write_patient_suv_csv"),
        patch("autods_pet.pipeline.write_patient_deauville_csv"),
    ):
        row = _run_pipeline_for_patient(cfg, "P001")

    assert row["patient_id"] == "P001"
    assert row["scores"] == {"FL_DS": 4}
    assert "extract_results" in row
    # Internal keys starting with _ should be excluded from extract_results
    assert all(not k.startswith("_") for k in row["extract_results"])


def test_run_pipeline_for_patient_convert_failure(tmp_path):
    """_run_pipeline_for_patient raises RuntimeError when convert fails."""
    from autods_pet.cli import _run_pipeline_for_patient

    cfg = _default_cfg(tmp_path)

    with (
        patch("autods_pet.patient.PatientCase"),
        patch(
            "autods_pet.pipeline.convert_images",
            side_effect=RuntimeError("CT missing"),
        ),
    ):
        with pytest.raises(RuntimeError, match="CT missing"):
            _run_pipeline_for_patient(cfg, "P001")


def test_run_pipeline_for_patient_extract_failure(tmp_path):
    """_run_pipeline_for_patient raises RuntimeError when extract_rois fails."""
    from autods_pet.cli import _run_pipeline_for_patient

    cfg = _default_cfg(tmp_path)

    with (
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.convert_images"),
        patch("autods_pet.pipeline.normalize_pet"),
        patch("autods_pet.pipeline.register_pet"),
        patch(
            "autods_pet.pipeline.segment_ct",
            return_value={"vb_available": False},
        ),
        patch(
            "autods_pet.pipeline.extract_rois",
            side_effect=ValueError("no ROI mask"),
        ),
    ):
        with pytest.raises(RuntimeError, match="ROI extraction failed"):
            _run_pipeline_for_patient(cfg, "P001")


def test_run_pipeline_for_patient_score_failure(tmp_path):
    """_run_pipeline_for_patient raises RuntimeError when scoring fails."""
    from autods_pet.cli import _run_pipeline_for_patient
    from autods_pet.results import ROIResult

    cfg = _default_cfg(tmp_path)

    mock_extract = {
        "Aorta MBP": ROIResult(stats={"median": 2.0}),
        "_roi_statuses": [("Aorta MBP", "ok", "")],
    }

    with (
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.convert_images"),
        patch("autods_pet.pipeline.normalize_pet"),
        patch("autods_pet.pipeline.register_pet"),
        patch(
            "autods_pet.pipeline.segment_ct",
            return_value={"vb_available": False},
        ),
        patch(
            "autods_pet.pipeline.extract_rois",
            return_value=mock_extract,
        ),
        patch(
            "autods_pet.pipeline.score_deauville",
            side_effect=ValueError("insufficient data"),
        ),
    ):
        with pytest.raises(RuntimeError, match="Deauville scoring failed"):
            _run_pipeline_for_patient(cfg, "P001")


def test_run_pipeline_for_patient_dict_rois(tmp_path):
    """_run_pipeline_for_patient handles plain dict ROI data (not ROIResult)."""
    from autods_pet.cli import _run_pipeline_for_patient

    cfg = _default_cfg(tmp_path)

    mock_extract = {
        "Aorta MBP": {"stats": {"median": 2.0}},
        "Liver": {"stats": {"median": 3.0}},
        "Focal lesion": {"stats": {"max": 5.0}},
        "_roi_statuses": [("Aorta MBP", "ok", ""), ("Liver", "ok", "")],
    }

    with (
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.convert_images"),
        patch("autods_pet.pipeline.normalize_pet"),
        patch("autods_pet.pipeline.register_pet"),
        patch(
            "autods_pet.pipeline.segment_ct",
            return_value={"vb_available": False},
        ),
        patch(
            "autods_pet.pipeline.extract_rois",
            return_value=mock_extract,
        ),
        patch(
            "autods_pet.pipeline.score_deauville",
            return_value={"FL_DS": 3},
        ),
        patch("autods_pet.pipeline.write_patient_suv_csv"),
        patch("autods_pet.pipeline.write_patient_deauville_csv"),
    ):
        row = _run_pipeline_for_patient(cfg, "P001")

    assert row["scores"] == {"FL_DS": 3}
    assert "extract_results" in row


def test_run_pipeline_for_patient_missing_rois(tmp_path):
    """_run_pipeline_for_patient handles missing ROI keys gracefully."""
    from autods_pet.cli import _run_pipeline_for_patient

    cfg = _default_cfg(tmp_path)

    # extract_results with no Aorta MBP, no Liver, no Focal lesion
    mock_extract = {
        "_roi_statuses": [],
    }

    with (
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline.convert_images"),
        patch("autods_pet.pipeline.normalize_pet"),
        patch("autods_pet.pipeline.register_pet"),
        patch(
            "autods_pet.pipeline.segment_ct",
            return_value={"vb_available": False},
        ),
        patch(
            "autods_pet.pipeline.extract_rois",
            return_value=mock_extract,
        ),
        patch(
            "autods_pet.pipeline.score_deauville",
            return_value={},
        ),
        patch("autods_pet.pipeline.write_patient_suv_csv"),
        patch("autods_pet.pipeline.write_patient_deauville_csv"),
    ):
        row = _run_pipeline_for_patient(cfg, "P001")

    assert row["scores"] == {}
    assert row["extract_results"] == {}


def test_normalize_command_handles_error(tmp_path):
    """normalize command catches and prints errors without crashing."""
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch(
            "autods_pet.pipeline.normalize_pet",
            side_effect=RuntimeError("bad header"),
        ),
    ):
        result = runner.invoke(app, ["normalize", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    assert "bad header" in result.output


def test_register_command_handles_error(tmp_path):
    """register command catches and prints errors without crashing."""
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch(
            "autods_pet.pipeline.register_pet",
            side_effect=RuntimeError("registration diverged"),
        ),
    ):
        result = runner.invoke(app, ["register", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    assert "registration diverged" in result.output


def test_segment_command_handles_error(tmp_path):
    """segment command catches and prints errors without crashing."""
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch(
            "autods_pet.pipeline.segment_ct",
            side_effect=RuntimeError("CUDA OOM"),
        ),
    ):
        result = runner.invoke(app, ["segment", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    assert "CUDA OOM" in result.output


def test_run_batch_multiple_patients(tmp_path):
    """run command with multiple patients uses progress bar batch path."""
    cfg = _default_cfg(tmp_path)
    output_dir = tmp_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch(
            "autods_pet.cli._load_and_resolve",
            return_value=(cfg, ["P001", "P002"]),
        ),
        patch("autods_pet.cli._init_logging"),
        patch(
            "autods_pet.cli._run_pipeline_for_patient",
            return_value={
                "patient_id": "P001",
                "scores": {},
                "extract_results": {},
            },
        ) as mock_run,
        patch(
            "autods_pet.io.save_dataframe",
            return_value=output_dir / "batch_results.csv",
        ),
    ):
        result = runner.invoke(app, ["run", "-c", "fake.ini", "-p", "P001,P002"])

    assert result.exit_code == 0
    assert mock_run.call_count == 2


def test_run_batch_with_failure(tmp_path):
    """run batch captures per-patient errors without crashing."""
    cfg = _default_cfg(tmp_path)
    output_dir = tmp_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    call_count = 0

    def _side_effect(cfg, pid, fast=False):
        nonlocal call_count
        call_count += 1
        if pid == "P002":
            raise RuntimeError("P002 failed")
        return {
            "patient_id": pid,
            "scores": {},
            "extract_results": {},
        }

    with (
        patch(
            "autods_pet.cli._load_and_resolve",
            return_value=(cfg, ["P001", "P002"]),
        ),
        patch("autods_pet.cli._init_logging"),
        patch(
            "autods_pet.cli._run_pipeline_for_patient",
            side_effect=_side_effect,
        ),
        patch(
            "autods_pet.io.save_dataframe",
            return_value=output_dir / "batch_results.csv",
        ),
    ):
        result = runner.invoke(app, ["run", "-c", "fake.ini", "-p", "P001,P002"])

    assert result.exit_code == 0
    assert call_count == 2


def test_run_fast_flag_sets_config(tmp_path):
    """run --fast sets totalsegmentator.fast in the config."""
    cfg = _default_cfg(tmp_path)
    output_dir = tmp_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    captured_cfg = {}

    def _capture(c, pid, fast=False):
        captured_cfg.update(c)
        return {
            "patient_id": pid,
            "scores": {},
            "extract_results": {},
        }

    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch(
            "autods_pet.cli._run_pipeline_for_patient",
            side_effect=_capture,
        ),
        patch(
            "autods_pet.io.save_dataframe",
            return_value=output_dir / "batch_results.csv",
        ),
    ):
        result = runner.invoke(app, ["run", "-c", "fake.ini", "-p", "P001", "--fast"])

    assert result.exit_code == 0
    assert captured_cfg.get("totalsegmentator", {}).get("fast") is True


def test_main_entry_point():
    """main() invokes the Typer app."""
    from autods_pet.cli import main

    with patch("autods_pet.cli.app") as mock_app:
        main()
    mock_app.assert_called_once()


def test_extract_command_handles_error(tmp_path):
    """extract command catches and prints errors without crashing."""
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline._has_new_target_masks", return_value=True),
        patch(
            "autods_pet.pipeline.extract_rois",
            side_effect=RuntimeError("mask not found"),
        ),
    ):
        result = runner.invoke(app, ["extract", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    assert "mask not found" in result.output


def test_score_command_handles_error(tmp_path):
    """score command catches and prints errors without crashing."""
    cfg = _default_cfg(tmp_path)
    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline._has_new_target_masks", return_value=True),
        patch(
            "autods_pet.pipeline.extract_rois",
            side_effect=RuntimeError("extraction failure"),
        ),
    ):
        result = runner.invoke(app, ["score", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    assert "extraction failure" in result.output


def test_run_uses_config_output_dir(tmp_path):
    """run command uses output_dir from config when specified."""
    custom_output = tmp_path / "custom_output"
    cfg = {
        "paths": {
            "basepath": str(tmp_path),
            "output_dir": str(custom_output),
        },
        "totalsegmentator": {"fast": False},
    }

    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch(
            "autods_pet.cli._run_pipeline_for_patient",
            return_value={
                "patient_id": "P001",
                "scores": {},
                "extract_results": {},
            },
        ),
        patch(
            "autods_pet.io.save_dataframe",
            return_value=custom_output / "batch_results.csv",
        ) as mock_save,
    ):
        result = runner.invoke(app, ["run", "-c", "fake.ini", "-p", "P001"])

    assert result.exit_code == 0
    # Verify the output path passed to save_dataframe uses custom_output
    saved_path = mock_save.call_args[0][1]
    assert str(custom_output) in str(saved_path)


def test_run_uses_relative_output_dir(tmp_path):
    """run command resolves relative output_dir to CWD (verified via path check)."""
    from pathlib import Path

    from autods_pet.config import resolve_output_dir

    cfg = {
        "paths": {
            "basepath": str(tmp_path),
            "output_dir": "my_results",
        },
    }
    # Verify the resolution logic directly (no file I/O).
    resolved = resolve_output_dir(cfg)
    assert resolved == Path.cwd() / "my_results"


def test_run_save_masks_flag_sets_config(tmp_path):
    """run --save-masks sets both output flags in config."""
    cfg = _default_cfg(tmp_path)
    output_dir = tmp_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    captured_cfg = {}

    def _capture(c, pid, fast=False):
        captured_cfg.update(c)
        return {
            "patient_id": pid,
            "scores": {},
            "extract_results": {},
        }

    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch(
            "autods_pet.cli._run_pipeline_for_patient",
            side_effect=_capture,
        ),
        patch(
            "autods_pet.io.save_dataframe",
            return_value=output_dir / "batch_results.csv",
        ),
    ):
        result = runner.invoke(
            app, ["run", "-c", "fake.ini", "-p", "P001", "--save-masks"]
        )

    assert result.exit_code == 0
    assert captured_cfg.get("output", {}).get("save_raw_masks") is True
    assert captured_cfg.get("output", {}).get("save_refined_masks") is True


def test_run_save_raw_masks_flag_only(tmp_path):
    """run --save-raw-masks sets only save_raw_masks."""
    cfg = _default_cfg(tmp_path)
    output_dir = tmp_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    captured_cfg = {}

    def _capture(c, pid, fast=False):
        captured_cfg.update(c)
        return {
            "patient_id": pid,
            "scores": {},
            "extract_results": {},
        }

    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch(
            "autods_pet.cli._run_pipeline_for_patient",
            side_effect=_capture,
        ),
        patch(
            "autods_pet.io.save_dataframe",
            return_value=output_dir / "batch_results.csv",
        ),
    ):
        result = runner.invoke(
            app, ["run", "-c", "fake.ini", "-p", "P001", "--save-raw-masks"]
        )

    assert result.exit_code == 0
    assert captured_cfg.get("output", {}).get("save_raw_masks") is True


def test_extract_save_refined_flag(tmp_path):
    """extract --save-refined-masks sets save_refined_masks in config."""
    cfg = _default_cfg(tmp_path)

    captured_cfg = {}

    def _mock_extract(c, patient, seg_result=None):
        captured_cfg.update(c)
        return {"_roi_statuses": []}

    with (
        patch("autods_pet.cli._load_and_resolve", return_value=(cfg, ["P001"])),
        patch("autods_pet.cli._init_logging"),
        patch("autods_pet.patient.PatientCase"),
        patch("autods_pet.pipeline._has_new_target_masks", return_value=True),
        patch(
            "autods_pet.pipeline.extract_rois",
            side_effect=_mock_extract,
        ),
    ):
        result = runner.invoke(
            app, ["extract", "-c", "fake.ini", "-p", "P001", "--save-refined-masks"]
        )

    assert result.exit_code == 0
    assert captured_cfg.get("output", {}).get("save_refined_masks") is True


def test_create_config_command(tmp_path):
    out = tmp_path / "new_config.ini"
    result = runner.invoke(app, ["create-config", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_create_config_refuses_overwrite(tmp_path):
    out = tmp_path / "existing.ini"
    out.write_text("[paths]\n")
    result = runner.invoke(app, ["create-config", "--output", str(out)])
    assert result.exit_code == 1


def test_create_config_force_overwrites(tmp_path):
    out = tmp_path / "existing.ini"
    out.write_text("[paths]\n")
    original_size = out.stat().st_size
    result = runner.invoke(app, ["create-config", "--output", str(out), "--force"])
    assert result.exit_code == 0
    assert out.stat().st_size > original_size


def test_create_config_with_profile(tmp_path):
    out = tmp_path / "quick.ini"
    result = runner.invoke(
        app, ["create-config", "--output", str(out), "--profile", "quick"]
    )
    assert result.exit_code == 0
    assert out.exists()
    text = out.read_text()
    assert "Profile: quick" in text
    assert "fast = true" in text


def test_create_config_with_short_profile_flag(tmp_path):
    out = tmp_path / "full.ini"
    result = runner.invoke(app, ["create-config", "--output", str(out), "-p", "full"])
    assert result.exit_code == 0
    assert out.exists()
    text = out.read_text()
    assert "Profile: full" in text


def test_create_config_invalid_profile(tmp_path):
    out = tmp_path / "bad.ini"
    result = runner.invoke(
        app, ["create-config", "--output", str(out), "-p", "nonexistent"]
    )
    assert result.exit_code == 1
    assert "Unknown profile" in result.output


def test_validate_config_valid_file(tmp_path):
    ini = tmp_path / "good.ini"
    ini.write_text("")  # empty = defaults = valid
    result = runner.invoke(app, ["validate-config", str(ini)])
    assert result.exit_code == 0


def test_validate_config_invalid_file(tmp_path):
    ini = tmp_path / "bad.ini"
    ini.write_text("[lumbar_vb]\nstats = foobar\n")
    result = runner.invoke(app, ["validate-config", str(ini)])
    assert result.exit_code == 1


def test_validate_config_multiple_errors(tmp_path):
    ini = tmp_path / "bad.ini"
    ini.write_text(
        "[lumbar_vb]\nstats = foobar\nerosion_mm = -1.0\n[liver]\nstats = badstat\n"
    )
    result = runner.invoke(app, ["validate-config", str(ini)])
    assert result.exit_code == 1


def test_validate_config_missing_file():
    result = runner.invoke(app, ["validate-config", "/nonexistent/config.ini"])
    assert result.exit_code == 1


def test_validate_config_unknown_section(tmp_path):
    ini = tmp_path / "bad.ini"
    ini.write_text("[nonexistent_section]\nfoo = bar\n")
    result = runner.invoke(app, ["validate-config", str(ini)])
    assert result.exit_code == 1


def test_list_segments_cmd_displays_table(tmp_path):
    """list-segments command displays a table of segments."""
    fake_dcm = tmp_path / "seg.dcm"
    fake_dcm.write_bytes(b"fake")
    segments = [
        {"number": 1, "label": "Tumor", "description": "Primary lesion"},
        {"number": 2, "label": "Necrosis", "description": ""},
    ]
    with patch("autods_pet.ops.dicom_seg.list_segments", return_value=segments):
        result = runner.invoke(app, ["list-segments", str(fake_dcm)])
    assert result.exit_code == 0
    assert "Tumor" in result.output
    assert "Necrosis" in result.output


def test_list_segments_cmd_empty_exits(tmp_path):
    """list-segments command exits with code 1 when no segments found."""
    fake_dcm = tmp_path / "seg.dcm"
    fake_dcm.write_bytes(b"fake")
    with patch("autods_pet.ops.dicom_seg.list_segments", return_value=[]):
        result = runner.invoke(app, ["list-segments", str(fake_dcm)])
    assert result.exit_code == 1
