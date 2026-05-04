"""Tests for autods_pet.segmentation - TotalSegmentator CLI wrapper."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from autods_pet.imaging.segmentation import TOTSEG_FILENAME, run_totalsegmentator


def test_segmentation_creates_output_dir(tmp_path):
    out_dir = tmp_path / "new_dir"
    nii_file = out_dir / TOTSEG_FILENAME

    def fake_run(cmd, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        nii_file.write_text("fake")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("autods_pet.imaging.segmentation.subprocess.run", side_effect=fake_run):
        result = run_totalsegmentator(tmp_path / "ct.nii", out_dir)
    assert out_dir.exists()
    assert result == nii_file


def test_segmentation_returns_nii_gz_path(tmp_path):
    out_dir = tmp_path / "seg"
    out_dir.mkdir()
    nii_file = out_dir / TOTSEG_FILENAME
    nii_file.write_text("fake")

    with patch("autods_pet.imaging.segmentation.subprocess.run"):
        result = run_totalsegmentator(tmp_path / "ct.nii", out_dir)
    assert result == nii_file


def test_segmentation_raises_when_output_missing(tmp_path):
    out_dir = tmp_path / "seg"
    out_dir.mkdir()

    with patch("autods_pet.imaging.segmentation.subprocess.run"):
        with pytest.raises(
            FileNotFoundError, match="TotalSegmentator output not found"
        ):
            run_totalsegmentator(tmp_path / "ct.nii", out_dir)


@pytest.mark.parametrize(
    "task,fast,expected_fragments",
    [
        ("total", False, ["--task", "total", "--ml"]),
        ("body", False, ["--task", "body", "--ml"]),
        ("total", True, ["--fast", "--ml"]),
    ],
)
def test_segmentation_cmd_args(tmp_path, task, fast, expected_fragments):
    out_dir = tmp_path / "seg"
    out_dir.mkdir()
    (out_dir / TOTSEG_FILENAME).write_text("fake")

    captured_cmd = []

    def capture_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch(
        "autods_pet.imaging.segmentation.subprocess.run", side_effect=capture_run
    ):
        run_totalsegmentator(tmp_path / "ct.nii", out_dir, task=task, fast=fast)

    for frag in expected_fragments:
        assert frag in captured_cmd

    # Verify -o points to a file path (not directory).
    o_idx = captured_cmd.index("-o")
    o_val = captured_cmd[o_idx + 1]
    assert o_val.endswith(TOTSEG_FILENAME)


def test_segmentation_subprocess_failure_raises(tmp_path):
    out_dir = tmp_path / "seg"
    out_dir.mkdir()

    with patch(
        "autods_pet.imaging.segmentation.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "TotalSegmentator"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            run_totalsegmentator(tmp_path / "ct.nii", out_dir)


def test_segmentation_path_types_coerced(tmp_path):
    out_dir = tmp_path / "seg"
    out_dir.mkdir()
    (out_dir / TOTSEG_FILENAME).write_text("fake")

    with patch("autods_pet.imaging.segmentation.subprocess.run"):
        result = run_totalsegmentator(str(tmp_path / "ct.nii"), str(out_dir))
    assert isinstance(result, Path)
