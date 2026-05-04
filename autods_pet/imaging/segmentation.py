"""Thin wrapper around TotalSegmentator CLI."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

#: Default filename for the multilabel segmentation output.
TOTSEG_FILENAME = "TotSeg_multilabel.nii.gz"


def run_totalsegmentator(
    ct_path: Path,
    output_dir: Path,
    task: str = "total",
    fast: bool = False,
    output_filename: str | None = None,
) -> Path:
    """Run TotalSegmentator on a CT image.

    Uses the ``--ml`` flag to produce a single multilabel NIfTI file
    named :data:`TOTSEG_FILENAME` inside *output_dir*.

    .. note::
        Paths are passed as subprocess arguments (list form, no shell).
        This mitigates OS command injection (CWE-78), but path traversal
        (CWE-22) is not validated.  If this function is ever exposed via
        a network service, add explicit path validation.

    Parameters
    ----------
    ct_path : Path
        Path to the input CT NIfTI file.
    output_dir : Path
        Directory where the segmentation output will be written.
    task : str
        TotalSegmentator task name (default ``"total"``).
    fast : bool
        If True, use the ``--fast`` flag for lower resolution but faster inference.
    output_filename : str or None
        Custom filename for the output segmentation file.  When ``None``
        (default), uses :data:`TOTSEG_FILENAME`.

    Returns
    -------
    Path
        Path to the output segmentation file.
    """
    ct_path = Path(ct_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # With --ml, -o is a FILE path (not directory).
    output_file = output_dir / (output_filename or TOTSEG_FILENAME)

    cmd = [
        "TotalSegmentator",
        "-i",
        str(ct_path),
        "-o",
        str(output_file),
        "--task",
        task,
        "--ml",
    ]
    if fast:
        cmd.append("--fast")

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if result.stdout:
        log.debug("TotalSegmentator stdout: %s", result.stdout)
    if result.stderr:
        log.debug("TotalSegmentator stderr: %s", result.stderr)

    if output_file.exists():
        return output_file

    raise FileNotFoundError(
        f"TotalSegmentator output not found: expected {output_file}"
    )
