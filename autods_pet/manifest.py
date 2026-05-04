"""Reproducibility manifest for autods_pet pipeline runs."""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_TRACKED_PACKAGES = (
    "SimpleITK",
    "TotalSegmentator",
    "numpy",
    "pandas",
    "pydicom",
    "typer",
    "rich",
    "openpyxl",
    "tqdm",
)


def collect_environment() -> dict[str, Any]:
    """Gather software environment information.

    Returns
    -------
    dict
        Contains ``autods_pet_version``, ``python_version``,
        ``platform``, and ``packages`` (name -> version mapping).
    """
    from autods_pet import __version__

    packages: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            packages[pkg] = version(pkg)
        except PackageNotFoundError:
            packages[pkg] = "not installed"

    return {
        "autods_pet_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def write_manifest(cfg: dict[str, Any], output_dir: Path) -> Path | None:
    """Write a reproducibility manifest to *output_dir*/``manifest.json``.

    Parameters
    ----------
    cfg : dict
        Full configuration dict used for the pipeline run.
    output_dir : Path
        Root output directory where the manifest is saved.

    Returns
    -------
    Path or None
        Path to the written file, or *None* if writing failed.
    """
    manifest = collect_environment()
    manifest["config"] = cfg
    manifest["timestamp"] = datetime.now(timezone.utc).isoformat()

    path = output_dir / "manifest.json"
    try:
        path.write_text(json.dumps(manifest, indent=2, default=str))
    except OSError:
        log.warning("Could not write manifest to %s", path, exc_info=True)
        return None
    return path
