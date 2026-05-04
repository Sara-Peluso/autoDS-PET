"""Programmatic pipeline for Deauville Score computation.

This module contains the :class:`DeauvillePipeline` class and per-stage
helper functions.  It has **no** dependency on Rich or Typer, so it can
be used from Python scripts, notebooks, or tests without a terminal.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from autods_pet.patient import PatientCase
    from autods_pet.results import (
        DeauvilleResult,
    )

log = logging.getLogger(__name__)

# Required columns for SUV normalization metadata.
METADATA_COLUMNS = [
    "PatientID",
    "StudyDate",
    "AcquisitionTime",
    "RadiopharmaceuticalStartTime",
    "RadionuclideTotalDose",
    "RadionuclideHalfLife",
    "DecayCorrection",
    "PatientWeight",
]

# Standardized Deauville Score short names.
DS_TARGET_NAMES: dict[str, str] = {
    "Focal lesion": "FL_DS",
    "Paramedullary": "PM_DS",
    "Extramedullary": "EM_DS",
}
DS_RESEARCH_NAMES: dict[str, str] = {
    "Lumbar VB": "BM_DS",
    "Long bones": "LB_DS",
}
# Fixed column order for batch output.
DS_COLUMN_ORDER = ["BM_DS", "LB_DS", "FL_DS", "PM_DS", "EM_DS", "BLR"]


def _resolve_csv_path(cfg: dict[str, Any]) -> Path | None:
    """Return the resolved metadata CSV path, or *None* if not configured."""
    paths = cfg.get("paths", {})
    raw = paths.get("metadata_csv", "")
    if not raw:
        return None
    if Path(raw).is_absolute():
        return Path(raw)
    from autods_pet.config import resolve_output_dir

    return resolve_output_dir(cfg) / raw


def _should_skip(cfg: dict[str, Any]) -> bool:
    """Return *True* unless force mode is enabled."""
    return not cfg.get("pipeline", {}).get("force", False)


def _output_skip_dirs(cfg: dict[str, Any], patient: PatientCase) -> set[Path]:
    """Return absolute directories to exclude from mask discovery walks.

    Currently this is the per-patient output directory (so previously
    written `<stem>.nii.gz` masks under `seg_dir/` are not rediscovered)
    plus the global output dir.
    """
    skip: set[Path] = set()
    out = patient.output_dir
    if out is not None:
        skip.add(out)
    seg_dir = patient.seg_dir
    if seg_dir is not None:
        skip.add(seg_dir)
    return skip


def _discover_patient_masks(
    cfg: dict[str, Any], patient: PatientCase
) -> tuple[dict[str, Any], list[str]]:
    """Run :func:`mask_discovery.discover_all_masks` for *patient*.

    Returns ``(target_name -> DiscoveredMask, warnings)``.  Warnings are
    purely informational here; callers decide whether to log them.
    """
    from autods_pet.config import get_all_targets
    from autods_pet.ops.mask_discovery import discover_all_masks

    targets = get_all_targets(cfg)
    return discover_all_masks(
        patient.input_dir,
        targets,
        patient.pet_series_uid,
        skip_dirs=_output_skip_dirs(cfg, patient),
    )


_SECTION_TO_DS: dict[str, str] = {
    "focal_lesion": "FL_DS",
    "paramedullary": "PM_DS",
    "extramedullary": "EM_DS",
}


def _mask_needs_reprocessing(
    mask: Any,
    output_mask: Path,
    ds_key: str,
    existing_ds: set[str],
) -> bool:
    """Return *True* if a single target mask needs (re-)processing."""
    if not output_mask.exists():
        return True
    try:
        if os.path.getmtime(mask.path) > os.path.getmtime(output_mask):
            return True
    except OSError:
        return True
    return ds_key not in existing_ds


def _has_new_target_masks(cfg: dict[str, Any], patient: PatientCase) -> bool:
    """Check whether any configured target mask needs (re-)processing.

    Returns *True* if any configured target has:
    - A mask source (file or DICOM SEG) newer than its output copy, OR
    - A mask source on disk but its DS is missing from the per-patient CSV.

    Discovery warnings (e.g. "DICOM SEG found but no segments matched the
    configured labels", or "no mask found anywhere") are always logged at
    WARNING level here so that callers like the ``extract``/``score`` CLI
    commands cannot silently skip a patient without explaining why.
    Downstream :func:`_extract_targets` re-runs discovery but does *not*
    re-log these warnings to avoid duplication.
    """
    from autods_pet.config import get_all_targets

    discovered, warnings = _discover_patient_masks(cfg, patient)
    for msg in warnings:
        log.warning("[%s] %s", patient.patient_id, msg)
    if not discovered:
        return False

    # Load existing DS results to detect missing scores.
    existing_ds: set[str] = set()
    ds_path = patient.deauville_csv_path
    if ds_path.exists():
        with open(ds_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ds.add(row.get("Target", ""))

    # Build a name -> primary stem map so we can locate the cached output.
    target_stems: dict[str, str] = {}
    for tcfg in get_all_targets(cfg):
        stems = tcfg.get("mask_filename") or []
        if isinstance(stems, str):
            stems = [stems]
        if stems:
            target_stems[tcfg["name"]] = stems[0]
        else:
            target_stems[tcfg["name"]] = tcfg["name"]

    for tname, mask in discovered.items():
        out_stem = target_stems.get(tname, tname)
        output_mask = patient.seg_dir / f"{out_stem}.nii.gz"
        ds_key = _SECTION_TO_DS.get(tname, tname)
        if _mask_needs_reprocessing(mask, output_mask, ds_key, existing_ds):
            return True

    return False


def _load_cached_references(patient: PatientCase) -> dict[str, Any] | None:
    """Load cached reference ROI values from an existing ``SUV_values.csv``.

    Reads the CSV at ``patient.suv_csv_path`` and reconstructs a dict
    compatible with :func:`score_deauville`.  Returns
    :class:`~autods_pet.results.ROIResult` objects for **all** ROIs found
    in the CSV (e.g. Aorta MBP, Liver, Brain, Long bones, Lumbar VB), or
    *None* if the file is missing or the required Aorta MBP and Liver
    median references are absent.
    """
    from autods_pet.results import ROIResult

    csv_path = patient.suv_csv_path
    if not csv_path.exists():
        return None

    # Parse CSV rows into {display_name: {stat: value}}
    roi_stats: dict[str, dict[str, float | None]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert underscored ROI names back to display names
            roi_name = row["ROI"].replace("_", " ")
            stat = row["Statistic"]
            try:
                value: float | None = float(row["Value"])
            except (ValueError, TypeError):
                value = None
            roi_stats.setdefault(roi_name, {})[stat] = value

    # We need at least Aorta MBP median and Liver median
    aorta_stats = roi_stats.get("Aorta MBP", {})
    liver_stats = roi_stats.get("Liver", {})
    if "median" not in aorta_stats or "median" not in liver_stats:
        return None

    # Build result dict with ROIResult objects for all cached ROIs
    result: dict[str, Any] = {}
    for roi_name, stats in roi_stats.items():
        result[roi_name] = ROIResult(stats=stats)
    return result


def extract_new_targets_only(
    cfg: dict[str, Any], patient: PatientCase
) -> dict[str, Any]:
    """Extract only target ROIs and merge with cached reference values.

    Loads the registered PET, reads cached reference values from the
    existing ``SUV_values.csv``, extracts only target masks, and computes
    Deauville Scores.  Falls back to a full :func:`extract_rois` if
    cached references are unavailable.

    Parameters
    ----------
    cfg : dict
        Configuration dict.
    patient : PatientCase
        Patient case with resolved paths.

    Returns
    -------
    dict
        Keys ``"scores"`` and ``"extract_results"`` (stats-only dict).
    """
    import SimpleITK as sitk

    cached_refs = _load_cached_references(patient)
    if cached_refs is None:
        # Fall back to full extraction
        extract_results = extract_rois(cfg, patient)
        scores = score_deauville(cfg, extract_results)
        write_patient_suv_csv(extract_results, patient.suv_csv_path)
        write_patient_deauville_csv(scores, patient.deauville_csv_path)
        from autods_pet.results import ROIResult

        return {
            "scores": scores,
            "extract_results": {
                k: {
                    "stats": v.stats if isinstance(v, ROIResult) else v.get("stats", {})
                }
                for k, v in extract_results.items()
                if not k.startswith("_")
            },
        }

    # Load the PET image (prefer registered, fall back to SUV)
    if patient.pet_registered_path.exists():
        pet = sitk.ReadImage(str(patient.pet_registered_path))
    else:
        pet = sitk.ReadImage(str(patient.pet_suv_path))

    # Extract only target masks
    target_results, _target_statuses = _extract_targets(cfg, pet, patient)

    # Merge target results with cached references
    merged = dict(cached_refs)
    merged.update(target_results)

    scores = score_deauville(cfg, merged)

    # Update per-patient CSVs (merge mode)
    write_patient_suv_csv(merged, patient.suv_csv_path)
    write_patient_deauville_csv(scores, patient.deauville_csv_path)

    from autods_pet.results import ROIResult

    return {
        "scores": scores,
        "extract_results": {
            k: {"stats": v.stats if isinstance(v, ROIResult) else v.get("stats", {})}
            for k, v in merged.items()
            if not k.startswith("_")
        },
    }


def _load_csv_row(cfg: dict[str, Any], patient_id: str) -> dict[str, Any] | None:
    """Load a single patient's row from the metadata CSV.

    Returns *None* if the CSV is not configured, does not exist, or
    does not contain a row for *patient_id*.
    """
    csv_full = _resolve_csv_path(cfg)
    if csv_full is None or not csv_full.exists():
        return None
    import pandas as pd

    df = pd.read_csv(csv_full)
    row = df[df["PatientID"].astype(str) == str(patient_id)]
    if row.empty:
        return None
    d = row.iloc[0].to_dict()
    # Convert NaN to None for consistent null handling.
    return {k: (None if (isinstance(v, float) and v != v) else v) for k, v in d.items()}


def _collect_patient_metadata(
    cfg: dict[str, Any],
    patient_ids: list[str],
    existing: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build metadata rows for patients with incomplete data.

    For each patient, merges JSON sidecar values with existing CSV values.
    Returns only patients that still have missing required columns.
    """
    from autods_pet.patient import resolve_paths

    rows: dict[str, dict[str, Any]] = {}
    for pid in patient_ids:
        paths = resolve_paths(cfg, pid)
        metadata_path = paths.get("pet_metadata")

        # Collect known values from JSON sidecar.
        known: dict[str, Any] = {}
        if metadata_path and metadata_path.exists():
            sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
            for col in METADATA_COLUMNS:
                val = sidecar.get(col)
                if val is not None:
                    known[col] = val

        known["PatientID"] = pid

        # Build patient_row: prefer existing CSV value, then JSON, then empty.
        patient_row: dict[str, Any] = {}
        for col in METADATA_COLUMNS:
            csv_val = existing.get(pid, {}).get(col, "")
            if csv_val not in ("", None):
                patient_row[col] = csv_val
            elif known.get(col) is not None:
                patient_row[col] = known[col]
            else:
                patient_row[col] = ""

        # Skip if all required columns are filled.
        if not any(patient_row[c] in ("", None) for c in METADATA_COLUMNS):
            continue

        rows[pid] = patient_row
    return rows


def generate_metadata_template(
    cfg: dict[str, Any],
    patient_ids: list[str],
) -> Path | None:
    """Generate (or update) a metadata CSV template for patients with gaps.

    For each patient, the JSON sidecar is checked.  If any required field is
    missing or ``null``, a row is added to the template with known values
    pre-filled and missing values left empty.  An existing CSV is merged so
    that user-provided values are preserved.

    Returns the path written, or *None* if every patient already has
    complete metadata.
    """
    from autods_pet.config import resolve_output_dir

    # Determine output path.
    csv_path = _resolve_csv_path(cfg)
    if csv_path is None:
        csv_path = resolve_output_dir(cfg) / "metadata.csv"

    # Load existing CSV rows (preserve user edits).
    existing: dict[str, dict[str, Any]] = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("PatientID", "")
                if pid:
                    existing[pid] = row

    rows = _collect_patient_metadata(cfg, patient_ids, existing)
    if not rows:
        return None

    # Write CSV (preserving rows for patients not in our list).
    all_rows: dict[str, dict[str, Any]] = {}
    all_rows.update(existing)
    all_rows.update(rows)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for pid in sorted(all_rows):
            writer.writerow(all_rows[pid])

    log.info("Wrote metadata template: %s (%d patients)", csv_path, len(rows))
    return csv_path


def _detect_format(path: Path) -> str:
    """Auto-detect input format from *path*.

    Returns ``"dicom"``, ``"nifti"``, or ``"nrrd"``.
    """
    if path.is_dir():
        return "dicom"
    suffix = path.suffix.lower()
    if suffix == ".nrrd":
        return "nrrd"
    if suffix == ".nii" or path.name.endswith(".nii.gz"):
        return "nifti"
    return "dicom"


def _ensure_totalseg_license(cfg: dict[str, Any]) -> None:
    """Run ``totalseg_set_license`` if a license key is configured.

    The license key is passed via the ``-l`` CLI flag to
    ``totalseg_set_license``.  It will be visible in process listings;
    this is a limitation of the TotalSegmentator CLI interface.
    """
    license_key = cfg.get("totalsegmentator", {}).get("license", "")
    if not license_key:
        return
    try:
        subprocess.run(
            ["totalseg_set_license", "-l", license_key],
            check=True,
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "TOTALSEG_LICENSE": license_key},
        )
        log.info("TotalSegmentator license set successfully.")
    except FileNotFoundError:
        log.warning(
            "totalseg_set_license command not found. "
            "Ensure TotalSegmentator is installed."
        )
    except subprocess.CalledProcessError:
        log.warning("Failed to set TotalSegmentator license (see stderr for details).")


def _find_nifti_sources(
    input_dir: Path, ct_dst: Path, pet_dst: Path
) -> list[tuple[Path, Path]]:
    """Match NIfTI files in *input_dir* to output destinations.

    Looks for common naming patterns (CT.nii, CT.nii.gz, PET.nii, PT.nii, etc.)
    and returns a list of (source, destination) pairs.
    """
    pairs: list[tuple[Path, Path]] = []
    ct_names = ["CT.nii", "CT.nii.gz"]
    pet_names = ["PET.nii.gz", "PET.nii", "PT.nii", "PT.nii.gz"]

    for name in ct_names:
        src = input_dir / name
        if src.exists():
            pairs.append((src, ct_dst))
            break

    for name in pet_names:
        src = input_dir / name
        if src.exists():
            pairs.append((src, pet_dst))
            break

    return pairs


def _convert_dicom(
    cfg: dict[str, Any],
    patient: PatientCase,
    input_dir: Path,
    ct_path: Path,
    pet_path: Path,
) -> dict[str, Any]:
    """Convert DICOM series to NIfTI and extract PET metadata."""
    from autods_pet.imaging.dicom import (
        dicom_series_to_nifti,
        extract_patient_weight,
        extract_pet_tags,
        find_series_by_modality,
    )

    size_threshold = cfg.get("dicom", {}).get("size_threshold_kb", 100)
    series = find_series_by_modality(input_dir, size_threshold)

    if not series["CT"]:
        raise FileNotFoundError(f"No CT DICOM series found in {input_dir}")
    if not series["PT"]:
        raise FileNotFoundError(f"No PT DICOM series found in {input_dir}")

    ct_path.parent.mkdir(parents=True, exist_ok=True)
    dicom_series_to_nifti(series["CT"], ct_path)
    dicom_series_to_nifti(series["PT"], pet_path)

    pet_tags = extract_pet_tags(series["PT"][0])
    weight = extract_patient_weight(series["PT"][0])
    metadata = {**pet_tags, "PatientWeight": weight}
    metadata_path = patient.metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    return {"ct_path": ct_path, "pet_path": pet_path, "metadata": metadata}


def _convert_nrrd(
    input_dir: Path,
    ct_path: Path,
    pet_path: Path,
) -> dict[str, Any]:
    """Convert NRRD files to NIfTI, matching CT and PET by filename patterns."""
    import SimpleITK as sitk

    ct_patterns = ["*CT*.nrrd", "*ct*.nrrd"]
    pet_patterns = ["*PT*.nrrd", "*PET*.nrrd", "*pet*.nrrd", "*pt*.nrrd"]
    for patterns, dst_path, label in [
        (ct_patterns, ct_path, "CT"),
        (pet_patterns, pet_path, "PET"),
    ]:
        src = None
        for pat in patterns:
            candidates = list(input_dir.glob(pat))
            if candidates:
                src = candidates[0]
                break
        if src is None:
            all_nrrd = sorted(input_dir.glob("*.nrrd"))
            if len(all_nrrd) == 2:
                log.warning(
                    "Assuming sorted NRRD file order: %s=CT, %s=PET. "
                    "Rename files to include 'CT'/'PT' for explicit matching.",
                    all_nrrd[0].name,
                    all_nrrd[1].name,
                )
                src = all_nrrd[0] if label == "CT" else all_nrrd[1]
            elif len(all_nrrd) == 1:
                raise FileNotFoundError(
                    f"Only one NRRD file found in {input_dir}; "
                    "cannot determine CT vs PET. "
                    "Rename files to include 'CT' or 'PT' in the filename."
                )
        if src is not None:
            img = sitk.ReadImage(str(src))
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(img, str(dst_path))
            log.info("Converted NRRD %s → %s", src, dst_path)
        else:
            raise FileNotFoundError(f"No {label} NRRD file found in {input_dir}")

    return {"ct_path": ct_path, "pet_path": pet_path}


def _convert_nifti(
    input_dir: Path,
    ct_path: Path,
    pet_path: Path,
) -> dict[str, Any]:
    """Copy NIfTI source files to the standard output layout.

    Uses SimpleITK read+write (not raw copy) to ensure proper
    gzip compression when the output path ends with ``.nii.gz``.
    """
    import SimpleITK as sitk

    ct_path.parent.mkdir(parents=True, exist_ok=True)
    for src, dst in _find_nifti_sources(input_dir, ct_path, pet_path):
        if not dst.exists():
            img = sitk.ReadImage(str(src))
            sitk.WriteImage(img, str(dst))
            log.info("Converted %s → %s", src, dst)
    return {"ct_path": ct_path, "pet_path": pet_path}


def _ensure_metadata_from_csv(cfg: dict[str, Any], patient: PatientCase) -> None:
    """Create PET_metadata.json from CSV if the sidecar is missing."""
    metadata_path = patient.metadata_path
    if not metadata_path.exists():
        csv_row = _load_csv_row(cfg, patient.patient_id)
        if csv_row:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(csv_row, indent=2, default=str), encoding="utf-8"
            )
            log.info("Created PET metadata for %s from CSV.", patient.patient_id)


def convert_images(
    cfg: dict[str, Any],
    patient: PatientCase,
) -> dict[str, Any]:
    """Convert input images (DICOM, NRRD, or NIfTI) to the standard NIfTI layout.

    Parameters
    ----------
    cfg : dict
        Configuration dict (from :func:`autods_pet.config.load_config`).
    patient : PatientCase
        Patient case with resolved paths.

    Returns
    -------
    dict[str, Any]
        Keys include ``ct_path``, ``pet_path``, ``skipped`` (bool), and
        optionally ``metadata`` (PET DICOM tags).

    Raises
    ------
    FileNotFoundError
        If the expected CT or PET input files are missing.
    """
    input_dir = patient.input_dir
    ct_path = patient.ct_path
    pet_path = patient.pet_path

    # Skip if output NIfTI already exist.
    if _should_skip(cfg) and ct_path.exists() and pet_path.exists():
        fmt = _detect_format(ct_path)
        if fmt == "nifti":
            log.info(
                "NIfTI files already exist for %s, skipping conversion.",
                patient.patient_id,
            )
            _ensure_metadata_from_csv(cfg, patient)
            return {"ct_path": ct_path, "pet_path": pet_path, "skipped": True}

    # Detect format from input directory.
    fmt = _detect_format(input_dir)
    has_dcm = fmt == "dicom" and any(input_dir.rglob("*.dcm"))
    if not has_dcm:
        nifti_pairs = _find_nifti_sources(input_dir, ct_path, pet_path)
        if nifti_pairs:
            fmt = "nifti"

    if fmt == "dicom":
        result = _convert_dicom(cfg, patient, input_dir, ct_path, pet_path)
    elif fmt == "nrrd":
        result = _convert_nrrd(input_dir, ct_path, pet_path)
    elif fmt == "nifti":
        result = _convert_nifti(input_dir, ct_path, pet_path)
    else:
        result = {"ct_path": ct_path, "pet_path": pet_path}

    _ensure_metadata_from_csv(cfg, patient)

    result.setdefault("skipped", False)
    return result


def _resolve_pet_metadata(cfg: dict[str, Any], patient: PatientCase) -> dict[str, Any]:
    """Merge PET metadata from JSON sidecar and CSV fallback.

    Raises
    ------
    ValueError
        If no metadata is available for the patient.
    """
    metadata: dict[str, Any] = {}
    metadata_path = patient.metadata_path
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    csv_row = _load_csv_row(cfg, patient.patient_id)
    if csv_row:
        for key, value in csv_row.items():
            if metadata.get(key) is None and value is not None:
                metadata[key] = value

    if not metadata:
        raise ValueError(
            f"No PET metadata found for {patient.patient_id}. "
            "Run 'autods-pet convert' first or provide a metadata_csv."
        )
    return metadata


def normalize_pet(
    cfg: dict[str, Any],
    patient: PatientCase,
) -> dict[str, Any]:
    """Compute SUV body-weight from a raw PET image.

    Parameters
    ----------
    cfg : dict
        Configuration dict.
    patient : PatientCase
        Patient case with resolved paths.

    Returns
    -------
    dict[str, Any]
        Keys: ``pet_suv_path`` (:class:`~pathlib.Path`), ``skipped`` (bool).

    Raises
    ------
    ValueError
        If no PET metadata is available for the patient.
    """
    import SimpleITK as sitk

    from autods_pet.imaging.normalization import (
        compute_suvbw,
        effective_dose,
        parse_dicom_date,
        parse_dicom_time,
        seconds_between,
    )

    pet_suv_path = patient.pet_suv_path

    if _should_skip(cfg) and pet_suv_path.exists():
        log.info(
            "PET SUV already exists for %s, skipping normalization.",
            patient.patient_id,
        )
        return {"pet_suv_path": pet_suv_path, "skipped": True}

    metadata = _resolve_pet_metadata(cfg, patient)

    study_date = parse_dicom_date(str(metadata["StudyDate"]))
    acq_time = parse_dicom_time(str(metadata["AcquisitionTime"]))
    inj_time = parse_dicom_time(str(metadata["RadiopharmaceuticalStartTime"]))

    # Use RadiopharmaceuticalStartDateTime (0018,1078) when available - it
    # encodes the full date+time so multi-day protocols are handled correctly.
    # Fall back to combining injection time with StudyDate (assumes same day).
    rph_start_dt = metadata.get("RadiopharmaceuticalStartDateTime")
    if rph_start_dt and str(rph_start_dt).strip():
        try:
            rph_str = str(rph_start_dt).strip()
            inj_date = parse_dicom_date(rph_str[:8])
            inj_time_full = (
                parse_dicom_time(rph_str[8:]) if len(rph_str) > 8 else inj_time
            )
            injection_dt = datetime.combine(inj_date, inj_time_full)
        except (ValueError, IndexError):
            injection_dt = datetime.combine(study_date, inj_time)
    else:
        injection_dt = datetime.combine(study_date, inj_time)

    acquisition_dt = datetime.combine(study_date, acq_time)
    elapsed_s = seconds_between(injection_dt, acquisition_dt)

    dose_bq = float(metadata["RadionuclideTotalDose"])
    half_life_s = float(metadata["RadionuclideHalfLife"])
    decay_corr = str(metadata.get("DecayCorrection", "NONE"))
    raw_weight = metadata.get("PatientWeight")
    if raw_weight is None:
        raise ValueError(
            f"No PatientWeight for {patient.patient_id}. "
            "Add it to PET_metadata.json or the metadata CSV."
        )
    weight_kg = float(raw_weight)

    eff_dose = effective_dose(dose_bq, half_life_s, elapsed_s, decay_corr)

    pet_img = sitk.ReadImage(str(patient.pet_path))
    suv_img = compute_suvbw(pet_img, weight_kg, eff_dose)

    pet_suv_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(suv_img, str(pet_suv_path))
    log.info("Wrote SUV image: %s", pet_suv_path)

    return {"pet_suv_path": pet_suv_path, "skipped": False}


def register_pet(
    cfg: dict[str, Any],
    patient: PatientCase,
) -> dict[str, Any]:
    """Rigidly register PET SUV image onto the CT grid.

    Parameters
    ----------
    cfg : dict
        Configuration dict.
    patient : PatientCase
        Patient case with resolved paths.

    Returns
    -------
    dict[str, Any]
        Keys: ``pet_registered_path`` (:class:`~pathlib.Path`), ``skipped`` (bool).
    """
    import SimpleITK as sitk

    from autods_pet.imaging.registration import rigid_register_pet_to_ct

    pet_reg_path = patient.pet_registered_path

    if _should_skip(cfg) and pet_reg_path.exists():
        log.info(
            "Registered PET already exists for %s, skipping.",
            patient.patient_id,
        )
        return {"pet_registered_path": pet_reg_path, "skipped": True}

    ct = sitk.ReadImage(str(patient.ct_path))
    pet_suv = sitk.ReadImage(str(patient.pet_suv_path))

    registered = rigid_register_pet_to_ct(
        ct, pet_suv, report_path=patient.elastix_report_path
    )

    pet_reg_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(registered, str(pet_reg_path))
    log.info("Wrote registered PET: %s", pet_reg_path)

    return {"pet_registered_path": pet_reg_path, "skipped": False}


def _copy_preexisting_segs(
    patient: PatientCase,
    seg_dir: Path,
    vb_path: Path,
    totseg_filename: str,
) -> None:
    """Copy pre-existing segmentations from ``input_seg_dir`` to output."""
    import shutil

    input_seg = patient.input_seg_dir
    if not input_seg.exists():
        return

    seg_dir.mkdir(parents=True, exist_ok=True)

    # Copy TotSeg multilabel
    for src_name in [totseg_filename, "whole_seg.nii", "whole_seg.nii.gz"]:
        src = input_seg / src_name
        if src.exists() and not (seg_dir / totseg_filename).exists():
            shutil.copy2(src, seg_dir / totseg_filename)
            log.info("Copied %s from input to output.", totseg_filename)
            break

    # Copy vertebral body
    for src_name in ["vertebral_body.nii.gz", "vertebral_body.nii"]:
        src = input_seg / src_name
        if src.exists() and not vb_path.exists():
            vb_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, vb_path)
            log.info("Copied vertebral_body from input to output.")
            break

    # Copy refined masks
    input_refined = input_seg / "refined"
    if input_refined.exists():
        output_refined = seg_dir / "refined"
        output_refined.mkdir(parents=True, exist_ok=True)
        for f in input_refined.iterdir():
            if f.is_file():
                dst = output_refined / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    log.info("Copied refined mask %s from input.", f.name)


def segment_ct(
    cfg: dict[str, Any],
    patient: PatientCase,
) -> dict[str, Any]:
    """Run TotalSegmentator on the CT image.

    Parameters
    ----------
    cfg : dict
        Configuration dict (reads ``totalsegmentator.fast`` and license key).
    patient : PatientCase
        Patient case with resolved paths.

    Returns
    -------
    dict[str, Any]
        Keys: ``seg_multilabel`` (:class:`~pathlib.Path`),
        ``vb_available`` (bool), and optionally ``vert_body_seg``.

    Raises
    ------
    RuntimeError
        If the main ``total`` segmentation task fails.
    """
    from autods_pet.imaging.segmentation import TOTSEG_FILENAME, run_totalsegmentator

    _ensure_totalseg_license(cfg)

    fast = cfg.get("totalsegmentator", {}).get("fast", False)
    seg_dir = patient.seg_dir
    ct_path = patient.ct_path
    vb_path = patient.vert_body_seg_path

    _copy_preexisting_segs(patient, seg_dir, vb_path, TOTSEG_FILENAME)

    result: dict[str, Any] = {}

    seg_file = seg_dir / TOTSEG_FILENAME
    if _should_skip(cfg) and seg_file.exists():
        log.info(
            "Segmentation already exists for %s, skipping total task.",
            patient.patient_id,
        )
        result["seg_multilabel"] = seg_file
    else:
        try:
            seg_path = run_totalsegmentator(ct_path, seg_dir, task="total", fast=fast)
            result["seg_multilabel"] = seg_path
        except Exception as exc:
            raise RuntimeError(f"TotalSegmentator (total) failed: {exc}") from exc

    if _should_skip(cfg) and vb_path.exists():
        log.info(
            "Vertebral body seg already exists for %s, skipping.",
            patient.patient_id,
        )
        result["vert_body_seg"] = vb_path
        result["vb_available"] = True
    else:
        try:
            run_totalsegmentator(
                ct_path,
                seg_dir,
                task="vertebrae_body",
                fast=False,  # vertebrae_body does not support fast mode
                output_filename="vertebral_body.nii.gz",
            )
            result["vert_body_seg"] = vb_path
            result["vb_available"] = True
        except Exception as exc:
            log.warning(
                "Vertebrae body segmentation failed (license may be required): %s",
                exc,
            )
            result["vb_available"] = False

    return result


def _extract_aorta_mbp(
    cfg: dict[str, Any], whole_seg: Any, pet: Any
) -> tuple[Any | None, tuple[str, str, str]]:
    """Extract Aorta MBP ROI, returning (result_or_None, status_tuple)."""
    from autods_pet.config import get_roi_config
    from autods_pet.roi import AortaMBP

    try:
        aorta_cfg = get_roi_config(cfg, "aorta_mbp")
        aorta = AortaMBP(
            vertebra_labels=aorta_cfg.get("vertebra_labels"),
            slab_axis=aorta_cfg.get("slab_axis", 2),
            heart_exclusion_mode=aorta_cfg.get(
                "heart_exclusion_mode", "dilate_intersection"
            ),
            heart_dilation_mm=aorta_cfg.get("heart_dilation_mm", 6.0),
            heart_distance_mm=aorta_cfg.get("heart_distance_mm", 12.0),
            aorta_erosion_mm=aorta_cfg.get("aorta_erosion_mm", 4.0),
            stats=aorta_cfg.get("stats"),
        )
        result = aorta.extract(whole_seg, pet)
        return result, ("Aorta MBP", "ok", "")
    except Exception as exc:
        log.error("Aorta MBP extraction failed: %s", exc)
        return None, ("Aorta MBP", "error", str(exc))


def _extract_liver(
    cfg: dict[str, Any], whole_seg: Any, pet: Any
) -> tuple[Any | None, tuple[str, str, str]]:
    """Extract Liver ROI, returning (result_or_None, status_tuple)."""
    from autods_pet.config import get_roi_config
    from autods_pet.roi import LiverROI

    try:
        liver_cfg = get_roi_config(cfg, "liver")
        liver = LiverROI(
            erosion_mm=liver_cfg.get("erosion_mm", 10.0),
            max_hole_volume_mm3=liver_cfg.get("max_hole_volume_mm3"),
            stats=liver_cfg.get("stats"),
        )
        result = liver.extract(whole_seg, pet)
        return result, ("Liver", "ok", "")
    except Exception as exc:
        log.error("Liver extraction failed: %s", exc)
        return None, ("Liver", "error", str(exc))


def _extract_brain(
    cfg: dict[str, Any], whole_seg: Any, pet: Any
) -> tuple[Any | None, tuple[str, str, str]]:
    """Extract Brain ROI, returning (result_or_None, status_tuple)."""
    from autods_pet.config import get_roi_config
    from autods_pet.roi import BrainROI

    try:
        brain_cfg = get_roi_config(cfg, "brain")
        brain = BrainROI(
            brain_label=brain_cfg.get("label", 90),
            grey_matter_only=brain_cfg.get("grey_matter_only", True),
            cortical_thickness_mm=brain_cfg.get("cortical_thickness_mm", 5.0),
            stats=brain_cfg.get("stats"),
        )
        result = brain.extract(whole_seg, pet)
        if result is None:
            return None, ("Brain", "skip", "brain label not in FOV")
        return result, ("Brain", "ok", "")
    except Exception as exc:
        log.error("Brain extraction failed: %s", exc)
        return None, ("Brain", "error", str(exc))


def _extract_lumbar_vb(
    cfg: dict[str, Any],
    whole_seg: Any,
    pet: Any,
    patient: PatientCase,
    seg_result: dict[str, Any] | None,
) -> tuple[Any | None, tuple[str, str, str]]:
    """Extract Lumbar VB ROI, returning (result_or_None, status_tuple)."""
    from autods_pet.config import get_roi_config
    from autods_pet.roi import LumbarVB

    vb_available = (
        seg_result.get("vb_available", False) if seg_result else False
    ) or patient.vert_body_seg_path.exists()

    if not vb_available:
        return None, ("Lumbar VB (L3-L5)", "skip", "no VB seg")

    try:
        lumbar_cfg = get_roi_config(cfg, "lumbar_vb")
        vert_body = patient.load_vert_body_seg()
        if vert_body is None:
            return None, ("Lumbar VB (L3-L5)", "skip", "VB seg file missing")
        lumbar = LumbarVB(
            lumbar_labels=lumbar_cfg.get("labels"),
            erosion_mm=lumbar_cfg.get("erosion_mm", 3.0),
            stats=lumbar_cfg.get("stats"),
        )
        result = lumbar.extract(whole_seg, vert_body, pet)
        return result, ("Lumbar VB (L3-L5)", "ok", "")
    except Exception as exc:
        log.error("Lumbar VB extraction failed: %s", exc)
        return None, ("Lumbar VB (L3-L5)", "error", str(exc))


def _extract_long_bones(
    cfg: dict[str, Any], whole_seg: Any, pet: Any
) -> tuple[Any | None, tuple[str, str, str]]:
    """Extract Long Bones ROI, returning (result_or_None, status_tuple)."""
    from autods_pet.config import get_roi_config
    from autods_pet.roi import LongBonesROI

    try:
        lb_cfg = get_roi_config(cfg, "long_bones")
        bones_cfg = lb_cfg.get("bones", [])
        bones_tuples = [
            (b["name"], b["label"], b.get("erosion_mm", 0.0)) for b in bones_cfg
        ]
        long_bones = LongBonesROI(
            bones=bones_tuples if bones_tuples else None,
            diaphysis_keep_pct=lb_cfg.get("diaphysis_keep_pct", 60),
            stats=lb_cfg.get("stats"),
        )
        result = long_bones.extract(whole_seg, pet)
        return result, ("Long bones", "ok", "")
    except Exception as exc:
        log.error("Long bones extraction failed: %s", exc)
        return None, ("Long bones", "error", str(exc))


def _resolve_mask_geometry(
    mask: Any,
    pet: Any,
    patient: PatientCase,
    display_name: str,
) -> Any:
    """Align *mask* to the *pet* (CT-registered) grid if geometries differ.

    Handles four cases: full CT match (no-op), full PET match (Elastix),
    CT sub-volume (resample), PET sub-volume (resample + Elastix).
    """
    import SimpleITK as sitk

    from autods_pet.imaging.geometry import (
        check_same_geometry,
        check_sub_geometry,
        resample_to_reference,
    )

    if check_same_geometry(mask, pet):
        return mask

    pet_suv = sitk.ReadImage(str(patient.pet_suv_path))

    if check_same_geometry(mask, pet_suv):
        mask = _apply_elastix(mask, patient, display_name, "PET to CT")
    elif check_sub_geometry(mask, pet):
        mask = resample_to_reference(mask, pet)
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        log.info("Resampled %s sub-volume mask onto CT grid.", display_name)
    elif check_sub_geometry(mask, pet_suv):
        mask = resample_to_reference(mask, pet_suv)
        mask = _apply_elastix(mask, patient, display_name, "PET grid, then to CT")
    else:
        raise ValueError(
            f"Mask geometry doesn't match CT or PET for {display_name}. "
            f"Mask size={mask.GetSize()}, "
            f"CT size={pet.GetSize()}, "
            f"PET size={pet_suv.GetSize()}"
        )
    return mask


def _apply_elastix(
    mask: Any, patient: PatientCase, display_name: str, description: str
) -> Any:
    """Apply the saved Elastix transform to *mask*, returning a binary uint8."""
    import SimpleITK as sitk

    from autods_pet.imaging.registration import apply_transform

    transform_path = patient.elastix_report_path
    if not transform_path.exists():
        raise FileNotFoundError(
            f"Elastix transform not found at {transform_path}. "
            "Run 'autods-pet register' first."
        )
    mask = apply_transform(mask, transform_path, nearest_neighbor=True)
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    log.info("Auto-registered %s mask from %s space.", display_name, description)
    return mask


def _extract_targets(
    cfg: dict[str, Any],
    pet: Any,
    patient: PatientCase,
) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    """Extract target ROIs (FL, PM, EM, custom).

    Returns ``(results_dict, statuses_list)``.
    """
    import SimpleITK as sitk

    from autods_pet.config import get_all_targets
    from autods_pet.roi import TargetROI

    results: dict[str, Any] = {}
    statuses: list[tuple[str, str, str]] = []

    targets = get_all_targets(cfg)
    target_display = {
        "focal_lesion": "Focal lesion",
        "paramedullary": "Paramedullary",
        "extramedullary": "Extramedullary",
    }
    all_named = list(target_display.keys())
    seen_names = {t["name"] for t in targets}

    # Discovery warnings are surfaced by the CLI gate (_has_new_target_masks)
    # before this function is reached, so we deliberately do not re-log them
    # here to avoid duplicate output.
    discovered, _warnings = _discover_patient_masks(cfg, patient)

    def _primary_stem(target_cfg: dict[str, Any]) -> str:
        stems = target_cfg.get("mask_filename") or []
        if isinstance(stems, str):
            stems = [stems]
        if stems:
            return stems[0]
        return target_cfg["name"]

    for target_cfg in targets:
        tname = target_cfg["name"]
        display_name = target_display.get(tname, tname)
        mask_info = discovered.get(tname)

        if mask_info is None:
            statuses.append((display_name, "skip", "no mask"))
            continue

        try:
            if mask_info.format == "dicom_seg":
                from autods_pet.ops.dicom_seg import read_dicom_seg

                mask = read_dicom_seg(
                    mask_info.path, segment_label=mask_info.segment_label
                )
            else:
                mask = sitk.ReadImage(str(mask_info.path))

            mask = _resolve_mask_geometry(mask, pet, patient, display_name)

            target = TargetROI(stats=target_cfg.get("stats"))
            result = target.extract(mask, pet)
            results[display_name] = result

            # Copy mask to output segmentations folder.  Use the primary
            # mask_filename stem when available so that the cached output
            # name remains predictable across runs.
            out_stem = _primary_stem(target_cfg)
            out_mask = patient.seg_dir / f"{out_stem}.nii.gz"
            out_mask.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(result.refined_mask, str(out_mask))

            statuses.append((display_name, "ok", ""))
        except Exception as exc:
            log.error("%s extraction failed: %s", display_name, exc)
            statuses.append((display_name, "error", str(exc)))

    for named in all_named:
        if named not in seen_names:
            display = target_display[named]
            if not any(s[0] == display for s in statuses):
                statuses.append((display, "skip", "no mask"))

    return results, statuses


# ROI keys that are NOT targets (used to identify target masks for subtraction)
_NON_TARGET_KEYS = frozenset(
    {
        "Aorta MBP",
        "Liver",
        "Brain",
        "Long bones",
        "Lumbar VB",
    }
)

# Marrow ROI keys eligible for lesion subtraction
_MARROW_KEYS = ("Lumbar VB", "Long bones")

# Config section name for each marrow key (for stats lookup)
_MARROW_CONFIG_SECTION = {
    "Lumbar VB": "lumbar_vb",
    "Long bones": "long_bones",
}


def _subtract_lesions_from_marrow(
    results: dict[str, Any],
    pet: Any,
    cfg: dict[str, Any],
) -> None:
    """Subtract target lesion masks from marrow ROIs and recompute stats.

    Modifies *results* in place.  If a marrow ROI becomes empty after
    subtraction, the original mask and stats are kept and a warning is
    logged.
    """
    import SimpleITK as sitk

    from autods_pet.ops.masks import subtract_mask
    from autods_pet.ops.stats import compute_stats, count_voxels

    # Collect all target refined masks.
    target_masks = []
    for key, roi_result in results.items():
        if key.startswith("_") or key in _NON_TARGET_KEYS:
            continue
        if hasattr(roi_result, "refined_mask") and roi_result.refined_mask is not None:
            target_masks.append(roi_result.refined_mask)

    if not target_masks:
        log.info("No target lesion masks available - skipping marrow subtraction.")
        return

    # Union all target masks into a single combined lesion mask.
    combined = target_masks[0]
    for m in target_masks[1:]:
        combined = sitk.Or(
            sitk.Cast(combined != 0, sitk.sitkUInt8),
            sitk.Cast(m != 0, sitk.sitkUInt8),
        )
    combined = sitk.Cast(combined != 0, sitk.sitkUInt8)

    for marrow_key in _MARROW_KEYS:
        if marrow_key not in results:
            continue
        roi_result = results[marrow_key]
        if roi_result.refined_mask is None:
            continue

        original_mask = roi_result.refined_mask
        original_count = count_voxels(original_mask)
        subtracted = subtract_mask(original_mask, combined)
        new_count = count_voxels(subtracted)

        if new_count == 0:
            log.warning(
                "%s mask is empty after lesion subtraction - keeping original.",
                marrow_key,
            )
            continue

        log.info(
            "%s: subtracted lesion voxels (%d -> %d, removed %d).",
            marrow_key,
            original_count,
            new_count,
            original_count - new_count,
        )
        roi_result.refined_mask = subtracted
        section = _MARROW_CONFIG_SECTION.get(marrow_key, marrow_key.lower())
        stat_names = cfg.get(section, {}).get("stats", ["p95"])
        roi_result.stats = compute_stats(stat_names, pet, subtracted)


def extract_rois(
    cfg: dict[str, Any],
    patient: PatientCase,
    seg_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract ROI statistics from PET.

    Returns a dict mapping ROI display names to :class:`ROIResult` instances,
    plus a ``"_roi_statuses"`` key with status tuples.
    """
    import SimpleITK as sitk

    from autods_pet.ops.save_masks import _REFINED_NAME_MAP
    from autods_pet.ops.stats import compute_stats
    from autods_pet.results import ROIResult

    # Map ROI display name to config section for stats lookup
    _ROI_CONFIG_SECTION = {
        "Aorta MBP": "aorta_mbp",
        "Liver": "liver",
        "Brain": "brain",
        "Long bones": "long_bones",
    }

    results: dict[str, Any] = {}
    roi_statuses: list[tuple[str, str, str]] = []

    whole_seg = patient.load_segmentation()
    pet = patient.load_pet_registered()

    # Reference ROIs
    for extract_fn, key, args in [
        (_extract_aorta_mbp, "Aorta MBP", (cfg, whole_seg, pet)),
        (_extract_liver, "Liver", (cfg, whole_seg, pet)),
        (_extract_brain, "Brain", (cfg, whole_seg, pet)),
        (_extract_long_bones, "Long bones", (cfg, whole_seg, pet)),
    ]:
        refined_filename = _REFINED_NAME_MAP.get(key, key.lower().replace(" ", "_"))
        refined_path = patient.seg_dir / "refined" / f"{refined_filename}.nii.gz"
        if _should_skip(cfg) and refined_path.exists():
            mask = sitk.ReadImage(str(refined_path))
            section = _ROI_CONFIG_SECTION.get(key, key.lower())
            stats_names = cfg.get(section, {}).get("stats", ["median"])
            stats = compute_stats(stats_names, pet, mask)
            results[key] = ROIResult(stats=stats, refined_mask=mask)
            roi_statuses.append((key, "ok", ""))
            log.info(
                "Loaded pre-existing refined mask for %s, skipping extraction.", key
            )
            continue
        roi_result, status = extract_fn(*args)
        if roi_result is not None:
            results[key] = roi_result
        roi_statuses.append(status)

    # Lumbar VB (needs extra args)
    lumbar_refined_filename = _REFINED_NAME_MAP.get("Lumbar VB", "lumbar_vb")
    lumbar_refined_path = (
        patient.seg_dir / "refined" / f"{lumbar_refined_filename}.nii.gz"
    )
    if _should_skip(cfg) and lumbar_refined_path.exists():
        mask = sitk.ReadImage(str(lumbar_refined_path))
        stats_names = cfg.get("lumbar_vb", {}).get("stats", ["p95"])
        stats = compute_stats(stats_names, pet, mask)
        results["Lumbar VB"] = ROIResult(stats=stats, refined_mask=mask)
        roi_statuses.append(("Lumbar VB (L3-L5)", "ok", ""))
        log.info("Loaded pre-existing refined mask for Lumbar VB, skipping extraction.")
    else:
        vb_result, vb_status = _extract_lumbar_vb(
            cfg, whole_seg, pet, patient, seg_result
        )
        if vb_result is not None:
            results["Lumbar VB"] = vb_result
        roi_statuses.append(vb_status)

    # Target ROIs
    target_results, target_statuses = _extract_targets(cfg, pet, patient)
    results.update(target_results)
    roi_statuses.extend(target_statuses)

    # Optionally subtract lesion masks from marrow ROIs.
    if cfg.get("output", {}).get("subtract_lesions_from_marrow", False):
        _subtract_lesions_from_marrow(results, pet, cfg)

    results["_roi_statuses"] = roi_statuses

    output_cfg = cfg.get("output", {})
    if output_cfg.get("save_raw_masks", False):
        from autods_pet.ops.save_masks import save_raw_masks as _save_raw

        _save_raw(whole_seg, patient.seg_dir)

    if output_cfg.get("save_refined_masks", False):
        from autods_pet.ops.save_masks import save_refined_masks as _save_refined

        _save_refined(results, patient.seg_dir)

    return results


def _get_roi_stats(data: Any) -> dict[str, float | None]:
    """Extract stats dict from ROIResult or plain dict."""
    from autods_pet.results import ROIResult

    if data is None:
        return {}
    if isinstance(data, ROIResult):
        return data.stats
    return data.get("stats", {})


def _pick_best_stat(
    stats: dict[str, float | None], priority: tuple[str, ...]
) -> float | None:
    """Return the first non-None stat value matching *priority* order."""
    for key in priority:
        if key in stats and stats[key] is not None:
            return stats[key]
    return None


def _score_named_targets(
    extract_results: dict[str, Any],
    mbp: float,
    liver: float,
    liver_multiplier: float,
) -> dict[str, int]:
    """Compute DS for named target ROIs (FL, PM, EM)."""
    from autods_pet.deauville import assign_ds

    scores: dict[str, int] = {}
    for roi_name, ds_key in DS_TARGET_NAMES.items():
        stats = _get_roi_stats(extract_results.get(roi_name))
        target_value = _pick_best_stat(stats, ("max", "p95", "median", "mean"))
        if target_value is not None:
            scores[ds_key] = assign_ds(
                target_value,
                mbp,
                liver,
                allow_ds1=(ds_key == "FL_DS"),
                liver_multiplier=liver_multiplier,
            )
    return scores


def _score_research_metrics(
    extract_results: dict[str, Any],
    mbp: float,
    liver: float,
    liver_multiplier: float,
) -> dict[str, int]:
    """Compute DS-equivalent for lumbar VB and long bones."""
    from autods_pet.deauville import assign_ds

    scores: dict[str, int] = {}
    for roi_name, ds_key in DS_RESEARCH_NAMES.items():
        stats = _get_roi_stats(extract_results.get(roi_name))
        target_value = _pick_best_stat(stats, ("p95", "max", "median", "mean"))
        if target_value is not None:
            scores[ds_key] = assign_ds(
                target_value,
                mbp,
                liver,
                liver_multiplier=liver_multiplier,
            )
    return scores


def _score_custom_targets(
    extract_results: dict[str, Any],
    mbp: float,
    liver: float,
    liver_multiplier: float,
    skip_rois: set[str],
) -> dict[str, int]:
    """Compute DS for user-defined custom target ROIs."""
    from autods_pet.deauville import assign_ds

    scores: dict[str, int] = {}
    for roi_name, roi_data in extract_results.items():
        if roi_name.startswith("_") or roi_name in skip_rois:
            continue
        stats = _get_roi_stats(roi_data)
        target_value = next(iter(stats.values()), None) if stats else None
        if target_value is not None:
            scores[roi_name] = assign_ds(
                target_value,
                mbp,
                liver,
                liver_multiplier=liver_multiplier,
            )
    return scores


def _compute_blr(extract_results: dict[str, Any], liver_median: float) -> float | None:
    """Compute Brain-to-Liver Ratio, or None if brain median is unavailable."""
    brain_median = _get_roi_stats(extract_results.get("Brain")).get("median")
    if brain_median is not None and liver_median > 0:
        return round(brain_median / liver_median, 4)
    return None


def score_deauville(
    cfg: dict[str, Any],
    extract_results: dict[str, Any],
) -> dict[str, int | float]:
    """Assign Deauville Scores from extracted ROI statistics.

    Parameters
    ----------
    cfg : dict
        Configuration dict.  Reads ``cfg["deauville"]["liver_multiplier"]``
        (default 2.0) for the DS 4 / DS 5 cutoff.
    extract_results : dict[str, Any]
        Output of :func:`extract_rois`, mapping ROI display names to
        :class:`~autods_pet.results.ROIResult` instances.

    Returns
    -------
    dict[str, int | float]
        Mapping of score keys (e.g. ``"FL_DS"``, ``"BM_DS"``, ``"BLR"``)
        to Deauville Scores (int, 1--5) or the Brain-to-Liver Ratio
        (float).  Empty if references are unavailable.
    """
    scores: dict[str, int | float] = {}
    liver_multiplier = cfg.get("deauville", {}).get("liver_multiplier", 2.0)

    mbp_median = _get_roi_stats(extract_results.get("Aorta MBP")).get("median")
    liver_median = _get_roi_stats(extract_results.get("Liver")).get("median")

    if mbp_median is None or liver_median is None:
        log.warning(
            "Cannot compute DS: MBP median=%s, Liver median=%s",
            mbp_median,
            liver_median,
        )
        return scores

    skip_rois = (
        set(DS_TARGET_NAMES) | set(DS_RESEARCH_NAMES) | {"Aorta MBP", "Liver", "Brain"}
    )

    scores.update(
        _score_named_targets(
            extract_results, mbp_median, liver_median, liver_multiplier
        )
    )
    scores.update(
        _score_research_metrics(
            extract_results, mbp_median, liver_median, liver_multiplier
        )
    )
    scores.update(
        _score_custom_targets(
            extract_results, mbp_median, liver_median, liver_multiplier, skip_rois
        )
    )

    blr = _compute_blr(extract_results, liver_median)
    if blr is not None:
        scores["BLR"] = blr

    return scores


def write_patient_suv_csv(
    extract_results: dict[str, Any],
    path: Path,
) -> None:
    """Write per-patient SUV values to a CSV file (merge mode).

    Reads the existing file (if any), merges new entries (update existing,
    add new), and writes back.  One row per (ROI, stat) pair.
    Columns: ROI, Statistic, Value.
    """
    from autods_pet.results import ROIResult

    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing entries keyed by (ROI, Statistic)
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["ROI"], row["Statistic"])
                existing[key] = row

    # Build new entries
    for roi_name, roi_data in extract_results.items():
        if roi_name.startswith("_"):
            continue
        if isinstance(roi_data, ROIResult) and roi_data.stats:
            for stat_name, value in roi_data.stats.items():
                key = (roi_name.replace(" ", "_"), stat_name)
                existing[key] = {
                    "ROI": key[0],
                    "Statistic": key[1],
                    "Value": value,
                }

    rows = list(existing.values())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ROI", "Statistic", "Value"])
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote SUV values: %s (%d entries)", path, len(rows))


def write_patient_deauville_csv(
    scores: dict[str, int | float],
    path: Path,
) -> None:
    """Write per-patient Deauville scores to a CSV file (merge mode).

    Reads the existing file (if any), merges new scores (update existing,
    add new), and writes back.  One row per target.
    Columns: Target, DeauvilleScore.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing entries keyed by Target
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["Target"]] = row

    # Merge new scores
    for k, v in scores.items():
        existing[k] = {"Target": k, "DeauvilleScore": v}

    rows = list(existing.values())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Target", "DeauvilleScore"])
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote Deauville scores: %s (%d targets)", path, len(rows))


def save_batch_csv(
    new_df: Any,
    path: Path,
    output_format: str = "csv",
    int_columns: list[str] | None = None,
) -> Path:
    """Save batch results, merging with an existing file if present.

    Existing patients are updated (replaced), new patients are appended.

    Parameters
    ----------
    new_df : pandas.DataFrame
        New batch results (must have a ``patient_id`` column).
    path : Path
        Destination file path.
    output_format : str
        ``"csv"`` or ``"xlsx"``.
    int_columns : list[str] or None
        Columns to cast to nullable ``Int64`` after merge/concat.
    """
    import pandas as pd

    from autods_pet.io import save_dataframe

    if new_df.empty or "patient_id" not in new_df.columns:
        return path  # Nothing to write.

    if path.exists():
        existing = pd.read_csv(path)
        if "patient_id" not in existing.columns:
            existing = pd.DataFrame()
        else:
            existing["patient_id"] = existing["patient_id"].astype(str)
        new_df["patient_id"] = new_df["patient_id"].astype(str)
        existing = existing[~existing["patient_id"].isin(new_df["patient_id"])]
        new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df = new_df.sort_values("patient_id").reset_index(drop=True)
    if int_columns:
        for c in int_columns:
            if c in new_df.columns:
                new_df[c] = new_df[c].astype("Int64")
    return save_dataframe(new_df, path, output_format)


def _result_to_batch_dfs(
    result: DeauvilleResult,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one-row DS and SUV DataFrames from a single DeauvilleResult.

    Parameters
    ----------
    result : DeauvilleResult
        A successful pipeline result (``result.error`` should be ``None``).
    cfg : dict
        Configuration dict (unused currently, kept for future flexibility).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(ds_df, suv_df)`` each with a single row.
    """
    import pandas as pd

    from autods_pet.results import ROIResult

    # DS row
    ds_row: dict[str, Any] = {"patient_id": result.patient_id}
    ds_row.update(result.scores)
    ds_df = pd.DataFrame([ds_row])
    cols = ["patient_id"] + list(DS_COLUMN_ORDER)
    for c in cols:
        if c not in ds_df.columns:
            ds_df[c] = pd.NA
    for c in DS_COLUMN_ORDER:
        if c != "BLR" and c in ds_df.columns:
            ds_df[c] = ds_df[c].astype("Int64")
    ds_df = ds_df[cols]

    # SUV row
    suv_row: dict[str, Any] = {"patient_id": result.patient_id}
    for roi_name, roi_data in result.rois.items():
        if isinstance(roi_data, ROIResult) and roi_data.stats:
            for stat_name, value in roi_data.stats.items():
                suv_row[f"{roi_name.replace(' ', '_')}_{stat_name}"] = value
    suv_df = pd.DataFrame([suv_row])

    return ds_df, suv_df


class DeauvillePipeline:
    """Orchestrates the full DS computation pipeline.

    Parameters
    ----------
    cfg : dict
        Configuration dict (from :func:`autods_pet.config.load_config`).
    force : bool
        When *True*, re-run every stage even if outputs already exist
        (sets ``cfg["pipeline"]["force"] = True``).  Defaults to *False*,
        which skips stages whose output files are already on disk.

    Examples
    --------
    >>> from autods_pet import DeauvillePipeline, load_config
    >>> cfg = load_config("config.ini")
    >>> pipeline = DeauvillePipeline(cfg)
    >>> result = pipeline.run("PATIENT_001")
    >>> print(result.scores)
    """

    def __init__(self, cfg: dict[str, Any], force: bool = False) -> None:
        self.cfg = cfg
        if force:
            self.cfg.setdefault("pipeline", {})["force"] = True

    def run(self, patient_id: str) -> DeauvilleResult:
        """Run the full pipeline for one patient.

        Returns
        -------
        DeauvilleResult
            Contains ``patient_id``, ``scores``, ``rois``, and ``error``.
        """
        from autods_pet.config import resolve_output_dir
        from autods_pet.patient import PatientCase
        from autods_pet.results import DeauvilleResult, ROIResult

        patient = PatientCase(self.cfg, patient_id)

        try:
            self.convert(patient)
            self.normalize(patient)
            self.register(patient)
            seg_result = self.segment(patient)
            extract_results = self.extract(patient, seg_result)
            scores = self.score(extract_results)

            # Write per-patient result CSVs.
            write_patient_suv_csv(extract_results, patient.suv_csv_path)
            write_patient_deauville_csv(scores, patient.deauville_csv_path)

            # Convert extract_results to ROIResult dict (filter internals)
            rois = {
                k: v
                for k, v in extract_results.items()
                if not k.startswith("_") and isinstance(v, ROIResult)
            }

            result = DeauvilleResult(
                patient_id=patient_id,
                scores=scores,
                rois=rois,
            )

            # Append to batch CSVs
            output_dir = resolve_output_dir(self.cfg)
            output_dir.mkdir(parents=True, exist_ok=True)
            ds_df, suv_df = _result_to_batch_dfs(result, self.cfg)
            _ds_int_cols = [c for c in DS_COLUMN_ORDER if c != "BLR"]
            save_batch_csv(
                ds_df, output_dir / "batch_results_DS.csv", int_columns=_ds_int_cols
            )
            save_batch_csv(suv_df, output_dir / "batch_results_SUV.csv")

            return result
        except Exception as exc:
            log.error("Pipeline failed for %s: %s", patient_id, exc)
            return DeauvilleResult(patient_id=patient_id, error=str(exc))

    def convert(self, patient: PatientCase) -> dict[str, Any]:
        """Convert input images to NIfTI format."""
        return convert_images(self.cfg, patient)

    def normalize(self, patient: PatientCase) -> dict[str, Any]:
        """Compute SUVbw from raw PET."""
        return normalize_pet(self.cfg, patient)

    def register(self, patient: PatientCase) -> dict[str, Any]:
        """Rigidly register PET SUV to CT grid."""
        return register_pet(self.cfg, patient)

    def segment(self, patient: PatientCase) -> dict[str, Any]:
        """Run TotalSegmentator on CT."""
        return segment_ct(self.cfg, patient)

    def extract(
        self, patient: PatientCase, seg_result: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Extract ROI statistics from registered PET."""
        return extract_rois(self.cfg, patient, seg_result)

    def score(self, extract_results: dict[str, Any]) -> dict[str, int | float]:
        """Assign Deauville Scores from extracted ROI stats."""
        return score_deauville(self.cfg, extract_results)

    def update_targets(self, patient_id: str) -> dict[str, Any]:
        """Re-extract only target ROIs for a patient, using cached references.

        Parameters
        ----------
        patient_id : str
            Patient identifier.

        Returns
        -------
        dict
            Keys ``"scores"`` and ``"extract_results"``.
        """
        from autods_pet.patient import PatientCase

        patient = PatientCase(self.cfg, patient_id)
        return extract_new_targets_only(self.cfg, patient)

    def run_batch(self, patients: list[str] | str | Path) -> list[DeauvilleResult]:
        """Run the full pipeline for multiple patients.

        Errors are captured per-patient - the batch never stops early.

        Parameters
        ----------
        patients : list[str] or str or Path
            Either a list of patient ID strings, or a path to a text file
            containing one patient ID per line (``#`` comments allowed).

        Returns
        -------
        list[DeauvilleResult]
            One result per patient (check ``.error`` for failures).
        """
        if isinstance(patients, (str, Path)):
            from autods_pet.io import read_patient_list

            patient_ids = read_patient_list(Path(patients))
        else:
            patient_ids = patients

        results = [self.run(pid) for pid in patient_ids]

        import pandas as pd

        from autods_pet.config import resolve_output_dir
        from autods_pet.manifest import write_manifest

        output_dir = resolve_output_dir(self.cfg)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build batch DataFrames using the shared helper.
        ds_dfs = []
        suv_dfs = []
        error_rows = []
        for r in results:
            if r.error:
                error_rows.append({"patient_id": r.patient_id, "error": r.error})
                continue
            ds_df, suv_df = _result_to_batch_dfs(r, self.cfg)
            ds_dfs.append(ds_df)
            suv_dfs.append(suv_df)

        if ds_dfs:
            _ds_int_cols = [c for c in DS_COLUMN_ORDER if c != "BLR"]
            save_batch_csv(
                pd.concat(ds_dfs, ignore_index=True),
                output_dir / "batch_results_DS.csv",
                int_columns=_ds_int_cols,
            )

        if suv_dfs:
            save_batch_csv(
                pd.concat(suv_dfs, ignore_index=True),
                output_dir / "batch_results_SUV.csv",
            )

        if error_rows:
            save_batch_csv(pd.DataFrame(error_rows), output_dir / "batch_errors.csv")

        write_manifest(self.cfg, output_dir)

        return results

    @staticmethod
    def to_dataframe(results: list[DeauvilleResult]) -> pd.DataFrame:
        """Convert a list of :class:`DeauvilleResult` into a summary DataFrame.

        Columns include ``patient_id``, ``error``, per-ROI statistics
        (e.g. ``Liver_median``), and DS scores (e.g. ``DS_FL_DS``).

        Parameters
        ----------
        results : list[DeauvilleResult]
            Results from :meth:`run` or :meth:`run_batch`.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        rows = []
        for r in results:
            base: dict[str, Any] = {
                "patient_id": r.patient_id,
                "error": r.error or "",
            }
            if r.error:
                rows.append(base)
                continue

            for roi_name, roi_result in r.rois.items():
                for stat_name, value in roi_result.stats.items():
                    base[f"{roi_name}_{stat_name}"] = value

            for target_name, ds in r.scores.items():
                base[f"DS_{target_name}"] = ds

            rows.append(base)

        return pd.DataFrame(rows)
