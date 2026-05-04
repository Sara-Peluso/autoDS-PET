"""DICOM discovery, tag extraction, and NIfTI conversion for autods_pet.

Provides functions to find DICOM files, extract PET/CT metadata tags,
and convert DICOM series to NIfTI format using pydicom and SimpleITK.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pydicom
import SimpleITK as sitk

log = logging.getLogger(__name__)


def _safe_get(
    ds: pydicom.Dataset,
    tag_name: str,
    convert: Callable[..., Any] = str,
    default: Any = None,
) -> Any:
    """Safely extract a DICOM tag value with type conversion.

    Returns *default* if the tag is missing or conversion fails.
    """
    val = ds.get(tag_name, None)
    if val is None:
        return default
    try:
        return convert(val)
    except (ValueError, TypeError):
        return default


def resolve_patient_folder(basepath: Path, patient_id: str) -> Path | None:
    """Case-insensitive lookup for a patient subfolder.

    Returns the resolved :class:`Path` or *None* if no match is found.
    """
    basepath = Path(basepath)
    if not basepath.is_dir():
        return None
    folder_map = {d.name.lower(): d for d in basepath.iterdir() if d.is_dir()}
    actual = folder_map.get(patient_id.lower())
    return actual if actual is not None else None


def find_dicom_files(
    patient_dir: str | Path,
    size_threshold_kb: int = 100,
) -> list[Path]:
    """Recursively find ``.dcm`` files above a size threshold.

    Parameters
    ----------
    patient_dir : Path
        Root directory to search.
    size_threshold_kb : int
        Minimum file size in KB.  Files smaller than this are skipped
        (e.g. tiny overlay or presentation-state files).

    Returns
    -------
    list[Path]
        Sorted list of matching file paths.
    """
    patient_dir = Path(patient_dir)
    threshold_bytes = size_threshold_kb * 1024
    files = []
    for f in patient_dir.rglob("*.dcm"):
        if f.is_file() and f.stat().st_size >= threshold_bytes:
            files.append(f)
    files.sort()
    return files


def find_series_by_modality(
    patient_dir: str | Path,
    size_threshold_kb: int = 100,
) -> dict[str, list[Path]]:
    """Find the primary CT and PT DICOM series in a patient directory.

    Groups ``.dcm`` files by ``(Modality, SeriesInstanceUID)``, then picks
    the series with the most files for each modality.

    .. note::
        This function reads DICOM headers for every ``.dcm`` file
        sequentially (O(n) disk reads, no parallelism).  For directories
        with 1000+ files on network-attached storage this may be slow.
        Consider parallel header reads for high-throughput deployments.

    Returns
    -------
    dict
        ``{"CT": [Path, ...], "PT": [Path, ...]}``.
        Empty lists for modalities that are not found.
    """
    dcm_files = find_dicom_files(patient_dir, size_threshold_kb)

    # Group: (modality, series_uid) -> [paths]
    groups: dict[tuple[str, str], list[Path]] = {}
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            modality = str(getattr(ds, "Modality", "")).upper()
            series_uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
            groups.setdefault((modality, series_uid), []).append(f)
        except Exception:
            log.debug("Skipping %s: could not read DICOM header", f)
            continue

    result: dict[str, list[Path]] = {"CT": [], "PT": []}
    for modality_key in ("CT", "PT"):
        # Find all series for this modality, pick the one with most files
        candidates = {
            uid: paths for (mod, uid), paths in groups.items() if mod == modality_key
        }
        if candidates:
            best_uid = max(candidates, key=lambda uid: len(candidates[uid]))
            result[modality_key] = candidates[best_uid]

    if not result["CT"]:
        log.warning("No CT series found in %s", patient_dir)
    if not result["PT"]:
        log.warning("No PT series found in %s", patient_dir)

    return result


def extract_pet_tags(dicom_path: str | Path) -> dict[str, Any]:
    """Extract PET metadata tags from a single DICOM file.

    Handles nested ``RadiopharmaceuticalInformationSequence`` with
    top-level fallback for dose and half-life tags.

    Returns
    -------
    dict
        Keys match the metadata expected by :func:`~autods_pet.pipeline.normalize_pet`:
        ``PatientID``, ``StudyDate``, ``AcquisitionTime``, ``Units``,
        ``DecayCorrection``, ``RadiopharmaceuticalStartTime``,
        ``RadionuclideTotalDose``, ``RadionuclideHalfLife``,
        ``SeriesInstanceUID``, ``StudyInstanceUID``.
        Missing tags are set to *None*.
    """
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, force=True)

    # Radiopharmaceutical info from nested sequence
    rph_start = None
    half_life = None
    total_dose = None

    seq = ds.get("RadiopharmaceuticalInformationSequence", None)
    if seq is not None and len(seq) > 0:
        item = seq[0]
        rph_start = _safe_get(item, "RadiopharmaceuticalStartTime")
        half_life = _safe_get(item, "RadionuclideHalfLife", convert=float)
        total_dose = _safe_get(item, "RadionuclideTotalDose", convert=float)

    # Top-level fallbacks
    if total_dose is None:
        total_dose = _safe_get(ds, "RadionuclideTotalDose", convert=float)
    if half_life is None:
        half_life = _safe_get(ds, "RadionuclideHalfLife", convert=float)
    if rph_start is None:
        rph_start = _safe_get(ds, "RadiopharmaceuticalStartTime")

    return {
        "PatientID": _safe_get(ds, "PatientID"),
        "StudyDate": _safe_get(ds, "StudyDate"),
        "AcquisitionTime": _safe_get(ds, "AcquisitionTime"),
        "Units": _safe_get(ds, "Units"),
        "DecayCorrection": _safe_get(ds, "DecayCorrection"),
        "RadiopharmaceuticalStartTime": rph_start,
        "RadionuclideTotalDose": total_dose,
        "RadionuclideHalfLife": half_life,
        "SeriesInstanceUID": _safe_get(ds, "SeriesInstanceUID"),
        "StudyInstanceUID": _safe_get(ds, "StudyInstanceUID"),
    }


def extract_ct_tags(dicom_path: str | Path) -> dict[str, Any]:
    """Extract CT metadata tags from a single DICOM file.

    Returns
    -------
    dict
        CT-relevant tags.  Missing tags are set to *None*.
    """
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, force=True)
    return {
        "PatientID": _safe_get(ds, "PatientID"),
        "StudyDate": _safe_get(ds, "StudyDate"),
        "Manufacturer": _safe_get(ds, "Manufacturer"),
        "ManufacturerModelName": _safe_get(ds, "ManufacturerModelName"),
        "SliceThickness": _safe_get(ds, "SliceThickness", convert=float),
        "PixelSpacing": _safe_get(ds, "PixelSpacing"),
        "KVP": _safe_get(ds, "KVP", convert=float),
        "ConvolutionKernel": _safe_get(ds, "ConvolutionKernel"),
        "Rows": _safe_get(ds, "Rows", convert=int),
        "Columns": _safe_get(ds, "Columns", convert=int),
    }


def extract_patient_weight(dicom_path: str | Path) -> float | None:
    """Extract patient weight (kg) from a DICOM file.

    Returns
    -------
    float or None
        Weight in kg, or *None* if the tag is absent or zero.
    """
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, force=True)
    weight = _safe_get(ds, "PatientWeight", convert=float)
    if weight is not None and weight > 0:
        return weight
    return None


def dicom_series_to_nifti(
    dicom_files: Sequence[str | Path],
    output_path: str | Path,
) -> sitk.Image:
    """Convert a DICOM series to a NIfTI file.

    Uses SimpleITK's ``ImageSeriesReader`` with GDCM-based spatial
    ordering to ensure correct slice order.

    Parameters
    ----------
    dicom_files : list
        File paths belonging to a single DICOM series.
    output_path : str or Path
        Where to write the NIfTI file.

    Returns
    -------
    sitk.Image
        The loaded image (also written to *output_path*).
    """
    if not dicom_files:
        raise ValueError("dicom_files is empty - cannot convert.")

    # Read SeriesInstanceUID from first file to let GDCM sort properly
    first_file = Path(dicom_files[0])
    ds = pydicom.dcmread(str(first_file), stop_before_pixels=True, force=True)
    series_uid = str(getattr(ds, "SeriesInstanceUID", ""))

    # Check if all files share the same parent directory
    parents = {Path(f).parent for f in dicom_files}
    if len(parents) == 1:
        # Single directory: use GDCM sorting for correct spatial ordering
        dicom_dir = str(parents.pop())
        sorted_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            dicom_dir, series_uid
        )  # type: ignore[arg-type]
        if not sorted_files:
            log.warning("GDCM sorting returned no files; using provided file list.")
            sorted_files = [str(f) for f in dicom_files]
    else:
        # Multiple directories: GDCM directory scan would miss files
        log.warning(
            "DICOM files span %d directories; using provided file order.", len(parents)
        )
        sorted_files = [str(f) for f in dicom_files]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))
    log.info("Wrote NIfTI: %s (%d files)", output_path, len(sorted_files))

    return image
