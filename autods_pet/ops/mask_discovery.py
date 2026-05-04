"""Discover manual lesion masks (NIfTI/NRRD/DICOM SEG) for a patient.

This module is the single source of truth for finding the file (or
DICOM segment) that backs each configured target ROI section.

Two complementary lookup mechanisms run in parallel:

* **DICOM SEG path** - recursively walks the patient input directory,
  filters ``.dcm`` files via :func:`is_dicom_seg`, keeps only those
  whose ``ReferencedSeriesSequence`` includes the patient's PET
  ``SeriesInstanceUID``, then matches each segment's ``SegmentLabel``
  (case-insensitive) against the ``segment_label`` keys configured in
  the target sections.  One SEG file may supply multiple targets.

* **File path** - recursively walks for ``.nii.gz`` / ``.nii`` /
  ``.nrrd`` files whose stem matches one of the ``mask_filename``
  values configured in the target sections.

When a target has both a DICOM SEG match and a file match, **DICOM SEG
wins**.  The walker is bounded by ``max_depth`` and skips the global
``output_dir`` (when nested under ``input_dir``) so that previously
written result masks are not rediscovered as inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

log = logging.getLogger(__name__)

# Default depth for recursive walks (patient_root / study / series / file).
_DEFAULT_MAX_DEPTH = 4

_FILE_EXTENSIONS: tuple[tuple[str, str], ...] = (
    (".nii.gz", "nifti"),
    (".nii", "nifti"),
    (".nrrd", "nrrd"),
)


@dataclass(frozen=True)
class DiscoveredMask:
    """A resolved manual mask for a single target ROI."""

    target_name: str
    """Config section name (``"focal_lesion"``, ``"paramedullary"``,
    ``"extramedullary"``, or a custom ``targets.<name>`` key)."""

    path: Path
    """File on disk that contains the mask."""

    format: Literal["dicom_seg", "nifti", "nrrd"]
    """Storage format."""

    segment_label: str | None = None
    """For DICOM SEG, the matched ``SegmentLabel`` (as written in the
    SEG, not the user pattern).  ``None`` for NIfTI/NRRD."""

    segment_number: int | None = None
    """For DICOM SEG, the matched segment number.  ``None`` for
    NIfTI/NRRD."""


def _iter_files(
    root: Path,
    *,
    max_depth: int,
    skip_dirs: set[Path],
    suffixes: tuple[str, ...] | None = None,
) -> Iterable[Path]:
    """Yield files under *root* up to *max_depth* (root itself = depth 0).

    *skip_dirs* contains absolute paths whose entire subtree is excluded.
    *suffixes* is an optional whitelist of lowercased suffix strings
    (matched against the full filename, e.g. ``".nii.gz"``).
    """
    if not root.exists() or not root.is_dir():
        return

    skip_resolved = {p.resolve() for p in skip_dirs}

    def _matches_suffix(name: str) -> bool:
        if suffixes is None:
            return True
        lower = name.lower()
        return any(lower.endswith(s) for s in suffixes)

    def _walk(directory: Path, depth: int) -> Iterable[Path]:
        try:
            resolved = directory.resolve()
        except OSError:
            return
        if resolved in skip_resolved:
            return
        try:
            entries = list(directory.iterdir())
        except (PermissionError, OSError) as exc:
            log.debug("Skipping %s: %s", directory, exc)
            return
        for entry in entries:
            if entry.is_dir():
                if depth < max_depth:
                    yield from _walk(entry, depth + 1)
            elif _matches_suffix(entry.name):
                yield entry

    yield from _walk(root, 0)


def _normalize_pattern_list(value: Any) -> list[str]:
    """Coerce a config value into a list of non-empty stripped strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, (list, tuple)):
        return [str(s).strip() for s in value if str(s).strip()]
    return []


def _stem_no_ext(path: Path) -> str:
    """Return the filename stem with all known extensions stripped."""
    name = path.name
    lower = name.lower()
    for ext, _ in _FILE_EXTENSIONS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def _file_format(path: Path) -> Literal["nifti", "nrrd"] | None:
    lower = path.name.lower()
    for ext, fmt in _FILE_EXTENSIONS:
        if lower.endswith(ext):
            return fmt  # type: ignore[return-value]
    return None


def discover_file_masks(
    input_dir: Path,
    targets: list[dict[str, Any]],
    *,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    skip_dirs: set[Path] | None = None,
) -> dict[str, DiscoveredMask]:
    """Recursively find NIfTI/NRRD masks for each target by filename stem.

    Parameters
    ----------
    input_dir : Path
        Patient input directory to walk recursively.
    targets : list[dict]
        Target config entries (from :func:`autods_pet.config.get_all_targets`).
    max_depth : int, optional
        Maximum recursion depth.
    skip_dirs : set[Path] or None
        Absolute directories to exclude from the walk (e.g. the global
        ``output_dir`` when nested under ``input_dir``).

    Returns
    -------
    dict[str, DiscoveredMask]
        Map from target name to the first matching file found.
    """
    skip_dirs = skip_dirs or set()
    # Build stem -> target_name mapping. First match wins per target.
    stem_targets: dict[str, str] = {}
    target_order: dict[str, list[str]] = {}
    for tgt in targets:
        name = tgt["name"]
        stems = _normalize_pattern_list(tgt.get("mask_filename"))
        target_order[name] = stems
        for stem in stems:
            stem_targets.setdefault(stem.lower(), name)

    if not stem_targets:
        return {}

    # Walk once, collect every match keyed by (target_name, stem).
    found: dict[tuple[str, str], Path] = {}
    suffixes = tuple(ext for ext, _ in _FILE_EXTENSIONS)
    for path in _iter_files(
        input_dir, max_depth=max_depth, skip_dirs=skip_dirs, suffixes=suffixes
    ):
        stem = _stem_no_ext(path).lower()
        target = stem_targets.get(stem)
        if target is None:
            continue
        key = (target, stem)
        if key not in found:
            found[key] = path

    # Resolve to one mask per target, honouring the order in mask_filename.
    result: dict[str, DiscoveredMask] = {}
    for target_name, stems in target_order.items():
        for stem in stems:
            key = (target_name, stem.lower())
            if key in found:
                path = found[key]
                fmt = _file_format(path)
                if fmt is None:
                    continue
                result[target_name] = DiscoveredMask(
                    target_name=target_name,
                    path=path,
                    format=fmt,
                )
                break
    return result


def discover_dicom_seg_masks(
    input_dir: Path,
    targets: list[dict[str, Any]],
    pet_series_uid: str | None,
    *,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    skip_dirs: set[Path] | None = None,
) -> tuple[dict[str, DiscoveredMask], list[str]]:
    """Recursively find DICOM SEG masks for each target by SegmentLabel.

    Parameters
    ----------
    input_dir : Path
        Patient input directory to walk recursively.
    targets : list[dict]
        Target config entries.
    pet_series_uid : str or None
        The patient's PET ``SeriesInstanceUID``.  When *None* the UID
        match is bypassed (any DICOM SEG is considered) and an
        informational message is added to the warnings list.
    max_depth : int, optional
        Maximum recursion depth.
    skip_dirs : set[Path] or None
        Absolute directories to exclude from the walk.

    Returns
    -------
    tuple[dict[str, DiscoveredMask], list[str]]
        ``(matches, info_messages)``.  Info messages capture
        unmatched-segment notes and "found SEG but UID didn't match"
        notes that callers may want to log.
    """
    skip_dirs = skip_dirs or set()
    info: list[str] = []

    # Build label -> target_name mapping (case-insensitive).
    label_targets: dict[str, str] = {}
    target_order: dict[str, list[str]] = {}
    for tgt in targets:
        name = tgt["name"]
        labels = _normalize_pattern_list(tgt.get("segment_label"))
        target_order[name] = labels
        for label in labels:
            label_targets.setdefault(label.lower(), name)

    if not label_targets:
        return {}, info

    # Lazy imports - pydicom is always available; highdicom is optional.
    from autods_pet.ops.dicom_seg import (
        is_dicom_seg,
        list_segments,
        read_referenced_series_uids,
    )

    result: dict[str, DiscoveredMask] = {}

    for path in _iter_files(
        input_dir, max_depth=max_depth, skip_dirs=skip_dirs, suffixes=(".dcm",)
    ):
        if not is_dicom_seg(path):
            continue
        _match_seg_file(
            path,
            label_targets,
            pet_series_uid,
            result,
            info,
            is_dicom_seg,
            list_segments,
            read_referenced_series_uids,
        )

    return result, info


def _match_seg_file(
    path: Path,
    label_targets: dict[str, str],
    pet_series_uid: str | None,
    result: dict[str, DiscoveredMask],
    info: list[str],
    is_dicom_seg: Any,
    list_segments: Any,
    read_referenced_series_uids: Any,
) -> None:
    """Process a single DICOM SEG file, updating *result* and *info* in place."""
    # UID gating.
    if pet_series_uid is not None:
        refs = read_referenced_series_uids(path)
        if pet_series_uid not in refs:
            info.append(
                f"DICOM SEG at {path} does not reference PET SeriesInstanceUID "
                f"{pet_series_uid} (refs: {refs or '∅'}); skipping."
            )
            return
    else:
        info.append(
            f"PET SeriesInstanceUID unknown; matching DICOM SEG at {path} by label only."
        )

    try:
        segments = list_segments(path)
    except ModuleNotFoundError as exc:
        info.append(
            f"Cannot inspect DICOM SEG at {path}: {exc}.  "
            "Install with: pip install autods-pet[dicom-seg]"
        )
        return
    except Exception as exc:  # noqa: BLE001
        info.append(f"Failed to read DICOM SEG at {path}: {exc}")
        return

    matched_any = False
    for seg in segments:
        label = str(seg.get("label", ""))
        target = label_targets.get(label.lower())
        if target is None:
            continue
        if target in result:
            continue
        result[target] = DiscoveredMask(
            target_name=target,
            path=path,
            format="dicom_seg",
            segment_label=label,
            segment_number=int(seg.get("number", 0)) or None,
        )
        matched_any = True

    if not matched_any:
        seg_labels = ", ".join(s.get("label", "?") for s in segments) or "∅"
        info.append(
            f"DICOM SEG at {path} contains segments [{seg_labels}] but none "
            "matched any configured target segment_label patterns."
        )


def discover_all_masks(
    input_dir: Path,
    targets: list[dict[str, Any]],
    pet_series_uid: str | None,
    *,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    skip_dirs: set[Path] | None = None,
) -> tuple[dict[str, DiscoveredMask], list[str]]:
    """Discover masks for every target via both DICOM SEG and file paths.

    DICOM SEG matches override file matches when both formats resolve
    the same target.

    Returns
    -------
    tuple[dict[str, DiscoveredMask], list[str]]
        ``(target_name -> DiscoveredMask, warnings)``.  ``warnings``
        contains caller-actionable messages: missing-mask warnings (one
        per target whose section is enabled but no match was found),
        unmatched-segment notes, and conflict notes when both formats
        matched the same target.
    """
    file_matches = discover_file_masks(
        input_dir, targets, max_depth=max_depth, skip_dirs=skip_dirs
    )
    dicom_matches, info = discover_dicom_seg_masks(
        input_dir,
        targets,
        pet_series_uid,
        max_depth=max_depth,
        skip_dirs=skip_dirs,
    )

    warnings = list(info)
    merged: dict[str, DiscoveredMask] = {}

    for tgt in targets:
        name = tgt["name"]
        in_dicom = dicom_matches.get(name)
        in_file = file_matches.get(name)
        if in_dicom is not None:
            merged[name] = in_dicom
            if in_file is not None:
                warnings.append(
                    f"Target [{name}] matched both a DICOM SEG "
                    f"({in_dicom.path}, segment '{in_dicom.segment_label}') "
                    f"and a file mask ({in_file.path}); using the DICOM SEG."
                )
        elif in_file is not None:
            merged[name] = in_file
        else:
            mf = _normalize_pattern_list(tgt.get("mask_filename"))
            sl = _normalize_pattern_list(tgt.get("segment_label"))
            if not mf and not sl:
                # Section is effectively disabled - no warning needed.
                continue
            parts: list[str] = []
            if sl:
                uid_clause = (
                    f"referencing PET SeriesInstanceUID {pet_series_uid}"
                    if pet_series_uid
                    else "(PET SeriesInstanceUID unknown)"
                )
                parts.append(f"DICOM SEG segments matching {sl} {uid_clause}")
            if mf:
                parts.append(
                    f"NIfTI/NRRD files named {{{', '.join(mf)}}}"
                    f".{{nii.gz,nii,nrrd}} under {input_dir}"
                )
            warnings.append(
                f"Target [{name}] is enabled in config but no mask was found: "
                f"searched for {' and '.join(parts)}.  "
                f"The corresponding Deauville score will not be computed."
            )

    return merged, warnings
