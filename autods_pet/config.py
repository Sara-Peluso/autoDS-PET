"""Configuration loading and validation for autods_pet.

Reads an INI file with per-ROI parameters and statistics choices.
Any key not provided in the user file falls back to built-in defaults.
Uses only stdlib ``configparser`` - no external dependencies.
"""

from __future__ import annotations

import configparser
import copy
import dataclasses
import re
from pathlib import Path
from typing import Any

_DEFAULTS: dict[str, Any] = {
    "paths": {
        "basepath": "",
        "patient_list": "",
        "metadata_csv": "",
        "output_dir": "",
        "ct_nifti": "{patient_id}_results/images/CT.nii.gz",
        "pet_nifti": "{patient_id}_results/images/PET.nii.gz",
        "pet_suv": "{patient_id}_results/images/PET_SUV.nii.gz",
        "pet_registered": "{patient_id}_results/images/PET_SUV_reg.nii.gz",
        "seg_dir": "{patient_id}_results/segmentations",
        "vert_body_seg": "{patient_id}_results/segmentations/vertebral_body.nii.gz",
        "pet_metadata": "{patient_id}_results/metadata/PET_metadata.json",
        "elastix_report": "{patient_id}_results/metadata/elastix_transform.txt",
        "deauville_csv": "{patient_id}_results/DeauvilleScores/deauville_scores.csv",
        "suv_csv": "{patient_id}_results/SUV/SUV_values.csv",
        "input_seg_dir": "{patient_id}/segmentations",
    },
    "totalsegmentator": {
        "license": "",
        "fast": False,
    },
    "lumbar_vb": {
        "labels": [29, 28, 27],
        "erosion_mm": 3.0,
        "stats": ["p95"],
    },
    "aorta_mbp": {
        "vertebra_labels": [36, 37, 38, 39, 40],
        "slab_axis": 2,
        "heart_exclusion_mode": "dilate_intersection",
        "heart_dilation_mm": 6.0,
        "heart_distance_mm": 12.0,
        "aorta_erosion_mm": 4.0,
        "stats": ["median"],
    },
    "liver": {
        "erosion_mm": 10.0,
        "stats": ["median"],
    },
    "brain": {
        "label": 90,
        "grey_matter_only": True,
        "cortical_thickness_mm": 5.0,
        "stats": ["median"],
    },
    "long_bones": {
        "bones": [
            {"name": "femur_L", "erosion_mm": 5.0},
            {"name": "femur_R", "erosion_mm": 5.0},
            {"name": "humerus_L", "erosion_mm": 4.0},
            {"name": "humerus_R", "erosion_mm": 4.0},
        ],
        "diaphysis_keep_pct": 60,
        "stats": ["p95"],
    },
    "deauville": {
        "liver_multiplier": 2.0,
    },
    "targets": {},  # custom targets populated from [targets.*] sections
    "output": {
        "save_raw_masks": False,
        "save_refined_masks": True,
        "subtract_lesions_from_marrow": False,
    },
    "dicom": {
        "size_threshold_kb": 100,
    },
}

# Schema: maps (section, key) -> expected type for typed parsing
# Keys not listed here are kept as strings.
_TYPED_KEYS: dict[tuple[str, str], str] = {
    ("lumbar_vb", "labels"): "int_list",
    ("lumbar_vb", "erosion_mm"): "float",
    ("lumbar_vb", "stats"): "str_list",
    ("aorta_mbp", "vertebra_labels"): "int_list",
    ("aorta_mbp", "slab_axis"): "int",
    ("aorta_mbp", "heart_dilation_mm"): "float",
    ("aorta_mbp", "heart_distance_mm"): "float",
    ("aorta_mbp", "aorta_erosion_mm"): "float",
    ("aorta_mbp", "stats"): "str_list",
    ("liver", "erosion_mm"): "float",
    ("liver", "max_hole_volume_mm3"): "float_or_none",
    ("liver", "stats"): "str_list",
    ("brain", "label"): "int",
    ("brain", "grey_matter_only"): "bool",
    ("brain", "cortical_thickness_mm"): "float",
    ("brain", "stats"): "str_list",
    ("long_bones", "diaphysis_keep_pct"): "int",
    ("long_bones", "stats"): "str_list",
    ("focal_lesion", "stats"): "str_list",
    ("focal_lesion", "mask_filename"): "str_list",
    ("focal_lesion", "segment_label"): "str_list",
    ("paramedullary", "stats"): "str_list",
    ("paramedullary", "mask_filename"): "str_list",
    ("paramedullary", "segment_label"): "str_list",
    ("extramedullary", "stats"): "str_list",
    ("extramedullary", "mask_filename"): "str_list",
    ("extramedullary", "segment_label"): "str_list",
    ("deauville", "liver_multiplier"): "float",
    ("dicom", "size_threshold_kb"): "int",
    ("totalsegmentator", "fast"): "bool",
    ("output", "save_raw_masks"): "bool",
    ("output", "save_refined_masks"): "bool",
    ("output", "subtract_lesions_from_marrow"): "bool",
}

# Bone sub-section keys
_BONE_TYPED_KEYS: dict[str, str] = {
    "erosion_mm": "float",
}

# Fixed labels from TotalSegmentator (not user-configurable)
BONE_LABELS: dict[str, int] = {
    "femur_L": 75,
    "femur_R": 76,
    "humerus_L": 69,
    "humerus_R": 70,
}

# Valid stat names (p<N> is validated separately via regex)
_FIXED_STATS = {"mean", "median", "min", "max"}
_PERCENTILE_RE = re.compile(r"^p(\d+(?:\.\d+)?)$")

_LONG_BONES_PREFIX = "long_bones."
_TARGETS_PREFIX = "targets."

# Named target ROI sections (opt-in: only processed when present in INI)
NAMED_TARGET_SECTIONS = ("focal_lesion", "paramedullary", "extramedullary")

# Typed keys for custom [targets.*] sub-sections
_TARGET_TYPED_KEYS: dict[str, str] = {
    "stats": "str_list",
    "mask_filename": "str_list",
    "segment_label": "str_list",
}

# -- Profile definitions -------------------------------------------------

_PROFILE_DESCRIPTIONS: dict[str, str] = {
    "quick": (
        "Fast TotalSeg, no license needed, no target masks, no saved masks.\n"
        "; For pipeline testing, QC, and rapid cohort screening."
    ),
    "standard": (
        "High-res TotalSeg, no license needed, no target masks, saves refined masks.\n"
        "; Balanced starting point - produces LB_DS and reference SUV values."
    ),
    "advanced": (
        "High-res TotalSeg, license required (BM_DS enabled), no target masks,\n"
        "; saves refined masks. Adds vertebral body segmentation for Bone Marrow DS."
    ),
    "full": (
        "High-res TotalSeg, license required, all target masks (FL/PM/EM),\n"
        "; saves raw + refined masks. Complete analysis - all five DS + BLR."
    ),
    "brain": (
        "High-res TotalSeg, no license needed, brain + liver ROIs only.\n"
        "; Produces Brain-to-Liver Ratio (BLR) for neurological assessment."
    ),
}

_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "quick": {
        "totalsegmentator": {"fast": True},
        "output": {"save_raw_masks": False, "save_refined_masks": False},
    },
    "standard": {},
    "advanced": {
        "totalsegmentator": {"license": "YOUR_LICENSE_KEY_HERE"},
    },
    "full": {
        "totalsegmentator": {"license": "YOUR_LICENSE_KEY_HERE"},
        "liver": {"max_hole_volume_mm3": 500.0},
        "output": {"save_raw_masks": True, "save_refined_masks": True},
    },
    "brain": {},
}

# Sections to omit from the generated config for a given profile.
_PROFILE_SKIP_SECTIONS: dict[str, set[str]] = {
    "brain": {"lumbar_vb", "aorta_mbp", "long_bones"},
}

# Default values for named target sections when enabled (full profile).
_NAMED_TARGET_DEFAULTS: dict[str, dict[str, str]] = {
    "focal_lesion": {
        "mask_filename": "focal_lesion",
        "segment_label": "Focal lesion, FL, focal_lesion",
        "stats": "max, p90",
    },
    "paramedullary": {
        "mask_filename": "PM_lesion",
        "segment_label": "Paramedullary, PM, PM_lesion",
        "stats": "max, p90",
    },
    "extramedullary": {
        "mask_filename": "EM_lesion",
        "segment_label": "Extramedullary, EM, EM_lesion",
        "stats": "max, p90",
    },
}

# Profiles that enable named target sections (uncommented in generated INI).
_PROFILES_WITH_TARGETS: set[str] = {"full"}

PROFILE_NAMES: tuple[str, ...] = tuple(_PROFILE_DESCRIPTIONS)


def _merge_profile(profile: str) -> dict[str, Any]:
    """Return *_DEFAULTS* deep-merged with the overrides for *profile*."""
    merged = copy.deepcopy(_DEFAULTS)
    for section, values in _PROFILE_OVERRIDES.get(profile, {}).items():
        if section in merged and isinstance(merged[section], dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged


def _parse_str_list(raw: str) -> list[str]:
    """Parse a comma-separated string into a list of stripped strings."""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated string into a list of ints."""
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def _cast_value(raw: str, typ: str) -> Any:
    """Cast a raw INI string to the expected Python type."""
    if typ == "bool":
        return raw.strip().lower() in ("true", "1", "yes")
    if typ == "int":
        return int(raw)
    if typ == "float":
        return float(raw)
    if typ == "float_or_none":
        stripped = raw.strip().lower()
        if stripped in ("none", ""):
            return None
        return float(raw)
    if typ == "int_list":
        return _parse_int_list(raw)
    if typ == "str_list":
        return _parse_str_list(raw)
    return raw


def default_config() -> dict[str, Any]:
    """Return a deep copy of the built-in default configuration."""
    return copy.deepcopy(_DEFAULTS)


def _load_paths(ini: configparser.ConfigParser, cfg: dict[str, Any]) -> None:
    """Load the ``[paths]`` section (all strings, no type coercion)."""
    if ini.has_section("paths"):
        for key in ini.options("paths"):
            cfg["paths"][key] = ini.get("paths", key)


def _load_totalsegmentator(ini: configparser.ConfigParser, cfg: dict[str, Any]) -> None:
    """Load the ``[totalsegmentator]`` section with typed values."""
    if ini.has_section("totalsegmentator"):
        for key in ini.options("totalsegmentator"):
            raw = ini.get("totalsegmentator", key)
            typ = _TYPED_KEYS.get(("totalsegmentator", key))
            cfg["totalsegmentator"][key] = _cast_value(raw, typ) if typ else raw


def _load_sections(
    ini: configparser.ConfigParser, cfg: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load ROI, bone-sub, and target-sub sections.

    Returns ``(bone_sections, target_sections)`` for post-processing.
    """
    bone_sections: dict[str, dict[str, Any]] = {}
    target_sections: dict[str, dict[str, Any]] = {}

    for section in ini.sections():
        if section in ("paths", "totalsegmentator"):
            continue

        # Bone sub-sections: [long_bones.femur_L] etc.
        if section.startswith(_LONG_BONES_PREFIX):
            bone_name = section[len(_LONG_BONES_PREFIX) :]
            bone: dict[str, Any] = {"name": bone_name}
            for key in ini.options(section):
                raw = ini.get(section, key)
                typ = _BONE_TYPED_KEYS.get(key)
                bone[key] = _cast_value(raw, typ) if typ else raw
            bone_sections[bone_name] = bone
            continue

        # Custom target sub-sections: [targets.my_roi] etc.
        if section.startswith(_TARGETS_PREFIX):
            target_name = section[len(_TARGETS_PREFIX) :]
            target: dict[str, Any] = {"name": target_name}
            for key in ini.options(section):
                raw = ini.get(section, key)
                typ = _TARGET_TYPED_KEYS.get(key)
                target[key] = _cast_value(raw, typ) if typ else raw
            target_sections[target_name] = target
            continue

        # Named target sections (opt-in, not in _DEFAULTS)
        if section in NAMED_TARGET_SECTIONS and section not in cfg:
            cfg[section] = {}

        # Regular ROI section
        if section not in cfg:
            raise ValueError(f"Unknown config section: {section!r}")

        for key in ini.options(section):
            raw = ini.get(section, key)
            typ = _TYPED_KEYS.get((section, key))
            cfg[section][key] = _cast_value(raw, typ) if typ else raw

    return bone_sections, target_sections


def _inject_bone_labels(
    cfg: dict[str, Any], bone_sections: dict[str, dict[str, Any]]
) -> None:
    """Reconstruct bones list from sub-sections and inject fixed labels."""
    if bone_sections:
        for bone_name, bone_dict in bone_sections.items():
            if bone_name in BONE_LABELS:
                bone_dict["label"] = BONE_LABELS[bone_name]
        cfg["long_bones"]["bones"] = list(bone_sections.values())
    else:
        for bone_dict in cfg["long_bones"]["bones"]:
            name = bone_dict["name"]
            if name in BONE_LABELS:
                bone_dict["label"] = BONE_LABELS[name]


def load_config(
    path: str | Path | None = None, *, validate: bool = True
) -> dict[str, Any]:
    """Load an INI configuration file and merge it with built-in defaults.

    Parameters
    ----------
    path : str, Path, or None
        Path to an INI file.  If *None*, returns the built-in defaults.
    validate : bool
        When *True* (default) the merged config is validated and a
        ``ValueError`` is raised on the first problem.  Set to *False*
        to skip validation - useful when you want to run
        :class:`ConfigValidator` yourself to collect **all** issues.

    Returns
    -------
    dict
        Merged configuration (user values override defaults).
    """
    cfg = default_config()
    if path is None:
        return cfg

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    ini = configparser.ConfigParser()
    ini.read(path, encoding="utf-8")

    _load_paths(ini, cfg)
    _load_totalsegmentator(ini, cfg)
    bone_sections, target_sections = _load_sections(ini, cfg)

    if target_sections:
        cfg["targets"] = target_sections

    _inject_bone_labels(cfg, bone_sections)
    if validate:
        _validate(cfg)
    return cfg


def parse_stat(name: str) -> tuple[str, float | None]:
    """Parse a stat name into ``(kind, param)``.

    Parameters
    ----------
    name : str
        Stat name such as ``"mean"``, ``"median"``, ``"p95"``.

    Returns
    -------
    tuple[str, float | None]
        ``("percentile", 95.0)`` for ``"p95"``,
        ``("mean", None)`` for ``"mean"``, etc.

    Raises
    ------
    ValueError
        If *name* is not a recognised statistic.

    Examples
    --------
    >>> from autods_pet.config import parse_stat
    >>> parse_stat("mean")
    ('mean', None)
    >>> parse_stat("p95")
    ('percentile', 95.0)
    >>> parse_stat("median")
    ('median', None)
    """
    m = _PERCENTILE_RE.match(name)
    if m:
        return ("percentile", float(m.group(1)))
    if name in _FIXED_STATS:
        return (name, None)
    raise ValueError(
        f"Unknown stat {name!r}.  "
        f"Valid: {sorted(_FIXED_STATS)} or p<N> (e.g. p90, p95)."
    )


def get_roi_config(cfg: dict[str, Any], roi: str) -> dict[str, Any]:
    """Return the sub-dict for a single ROI, raising on unknown names."""
    if roi not in cfg:
        raise KeyError(f"No config section for ROI {roi!r}")
    result: dict[str, Any] = cfg[roi]
    return result


def get_all_targets(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a list of all configured target ROIs (named + custom).

    Each entry is a dict with at least ``name``, ``mask_filename``, ``stats``.
    Named targets (focal_lesion, paramedullary, extramedullary) are only
    included if the user explicitly added the section to the INI file.
    Custom targets from ``[targets.*]`` sections are always included.
    """
    targets = []
    for section_name in NAMED_TARGET_SECTIONS:
        if section_name in cfg:
            entry = cfg[section_name].copy()
            entry["name"] = section_name
            targets.append(entry)
    for target_name, target_cfg in cfg.get("targets", {}).items():
        entry = target_cfg.copy()
        entry.setdefault("name", target_name)
        targets.append(entry)
    return targets


@dataclasses.dataclass
class ValidationIssue:
    """A single validation problem found in a configuration."""

    section: str
    key: str | None
    level: str  # "error" or "warning"
    message: str


class ConfigValidator:
    """Collect all validation issues from a merged configuration dict.

    Unlike :func:`load_config`, which raises on the first error, this class
    accumulates every problem so that users can fix them all at once.

    Parameters
    ----------
    cfg : dict
        A merged configuration dict (as returned by :func:`load_config`
        with ``validate=False``).

    Examples
    --------
    >>> from autods_pet.config import load_config, ConfigValidator
    >>> cfg = load_config("config.ini", validate=False)
    >>> v = ConfigValidator(cfg)
    >>> v.validate()
    >>> if not v.is_valid:
    ...     for issue in v.errors:
    ...         print(issue)
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self.issues: list[ValidationIssue] = []

    # -- helpers ----------------------------------------------------------

    def _add(
        self,
        section: str,
        key: str | None,
        level: str,
        message: str,
    ) -> None:
        self.issues.append(ValidationIssue(section, key, level, message))

    # -- public API -------------------------------------------------------

    @property
    def errors(self) -> list[ValidationIssue]:
        """Return only error-level issues."""
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Return only warning-level issues."""
        return [i for i in self.issues if i.level == "warning"]

    @property
    def is_valid(self) -> bool:
        """Return *True* when no errors were found."""
        return not self.errors

    def validate(self) -> list[ValidationIssue]:
        """Run all checks and return the list of issues found.

        The same list is also available as :attr:`issues`.
        """
        self.issues.clear()
        self._check_roi_sections()
        self._check_custom_targets()
        return self.issues

    # -- internal checks --------------------------------------------------

    def _check_stats(self, section: str, values: dict[str, Any]) -> bool:
        """Validate stats presence and names. Returns True if stats are valid."""
        stats = values.get("stats")
        if stats is None:
            self._add(
                section, "stats", "error", f"Section {section!r} is missing 'stats'"
            )
            return False
        if not isinstance(stats, list) or len(stats) == 0:
            self._add(
                section,
                "stats",
                "error",
                f"Section {section!r}: 'stats' must be a non-empty list",
            )
            return False
        for s in stats:
            try:
                parse_stat(s)
            except ValueError as exc:
                self._add(section, "stats", "error", str(exc))
        return True

    def _check_reference_roi(self, section: str, values: dict[str, Any]) -> None:
        """Validate section-specific constraints for reference ROIs."""
        if section == "lumbar_vb":
            self._check_non_negative(values, "erosion_mm", section)
        elif section == "aorta_mbp":
            self._check_non_negative(values, "aorta_erosion_mm", section)
            if values.get("heart_exclusion_mode") not in (
                "dilate_intersection",
                "distance",
            ):
                self._add(
                    section,
                    "heart_exclusion_mode",
                    "error",
                    "aorta_mbp.heart_exclusion_mode must be "
                    "'dilate_intersection' or 'distance'",
                )
        elif section == "liver":
            self._check_non_negative(values, "erosion_mm", section)
        elif section == "brain":
            self._check_non_negative(values, "cortical_thickness_mm", section)
        elif section == "long_bones":
            pct = values.get("diaphysis_keep_pct", 60)
            if not (1 <= pct <= 100):
                self._add(
                    section,
                    "diaphysis_keep_pct",
                    "error",
                    f"long_bones.diaphysis_keep_pct must be 1..100, got {pct}",
                )
        elif section in NAMED_TARGET_SECTIONS:
            self._check_target_identity(section, values)

    def _check_roi_sections(self) -> None:
        cfg = self._cfg
        roi_sections = [
            k
            for k in cfg
            if k
            not in (
                "paths",
                "targets",
                "dicom",
                "totalsegmentator",
                "output",
                "deauville",
            )
        ]
        for section in roi_sections:
            values = cfg[section]
            if not self._check_stats(section, values):
                continue
            self._check_reference_roi(section, values)

    def _check_target_identity(self, section: str, values: dict[str, Any]) -> None:
        """Validate that a target section has at least one identity key set.

        A section is valid when it sets ``mask_filename`` (NIfTI/NRRD
        discovery) and/or ``segment_label`` (DICOM SEG discovery).  Empty
        lists do not count.
        """
        mf = values.get("mask_filename")
        sl = values.get("segment_label")
        mf_set = isinstance(mf, list) and len(mf) > 0
        sl_set = isinstance(sl, list) and len(sl) > 0
        if not (mf_set or sl_set):
            self._add(
                section,
                "mask_filename",
                "error",
                f"Section {section!r} must set at least one of "
                "'mask_filename' (NIfTI/NRRD) or 'segment_label' (DICOM SEG)",
            )

    def _check_custom_targets(self) -> None:
        for target_name, target_cfg in self._cfg.get("targets", {}).items():
            self._check_target_identity(f"targets.{target_name}", target_cfg)
            stats = target_cfg.get("stats")
            if stats is None:
                self._add(
                    f"targets.{target_name}",
                    "stats",
                    "error",
                    f"Custom target {target_name!r} is missing 'stats'",
                )
            elif not isinstance(stats, list) or len(stats) == 0:
                self._add(
                    f"targets.{target_name}",
                    "stats",
                    "error",
                    f"Custom target {target_name!r}: 'stats' must be a non-empty list",
                )
            else:
                for s in stats:
                    try:
                        parse_stat(s)
                    except ValueError as exc:
                        self._add(f"targets.{target_name}", "stats", "error", str(exc))

    def _check_non_negative(self, d: dict[str, Any], key: str, section: str) -> None:
        val = d.get(key)
        if val is not None and val < 0:
            self._add(section, key, "error", f"{section}.{key} must be >= 0, got {val}")


def _validate(cfg: dict[str, Any]) -> None:
    """Validate the merged config, raising ValueError on the first problem."""
    validator = ConfigValidator(cfg)
    validator.validate()
    if validator.errors:
        raise ValueError(validator.errors[0].message)


def _fmt_value(value: object) -> str:
    """Format a config value for INI output."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _write_paths_section(lines: list[str], cfg: dict[str, Any]) -> None:
    lines.append("; Base directory that contains one sub-folder per patient.")
    lines.append("[paths]")
    for key, val in cfg["paths"].items():
        if key == "input_seg_dir":
            lines.append("; Input segmentations folder (relative to basepath).")
            lines.append(
                "; Pre-existing TotalSeg and refined masks here are copied to output."
            )
        lines.append(f"{key} = {val}")
    lines.append("")


def _write_totalsegmentator_section(lines: list[str], cfg: dict[str, Any]) -> None:
    lines.append("; TotalSegmentator settings (license key and fast mode).")
    lines.append("[totalsegmentator]")
    ts = cfg["totalsegmentator"]
    if ts.get("license"):
        lines.append(
            "; Enter your TotalSegmentator license key to enable vertebral body segmentation."
        )
    else:
        lines.append(
            "; No license - vertebral body segmentation will be skipped (BM_DS unavailable)."
        )
    for key, val in ts.items():
        lines.append(f"{key} = {_fmt_value(val)}")
    lines.append("")


def _write_roi_sections(lines: list[str], cfg: dict[str, Any], skip: set[str]) -> None:
    if "lumbar_vb" not in skip:
        lines.append("; Lumbar vertebral body ROI.")
        lines.append("[lumbar_vb]")
        for key, val in cfg["lumbar_vb"].items():
            lines.append(f"{key} = {_fmt_value(val)}")
        lines.append("")

    if "aorta_mbp" not in skip:
        lines.append("; Aorta medial blood pool ROI.")
        lines.append("[aorta_mbp]")
        for key, val in cfg["aorta_mbp"].items():
            lines.append(f"{key} = {_fmt_value(val)}")
        lines.append("")

    lines.append("; Liver ROI.")
    lines.append("[liver]")
    for key, val in cfg["liver"].items():
        lines.append(f"{key} = {_fmt_value(val)}")
    lines.append("")

    lines.append("; Brain ROI (for Brain-to-Liver Ratio).")
    lines.append("; When grey_matter_only = true, extracts cortical shell")
    lines.append("; (original minus eroded by cortical_thickness_mm).")
    lines.append("[brain]")
    for key, val in cfg["brain"].items():
        lines.append(f"{key} = {_fmt_value(val)}")
    lines.append("")

    if "long_bones" not in skip:
        lines.append("; Long bones ROI (diaphysis extraction).")
        lines.append("[long_bones]")
        lb = cfg["long_bones"]
        lines.append(f"diaphysis_keep_pct = {lb['diaphysis_keep_pct']}")
        lines.append(f"stats = {_fmt_value(lb['stats'])}")
        lines.append("")
        for bone in lb["bones"]:
            lines.append(f"[long_bones.{bone['name']}]")
            lines.append(f"erosion_mm = {bone['erosion_mm']}")
            lines.append("")


_TARGET_DOC = [
    "; Target ROI sections - manual lesion masks.",
    ";",
    "; Mask discovery is recursive: place files anywhere under the patient",
    "; input directory.  Both NIfTI/NRRD and DICOM SEG are supported.",
    ";",
    ";   - mask_filename : stem(s) for .nii.gz / .nii / .nrrd files.  A",
    ';                     single stem (e.g. "focal_lesion") OR a comma',
    ';                     list (e.g. "focal_lesion, FL_mask, GTV").',
    ";                     Searched recursively under input_dir.  This is",
    ";                     the only key NIfTI/NRRD users need.",
    ";   - segment_label : SegmentLabel(s) inside a DICOM SEG (single",
    ";                     value or comma list, case-insensitive).  The",
    ";                     SEG file is identified by matching its",
    ";                     ReferencedSeriesSequence to the PET",
    ";                     SeriesInstanceUID - no filename or location",
    ";                     required.  Only consulted for .dcm files.",
    ";",
    ";   Set either, both, or neither.  DICOM SEG wins when both formats",
    ";   match the same target for the same patient.  Geometry is",
    ";   auto-detected and PET-space masks are auto-registered to CT.",
]


def _write_target_sections(lines: list[str], targets_enabled: bool) -> None:
    lines.extend(_TARGET_DOC)
    if targets_enabled:
        for name, defaults in _NAMED_TARGET_DEFAULTS.items():
            lines.append(f"[{name}]")
            for key, val in defaults.items():
                lines.append(f"{key} = {val}")
            lines.append("")
    else:
        lines.append("; (sections below are commented out - uncomment to enable)")
        for name, defaults in _NAMED_TARGET_DEFAULTS.items():
            lines.append(";")
            lines.append(f"; [{name}]")
            for key, val in defaults.items():
                lines.append(f"; {key} = {val}")
            lines.append("")

    lines.append("; Custom target ROIs: add [targets.<name>] sections.")
    lines.append("; [targets.my_custom_roi]")
    lines.append("; mask_filename = my_roi")
    lines.append("; segment_label = my tumor label")
    lines.append("; stats = max, median")
    lines.append("")


def _write_output_and_dicom_sections(lines: list[str], cfg: dict[str, Any]) -> None:
    lines.append("; Mask saving and extraction options.")
    lines.append("[output]")
    for key, val in cfg["output"].items():
        if key == "subtract_lesions_from_marrow":
            lines.append(
                "; Subtract target lesion masks (FL, PM, EM, custom) from marrow"
            )
            lines.append("; ROIs (Lumbar VB, Long bones) before computing statistics.")
        lines.append(f"{key} = {_fmt_value(val)}")
    lines.append("")

    lines.append("; DICOM conversion settings.")
    lines.append("[dicom]")
    for key, val in cfg["dicom"].items():
        lines.append(f"{key} = {_fmt_value(val)}")
    lines.append("")


def create_default_config(path: str | Path, profile: str = "standard") -> Path:
    """Write a commented INI template for the given *profile*.

    The generated file can be loaded by :func:`load_config` without errors
    and serves as a starting point for users to customise.

    Parameters
    ----------
    path : str or Path
        Destination file path.
    profile : str
        Profile name (one of :data:`PROFILE_NAMES`).

    Returns
    -------
    Path
        The path that was written.
    """
    if profile not in _PROFILE_OVERRIDES:
        raise ValueError(
            f"Unknown profile {profile!r}. Choose from: {', '.join(PROFILE_NAMES)}"
        )

    path = Path(path)
    cfg = _merge_profile(profile)
    skip = _PROFILE_SKIP_SECTIONS.get(profile, set())
    targets_enabled = profile in _PROFILES_WITH_TARGETS
    lines: list[str] = []

    # Header
    lines.append(f"; autods-pet configuration - Profile: {profile}")
    lines.append(f"; {_PROFILE_DESCRIPTIONS[profile]}")
    lines.append(";")
    lines.append(
        f"; Edit as needed, then validate with: autods-pet validate-config {path.name}\n"
    )

    _write_paths_section(lines, cfg)
    _write_totalsegmentator_section(lines, cfg)
    _write_roi_sections(lines, cfg, skip)
    _write_target_sections(lines, targets_enabled)
    _write_output_and_dicom_sections(lines, cfg)

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def resolve_output_dir(cfg: dict[str, Any]) -> Path:
    """Return the resolved output directory from *cfg*.

    Uses ``cfg["paths"]["output_dir"]`` when set (absolute or relative to
    the current working directory), otherwise falls back to ``CWD / "results"``.
    """
    raw = cfg.get("paths", {}).get("output_dir", "")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else Path.cwd() / p
    return Path.cwd() / "results"
