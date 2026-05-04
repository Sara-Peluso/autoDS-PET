"""Patient case: resolved paths and lazy image loading for a single patient."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def resolve_paths(cfg: dict[str, Any], patient_id: str) -> dict[str, Path]:
    """Build concrete file paths for a patient from config path templates.

    Output paths (NIfTI, segmentations, metadata, results) are resolved
    relative to ``output_dir``.  The ``input_dir`` key points to the source
    data directory (``basepath / patient_id``).

    Parameters
    ----------
    cfg : dict
        Configuration dict (from :func:`autods_pet.config.load_config`).
    patient_id : str
        Patient identifier used for template substitution.

    Returns
    -------
    dict[str, Path]
        Mapping of logical names to resolved :class:`~pathlib.Path` objects.
    """
    from autods_pet.config import resolve_output_dir

    basepath = Path(cfg["paths"]["basepath"])
    output_root = resolve_output_dir(cfg)

    paths: dict[str, Path] = {
        "basepath": basepath,
        "input_dir": basepath / patient_id,
        "output_dir": output_root,
    }

    template_keys = (
        "ct_nifti",
        "pet_nifti",
        "pet_suv",
        "pet_registered",
        "seg_dir",
        "vert_body_seg",
        "pet_metadata",
        "elastix_report",
        "deauville_csv",
        "suv_csv",
    )
    for key in template_keys:
        template = cfg["paths"].get(key, "")
        if template:
            rel = template.format(patient_id=patient_id)
            paths[key] = output_root / rel
        else:
            paths[key] = output_root / f"{patient_id}_results" / f"{key}.nii.gz"

    input_seg_template = cfg["paths"].get("input_seg_dir", "{patient_id}/segmentations")
    paths["input_seg_dir"] = basepath / input_seg_template.format(patient_id=patient_id)

    return paths


class PatientCase:
    """Represents a single patient with resolved paths and lazy image loading.

    Parameters
    ----------
    cfg : dict
        Configuration dict (from :func:`autods_pet.config.load_config`).
    patient_id : str
        Patient identifier used for path resolution.
    """

    def __init__(self, cfg: dict[str, Any], patient_id: str) -> None:
        self.cfg = cfg
        self.patient_id = patient_id
        self.paths = resolve_paths(cfg, patient_id)
        self._cache: dict[str, Any] = {}

    @property
    def ct_path(self) -> Path:
        """Path to the CT NIfTI file."""
        return self.paths["ct_nifti"]

    @property
    def pet_path(self) -> Path:
        """Path to the PET NIfTI file."""
        return self.paths["pet_nifti"]

    @property
    def pet_suv_path(self) -> Path:
        """Path to the PET SUV file."""
        return self.paths["pet_suv"]

    @property
    def pet_registered_path(self) -> Path:
        """Path to the registered PET file."""
        return self.paths["pet_registered"]

    @property
    def seg_dir(self) -> Path:
        """Path to the segmentation directory."""
        return self.paths["seg_dir"]

    @property
    def vert_body_seg_path(self) -> Path:
        """Path to the vertebral body segmentation file."""
        return self.paths["vert_body_seg"]

    @property
    def metadata_path(self) -> Path:
        """Path to the PET metadata file."""
        return self.paths["pet_metadata"]

    @property
    def input_dir(self) -> Path:
        """Path to the source data directory (DICOM/NIfTI)."""
        return self.paths["input_dir"]

    @property
    def input_seg_dir(self) -> Path:
        """Path to the input segmentations directory."""
        return self.paths["input_seg_dir"]

    @property
    def output_dir(self) -> Path:
        """Path to the global output directory."""
        return self.paths["output_dir"]

    @property
    def elastix_report_path(self) -> Path:
        """Path to the Elastix registration report."""
        return self.paths["elastix_report"]

    @property
    def deauville_csv_path(self) -> Path:
        """Path to the per-patient Deauville scores CSV."""
        return self.paths["deauville_csv"]

    @property
    def suv_csv_path(self) -> Path:
        """Path to the per-patient SUV values CSV."""
        return self.paths["suv_csv"]

    @property
    def pet_series_uid(self) -> str | None:
        """Read the PET ``SeriesInstanceUID`` from ``PET_metadata.json``.

        Returns *None* when the metadata file does not exist or does not
        contain the field (e.g. patients converted before this field was
        captured).  Cached after the first read.
        """
        if "pet_series_uid" in self._cache:
            return self._cache["pet_series_uid"]

        import json

        uid: str | None = None
        meta_path = self.metadata_path
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                raw = data.get("SeriesInstanceUID")
                if raw is not None:
                    uid = str(raw)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning(
                    "Could not read SeriesInstanceUID from %s: %s", meta_path, exc
                )
        self._cache["pet_series_uid"] = uid
        return uid

    def _load_cached(self, key: str, path: Path) -> Any:
        """Load and cache a SimpleITK image."""
        if key not in self._cache:
            import SimpleITK as sitk

            self._cache[key] = sitk.ReadImage(str(path))
        return self._cache[key]

    def load_ct(self) -> Any:
        """Load the CT NIfTI image (cached)."""
        return self._load_cached("ct", self.ct_path)

    def load_pet_suv(self) -> Any:
        """Load the SUV-normalised PET image (cached)."""
        return self._load_cached("pet_suv", self.pet_suv_path)

    def load_pet_registered(self) -> Any:
        """Load the registered PET SUV image (cached)."""
        return self._load_cached("pet_registered", self.pet_registered_path)

    def load_segmentation(self) -> Any:
        """Load the TotalSegmentator multilabel segmentation (cached).

        Searches for :data:`~autods_pet.imaging.segmentation.TOTSEG_FILENAME`
        (or legacy ``whole_seg.nii[.gz]``) in *seg_dir*.
        """
        if "seg" not in self._cache:
            import SimpleITK as sitk

            from autods_pet.imaging.segmentation import TOTSEG_FILENAME

            for candidate in (TOTSEG_FILENAME, "whole_seg.nii", "whole_seg.nii.gz"):
                p = self.seg_dir / candidate
                if p.exists():
                    self._cache["seg"] = sitk.ReadImage(str(p))
                    return self._cache["seg"]
            raise FileNotFoundError(
                f"Segmentation not found in {self.seg_dir}. Run 'autods-pet segment' first."
            )
        return self._cache["seg"]

    def load_vert_body_seg(self) -> Any | None:
        """Load the vertebral body segmentation, or *None* if unavailable."""
        if "vb" not in self._cache:
            p = self.vert_body_seg_path
            if not p.exists():
                self._cache["vb"] = None
            else:
                import SimpleITK as sitk

                self._cache["vb"] = sitk.ReadImage(str(p))
        return self._cache["vb"]

    def clear_cache(self) -> None:
        """Release all cached images to free memory."""
        self._cache.clear()
