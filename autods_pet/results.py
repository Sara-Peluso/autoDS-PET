"""Structured result types for ROI extraction and Deauville Score computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ROIResult:
    """Result from a single ROI extraction.

    Attributes
    ----------
    stats : dict[str, float | None]
        Computed statistics (e.g. ``{"median": 2.8, "p95": 3.1}``).
    refined_mask : sitk.Image or None
        The refined binary mask, if available.
    shrinkage : dict[str, Any] or None
        Voxel-count and volume deltas from refinement (see
        :func:`~autods_pet.ops.stats.shrinkage_report`).

    Examples
    --------
    >>> from autods_pet.results import ROIResult
    >>> roi = ROIResult(stats={"median": 2.8, "p95": 3.1})
    >>> roi.stats["median"]
    2.8
    """

    stats: dict[str, float | None] = field(default_factory=dict)
    refined_mask: Any = None  # sitk.Image; Any to avoid eager import
    shrinkage: dict[str, Any] | None = None


@dataclass
class DeauvilleResult:
    """Result from a full pipeline run.

    Attributes
    ----------
    patient_id : str
        The patient identifier.
    scores : dict[str, int | float]
        Deauville Scores keyed by short name (e.g.
        ``{"FL_DS": 4, "BM_DS": 3}``) or float ratios (e.g.
        ``{"BLR": 1.8}``).
    rois : dict[str, ROIResult]
        Per-ROI extraction results.
    error : str or None
        Error message if the pipeline failed, else ``None``.

    Examples
    --------
    >>> from autods_pet.results import DeauvilleResult, ROIResult
    >>> result = DeauvilleResult(
    ...     patient_id="PAT001",
    ...     scores={"FL_DS": 4},
    ...     rois={"Liver": ROIResult(stats={"median": 2.8})},
    ... )
    >>> result.scores["FL_DS"]
    4
    """

    patient_id: str
    scores: dict[str, int | float] = field(default_factory=dict)
    rois: dict[str, ROIResult] = field(default_factory=dict)
    error: str | None = None
