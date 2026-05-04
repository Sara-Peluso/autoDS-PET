"""Deauville Score assignment from ROI statistics.

Implements the standard threshold ladder comparing a target ROI uptake
value against the mediastinal blood pool (MBP) and liver references.
"""

from __future__ import annotations

import math

# Small tolerance to avoid floating-point rounding flipping borderline scores
_DS_TOL = 1e-9


def assign_ds(
    target_value: float | None,
    mbp_value: float,
    liver_value: float,
    allow_ds1: bool = False,
    liver_multiplier: float = 2.0,
) -> int:
    """Assign a Deauville Score (1-5) from PET uptake values.

    **Clinical note on DS 1:** Per the Deauville criteria, DS 1 represents
    *absent residual uptake* (no visible lesion).  It is only assigned when
    ``target_value`` is ``None`` (or ``NaN``) and ``allow_ds1`` is ``True``.
    A target with any measurable uptake (even very low, e.g. SUV = 0.001)
    receives DS 2 or higher, not DS 1.  Callers should pass ``None`` to
    indicate the absence of a target lesion.

    The DS 4/5 boundary is set at ``liver_multiplier × liver_value``
    (default 2.0, per the Lugano 2014 consensus - Barrington et al.,
    *J Clin Oncol*, 2014).  Some protocols use 3× liver; pass
    ``liver_multiplier=3.0`` for those.

    Parameters
    ----------
    target_value : float or None
        Uptake statistic for the target ROI (e.g. p95, max).
        If None/NaN, interpreted as "no uptake" (DS 1 when *allow_ds1* is True).
    mbp_value : float
        Mediastinal blood pool reference (e.g. voxelwise median of aorta).
    liver_value : float
        Liver reference (e.g. voxelwise median of liver).
    allow_ds1 : bool
        If True, a missing/NaN *target_value* yields DS 1 (used for focal
        lesion scoring where absence of a lesion = DS 1).  If False, a
        missing target yields 0 (unassignable).
    liver_multiplier : float
        Multiplier applied to *liver_value* for the DS 4/5 threshold
        (default ``2.0`` per Lugano 2014).

    Returns
    -------
    int
        Deauville Score: 1-5, or 0 if the score cannot be assigned.

    Raises
    ------
    ValueError
        If *mbp_value* or *liver_value* is <= 0.

    Examples
    --------
    >>> from autods_pet.deauville import assign_ds
    >>> assign_ds(target_value=1.5, mbp_value=2.0, liver_value=3.0)
    2
    >>> assign_ds(target_value=2.5, mbp_value=2.0, liver_value=3.0)
    3
    >>> assign_ds(target_value=5.0, mbp_value=2.0, liver_value=3.0)
    4
    >>> assign_ds(target_value=7.0, mbp_value=2.0, liver_value=3.0)
    5
    >>> assign_ds(target_value=None, mbp_value=2.0, liver_value=3.0, allow_ds1=True)
    1
    """
    # Validate references
    if mbp_value <= 0:
        raise ValueError(f"mbp_value must be > 0, got {mbp_value}")
    if liver_value <= 0:
        raise ValueError(f"liver_value must be > 0, got {liver_value}")

    # Handle missing target
    if target_value is None or (
        isinstance(target_value, float) and math.isnan(target_value)
    ):
        return 1 if allow_ds1 else 0

    if target_value <= mbp_value + _DS_TOL:
        return 2
    if target_value <= liver_value + _DS_TOL:
        return 3
    if target_value <= liver_multiplier * liver_value + _DS_TOL:
        return 4
    return 5
