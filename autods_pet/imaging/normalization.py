"""SUV normalization for PET images.

Converts raw PET activity concentration (Bq/mL) to SUV body-weight
(SUVbw) using patient weight and injected dose.  Includes helpers for
parsing DICOM timing metadata and computing decay-corrected dose.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, time

import SimpleITK as sitk

log = logging.getLogger(__name__)


def parse_dicom_time(tstr: str) -> time:
    """Parse a DICOM TM string into a :class:`datetime.time`.

    Handles missing leading zeros, fractional seconds, and hours >= 24
    (wrapped modulo 24).  Out-of-range minutes and seconds are clamped to
    59 with a warning, so a leap second (``ss == 60``, permitted by the
    DICOM TM value representation) or a malformed field does not abort a
    run mid-cohort.

    Parameters
    ----------
    tstr : str
        DICOM TM-format string (e.g. ``"143012.123456"``).

    Returns
    -------
    datetime.time

    Raises
    ------
    ValueError
        If *tstr* is empty.

    Examples
    --------
    >>> from autods_pet.imaging.normalization import parse_dicom_time
    >>> parse_dicom_time("143012.000000")
    datetime.time(14, 30, 12)
    >>> parse_dicom_time("090000")
    datetime.time(9, 0)
    >>> parse_dicom_time("143060")  # leap second
    datetime.time(14, 30, 59)
    """
    s = str(tstr).replace(":", "").strip()
    if not s:
        raise ValueError("Empty DICOM time")

    if "." in s:
        base, frac = s.split(".", 1)
        frac = (frac + "000000")[:6]
    else:
        base, frac = s, "000000"

    base = base.zfill(6)

    hh = int(base[0:2])
    mm = int(base[2:4])
    ss = int(base[4:6])
    us = int(frac)

    if hh >= 24:
        log.warning(
            "DICOM time has hours=%d (>=24); wrapping to %d. Verify metadata.",
            hh,
            hh % 24,
        )
        hh = hh % 24

    if mm >= 60:
        log.warning(
            "DICOM time has minutes=%d (>=60); clamping to 59. Verify metadata.", mm
        )
        mm = 59

    if ss >= 60:
        log.warning(
            "DICOM time has seconds=%d (>=60); clamping to 59. Verify metadata.", ss
        )
        ss = 59

    return time(hour=hh, minute=mm, second=ss, microsecond=us)


def parse_dicom_date(dstr: str) -> date:
    """Parse a DICOM DA string (``YYYYMMDD``) into a :class:`datetime.date`.

    Parameters
    ----------
    dstr : str
        DICOM DA-format string (e.g. ``"20230415"``).

    Returns
    -------
    datetime.date

    Raises
    ------
    ValueError
        If *dstr* is too short or contains invalid month/day.

    Examples
    --------
    >>> from autods_pet.imaging.normalization import parse_dicom_date
    >>> parse_dicom_date("20230415")
    datetime.date(2023, 4, 15)
    """
    s = str(dstr).strip()
    if len(s) < 8:
        raise ValueError(f"DICOM date too short: {dstr!r}")
    yyyy, mm, dd = int(s[0:4]), int(s[4:6]), int(s[6:8])
    if not (1 <= mm <= 12):
        raise ValueError(f"Invalid DICOM date month: {mm} in {dstr!r}")
    if not (1 <= dd <= 31):
        raise ValueError(f"Invalid DICOM date day: {dd} in {dstr!r}")
    return date(yyyy, mm, dd)


def seconds_between(
    injection: datetime,
    acquisition: datetime,
    max_uptake_hours: float = 6.0,
) -> float:
    """Seconds elapsed from *injection* to *acquisition*, handling midnight wrap.

    If the elapsed time is slightly negative (0 to ``-max_uptake_hours``),
    assumes the acquisition crossed midnight and adds 24 h.  If more
    negative than ``-max_uptake_hours``, still corrects but logs a warning.

    Parameters
    ----------
    injection : datetime
        Radiopharmaceutical injection time.
    acquisition : datetime
        PET acquisition time.
    max_uptake_hours : float
        Maximum expected injection-to-scan interval in hours.  Negative
        elapsed times within this window are assumed to be midnight
        wraparounds.  The default of 6 h is appropriate for standard
        18F-FDG protocols (typical uptake ~60 min).  Increase for
        tracers with longer uptake periods (e.g. 68Ga-DOTATATE).

    Returns
    -------
    float
        Elapsed seconds (always >= 0 after midnight correction).

    Examples
    --------
    >>> from datetime import datetime
    >>> from autods_pet.imaging.normalization import seconds_between
    >>> seconds_between(datetime(2023, 1, 1, 9, 0), datetime(2023, 1, 1, 10, 0))
    3600.0
    """
    dt = (acquisition - injection).total_seconds()
    max_negative = -max_uptake_hours * 3600
    if max_negative <= dt < 0:
        dt += 24 * 3600  # likely midnight wrap
        log.info("Applied midnight-wrap correction: elapsed time %.1f h.", dt / 3600)
    elif dt < max_negative:
        corrected = dt + 24 * 3600
        if corrected < 0:
            raise ValueError(
                f"Elapsed time between injection and acquisition is "
                f"{dt / 3600:.1f} h ({corrected / 3600:.1f} h after midnight "
                f"correction). This likely indicates a metadata error in "
                f"injection/acquisition times."
            )
        dt = corrected
        log.warning(
            "Elapsed time was < -%.1f h (%.1f h after midnight correction); "
            "verify injection/acquisition times in metadata.",
            max_uptake_hours,
            dt / 3600,
        )
    return dt


def decay_dose(dose_bq: float, half_life_s: float, elapsed_s: float) -> float:
    """Apply radioactive decay: ``dose * exp(-lambda * t)``.

    Parameters
    ----------
    dose_bq : float
        Initial dose in Bq.
    half_life_s : float
        Radionuclide half-life in seconds.
    elapsed_s : float
        Time elapsed since injection in seconds.

    Returns
    -------
    float
        Decayed dose in Bq.

    Examples
    --------
    >>> from autods_pet.imaging.normalization import decay_dose
    >>> round(decay_dose(370e6, 6586.2, 3600), 1)
    253314180.2
    """
    if half_life_s <= 0:
        raise ValueError(f"half_life_s must be > 0, got {half_life_s}")
    lam = math.log(2.0) / half_life_s
    return dose_bq * math.exp(-lam * elapsed_s)


def effective_dose(
    total_dose_bq: float,
    half_life_s: float,
    elapsed_s: float,
    decay_correction: str,
) -> float:
    """Determine the effective reference dose depending on the DICOM DecayCorrection tag.

    Parameters
    ----------
    total_dose_bq : float
        Injected radionuclide dose in Bq.
    half_life_s : float
        Radionuclide half-life in seconds.
    elapsed_s : float
        Seconds between injection and acquisition.
    decay_correction : str
        DICOM DecayCorrection value (e.g. ``"NONE"``, ``"START"``, ``"ADMIN"``).

    Returns
    -------
    float
        Dose in Bq to use as the SUV denominator.

    Notes
    -----
    The denominator must be the activity present in the patient at the
    same reference time the pixel values are corrected to:

    ``ADMIN`` (and the non-standard ``YES``/``CORRECTED``) means the
    pixels are decay corrected back to the injection time, so the raw
    injected dose is already the activity at that reference time.

    ``START`` means the pixels are decay corrected to the acquisition
    start time, the dominant Siemens and GE whole-body FDG convention.
    The reference activity is then the injected dose decayed from
    injection to acquisition.

    ``NONE`` (and unknown values, handled conservatively the same way)
    means the pixels hold the activity measured at acquisition, so the
    dose must likewise be decayed to that time.
    """
    dc = str(decay_correction).upper().strip()

    if dc in ("NONE", ""):
        return decay_dose(total_dose_bq, half_life_s, elapsed_s)

    if dc == "START":
        return decay_dose(total_dose_bq, half_life_s, elapsed_s)

    if dc in ("ADMIN", "YES", "CORRECTED"):
        return total_dose_bq

    log.warning("Unknown DecayCorrection='%s', applying decay correction.", dc)
    return decay_dose(total_dose_bq, half_life_s, elapsed_s)


def compute_suvbw(
    pet_bqml: sitk.Image,
    weight_kg: float,
    dose_bq: float,
) -> sitk.Image:
    """Convert a PET image from Bq/mL to SUV body-weight.

    Parameters
    ----------
    pet_bqml : sitk.Image
        PET image in Bq/mL units.
    weight_kg : float
        Patient body weight in kg.
    dose_bq : float
        Effective injected dose in Bq (after decay correction if needed).

    Returns
    -------
    sitk.Image
        PET image in SUVbw units (float64).
    """
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if dose_bq <= 0:
        raise ValueError(f"dose_bq must be > 0, got {dose_bq}")

    scale = (weight_kg * 1000.0) / dose_bq
    pet = sitk.Cast(pet_bqml, sitk.sitkFloat64)
    return sitk.Cast(pet * scale, sitk.sitkFloat64)
