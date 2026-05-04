"""Tests for autods_pet.normalization - SUV normalization and DICOM helpers."""

import logging
from datetime import date, datetime, time

import numpy as np
import pytest
import SimpleITK as sitk

from autods_pet.imaging.normalization import (
    compute_suvbw,
    decay_dose,
    effective_dose,
    parse_dicom_date,
    parse_dicom_time,
    seconds_between,
)


def test_parse_dicom_date_valid():
    assert parse_dicom_date("20260323") == date(2026, 3, 23)


def test_parse_dicom_date_invalid_month():
    with pytest.raises(ValueError, match="month"):
        parse_dicom_date("20261300")


def test_parse_dicom_date_invalid_day():
    with pytest.raises(ValueError, match="day"):
        parse_dicom_date("20260332")


def test_parse_dicom_date_too_short():
    with pytest.raises(ValueError, match="too short"):
        parse_dicom_date("2026")


def test_parse_dicom_time_hhmmss():
    assert parse_dicom_time("143025") == time(14, 30, 25)


def test_parse_dicom_time_fractional_seconds():
    t = parse_dicom_time("143025.123456")
    assert t.hour == 14
    assert t.minute == 30
    assert t.second == 25
    assert t.microsecond == 123456


def test_parse_dicom_time_missing_leading_zero():
    # "90500" should be zero-padded to "090500"
    assert parse_dicom_time("90500") == time(9, 5, 0)


def test_parse_dicom_time_hour_ge_24_wraps():
    t = parse_dicom_time("250000")
    assert t.hour == 1  # 25 % 24 = 1


def test_parse_dicom_time_empty_raises():
    with pytest.raises(ValueError, match="Empty"):
        parse_dicom_time("")


def test_parse_dicom_time_with_colons():
    # Some DICOM implementations use colons
    assert parse_dicom_time("14:30:25") == time(14, 30, 25)


def test_seconds_between_normal():
    inj = datetime(2026, 1, 1, 10, 0, 0)
    acq = datetime(2026, 1, 1, 11, 0, 0)
    assert seconds_between(inj, acq) == 3600.0


def test_seconds_between_midnight_wrap():
    # Injection at 23:50, acquisition at 00:10 → -23h40m → corrected to +20min
    inj = datetime(2026, 1, 1, 23, 50, 0)
    acq = datetime(2026, 1, 1, 0, 10, 0)
    dt = seconds_between(inj, acq)
    assert dt == pytest.approx(20 * 60, abs=1)


def test_seconds_between_large_negative_warns(caplog):
    # Injection at 20:00, acquisition at 02:00 → -18h → corrected to +6h + warning
    inj = datetime(2026, 1, 1, 20, 0, 0)
    acq = datetime(2026, 1, 1, 2, 0, 0)
    with caplog.at_level(logging.WARNING):
        dt = seconds_between(inj, acq)
    assert dt > 0
    assert "verify injection/acquisition times" in caplog.text


def test_decay_dose_zero_elapsed():
    assert decay_dose(370e6, 6586.2, 0) == 370e6


def test_decay_dose_one_half_life():
    hl = 6586.2  # F-18 half-life in seconds
    result = decay_dose(370e6, hl, hl)
    assert result == pytest.approx(185e6, rel=1e-6)


def test_effective_dose_none():
    # DecayCorrection = "NONE" → apply decay
    result = effective_dose(370e6, 6586.2, 3600, "NONE")
    assert result < 370e6


def test_effective_dose_start():
    # DecayCorrection = "START" → return dose unchanged
    assert effective_dose(370e6, 6586.2, 3600, "START") == 370e6


def test_effective_dose_admin():
    assert effective_dose(370e6, 6586.2, 3600, "ADMIN") == 370e6


def test_effective_dose_unknown_warns(caplog):
    with caplog.at_level(logging.WARNING):
        result = effective_dose(370e6, 6586.2, 3600, "FOOBAR")
    assert result < 370e6
    assert "Unknown DecayCorrection" in caplog.text


def test_compute_suvbw_correct_scaling():
    arr = np.ones((2, 2, 2), dtype=np.float64) * 1000.0  # 1000 Bq/mL
    pet = sitk.GetImageFromArray(arr)

    weight_kg = 70.0
    dose_bq = 370e6
    expected_scale = (weight_kg * 1000.0) / dose_bq

    suv = compute_suvbw(pet, weight_kg, dose_bq)
    suv_arr = sitk.GetArrayFromImage(suv)
    assert suv_arr[0, 0, 0] == pytest.approx(1000.0 * expected_scale, rel=1e-9)


def test_compute_suvbw_zero_weight_raises():
    pet = sitk.Image([2, 2, 2], sitk.sitkFloat64)
    with pytest.raises(ValueError, match="weight_kg"):
        compute_suvbw(pet, 0, 370e6)


def test_compute_suvbw_zero_dose_raises():
    pet = sitk.Image([2, 2, 2], sitk.sitkFloat64)
    with pytest.raises(ValueError, match="dose_bq"):
        compute_suvbw(pet, 70, 0)


def test_seconds_between_small_negative_midnight_wrap():
    """Elapsed time between -6h and 0 triggers midnight-wrap correction."""
    inj = datetime(2026, 1, 1, 10, 0, 0)
    acq = datetime(2026, 1, 1, 9, 0, 0)
    dt = seconds_between(inj, acq)
    assert dt == pytest.approx(23 * 3600, abs=1)


def test_seconds_between_very_large_negative_raises():
    """Elapsed time where midnight correction still yields negative raises ValueError."""
    # dt must be < -24h so that corrected (dt + 24h) is still < 0.
    # This can happen with multi-day datetime differences.
    inj = datetime(2026, 1, 3, 12, 0, 0)
    acq = datetime(2026, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError, match="metadata error"):
        seconds_between(inj, acq)


def test_decay_dose_negative_half_life_raises():
    """Negative half_life_s raises ValueError."""
    with pytest.raises(ValueError, match="half_life_s"):
        decay_dose(370e6, -1.0, 3600)


def test_decay_dose_zero_half_life_raises():
    """Zero half_life_s raises ValueError."""
    with pytest.raises(ValueError, match="half_life_s"):
        decay_dose(370e6, 0.0, 3600)
