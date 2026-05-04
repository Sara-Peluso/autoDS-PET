"""Tests for autods_pet.deauville - Deauville Score assignment."""

import pytest

from autods_pet.deauville import assign_ds


def test_none_target_allow_ds1():
    assert assign_ds(None, 2.0, 3.0, allow_ds1=True) == 1


def test_none_target_disallow_ds1():
    assert assign_ds(None, 2.0, 3.0, allow_ds1=False) == 0


def test_nan_target_allow_ds1():
    assert assign_ds(float("nan"), 2.0, 3.0, allow_ds1=True) == 1


def test_nan_target_disallow_ds1():
    assert assign_ds(float("nan"), 2.0, 3.0, allow_ds1=False) == 0


def test_target_below_mbp():
    assert assign_ds(1.0, 2.0, 3.0) == 2


def test_target_equal_mbp():
    assert assign_ds(2.0, 2.0, 3.0) == 2


def test_target_between_mbp_and_liver():
    assert assign_ds(2.5, 2.0, 3.0) == 3


def test_target_equal_liver():
    assert assign_ds(3.0, 2.0, 3.0) == 3


def test_target_between_liver_and_2x_liver():
    assert assign_ds(4.5, 2.0, 3.0) == 4


def test_target_equal_2x_liver():
    assert assign_ds(6.0, 2.0, 3.0) == 4


def test_target_above_2x_liver():
    assert assign_ds(7.0, 2.0, 3.0) == 5


def test_target_barely_above_2x_liver_within_tolerance():
    # 2 * 3.0 = 6.0; target is 6.0 + 1e-10 - should still be DS 4
    assert assign_ds(6.0 + 1e-10, 2.0, 3.0) == 4


def test_target_clearly_above_2x_liver():
    assert assign_ds(6.01, 2.0, 3.0) == 5


def test_mbp_zero_raises():
    with pytest.raises(ValueError, match="mbp_value"):
        assign_ds(3.0, 0, 3.0)


def test_mbp_negative_raises():
    with pytest.raises(ValueError, match="mbp_value"):
        assign_ds(3.0, -1.0, 3.0)


def test_liver_zero_raises():
    with pytest.raises(ValueError, match="liver_value"):
        assign_ds(3.0, 2.0, 0)


def test_liver_negative_raises():
    with pytest.raises(ValueError, match="liver_value"):
        assign_ds(3.0, 2.0, -1.0)


def test_inf_target_returns_ds5():
    """Infinite target value yields DS 5."""
    assert assign_ds(float("inf"), 2.0, 3.0) == 5


def test_target_exactly_at_2x_liver_boundary():
    """Target exactly at 2*liver (within tolerance) yields DS 4, not DS 5."""
    assert assign_ds(6.0, 2.0, 3.0) == 4


def test_very_small_positive_target():
    """Very small positive target (e.g. 0.001) that is below MBP yields DS 2."""
    assert assign_ds(0.001, 2.0, 3.0) == 2
