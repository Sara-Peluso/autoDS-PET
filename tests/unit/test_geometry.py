"""Tests for autods_pet.geometry - geometry checks."""

import SimpleITK as sitk

from autods_pet.imaging.geometry import check_same_geometry


def test_geometry_identical():
    a = sitk.Image([3, 3, 3], sitk.sitkFloat64)
    b = sitk.Image([3, 3, 3], sitk.sitkFloat64)
    assert check_same_geometry(a, b) is True


def test_geometry_different_size():
    a = sitk.Image([3, 3, 3], sitk.sitkFloat64)
    b = sitk.Image([4, 3, 3], sitk.sitkFloat64)
    assert check_same_geometry(a, b) is False


def test_geometry_spacing_within_tolerance():
    a = sitk.Image([3, 3, 3], sitk.sitkFloat64)
    b = sitk.Image([3, 3, 3], sitk.sitkFloat64)
    b.SetSpacing([1.0 + 5e-5, 1.0, 1.0])
    assert check_same_geometry(a, b) is True


def test_geometry_spacing_beyond_tolerance():
    a = sitk.Image([3, 3, 3], sitk.sitkFloat64)
    b = sitk.Image([3, 3, 3], sitk.sitkFloat64)
    b.SetSpacing([1.0 + 5e-3, 1.0, 1.0])
    assert check_same_geometry(a, b) is False


def test_geometry_different_origin():
    a = sitk.Image([3, 3, 3], sitk.sitkFloat64)
    b = sitk.Image([3, 3, 3], sitk.sitkFloat64)
    b.SetOrigin([10.0, 0.0, 0.0])
    assert check_same_geometry(a, b) is False


def test_geometry_different_direction():
    a = sitk.Image([3, 3, 3], sitk.sitkFloat64)
    b = sitk.Image([3, 3, 3], sitk.sitkFloat64)
    # Flip the first axis
    b.SetDirection([-1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert check_same_geometry(a, b) is False
