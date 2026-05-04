"""Tests for autods_pet.ops.morphology - distance-transform based erosion/dilation."""

import numpy as np
import pytest
import SimpleITK as sitk

from autods_pet.ops.morphology import dilate_mask_mm, erode_mask_mm, signed_distance_mm
from autods_pet.ops.stats import count_voxels


@pytest.fixture()
def sphere_mask():
    """20×20×20 image with a solid sphere of radius 8 at the center."""
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    zz, yy, xx = np.ogrid[0:20, 0:20, 0:20]
    sphere = ((zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2) <= 64
    arr[sphere] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


@pytest.fixture()
def small_cube_mask():
    """10×10×10 image with a 6×6×6 solid cube at center."""
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:8, 2:8, 2:8] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def test_signed_distance_positive_inside(sphere_mask):
    dist = signed_distance_mm(sphere_mask)
    dist_arr = sitk.GetArrayFromImage(dist)
    # Interior voxels (excluding boundary) should be positive
    eroded = sitk.GetArrayFromImage(erode_mask_mm(sphere_mask, 1.5)).astype(bool)
    assert np.all(dist_arr[eroded] > 0)


def test_signed_distance_negative_outside(sphere_mask):
    dist = signed_distance_mm(sphere_mask)
    dist_arr = sitk.GetArrayFromImage(dist)
    # Voxels clearly outside (not on the boundary) should be negative
    # The corner voxel [0,0,0] is definitely outside
    assert dist_arr[0, 0, 0] < 0


def test_signed_distance_respects_anisotropic_spacing():
    # Use a larger asymmetric mask so boundary distances differ clearly
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[5:15, 5:15, 5:15] = 1
    img_iso = sitk.GetImageFromArray(arr.copy())
    img_iso.SetSpacing((1.0, 1.0, 1.0))
    img_aniso = sitk.GetImageFromArray(arr.copy())
    img_aniso.SetSpacing((1.0, 1.0, 3.0))  # z-spacing tripled

    dist_iso = sitk.GetArrayFromImage(signed_distance_mm(img_iso))
    dist_aniso = sitk.GetArrayFromImage(signed_distance_mm(img_aniso))
    # A voxel near the z-boundary (but not the center) should differ
    # Voxel at z=6 is 1 voxel from z-boundary: 1mm iso vs 3mm aniso
    assert dist_iso[6, 10, 10] != pytest.approx(dist_aniso[6, 10, 10], abs=0.5)


def test_erode_mask_reduces_voxels(sphere_mask):
    eroded = erode_mask_mm(sphere_mask, 2.0)
    assert count_voxels(eroded) < count_voxels(sphere_mask)
    assert count_voxels(eroded) > 0


def test_erode_mask_zero_radius_noop(small_cube_mask):
    eroded = erode_mask_mm(small_cube_mask, 0.0)
    assert count_voxels(eroded) == count_voxels(small_cube_mask)


def test_erode_mask_negative_radius_noop(small_cube_mask):
    eroded = erode_mask_mm(small_cube_mask, -5.0)
    assert count_voxels(eroded) == count_voxels(small_cube_mask)


def test_erode_mask_large_radius_empties(small_cube_mask):
    eroded = erode_mask_mm(small_cube_mask, 100.0)
    assert count_voxels(eroded) == 0


def test_erode_mask_precomputed_dist(sphere_mask):
    dist = signed_distance_mm(sphere_mask)
    eroded_auto = erode_mask_mm(sphere_mask, 2.0)
    eroded_pre = erode_mask_mm(sphere_mask, 2.0, dist=dist)
    assert count_voxels(eroded_auto) == count_voxels(eroded_pre)


def test_dilate_mask_increases_voxels(sphere_mask):
    dilated = dilate_mask_mm(sphere_mask, 2.0)
    assert count_voxels(dilated) > count_voxels(sphere_mask)


def test_dilate_mask_zero_radius_noop(small_cube_mask):
    dilated = dilate_mask_mm(small_cube_mask, 0.0)
    assert count_voxels(dilated) == count_voxels(small_cube_mask)


def test_dilate_mask_negative_radius_noop(small_cube_mask):
    dilated = dilate_mask_mm(small_cube_mask, -5.0)
    assert count_voxels(dilated) == count_voxels(small_cube_mask)


def test_dilate_mask_precomputed_dist(sphere_mask):
    dist = signed_distance_mm(sphere_mask)
    dilated_auto = dilate_mask_mm(sphere_mask, 2.0)
    dilated_pre = dilate_mask_mm(sphere_mask, 2.0, dist=dist)
    assert count_voxels(dilated_auto) == count_voxels(dilated_pre)
