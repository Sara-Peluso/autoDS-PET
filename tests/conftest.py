"""Top-level pytest configuration: custom markers, CLI options, shared fixtures."""

import numpy as np
import pytest
import SimpleITK as sitk

from autods_pet import labels


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def _build_seg_phantom():
    """Build a 30x30x30 multilabel segmentation with known label regions.

    Layout (numpy axes: z, y, x):
    - Label 52 (AORTA):     cylinder at (y=15, x=15, r=4) for z=2..19
    - Label 51 (HEART):     block y=0..7, x=0..7 for z=0..7
    - Labels 36-40 (T4-T8): block y=12..18, x=12..18 for z=4..9
    - Label 5  (LIVER):     block y=5..24, x=5..24 for z=10..19
    - Labels 29,28,27 (L3-L5): block y=12..18, x=12..18 for z=10..15
    - Label 75 (FEMUR_L):   column y=5, x=5 for z=0..29
    - Label 76 (FEMUR_R):   column y=5, x=25 for z=0..29
    - Label 69 (HUMERUS_L): column y=25, x=5 for z=0..29
    - Label 70 (HUMERUS_R): column y=25, x=25 for z=0..29

    NOTE: labels are painted in order; later labels overwrite earlier ones.
    Bones are thin columns (single voxel cross-section, 30 slices long).
    The aorta is a cylinder of radius 4 to survive erosion.
    """
    arr = np.zeros((30, 30, 30), dtype=np.uint8)

    # Bones - full z columns (painted first so others can overwrite)
    arr[:, 4:7, 4:7] = labels.FEMUR_L  # 75
    arr[:, 4:7, 23:26] = labels.FEMUR_R  # 76
    arr[:, 23:26, 4:7] = labels.HUMERUS_L  # 69
    arr[:, 23:26, 23:26] = labels.HUMERUS_R  # 70

    # Heart - block in lower-index corner
    arr[0:8, 0:8, 0:8] = labels.HEART  # 51

    # Aorta - cylinder center (y=15, x=15), radius 4, z=2..19
    yy, xx = np.ogrid[0:30, 0:30]
    cyl = ((yy - 15) ** 2 + (xx - 15) ** 2) <= 16  # r=4
    for z in range(2, 20):
        arr[z][cyl] = labels.AORTA  # 52

    # Thoracic vertebrae T4-T8 (labels 40..36) - block z=4..9
    t_labels = [labels.T4, labels.T5, labels.T6, labels.T7, labels.T8]  # 40,39,38,37,36
    for i, tl in enumerate(t_labels):
        arr[4 + i, 12:19, 12:19] = tl
    # Also put last label on slice 9 if only 5 labels for 6 slices
    arr[9, 12:19, 12:19] = labels.T8

    # Liver - large block
    arr[10:20, 5:25, 5:25] = labels.LIVER  # 5

    # Lumbar vertebrae L3-L5 - painted on top of liver in the overlap region
    arr[10:12, 12:19, 12:19] = labels.L3  # 29
    arr[12:14, 12:19, 12:19] = labels.L4  # 28
    arr[14:16, 12:19, 12:19] = labels.L5  # 27

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


@pytest.fixture(scope="session")
def seg_phantom():
    """30x30x30 multilabel segmentation phantom."""
    return _build_seg_phantom()


@pytest.fixture(scope="session")
def vert_body_seg():
    """Binary vertebral-body mask matching seg_phantom geometry.

    Covers the lumbar region (z=10..15, y=10..20, x=10..20) - slightly
    larger than L3-L5 labels so the intersection is meaningful.
    """
    arr = np.zeros((30, 30, 30), dtype=np.uint8)
    arr[10:16, 10:21, 10:21] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


@pytest.fixture(scope="session")
def pet_phantom():
    """Float64 PET image matching seg_phantom geometry with known SUV values.

    - Aorta region (cylinder): 2.0
    - Liver region: 3.0
    - Vertebral (L3-L5 overlap): 4.0
    - Bone columns: 5.0
    - Background: 0.5
    """
    arr = np.full((30, 30, 30), 0.5, dtype=np.float64)

    # Bones
    arr[:, 4:7, 4:7] = 5.0
    arr[:, 4:7, 23:26] = 5.0
    arr[:, 23:26, 4:7] = 5.0
    arr[:, 23:26, 23:26] = 5.0

    # Aorta
    yy, xx = np.ogrid[0:30, 0:30]
    cyl = ((yy - 15) ** 2 + (xx - 15) ** 2) <= 16
    for z in range(2, 20):
        arr[z][cyl] = 2.0

    # Liver
    arr[10:20, 5:25, 5:25] = 3.0

    # Lumbar overlap - higher uptake
    arr[10:16, 12:19, 12:19] = 4.0

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


@pytest.fixture(scope="session")
def empty_mask():
    """All-zeros mask, 30x30x30 at 1 mm spacing."""
    img = sitk.Image([30, 30, 30], sitk.sitkUInt8)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img
