"""Integration tests: ROI extraction → Deauville Score assignment."""

import numpy as np
import pytest
import SimpleITK as sitk

from autods_pet.deauville import assign_ds
from autods_pet.roi.aorta_mbp import AortaMBP
from autods_pet.roi.liver import LiverROI
from autods_pet.roi.target_roi import TargetROI


@pytest.mark.integration
class TestFullDeauvillePipeline:
    """End-to-end: extract reference ROI stats, then assign Deauville Scores."""

    def test_ds2_target_below_mbp(self, seg_phantom, pet_phantom):
        # Verifies: full pipeline → DS 2 when target < MBP
        mbp = AortaMBP(aorta_erosion_mm=0.0).extract(seg_phantom, pet_phantom)
        liver = LiverROI(erosion_mm=0.0).extract(seg_phantom, pet_phantom)

        # Build a target mask with low uptake (use background region)
        target_arr = np.zeros((30, 30, 30), dtype=np.uint8)
        target_arr[25:28, 0:3, 0:3] = 1
        target_mask = sitk.GetImageFromArray(target_arr)
        target_mask.SetSpacing((1.0, 1.0, 1.0))
        # PET in that region is 0.5 (background), which is below MBP (2.0)
        target = TargetROI(stats=["max"]).extract(target_mask, pet_phantom)

        ds = assign_ds(
            target.stats["max"],
            mbp.stats["median"],
            liver.stats["median"],
        )
        assert ds == 2

    def test_ds5_target_above_2x_liver(self, seg_phantom):
        # Verifies: full pipeline → DS 5 when target > 2×liver
        # Build a custom PET where target region has very high uptake
        pet_arr = np.full((30, 30, 30), 0.5, dtype=np.float64)
        # Aorta region: 2.0
        yy, xx = np.ogrid[0:30, 0:30]
        cyl = ((yy - 15) ** 2 + (xx - 15) ** 2) <= 16
        for z in range(2, 20):
            pet_arr[z][cyl] = 2.0
        # Liver region: 3.0
        pet_arr[10:20, 5:25, 5:25] = 3.0
        # Target region: 20.0 (way above 2×3.0 = 6.0)
        pet_arr[25:28, 0:3, 0:3] = 20.0

        pet = sitk.GetImageFromArray(pet_arr)
        pet.SetSpacing((1.0, 1.0, 1.0))

        mbp = AortaMBP(aorta_erosion_mm=0.0).extract(seg_phantom, pet)
        liver = LiverROI(erosion_mm=0.0).extract(seg_phantom, pet)

        target_arr = np.zeros((30, 30, 30), dtype=np.uint8)
        target_arr[25:28, 0:3, 0:3] = 1
        target_mask = sitk.GetImageFromArray(target_arr)
        target_mask.SetSpacing((1.0, 1.0, 1.0))
        target = TargetROI(stats=["max"]).extract(target_mask, pet)

        ds = assign_ds(
            target.stats["max"],
            mbp.stats["median"],
            liver.stats["median"],
        )
        assert ds == 5

    def test_ds4_target_between_liver_and_2x(self, seg_phantom):
        # Verifies: full pipeline → DS 4
        pet_arr = np.full((30, 30, 30), 0.5, dtype=np.float64)
        yy, xx = np.ogrid[0:30, 0:30]
        cyl = ((yy - 15) ** 2 + (xx - 15) ** 2) <= 16
        for z in range(2, 20):
            pet_arr[z][cyl] = 2.0
        pet_arr[10:20, 5:25, 5:25] = 3.0
        # Target: 4.5 (between 3.0 and 6.0)
        pet_arr[25:28, 0:3, 0:3] = 4.5

        pet = sitk.GetImageFromArray(pet_arr)
        pet.SetSpacing((1.0, 1.0, 1.0))

        mbp = AortaMBP(aorta_erosion_mm=0.0).extract(seg_phantom, pet)
        liver = LiverROI(erosion_mm=0.0).extract(seg_phantom, pet)

        target_arr = np.zeros((30, 30, 30), dtype=np.uint8)
        target_arr[25:28, 0:3, 0:3] = 1
        target_mask = sitk.GetImageFromArray(target_arr)
        target_mask.SetSpacing((1.0, 1.0, 1.0))
        target = TargetROI(stats=["max"]).extract(target_mask, pet)

        ds = assign_ds(
            target.stats["max"],
            mbp.stats["median"],
            liver.stats["median"],
        )
        assert ds == 4

    def test_ds1_no_target(self, seg_phantom, pet_phantom):
        # Verifies: full pipeline → DS 1 with allow_ds1=True
        mbp = AortaMBP(aorta_erosion_mm=0.0).extract(seg_phantom, pet_phantom)
        liver = LiverROI(erosion_mm=0.0).extract(seg_phantom, pet_phantom)
        ds = assign_ds(
            None,
            mbp.stats["median"],
            liver.stats["median"],
            allow_ds1=True,
        )
        assert ds == 1
