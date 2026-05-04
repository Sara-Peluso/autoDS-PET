"""Integration tests: extract all reference ROIs from a single phantom."""

import math

import pytest

from autods_pet.ops.stats import count_voxels
from autods_pet.roi.aorta_mbp import AortaMBP
from autods_pet.roi.liver import LiverROI
from autods_pet.roi.long_bones import LongBonesROI
from autods_pet.roi.lumbar_vb import LumbarVB


@pytest.mark.integration
class TestAllReferenceROIs:
    def test_all_reference_rois_produce_nonempty_masks(
        self,
        seg_phantom,
        vert_body_seg,
        pet_phantom,
    ):
        # All reference ROIs should produce non-empty masks from the phantom
        lumbar = LumbarVB(erosion_mm=0.0).extract(
            seg_phantom,
            vert_body_seg,
            pet_phantom,
        )
        aorta = AortaMBP(aorta_erosion_mm=0.0).extract(seg_phantom, pet_phantom)
        liver = LiverROI(erosion_mm=0.0).extract(seg_phantom, pet_phantom)
        # Use zero erosion for bones - phantom columns are too thin for default erosion
        bones = LongBonesROI(
            bones=[
                ("femur_L", 75, 0.0),
                ("femur_R", 76, 0.0),
                ("humerus_L", 69, 0.0),
                ("humerus_R", 70, 0.0),
            ],
            diaphysis_keep_pct=80,
        ).extract(seg_phantom, pet_phantom)

        assert count_voxels(lumbar.refined_mask) > 0
        assert count_voxels(aorta.refined_mask) > 0
        assert count_voxels(liver.refined_mask) > 0
        assert count_voxels(bones.refined_mask) > 0

    def test_reference_stats_are_finite_and_nonnegative(
        self,
        seg_phantom,
        vert_body_seg,
        pet_phantom,
    ):
        # All stats should be finite floats >= 0
        lumbar = LumbarVB(
            erosion_mm=0.0,
            stats=["mean", "p95"],
        ).extract(seg_phantom, vert_body_seg, pet_phantom)
        aorta = AortaMBP(
            aorta_erosion_mm=0.0,
            stats=["median", "mean"],
        ).extract(seg_phantom, pet_phantom)
        liver = LiverROI(
            erosion_mm=0.0,
            stats=["median", "mean"],
        ).extract(seg_phantom, pet_phantom)
        bones = LongBonesROI(
            bones=[
                ("femur_L", 75, 0.0),
                ("femur_R", 76, 0.0),
                ("humerus_L", 69, 0.0),
                ("humerus_R", 70, 0.0),
            ],
            diaphysis_keep_pct=80,
            stats=["mean", "p95"],
        ).extract(seg_phantom, pet_phantom)

        for result in [lumbar, aorta, liver, bones]:
            for name, value in result.stats.items():
                assert value is not None, f"stat {name} is None"
                assert math.isfinite(value), f"stat {name} = {value} is not finite"
                assert value >= 0, f"stat {name} = {value} is negative"

    def test_config_driven_extraction(self, seg_phantom, vert_body_seg, pet_phantom):
        # Verify that default config values produce valid results
        from autods_pet.config import default_config, get_roi_config

        cfg = default_config()

        lumbar_cfg = get_roi_config(cfg, "lumbar_vb")
        result = LumbarVB(
            lumbar_labels=lumbar_cfg["labels"],
            erosion_mm=lumbar_cfg["erosion_mm"],
            stats=lumbar_cfg["stats"],
        ).extract(seg_phantom, vert_body_seg, pet_phantom)
        assert result.stats is not None
        for stat_name in lumbar_cfg["stats"]:
            assert stat_name in result.stats
