"""Tests for autods_pet.roi - submodule and barrel imports work."""


def test_roi_submodule_imports():
    from autods_pet.roi.aorta_mbp import AortaMBP
    from autods_pet.roi.liver import LiverROI
    from autods_pet.roi.long_bones import LongBonesROI
    from autods_pet.roi.lumbar_vb import LumbarVB
    from autods_pet.roi.target_roi import TargetROI

    assert callable(AortaMBP)
    assert callable(LiverROI)
    assert callable(LongBonesROI)
    assert callable(LumbarVB)
    assert callable(TargetROI)


def test_roi_barrel_exports():
    from autods_pet.roi import AortaMBP, LiverROI, LongBonesROI, LumbarVB, TargetROI

    assert callable(AortaMBP)
    assert callable(LiverROI)
    assert callable(LongBonesROI)
    assert callable(LumbarVB)
    assert callable(TargetROI)
