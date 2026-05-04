"""Shared fixtures for unit tests - synthetic images and factory helpers."""

import numpy as np
import pytest
import SimpleITK as sitk


@pytest.fixture()
def make_image():
    """Factory: numpy array → sitk.Image with optional spacing."""

    def _make(arr, spacing=(1.0, 1.0, 1.0)):
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing)
        return img

    return _make


@pytest.fixture()
def make_mask():
    """Factory: shape → binary sitk.Image.  ones_slices selects which slices are 1."""

    def _make(shape, ones_slices=None, spacing=(1.0, 1.0, 1.0)):
        if ones_slices is None:
            arr = np.ones(shape, dtype=np.uint8)
        else:
            arr = np.zeros(shape, dtype=np.uint8)
            for s in ones_slices:
                arr[s] = 1
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing)
        return img

    return _make


@pytest.fixture()
def minimal_cfg(tmp_path):
    """Minimal configuration dict with basepath set to a temp directory."""
    from autods_pet.config import default_config

    cfg = default_config()
    cfg["paths"]["basepath"] = str(tmp_path / "data")
    cfg["paths"]["output_dir"] = str(tmp_path / "results")
    return cfg
