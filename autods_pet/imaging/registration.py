"""Rigid PET-to-CT registration using SimpleElastix.

Registers a PET image onto the CT grid (same spacing, origin, direction,
and size as CT) using mutual information with a rigid transform.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import SimpleITK as sitk

log = logging.getLogger(__name__)


def rigid_register_pet_to_ct(
    ct: sitk.Image,
    pet: sitk.Image,
    log_to_console: bool = False,
    report_path: Path | None = None,
) -> sitk.Image:
    """Rigidly register *pet* onto the *ct* grid.

    Parameters
    ----------
    ct : sitk.Image
        CT image (fixed / reference frame).
    pet : sitk.Image
        PET image (moving).
    log_to_console : bool
        If True, print Elastix iteration logs to stdout.
    report_path : Path or None
        If set, Elastix writes ``TransformParameters.0.txt`` to
        ``report_path.parent`` and it is renamed to *report_path*.

    Returns
    -------
    sitk.Image
        PET resampled onto the CT grid (float64).
    """
    if ct.GetNumberOfPixels() == 0:
        raise ValueError("CT image is empty (zero pixels)")
    if pet.GetNumberOfPixels() == 0:
        raise ValueError("PET image is empty (zero pixels)")
    if ct.GetDimension() != 3 or pet.GetDimension() != 3:
        raise ValueError("Both CT and PET must be 3D images")

    fixed = sitk.Cast(ct, sitk.sitkFloat64)
    moving = sitk.Cast(pet, sitk.sitkFloat64)

    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)

    pm = sitk.GetDefaultParameterMap("rigid")
    pm["Metric"] = ["AdvancedMattesMutualInformation"]
    pm["NumberOfResolutions"] = ["3"]
    pm["MaximumNumberOfIterations"] = ["512"]
    pm["ResampleInterpolator"] = ["FinalLinearInterpolator"]
    pm["ResultImagePixelType"] = ["double"]
    pm["DefaultPixelValue"] = ["0"]
    pm["WriteResultImage"] = ["false"]  # We save the image ourselves.

    elastix.SetParameterMap(pm)
    if log_to_console:
        elastix.LogToConsoleOn()
    else:
        elastix.LogToConsoleOff()

    if report_path is not None:
        # Redirect Elastix output files to the metadata directory.
        out_dir = report_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        elastix.SetOutputDirectory(str(out_dir))
        elastix.Execute()
        # Clean up result image (SimpleElastix writes it despite WriteResultImage=false).
        for leftover in out_dir.glob("result.*.nii"):
            leftover.unlink()
        # Rename the native TransformParameters file to the requested name.
        native = out_dir / "TransformParameters.0.txt"
        if native.exists() and native != report_path:
            native.rename(report_path)
        log.info("Wrote Elastix transform: %s", report_path)
    else:
        # No report requested - use a temp dir to catch side-effect files.
        with tempfile.TemporaryDirectory() as tmpdir:
            elastix.SetOutputDirectory(tmpdir)
            elastix.Execute()

    # Sanity-check registration plausibility: verify translation magnitude
    # does not exceed the fixed image's field-of-view diagonal.
    try:
        tp = elastix.GetTransformParameterMap()[0]
        params = [float(p) for p in tp["TransformParameters"]]
        # Rigid 3D: last 3 parameters are translation (tx, ty, tz).
        translation = params[-3:]
        import math as _math

        translation_mm = _math.sqrt(sum(t * t for t in translation))
        fov = [ct.GetSize()[i] * ct.GetSpacing()[i] for i in range(3)]
        fov_diag = _math.sqrt(sum(d * d for d in fov))
        if translation_mm > fov_diag:
            log.warning(
                "Registration translation (%.1f mm) exceeds FOV diagonal "
                "(%.1f mm) - the result may be misregistered. "
                "Verify PET-CT alignment visually.",
                translation_mm,
                fov_diag,
            )
        else:
            log.info(
                "Registration translation: %.1f mm (FOV diagonal: %.1f mm).",
                translation_mm,
                fov_diag,
            )
    except Exception:
        log.debug(
            "Could not read transform parameters for sanity check.", exc_info=True
        )

    return sitk.Cast(elastix.GetResultImage(), sitk.sitkFloat64)


def apply_transform(
    moving: sitk.Image,
    transform_path: Path,
    nearest_neighbor: bool = False,
) -> sitk.Image:
    """Apply a saved Elastix transform to an image using transformix.

    Parameters
    ----------
    moving : sitk.Image
        Image to transform.
    transform_path : Path
        Path to the Elastix ``TransformParameters`` text file.
    nearest_neighbor : bool
        If True, use nearest-neighbor interpolation (for binary masks).

    Returns
    -------
    sitk.Image
        Transformed image.
    """
    pm = sitk.ReadParameterFile(str(transform_path))
    if nearest_neighbor:
        pm["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

    transformix = sitk.TransformixImageFilter()
    transformix.SetMovingImage(moving)
    transformix.SetTransformParameterMap(pm)
    transformix.LogToConsoleOff()

    with tempfile.TemporaryDirectory() as tmpdir:
        transformix.SetOutputDirectory(tmpdir)
        transformix.Execute()

    return transformix.GetResultImage()
