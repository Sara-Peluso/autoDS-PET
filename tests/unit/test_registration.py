"""Tests for autods_pet.registration - rigid PET-to-CT registration."""

from unittest.mock import MagicMock, patch

import pytest
import SimpleITK as sitk

from autods_pet.imaging.registration import rigid_register_pet_to_ct


def test_register_empty_ct_raises():
    ct = sitk.Image([0, 0, 0], sitk.sitkFloat64)
    pet = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    with pytest.raises(ValueError, match="CT image is empty"):
        rigid_register_pet_to_ct(ct, pet)


def test_register_empty_pet_raises():
    ct = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    pet = sitk.Image([0, 0, 0], sitk.sitkFloat64)
    with pytest.raises(ValueError, match="PET image is empty"):
        rigid_register_pet_to_ct(ct, pet)


def test_register_2d_ct_raises():
    ct = sitk.Image([5, 5], sitk.sitkFloat64)
    pet = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    with pytest.raises(ValueError, match="3D"):
        rigid_register_pet_to_ct(ct, pet)


def test_register_2d_pet_raises():
    ct = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    pet = sitk.Image([5, 5], sitk.sitkFloat64)
    with pytest.raises(ValueError, match="3D"):
        rigid_register_pet_to_ct(ct, pet)


def _make_mock_elastix(ct_size, ct_spacing):
    """Create a mock ElastixImageFilter that returns a result image."""
    mock_filter = MagicMock()
    result = sitk.Image(ct_size, sitk.sitkFloat64)
    result.SetSpacing(ct_spacing)
    mock_filter.GetResultImage.return_value = result
    return mock_filter


@patch("autods_pet.imaging.registration.sitk.ElastixImageFilter", create=True)
@patch("autods_pet.imaging.registration.sitk.GetDefaultParameterMap", create=True)
def test_register_result_is_float64(mock_get_pm, mock_elastix_cls):
    ct = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    pet = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    mock_elastix_cls.return_value = _make_mock_elastix([5, 5, 5], (1.0, 1.0, 1.0))
    mock_get_pm.return_value = {}
    result = rigid_register_pet_to_ct(ct, pet)
    assert result.GetPixelIDValue() == sitk.sitkFloat64


@patch("autods_pet.imaging.registration.sitk.ElastixImageFilter", create=True)
@patch("autods_pet.imaging.registration.sitk.GetDefaultParameterMap", create=True)
def test_register_log_to_console_on(mock_get_pm, mock_elastix_cls):
    ct = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    pet = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    mock_instance = _make_mock_elastix([5, 5, 5], (1.0, 1.0, 1.0))
    mock_elastix_cls.return_value = mock_instance
    mock_get_pm.return_value = {}
    rigid_register_pet_to_ct(ct, pet, log_to_console=True)
    mock_instance.LogToConsoleOn.assert_called_once()


@patch("autods_pet.imaging.registration.sitk.ElastixImageFilter", create=True)
@patch("autods_pet.imaging.registration.sitk.GetDefaultParameterMap", create=True)
def test_register_log_to_console_off(mock_get_pm, mock_elastix_cls):
    ct = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    pet = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    mock_instance = _make_mock_elastix([5, 5, 5], (1.0, 1.0, 1.0))
    mock_elastix_cls.return_value = mock_instance
    mock_get_pm.return_value = {}
    rigid_register_pet_to_ct(ct, pet, log_to_console=False)
    mock_instance.LogToConsoleOff.assert_called_once()


@patch("autods_pet.imaging.registration.sitk.ElastixImageFilter", create=True)
@patch("autods_pet.imaging.registration.sitk.GetDefaultParameterMap", create=True)
def test_register_execute_called(mock_get_pm, mock_elastix_cls):
    ct = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    pet = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    mock_instance = _make_mock_elastix([5, 5, 5], (1.0, 1.0, 1.0))
    mock_elastix_cls.return_value = mock_instance
    mock_get_pm.return_value = {}
    rigid_register_pet_to_ct(ct, pet)
    mock_instance.Execute.assert_called_once()


@patch("autods_pet.imaging.registration.sitk.ElastixImageFilter", create=True)
@patch("autods_pet.imaging.registration.sitk.GetDefaultParameterMap", create=True)
def test_register_parameter_map_set(mock_get_pm, mock_elastix_cls):
    ct = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    pet = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    pm = {}
    mock_get_pm.return_value = pm
    mock_instance = _make_mock_elastix([5, 5, 5], (1.0, 1.0, 1.0))
    mock_elastix_cls.return_value = mock_instance
    rigid_register_pet_to_ct(ct, pet)
    mock_instance.SetParameterMap.assert_called_once_with(pm)
    assert pm["Metric"] == ["AdvancedMattesMutualInformation"]
    assert pm["NumberOfResolutions"] == ["3"]


@patch("autods_pet.imaging.registration.sitk.ElastixImageFilter", create=True)
@patch("autods_pet.imaging.registration.sitk.GetDefaultParameterMap", create=True)
def test_register_with_report_path(mock_get_pm, mock_elastix_cls, tmp_path):
    """report_path branch: writes transform, cleans up result.*.nii, renames."""
    ct = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    pet = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    report_path = tmp_path / "metadata" / "transform_params.txt"

    mock_instance = _make_mock_elastix([5, 5, 5], (1.0, 1.0, 1.0))
    mock_elastix_cls.return_value = mock_instance
    mock_get_pm.return_value = {}

    def create_side_effects():
        out_dir = report_path.parent
        (out_dir / "result.0.nii").write_text("fake")
        (out_dir / "TransformParameters.0.txt").write_text("params")

    mock_instance.Execute.side_effect = lambda: create_side_effects()

    result = rigid_register_pet_to_ct(ct, pet, report_path=report_path)

    mock_instance.SetOutputDirectory.assert_called_once_with(str(report_path.parent))
    assert not (report_path.parent / "result.0.nii").exists()  # cleaned up
    assert report_path.exists()  # renamed from TransformParameters.0.txt
    assert result.GetPixelIDValue() == sitk.sitkFloat64


@patch("autods_pet.imaging.registration.sitk.ElastixImageFilter", create=True)
@patch("autods_pet.imaging.registration.sitk.GetDefaultParameterMap", create=True)
def test_register_report_path_no_leftover_nii(mock_get_pm, mock_elastix_cls, tmp_path):
    """report_path branch when Execute produces no result.*.nii files."""
    ct = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    pet = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    report_path = tmp_path / "metadata" / "transform.txt"

    mock_instance = _make_mock_elastix([5, 5, 5], (1.0, 1.0, 1.0))
    mock_elastix_cls.return_value = mock_instance
    mock_get_pm.return_value = {}

    def create_side_effects():
        out_dir = report_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "TransformParameters.0.txt").write_text("params")

    mock_instance.Execute.side_effect = lambda: create_side_effects()

    rigid_register_pet_to_ct(ct, pet, report_path=report_path)
    assert report_path.exists()


@patch("autods_pet.imaging.registration.sitk.ElastixImageFilter", create=True)
@patch("autods_pet.imaging.registration.sitk.GetDefaultParameterMap", create=True)
def test_register_report_path_native_is_target(mock_get_pm, mock_elastix_cls, tmp_path):
    """When report_path IS TransformParameters.0.txt, no rename needed."""
    ct = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    pet = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    report_path = tmp_path / "TransformParameters.0.txt"

    mock_instance = _make_mock_elastix([5, 5, 5], (1.0, 1.0, 1.0))
    mock_elastix_cls.return_value = mock_instance
    mock_get_pm.return_value = {}

    def create_side_effects():
        report_path.write_text("params")

    mock_instance.Execute.side_effect = lambda: create_side_effects()

    rigid_register_pet_to_ct(ct, pet, report_path=report_path)
    assert report_path.exists()
    assert report_path.read_text() == "params"


@patch("autods_pet.imaging.registration.sitk.TransformixImageFilter", create=True)
@patch("autods_pet.imaging.registration.sitk.ReadParameterFile", create=True)
def test_apply_transform_calls_transformix(
    mock_read_pm, mock_transformix_cls, tmp_path
):
    """apply_transform reads the parameter file and executes transformix."""
    from autods_pet.imaging.registration import apply_transform

    mock_pm = {"ResampleInterpolator": ["FinalBSplineInterpolator"]}
    mock_read_pm.return_value = mock_pm

    mock_instance = MagicMock()
    fake_result = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    mock_instance.GetResultImage.return_value = fake_result
    mock_transformix_cls.return_value = mock_instance

    moving = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    transform_path = tmp_path / "transform.txt"
    transform_path.write_text("fake params")

    result = apply_transform(moving, transform_path)

    mock_read_pm.assert_called_once_with(str(transform_path))
    mock_instance.SetMovingImage.assert_called_once_with(moving)
    mock_instance.Execute.assert_called_once()
    assert result is fake_result


@patch("autods_pet.imaging.registration.sitk.TransformixImageFilter", create=True)
@patch("autods_pet.imaging.registration.sitk.ReadParameterFile", create=True)
def test_apply_transform_nearest_neighbor(mock_read_pm, mock_transformix_cls, tmp_path):
    """apply_transform sets nearest-neighbor interpolation when requested."""
    from autods_pet.imaging.registration import apply_transform

    mock_pm = {"ResampleInterpolator": ["FinalBSplineInterpolator"]}
    mock_read_pm.return_value = mock_pm

    mock_instance = MagicMock()
    mock_instance.GetResultImage.return_value = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    mock_transformix_cls.return_value = mock_instance

    moving = sitk.Image([5, 5, 5], sitk.sitkFloat64)
    transform_path = tmp_path / "transform.txt"
    transform_path.write_text("fake")

    apply_transform(moving, transform_path, nearest_neighbor=True)

    assert mock_pm["ResampleInterpolator"] == ["FinalNearestNeighborInterpolator"]
