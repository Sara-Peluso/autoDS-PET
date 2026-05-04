"""Tests for autods_pet.dicom - DICOM discovery, tag extraction, and helpers."""

import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

from autods_pet.imaging.dicom import (
    _safe_get,
    dicom_series_to_nifti,
    extract_ct_tags,
    extract_patient_weight,
    extract_pet_tags,
    find_dicom_files,
    find_series_by_modality,
    resolve_patient_folder,
)


def test_safe_get_str():
    ds = Dataset()
    ds.PatientID = "PAT001"
    assert _safe_get(ds, "PatientID") == "PAT001"


def test_safe_get_float():
    ds = Dataset()
    ds.RadionuclideTotalDose = "370000000"
    assert _safe_get(ds, "RadionuclideTotalDose", convert=float) == 370000000.0


def test_safe_get_missing_returns_default():
    ds = Dataset()
    assert _safe_get(ds, "NonexistentTag") is None
    assert _safe_get(ds, "NonexistentTag", default="N/A") == "N/A"


def test_safe_get_conversion_failure_returns_default():
    ds = Dataset()
    ds.PatientID = "not_a_number"
    assert _safe_get(ds, "PatientID", convert=float) is None


def _make_pet_dcm(tmp_path, *, with_sequence=True, weight=None):
    """Create a minimal PET DICOM file for testing."""
    ds = Dataset()
    ds.PatientID = "TEST001"
    ds.StudyDate = "20260325"
    ds.AcquisitionTime = "143025"
    ds.Units = "BQML"
    ds.DecayCorrection = "START"
    ds.Modality = "PT"

    if with_sequence:
        rph = Dataset()
        rph.RadiopharmaceuticalStartTime = "140000"
        rph.RadionuclideTotalDose = "370000000"
        rph.RadionuclideHalfLife = "6586.2"
        ds.RadiopharmaceuticalInformationSequence = Sequence([rph])
    else:
        # Top-level fallback
        ds.RadionuclideTotalDose = "370000000"
        ds.RadionuclideHalfLife = "6586.2"

    if weight is not None:
        ds.PatientWeight = str(weight)

    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    path = tmp_path / "test.dcm"
    ds.save_as(str(path))
    return path


def test_extract_pet_tags_with_sequence(tmp_path):
    dcm = _make_pet_dcm(tmp_path, with_sequence=True)
    tags = extract_pet_tags(dcm)
    assert tags["PatientID"] == "TEST001"
    assert tags["StudyDate"] == "20260325"
    assert tags["Units"] == "BQML"
    assert tags["RadiopharmaceuticalStartTime"] == "140000"
    assert tags["RadionuclideTotalDose"] == 370000000.0
    assert tags["RadionuclideHalfLife"] == 6586.2


def test_extract_pet_tags_fallback_no_sequence(tmp_path):
    dcm = _make_pet_dcm(tmp_path, with_sequence=False)
    tags = extract_pet_tags(dcm)
    assert tags["RadionuclideTotalDose"] == 370000000.0
    assert tags["RadionuclideHalfLife"] == 6586.2
    # RadiopharmaceuticalStartTime not available without sequence
    assert tags["RadiopharmaceuticalStartTime"] is None


def test_extract_pet_tags_missing_tags(tmp_path):
    ds = Dataset()
    ds.Modality = "PT"
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    path = tmp_path / "empty.dcm"
    ds.save_as(str(path))

    tags = extract_pet_tags(path)
    assert tags["PatientID"] is None
    assert tags["RadionuclideTotalDose"] is None


def test_extract_patient_weight_present(tmp_path):
    dcm = _make_pet_dcm(tmp_path, weight=70.5)
    assert extract_patient_weight(dcm) == 70.5


def test_extract_patient_weight_absent(tmp_path):
    dcm = _make_pet_dcm(tmp_path, weight=None)
    assert extract_patient_weight(dcm) is None


def test_extract_patient_weight_zero(tmp_path):
    dcm = _make_pet_dcm(tmp_path, weight=0)
    assert extract_patient_weight(dcm) is None


def test_find_dicom_files_filters_small(tmp_path):
    big = tmp_path / "big.dcm"
    big.write_bytes(b"\x00" * 200_000)  # 200 KB
    small = tmp_path / "small.dcm"
    small.write_bytes(b"\x00" * 50)  # 50 bytes

    files = find_dicom_files(tmp_path, size_threshold_kb=100)
    assert len(files) == 1
    assert files[0] == big


def test_find_dicom_files_empty_dir(tmp_path):
    assert find_dicom_files(tmp_path) == []


def test_find_dicom_files_no_dcm_extension(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_bytes(b"\x00" * 200_000)
    assert find_dicom_files(tmp_path) == []


def test_resolve_case_insensitive(tmp_path):
    (tmp_path / "PATIENT_001").mkdir()
    result = resolve_patient_folder(tmp_path, "patient_001")
    assert result is not None
    assert result.name == "PATIENT_001"


def test_resolve_exact_match(tmp_path):
    (tmp_path / "PAT001").mkdir()
    result = resolve_patient_folder(tmp_path, "PAT001")
    assert result is not None
    assert result.name == "PAT001"


def test_resolve_not_found(tmp_path):
    (tmp_path / "OTHER").mkdir()
    assert resolve_patient_folder(tmp_path, "PAT001") is None


def test_resolve_nonexistent_basepath(tmp_path):
    assert resolve_patient_folder(tmp_path / "nope", "PAT001") is None


def _make_ct_dcm(tmp_path, **overrides):
    """Create a minimal CT DICOM file for testing."""
    ds = Dataset()
    ds.Modality = "CT"
    ds.PatientID = overrides.pop("PatientID", "CT_TEST001")
    ds.StudyDate = overrides.pop("StudyDate", "20260325")
    ds.SeriesInstanceUID = overrides.pop(
        "SeriesInstanceUID", pydicom.uid.generate_uid()
    )
    ds.SliceThickness = overrides.pop("SliceThickness", "2.5")
    ds.KVP = overrides.pop("KVP", "120")
    ds.Rows = overrides.pop("Rows", 512)
    ds.Columns = overrides.pop("Columns", 512)
    ds.Manufacturer = overrides.pop("Manufacturer", "TestManufacturer")
    ds.ManufacturerModelName = overrides.pop("ManufacturerModelName", "TestModel")
    ds.ConvolutionKernel = overrides.pop("ConvolutionKernel", "STANDARD")
    ds.PixelSpacing = overrides.pop("PixelSpacing", [0.9765625, 0.9765625])

    # Apply any remaining overrides
    for key, value in overrides.items():
        setattr(ds, key, value)

    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    path = tmp_path / "ct_test.dcm"
    ds.save_as(str(path))
    return path


def _make_dicom_series(tmp_path, modality, series_uid, n_files):
    """Create *n_files* minimal DICOM files with the given modality and series UID."""
    paths = []
    sop_class = (
        "1.2.840.10008.5.1.4.1.1.128"
        if modality == "PT"
        else "1.2.840.10008.5.1.4.1.1.2"
    )
    for i in range(n_files):
        ds = Dataset()
        ds.Modality = modality
        ds.PatientID = "SERIES_TEST"
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = pydicom.uid.generate_uid()

        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = sop_class
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        path = tmp_path / f"{modality}_{series_uid}_{i:04d}.dcm"
        ds.save_as(str(path))
        paths.append(path)
    return paths


def test_find_series_ct_and_pt_found(tmp_path):
    """Both CT and PT series are found and returned."""
    _make_dicom_series(tmp_path, "CT", "1.2.3.100", 3)
    _make_dicom_series(tmp_path, "PT", "1.2.3.200", 5)
    result = find_series_by_modality(tmp_path, size_threshold_kb=0)
    assert len(result["CT"]) == 3
    assert len(result["PT"]) == 5


def test_find_series_picks_largest_series(tmp_path):
    """When multiple series of same modality exist, the largest is picked."""
    _make_dicom_series(tmp_path, "CT", "1.2.3.300", 2)
    _make_dicom_series(tmp_path, "CT", "1.2.3.400", 5)
    _make_dicom_series(tmp_path, "PT", "1.2.3.200", 3)
    result = find_series_by_modality(tmp_path, size_threshold_kb=0)
    assert len(result["CT"]) == 5


def test_find_series_no_ct_warns(tmp_path, caplog):
    """Warning logged when no CT series found."""
    _make_dicom_series(tmp_path, "PT", "1.2.3.200", 2)
    import logging

    with caplog.at_level(logging.WARNING):
        result = find_series_by_modality(tmp_path, size_threshold_kb=0)
    assert result["CT"] == []
    assert "No CT series" in caplog.text


def test_find_series_no_pt_warns(tmp_path, caplog):
    """Warning logged when no PT series found."""
    _make_dicom_series(tmp_path, "CT", "1.2.3.100", 2)
    import logging

    with caplog.at_level(logging.WARNING):
        result = find_series_by_modality(tmp_path, size_threshold_kb=0)
    assert result["PT"] == []
    assert "No PT series" in caplog.text


def test_find_series_empty_dir(tmp_path):
    """Empty directory returns empty lists for both modalities."""
    result = find_series_by_modality(tmp_path, size_threshold_kb=0)
    assert result == {"CT": [], "PT": []}


def test_find_series_corrupted_file_skipped(tmp_path):
    """Corrupted DICOM file is silently skipped."""
    _make_dicom_series(tmp_path, "CT", "1.2.3.100", 3)
    # Write a corrupt .dcm file
    (tmp_path / "corrupt.dcm").write_bytes(b"NOT A DICOM FILE AT ALL")
    result = find_series_by_modality(tmp_path, size_threshold_kb=0)
    assert len(result["CT"]) == 3  # corrupt file ignored


def test_extract_ct_tags_all_present(tmp_path):
    """All CT tags are extracted correctly."""
    path = _make_ct_dcm(tmp_path)
    tags = extract_ct_tags(path)
    assert tags["PatientID"] is not None
    assert isinstance(tags["SliceThickness"], float)
    assert isinstance(tags["KVP"], float)
    assert isinstance(tags["Rows"], int)
    assert isinstance(tags["Columns"], int)


def test_extract_ct_tags_missing_optional_returns_none(tmp_path):
    """Minimal CT DICOM returns None for optional tags."""
    ds = pydicom.Dataset()
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.Modality = "CT"
    ds.PatientID = "TEST"
    path = tmp_path / "minimal_ct.dcm"
    ds.save_as(str(path))
    tags = extract_ct_tags(path)
    assert tags["PatientID"] == "TEST"
    assert tags["SliceThickness"] is None
    assert tags["KVP"] is None


def test_extract_ct_tags_type_conversion(tmp_path):
    """SliceThickness and KVP are float; Rows and Columns are int."""
    path = _make_ct_dcm(
        tmp_path, SliceThickness="2.5", KVP="120", Rows=512, Columns=512
    )
    tags = extract_ct_tags(path)
    assert isinstance(tags["SliceThickness"], float)
    assert isinstance(tags["KVP"], float)
    assert isinstance(tags["Rows"], int)
    assert isinstance(tags["Columns"], int)


def test_dicom_to_nifti_empty_raises():
    """Empty file list raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        dicom_series_to_nifti([], "/tmp/out.nii")


def test_dicom_to_nifti_single_dir_uses_gdcm(tmp_path):
    """Single directory path uses GDCM sorting."""
    from unittest.mock import MagicMock, patch

    _make_dicom_series(tmp_path, "CT", "1.2.3", 3)
    dcm_files = sorted(tmp_path.glob("*.dcm"))
    output = tmp_path / "output" / "CT.nii"

    fake_image = MagicMock()
    with (
        patch("autods_pet.imaging.dicom.sitk.ImageSeriesReader") as MockReader,
        patch("autods_pet.imaging.dicom.sitk.WriteImage"),
    ):
        MockReader.GetGDCMSeriesFileNames.return_value = [str(f) for f in dcm_files]
        reader_instance = MockReader.return_value
        reader_instance.Execute.return_value = fake_image
        result = dicom_series_to_nifti(dcm_files, output)
    MockReader.GetGDCMSeriesFileNames.assert_called_once()
    assert result is fake_image


def test_dicom_to_nifti_multi_dir_fallback(tmp_path):
    """Files from multiple directories skip GDCM sorting."""
    from unittest.mock import MagicMock, patch

    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()
    _make_dicom_series(dir1, "CT", "1.2.3", 2)
    _make_dicom_series(dir2, "CT", "1.2.3", 2)
    dcm_files = sorted(dir1.glob("*.dcm")) + sorted(dir2.glob("*.dcm"))
    output = tmp_path / "output" / "CT.nii"

    fake_image = MagicMock()
    with (
        patch(
            "autods_pet.imaging.dicom.sitk.ImageSeriesReader.GetGDCMSeriesFileNames",
        ) as mock_gdcm,
        patch("autods_pet.imaging.dicom.sitk.ImageSeriesReader") as MockReader,
        patch("autods_pet.imaging.dicom.sitk.WriteImage"),
    ):
        reader_instance = MockReader.return_value
        reader_instance.Execute.return_value = fake_image
        dicom_series_to_nifti(dcm_files, output)
    mock_gdcm.assert_not_called()


def test_dicom_to_nifti_gdcm_empty_falls_back(tmp_path):
    """When GDCM returns empty list, falls back to provided file order."""
    from unittest.mock import MagicMock, patch

    _make_dicom_series(tmp_path, "CT", "1.2.3", 3)
    dcm_files = sorted(tmp_path.glob("*.dcm"))
    output = tmp_path / "output" / "CT.nii"

    fake_image = MagicMock()
    with (
        patch(
            "autods_pet.imaging.dicom.sitk.ImageSeriesReader.GetGDCMSeriesFileNames",
            return_value=[],
        ),
        patch("autods_pet.imaging.dicom.sitk.ImageSeriesReader") as MockReader,
        patch("autods_pet.imaging.dicom.sitk.WriteImage"),
    ):
        reader_instance = MockReader.return_value
        reader_instance.Execute.return_value = fake_image
        result = dicom_series_to_nifti(dcm_files, output)
    assert result is fake_image


def test_dicom_to_nifti_creates_output_dir(tmp_path):
    """Output directory is created if it doesn't exist."""
    from unittest.mock import MagicMock, patch

    _make_dicom_series(tmp_path, "CT", "1.2.3", 2)
    dcm_files = sorted(tmp_path.glob("*.dcm"))
    output = tmp_path / "deep" / "nested" / "output" / "CT.nii"

    fake_image = MagicMock()
    with (
        patch(
            "autods_pet.imaging.dicom.sitk.ImageSeriesReader.GetGDCMSeriesFileNames",
            return_value=[str(f) for f in dcm_files],
        ),
        patch("autods_pet.imaging.dicom.sitk.ImageSeriesReader") as MockReader,
        patch("autods_pet.imaging.dicom.sitk.WriteImage"),
    ):
        reader_instance = MockReader.return_value
        reader_instance.Execute.return_value = fake_image
        dicom_series_to_nifti(dcm_files, output)
    assert output.parent.exists()
