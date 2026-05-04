"""Tests for autods_pet.ops.dicom_seg -- DICOM SEG reading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

hd = pytest.importorskip("highdicom")
pydicom = pytest.importorskip("pydicom")

from pydicom.sr.codedict import codes  # noqa: E402
from pydicom.uid import generate_uid  # noqa: E402

from autods_pet.ops.dicom_seg import (  # noqa: E402
    is_dicom_seg,
    list_segments,
    read_dicom_seg,
    read_referenced_series_uids,
)


def _make_source_images(
    n_slices: int = 4,
    rows: int = 4,
    cols: int = 4,
    spacing: tuple[float, float] = (1.0, 1.0),
    slice_thickness: float = 2.0,
    orientation: list[float] | None = None,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[pydicom.Dataset]:
    """Build minimal DICOM source-image datasets for constructing a SEG."""
    if orientation is None:
        orientation = [1, 0, 0, 0, 1, 0]

    series_uid = generate_uid()
    study_uid = generate_uid()
    for_uid = generate_uid()

    sources: list[pydicom.Dataset] = []
    for i in range(n_slices):
        ds = pydicom.Dataset()
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.Modality = "CT"
        ds.ImageOrientationPatient = orientation
        ds.PixelSpacing = list(spacing)
        ds.SliceThickness = slice_thickness
        ds.PatientID = "TEST"
        ds.PatientName = "Test^Patient"
        ds.PatientBirthDate = "19900101"
        ds.PatientSex = "O"
        ds.StudyDate = "20240101"
        ds.StudyTime = "120000"
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.StudyID = "1"
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.FrameOfReferenceUID = for_uid
        ds.SOPInstanceUID = generate_uid()
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [
            origin[0],
            origin[1],
            origin[2] + i * slice_thickness,
        ]
        ds.PixelData = np.zeros((rows, cols), dtype=np.uint16).tobytes()
        ds.NumberOfFrames = 1
        sources.append(ds)
    return sources


def _make_seg_description(number: int, label: str) -> hd.seg.SegmentDescription:
    return hd.seg.SegmentDescription(
        segment_number=number,
        segment_label=label,
        segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
        segmented_property_type=codes.SCT.Neoplasm,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
    )


def _save_seg(seg: hd.seg.Segmentation, tmp_path: Path, name: str = "seg.dcm") -> Path:
    out = tmp_path / name
    seg.save_as(str(out))
    return out


@pytest.fixture()
def single_seg_path(tmp_path: Path) -> Path:
    """DICOM SEG file with one segment ('Tumor'), 4x4x4, cube at [1:3]."""
    sources = _make_source_images()
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[1:3, 1:3, 1:3] = 1
    seg = hd.seg.Segmentation(
        source_images=sources,
        pixel_array=arr,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[_make_seg_description(1, "Tumor")],
        series_instance_uid=generate_uid(),
        series_number=1,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="Test",
        manufacturer_model_name="Test",
        software_versions="1.0",
        device_serial_number="0",
        omit_empty_frames=False,
    )
    return _save_seg(seg, tmp_path)


@pytest.fixture()
def multi_seg_path(tmp_path: Path) -> Path:
    """DICOM SEG file with two segments: 'Tumor' and 'Necrosis'."""
    sources = _make_source_images()
    # Two-segment array: shape (4, 4, 4, 2)
    arr = np.zeros((4, 4, 4, 2), dtype=np.uint8)
    arr[1:3, 1:3, 1:3, 0] = 1  # Tumor
    arr[0, 0, 0, 1] = 1  # Necrosis - single voxel
    seg = hd.seg.Segmentation(
        source_images=sources,
        pixel_array=arr,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[
            _make_seg_description(1, "Tumor"),
            _make_seg_description(2, "Necrosis"),
        ],
        series_instance_uid=generate_uid(),
        series_number=1,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="Test",
        manufacturer_model_name="Test",
        software_versions="1.0",
        device_serial_number="0",
        omit_empty_frames=False,
    )
    return _save_seg(seg, tmp_path)


def test_is_dicom_seg_true(single_seg_path: Path) -> None:
    assert is_dicom_seg(single_seg_path) is True


def test_is_dicom_seg_false_for_ct(tmp_path: Path) -> None:
    ds = pydicom.Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT
    out = tmp_path / "ct.dcm"
    ds.save_as(str(out), little_endian=True, implicit_vr=True)
    assert is_dicom_seg(out) is False


def test_is_dicom_seg_non_dicom(tmp_path: Path) -> None:
    junk = tmp_path / "junk.dcm"
    junk.write_text("not a dicom file")
    assert is_dicom_seg(junk) is False


def test_read_referenced_series_uids_single_seg(single_seg_path: Path) -> None:
    """A SEG built from a single source series references that series's UID."""
    refs = read_referenced_series_uids(single_seg_path)
    assert len(refs) >= 1
    # Cross-check against the actual ReferencedSeriesSequence from highdicom.
    seg = hd.seg.segread(str(single_seg_path))
    expected = {str(item.SeriesInstanceUID) for item in seg.ReferencedSeriesSequence}
    assert set(refs) == expected


def test_read_referenced_series_uids_missing_sequence(tmp_path: Path) -> None:
    """Returns [] when the file has no ReferencedSeriesSequence."""
    ds = pydicom.Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
    out = tmp_path / "no_refs.dcm"
    ds.save_as(str(out), little_endian=True, implicit_vr=True)
    assert read_referenced_series_uids(out) == []


def test_read_referenced_series_uids_unparseable(tmp_path: Path) -> None:
    """Returns [] when the file cannot be parsed at all."""
    junk = tmp_path / "junk.dcm"
    junk.write_text("not a dicom file")
    assert read_referenced_series_uids(junk) == []


def test_list_segments_single(single_seg_path: Path) -> None:
    segs = list_segments(single_seg_path)
    assert len(segs) == 1
    assert segs[0]["number"] == 1
    assert segs[0]["label"] == "Tumor"


def test_list_segments_multi(multi_seg_path: Path) -> None:
    segs = list_segments(multi_seg_path)
    assert len(segs) == 2
    labels = {s["label"] for s in segs}
    assert labels == {"Tumor", "Necrosis"}


def test_read_single_segment_auto_select(single_seg_path: Path) -> None:
    """Single-segment file: auto-selects without segment_label."""
    img = read_dicom_seg(single_seg_path)
    assert isinstance(img, sitk.Image)
    assert img.GetSize() == (4, 4, 4)
    assert img.GetPixelIDTypeAsString() == "8-bit unsigned integer"

    arr = sitk.GetArrayFromImage(img)
    assert arr.sum() == 8  # 2x2x2 cube


def test_read_single_segment_explicit_label(single_seg_path: Path) -> None:
    img = read_dicom_seg(single_seg_path, segment_label="Tumor")
    assert sitk.GetArrayFromImage(img).sum() == 8


def test_read_single_segment_case_insensitive(single_seg_path: Path) -> None:
    img = read_dicom_seg(single_seg_path, segment_label="tumor")
    assert sitk.GetArrayFromImage(img).sum() == 8


def test_read_single_segment_spacing(single_seg_path: Path) -> None:
    img = read_dicom_seg(single_seg_path)
    sx, sy, sz = img.GetSpacing()
    assert sx == pytest.approx(1.0)
    assert sy == pytest.approx(1.0)
    assert sz == pytest.approx(2.0)


def test_read_multi_segment_with_label(multi_seg_path: Path) -> None:
    img = read_dicom_seg(multi_seg_path, segment_label="Tumor")
    assert sitk.GetArrayFromImage(img).sum() == 8


def test_read_multi_segment_other_label(multi_seg_path: Path) -> None:
    img = read_dicom_seg(multi_seg_path, segment_label="Necrosis")
    assert sitk.GetArrayFromImage(img).sum() == 1


def test_read_multi_segment_no_label_raises(multi_seg_path: Path) -> None:
    with pytest.raises(ValueError, match="contains 2 segments"):
        read_dicom_seg(multi_seg_path)


def test_read_wrong_label_raises(single_seg_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        read_dicom_seg(single_seg_path, segment_label="Nonexistent")


def test_missing_highdicom_raises() -> None:
    with patch.dict("sys.modules", {"highdicom": None}):
        from autods_pet.ops import dicom_seg

        with pytest.raises(ModuleNotFoundError, match="highdicom"):
            dicom_seg._import_highdicom()


@pytest.fixture()
def fractional_seg_path(tmp_path: Path) -> Path:
    """DICOM SEG file with FRACTIONAL segmentation type."""
    sources = _make_source_images()
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    arr[1:3, 1:3, 1:3] = 0.8  # above threshold
    arr[0, 0, 0] = 0.3  # below threshold
    seg = hd.seg.Segmentation(
        source_images=sources,
        pixel_array=arr,
        segmentation_type=hd.seg.SegmentationTypeValues.FRACTIONAL,
        segment_descriptions=[_make_seg_description(1, "Probability")],
        series_instance_uid=generate_uid(),
        series_number=1,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="Test",
        manufacturer_model_name="Test",
        software_versions="1.0",
        device_serial_number="0",
        max_fractional_value=1,
    )
    return _save_seg(seg, tmp_path, "fractional.dcm")


def test_read_fractional_seg_thresholds_at_half(fractional_seg_path: Path) -> None:
    """FRACTIONAL type: values >= 0.5 become 1, below become 0."""
    img = read_dicom_seg(fractional_seg_path)
    assert isinstance(img, sitk.Image)
    arr = sitk.GetArrayFromImage(img)
    assert arr.dtype == np.uint8
    # 2x2x2 cube above threshold = 8 voxels set to 1
    assert arr.sum() == 8


def test_read_dicom_seg_preserves_custom_spacing(tmp_path: Path) -> None:
    """Non-default spacing is preserved in the output SimpleITK image."""
    sources = _make_source_images(spacing=(0.5, 0.75), slice_thickness=3.0)
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[0, 0, 0] = 1
    seg = hd.seg.Segmentation(
        source_images=sources,
        pixel_array=arr,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[_make_seg_description(1, "ROI")],
        series_instance_uid=generate_uid(),
        series_number=1,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="Test",
        manufacturer_model_name="Test",
        software_versions="1.0",
        device_serial_number="0",
        omit_empty_frames=False,
    )
    path = _save_seg(seg, tmp_path, "custom_spacing.dcm")
    img = read_dicom_seg(path)
    sx, sy, sz = img.GetSpacing()
    # DICOM pixel spacing is (row, col); SimpleITK x=col, y=row
    assert sx == pytest.approx(0.75, abs=0.01)
    assert sy == pytest.approx(0.5, abs=0.01)
    assert sz == pytest.approx(3.0, abs=0.01)
