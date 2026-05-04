# Supported Segmentation Formats

autoDS-PET accepts user-provided segmentation masks in three formats:
**NIfTI**, **NRRD**, and **DICOM SEG**.

## Format Comparison

| Feature | NIfTI (.nii/.nii.gz) | NRRD (.nrrd) | DICOM SEG (.dcm) |
|---|---|---|---|
| Coordinate system | RAS | varies | LPS |
| Multi-segment | no (one file per mask) | no | yes (multiple segments in one file) |
| Compression | gzip (.nii.gz) | gzip / raw | various transfer syntaxes |
| Metadata | minimal header | key-value pairs | full DICOM tags |
| Typical producers | FSL, FreeSurfer, ITK-SNAP | 3D Slicer, ClearCanvas | 3D Slicer, OHIF, dcmqi |

> **Note:** SimpleITK handles coordinate system conversions internally.
> Regardless of input format, masks are correctly aligned in patient space.

## NIfTI (.nii, .nii.gz)

The most common neuroimaging and research format. One file per segmentation
mask. Produced by most segmentation tools.

- **Extension:** `.nii` (uncompressed) or `.nii.gz` (gzip-compressed)
- **Use:** Save the mask anywhere under the patient input directory with
  a filename whose stem matches `mask_filename` in the config (e.g.
  `focal_lesion.nii.gz`). The lookup is **recursive**, so nested study
  or series sub-folders are fine.

## NRRD (.nrrd)

Nearly Raw Raster Data format, common in 3D Slicer workflows.

- **Extension:** `.nrrd`
- **Use:** Same as NIfTI -- save anywhere under the patient input directory
  with a stem matching `mask_filename`.

## DICOM SEG (.dcm)

DICOM Segmentation objects store one or more binary/fractional segments in a
single DICOM file, with full provenance metadata linking back to the source
images.

- **Extension:** `.dcm`
- **Requires:** `pip install autods-pet[dicom-seg]` (installs
  [highdicom](https://github.com/ImagingDataCommons/highdicom)).
  Without this extra, encountering a `.dcm` mask will raise a clear error
  with install instructions.
- **Discovery is UID-based, not filename-based.** autoDS-PET walks the
  patient input directory recursively, validates `.dcm` files, and accepts
  any whose `ReferencedSeriesSequence` contains the patient's PET
  `SeriesInstanceUID` (read from `PET_metadata.json`, written at convert
  time). Filename and folder location are irrelevant - drop the SEG export
  wherever your viewer puts it (e.g. inside the original study folder).
- **Targets are matched by `SegmentLabel`.** Each target config section
  declares one or more labels (case-insensitive); discovery scans the
  segments inside every PET-referencing SEG and assigns segments to
  targets. **A single multi-segment SEG file can supply multiple
  targets** in one shot.

### Configuration Example

```ini
[focal_lesion]
segment_label = Focal lesion, FL, GTV    ; any of these labels match
stats = max, p90

[paramedullary]
segment_label = PM, Paramedullary
stats = max, p90
```

A single DICOM SEG containing both a "Focal lesion" segment and a "PM"
segment will populate `[focal_lesion]` and `[paramedullary]` from the
same file, with no renaming or relocation.

You can still set `mask_filename` alongside `segment_label`; that key
covers the NIfTI/NRRD path. When both formats match the same target for
the same patient, **DICOM SEG wins** and a note is logged.

### Migrating from filename-based DICOM SEG matching

Earlier releases looked for `.dcm` files by stem (e.g. `focal_lesion.dcm`)
in two fixed locations. That no longer applies - set `segment_label`
instead, and stop renaming exports.

### Listing Segments

Use the CLI to inspect a DICOM SEG file:

```bash
autods-pet list-segments path/to/segmentation.dcm
```

This prints a table of segment numbers, labels, and descriptions.

### Tools That Produce DICOM SEG

| Tool | Notes |
|---|---|
| **3D Slicer** | Export via QuantitativeReporting extension or `File > Export` |
| **OHIF Viewer** | Built-in segmentation export to DICOM SEG |
| **dcmqi** | `itkimage2segimage` converts NIfTI/NRRD to DICOM SEG |
| **Kaapana** | Platform for medical image analysis, outputs DICOM SEG |
| **MONAILabel** | Server-side inference, can output DICOM SEG |

### Converting Between Formats

**NIfTI to DICOM SEG** (using [dcmqi](https://github.com/QIICR/dcmqi)):

```bash
itkimage2segimage \
  --inputImageList mask.nii.gz \
  --inputDICOMDirectory source_dicoms/ \
  --inputMetadata seg_meta.json \
  --outputDICOM output_seg.dcm
```

**DICOM SEG to NIfTI** (using dcmqi):

```bash
segimage2itkimage \
  --inputDICOM seg.dcm \
  --outputDirectory output/ \
  --outputType nii
```

### Fractional Segmentation Type

DICOM SEG supports both `BINARY` and `FRACTIONAL` segment types. autoDS-PET
handles both: fractional values are thresholded at 0.5 to produce a binary
mask, with a warning logged.
