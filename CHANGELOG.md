# Changelog

All notable changes to autoDS-PET are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] - 2026-10-28

### Added

- `scoring_stat` key for target sections, selecting the single statistic a Deauville Score is computed from. Add one line to any target section to score on something else, for example `scoring_stat = p90` under `[focal_lesion]`. It defaults to `max` for `[focal_lesion]`, `[paramedullary]` and `[extramedullary]`, and to `p95` for `[targets.*]`. The chosen statistic is always computed, so `stats` now controls only what is additionally reported in `SUV_values.csv`.
- Support for DICOM inputs that previous releases silently ignored: uppercase `.DCM`, extensionless instances identified by their Part 10 header, and `RadiopharmaceuticalStartDateTime` (0018,1078), which is the only injection timestamp some Enhanced PET objects populate.
- `segment_number` on `read_dicom_seg`, to select a segment in a DICOM SEG whose labels are not unique.

### Changed

- **SUV normalization with `DecayCorrection = "START"` now decays the injected dose to the acquisition time.** `START` means the pixel values are decay corrected to the acquisition start, so the reference dose has to be decayed over the same interval; the raw injected dose is the reference for `ADMIN`.
- **Target scoring follows one documented rule instead of two implicit ones.** Named lesion sections previously used a fixed `max > p95 > median > mean` priority while custom `[targets.*]` sections used whichever statistic was listed first, so identical settings could yield different scores depending only on the section name. Custom targets whose first-listed statistic was not `p95` will be re-scored.

### Fixed

- **DICOM SEG masks with an oblique, gantry-tilted or rotated orientation were resampled against the wrong anatomy**, because the direction matrix was transposed. Axis-aligned axial SEGs, the common case, were never affected.
- **`extract` and `score` processed no patients at all on any config without target sections**, which is every profile except `full`. They now run the reference ROIs and produce `BM_DS`, `LB_DS` and `BLR` as documented.
- **`--force` on `extract` and `score` reused cached refined masks.** Changing a parameter such as `erosion_mm` and re-running with `--force` silently kept the stale masks.
- A lesion section configured without `max` produced no score and no warning. Any target that cannot be scored is now reported.
- A DICOM SEG containing two segments with the same label silently returned the last one. Duplicate labels now raise, naming the colliding segment numbers.
- Malformed or unusual DICOM metadata (leap seconds, missing injection tags, 2D masks) now fails with an actionable message instead of an opaque traceback mid-cohort.

### Documentation

- The `brain` profile does compute the reference ROIs and does produce `LB_DS`; only `BM_DS` is genuinely omitted. The previous description was self-contradictory, since BLR depends on the aorta reference it claimed was skipped.
- `create-config`, `validate-config` and `list-segments` do not accept `-c/--config`, and `validate-config` takes its config as a positional argument. The CLI reference now documents them separately from the pipeline commands.
- TotalSegmentator and SimpleITK-SimpleElastix are core dependencies installed with the package. Only the TotalSegmentator licence and model weights are a manual step.

## [0.1.0] - 2026-05-04

Initial release.
