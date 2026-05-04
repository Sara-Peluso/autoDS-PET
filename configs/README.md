# Configuration Profiles

Five pre-built profiles for common use cases. Generate one with `create-config`:

```bash
# Generate a config from a profile (defaults to "standard")
autods-pet create-config -p quick -o my_config.ini

# Edit paths, then validate and run:
autods-pet validate-config my_config.ini
autods-pet run -c my_config.ini
```

## Profiles

| File | TotalSeg | License | Targets | Masks saved | Use case |
|------|----------|---------|---------|-------------|----------|
| `quick.ini` | Fast | No | None | None | Rapid screening, QC, testing |
| `standard.ini` | High-res | No | None | Refined only | General-purpose research (default) |
| `advanced.ini` | High-res | Yes | None | Refined only | Research with bone marrow DS |
| `full.ini` | High-res | Yes | FL, PM, EM | Raw + refined | Complete clinical analysis |
| `brain.ini` | High-res | No | None | Refined only | Brain-to-Liver Ratio studies |

## Deauville Scores produced

| Score | quick | standard | advanced | full | brain |
|-------|-------|----------|----------|------|-------|
| BM_DS (bone marrow) | Skipped | Skipped | Yes | Yes | Skipped |
| LB_DS (long bones) | Yes | Yes | Yes | Yes | Skipped |
| FL_DS (focal lesion) | Skipped | Skipped | Skipped | Yes | Skipped |
| PM_DS (paramedullary) | Skipped | Skipped | Skipped | Yes | Skipped |
| EM_DS (extramedullary) | Skipped | Skipped | Skipped | Yes | Skipped |
| BLR (brain-to-liver) | Yes | Yes | Yes | Yes | Yes |

## Key differences

- **`quick.ini`**: Speed-optimized. Uses TotalSegmentator fast mode
  (~10x faster, lower resolution). No masks saved. Useful for initial cohort
  screening or pipeline testing before committing to full analysis.

- **`standard.ini`**: Balanced (default). High-resolution segmentation for
  accurate ROI extraction but no target masks or license required. Produces
  LB_DS and reference SUV values (Aorta MBP, Liver).

- **`advanced.ini`**: Adds bone marrow scoring. Requires a TotalSegmentator
  license key for vertebral body segmentation (BM_DS). Otherwise identical
  to standard.

- **`full.ini`**: Complete. Requires a TotalSegmentator license key. All
  three target masks enabled (focal lesion, paramedullary, extramedullary).
  Both raw and refined masks saved for clinical review and quality control.
  Liver hole-filling enabled (max 500 mm3).

- **`brain.ini`**: Focused on Brain-to-Liver Ratio (BLR). Only brain and
  liver ROIs are computed - other anatomical ROIs (lumbar vertebrae, aorta,
  long bones) are skipped. No license required.
