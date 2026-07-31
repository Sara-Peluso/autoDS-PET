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

| File | TotalSeg | License | Target masks | Masks saved | Use case |
|------|----------|---------|--------------|-------------|----------|
| `quick.ini` | Fast | No | None | None | Rapid screening, QC, testing |
| `standard.ini` | High-res | No | None | Refined only | General-purpose research (default) |
| `advanced.ini` | High-res | Yes | None | Refined only | Research with bone marrow DS |
| `full.ini` | High-res | Yes | FL, PM, EM | Raw + refined | Complete clinical analysis |
| `brain.ini` | High-res | No | None | Refined only | Brain-to-Liver Ratio studies |

"Target masks" lists which manual lesion mask sections the config enables, not which scores come out.

## Deauville Scores produced

| Profile | BM_DS | LB_DS | FL_DS | PM_DS | EM_DS | BLR |
|---------|-------|-------|-------|-------|-------|-----|
| `quick.ini` | No | **Yes** | No | No | No | **Yes** |
| `standard.ini` | No | **Yes** | No | No | No | **Yes** |
| `advanced.ini` | **Yes** | **Yes** | No | No | No | **Yes** |
| `full.ini` | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| `brain.ini` | No | **Yes** | No | No | No | **Yes** |

Two rules explain the whole table:

- `BM_DS` needs the lumbar vertebral body segmentation, which is a licensed TotalSegmentator task. Only the profiles with a license key produce it.
- `FL_DS`, `PM_DS` and `EM_DS` score user-provided lesion masks, so they only appear when the matching target section is enabled. Only `full.ini` enables them out of the box; adding a `[focal_lesion]`, `[paramedullary]` or `[extramedullary]` section to any other profile produces the corresponding score.

`LB_DS` and `BLR` come from the automatic reference ROIs, so every profile produces them, `brain.ini` included.

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

- **`brain.ini`**: Focused on Brain-to-Liver Ratio (BLR). The generated INI leaves out the `[aorta_mbp]`, `[long_bones]` and `[lumbar_vb]` sections to keep the file short, but those sections are refilled from the built-in defaults when the config is loaded, so the reference ROIs are still extracted at runtime and `LB_DS` is still produced. This is deliberate: the aorta mediastinal blood pool is the lower anchor of the Deauville ladder, so dropping it would suppress every score including BLR. The only score genuinely off is `BM_DS`, which needs the licensed vertebral body segmentation. No license required.
