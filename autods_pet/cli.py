"""Command-line interface for autods_pet.

Provides pipeline-stage commands for Deauville Score computation
from PET/CT images, with Rich progress panels and batch processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import pandas as pd

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from autods_pet import __version__, setup_logging

app = typer.Typer(
    name="autods-pet",
    help="Deauville Score computation from PET/CT images.",
    rich_markup_mode="rich",
    add_completion=False,
)

console = Console(stderr=True)


def _print_banner() -> None:
    """Print a styled banner with project name, version, and description."""
    console.print(
        Panel(
            f"[bold]autoDS-PET[/bold]  v{__version__}\n"
            "Deauville Score computation from PET/CT images",
            border_style="cyan",
        )
    )


log = logging.getLogger(__name__)

# Stage labels used for progress display
_STAGES = [
    ("convert", "Converting images"),
    ("normalize", "SUV normalization"),
    ("register", "PET-to-CT registration"),
    ("segment", "CT segmentation"),
    ("extract", "ROI extraction"),
    ("score", "Deauville scoring"),
]

# Common CLI options reused across commands
_opt_config = typer.Option(..., "--config", "-c", help="Path to INI config file.")
_opt_patients = typer.Option(
    None,
    "--patients",
    "-p",
    help="Patient(s): single ID, comma-separated IDs, or path to a .txt list.",
)
_opt_log_file = typer.Option(
    None, "--log-file", help="Write detailed logs to this file."
)
_opt_verbose = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging.")
_opt_force = typer.Option(False, "--force", help="Re-run even if outputs exist.")


def _apply_mask_flags(
    cfg: dict[str, Any],
    save_masks: bool = False,
    save_raw_masks: bool = False,
    save_refined_masks: bool = False,
    subtract_lesions: bool = False,
) -> None:
    """Apply mask-related CLI flags to the config dict."""
    if save_masks or save_raw_masks:
        cfg.setdefault("output", {})["save_raw_masks"] = True
    if save_masks or save_refined_masks:
        cfg.setdefault("output", {})["save_refined_masks"] = True
    if subtract_lesions:
        cfg.setdefault("output", {})["subtract_lesions_from_marrow"] = True


def _init_logging(verbose: bool, log_file: Path | None) -> None:
    """Set up Rich logging and optional file handler."""
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level, rich_console=console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(fh)


def _parse_patients(value: str) -> list[str]:
    """Parse a ``--patients`` value into a list of patient IDs.

    Accepts a path to a ``.txt`` file, comma-separated IDs, or a single ID.
    """
    from autods_pet.io import read_patient_list

    path = Path(value)
    if path.suffix == ".txt" and path.is_file():
        return read_patient_list(path)
    if "," in value:
        return [tok.strip() for tok in value.split(",") if tok.strip()]
    return [value.strip()]


def _load_and_resolve(
    config_path: Path, patients: str | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Load config and resolve the patient list.

    Returns
    -------
    tuple
        ``(cfg, patient_list)`` where *patient_list* is a list of patient
        ID strings.
    """
    from autods_pet.config import load_config
    from autods_pet.io import read_patient_list

    cfg = load_config(config_path)
    basepath = Path(cfg["paths"]["basepath"])

    if patients is not None:
        return cfg, _parse_patients(patients)

    patient_list_file = cfg["paths"].get("patient_list", "")
    if patient_list_file:
        pl_path = (
            Path(patient_list_file)
            if Path(patient_list_file).is_absolute()
            else basepath / patient_list_file
        )
        return cfg, read_patient_list(pl_path)

    # Auto-discover: every subdirectory of basepath is a patient
    # (excluding the output directory)
    if basepath.is_dir():
        from autods_pet.config import resolve_output_dir

        output_path = resolve_output_dir(cfg)
        patients_discovered = sorted(
            d.name
            for d in basepath.iterdir()
            if d.is_dir() and d.resolve() != output_path.resolve()
        )
        if patients_discovered:
            return cfg, patients_discovered

    console.print(
        "[red]No patients found.[/red] Provide --patients or set patient_list in config."
    )
    raise typer.Exit(1)


def _print_stage(stage_num: int, total: int, label: str, status: str) -> None:
    """Print a formatted stage status line."""
    if status == "ok":
        mark = "[green]\u2713[/green]"
    elif status == "skip":
        mark = "[yellow]\u2298[/yellow]"
    elif status == "error":
        mark = "[red]\u2717[/red]"
    else:
        mark = "[blue]\u2026[/blue]"
    console.print(
        f"  [{stage_num}/{total}] {label} {'.' * max(1, 35 - len(label))} {mark}"
    )


def _print_roi_status(name: str, status: str, detail: str = "") -> None:
    """Print an ROI extraction sub-status line."""
    if status == "ok":
        mark = "[green]\u2713[/green]"
    elif status == "skip":
        mark = f"[yellow]\u2298[/yellow] [dim]{detail}[/dim]"
    elif status == "warn":
        mark = f"[yellow]\u26a0[/yellow] [dim]{detail}[/dim]"
    else:
        mark = f"[red]\u2717[/red] [dim]{detail}[/dim]"
    console.print(f"    \u251c\u2500 {name} {'.' * max(1, 28 - len(name))} {mark}")


def _print_roi_status_last(name: str, status: str, detail: str = "") -> None:
    """Print the last ROI extraction sub-status line (with tree end marker)."""
    if status == "ok":
        mark = "[green]\u2713[/green]"
    elif status == "skip":
        mark = f"[yellow]\u2298[/yellow] [dim]{detail}[/dim]"
    elif status == "warn":
        mark = f"[yellow]\u26a0[/yellow] [dim]{detail}[/dim]"
    else:
        mark = f"[red]\u2717[/red] [dim]{detail}[/dim]"
    console.print(f"    \u2514\u2500 {name} {'.' * max(1, 28 - len(name))} {mark}")


def _print_patient_header(patient_id: str) -> None:
    """Print a header panel for a patient."""
    console.print()
    _print_banner()
    console.print(f"  Processing patient: [bold]{patient_id}[/bold]\n")


def _print_results_table(rois: dict[str, Any], scores: dict[str, int | float]) -> None:
    """Print a Rich table with per-ROI SUV stats and Deauville Scores."""
    from autods_pet.results import ROIResult

    table = Table(title="Results", border_style="cyan", show_lines=True)
    table.add_column("ROI", style="bold")
    table.add_column("Stat")
    table.add_column("SUV", justify="right")

    for roi_name, roi_data in rois.items():
        stats = (
            roi_data.stats
            if isinstance(roi_data, ROIResult)
            else roi_data.get("stats", {})
        )
        for stat_name, value in stats.items():
            table.add_row(
                roi_name, stat_name, f"{value:.2f}" if value is not None else "--"
            )

    for target_name, ds in scores.items():
        ds_str = str(ds) if ds > 0 else "--"
        table.add_row(f"DS ({target_name})", "", ds_str, style="bold magenta")

    console.print(table)


def _print_batch_summary(results: list[dict[str, Any]]) -> None:
    """Print a summary table for batch processing results."""
    table = Table(
        title="Batch Summary",
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Patient", style="bold")
    table.add_column("MBP (SUV)", justify="right")
    table.add_column("Liver (SUV)", justify="right")
    table.add_column("FL (SUV)", justify="right")
    table.add_column("DS (FL)", justify="right")
    table.add_column("Status")

    n_ok = 0
    n_fail = 0
    for row in results:
        if row.get("error"):
            n_fail += 1
            table.add_row(
                row["patient_id"],
                "--",
                "--",
                "--",
                "--",
                f"[red]\u2717 {row['error']}[/red]",
            )
        else:
            n_ok += 1
            mbp = row.get("mbp_median")
            liver = row.get("liver_median")
            fl = row.get("fl_value")
            ds = row.get("ds_fl")
            table.add_row(
                row["patient_id"],
                f"{mbp:.2f}" if mbp is not None else "--",
                f"{liver:.2f}" if liver is not None else "--",
                f"{fl:.2f}" if fl is not None else "--",
                str(ds) if ds is not None and ds > 0 else "--",
                "[green]\u2713 OK[/green]",
            )

    console.print()
    console.print(table)
    console.print(
        f"  {len(results)} patients processed | "
        f"[green]{n_ok} succeeded[/green] | "
        f"[red]{n_fail} failed[/red]"
    )


def _print_extract_statuses(roi_statuses: list[tuple[str, str, str]]) -> None:
    """Print ROI extraction status lines from pipeline results."""
    for i, (name, status, detail) in enumerate(roi_statuses):
        if i == len(roi_statuses) - 1:
            _print_roi_status_last(name, status, detail)
        else:
            _print_roi_status(name, status, detail)


def _run_stage(
    num: int,
    total: int,
    label: str,
    fn: Any,
) -> Any:
    """Run a single pipeline stage with Rich status output."""
    try:
        result = fn()
        _print_stage(num, total, label, "ok")
        return result
    except Exception as exc:
        _print_stage(num, total, label, "error")
        raise RuntimeError(f"{label} failed: {exc}") from exc


def _extract_roi_stats(data: Any) -> dict[str, float | None]:
    """Extract stats dict from ROIResult or plain dict."""
    from autods_pet.results import ROIResult

    if isinstance(data, ROIResult):
        return data.stats
    if isinstance(data, dict):
        return data.get("stats", {})
    return {}


def _run_pipeline_for_patient(
    cfg: dict[str, Any],
    patient_id: str,
    fast: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline for a single patient with Rich output."""
    from autods_pet.patient import PatientCase
    from autods_pet.pipeline import (
        convert_images,
        extract_rois,
        normalize_pet,
        register_pet,
        score_deauville,
        segment_ct,
        write_patient_deauville_csv,
        write_patient_suv_csv,
    )

    if force:
        cfg.setdefault("pipeline", {})["force"] = True

    patient = PatientCase(cfg, patient_id)
    total = 6

    _print_patient_header(patient_id)

    _run_stage(1, total, "Converting images", lambda: convert_images(cfg, patient))
    _run_stage(2, total, "SUV normalization", lambda: normalize_pet(cfg, patient))
    _run_stage(3, total, "PET-to-CT registration", lambda: register_pet(cfg, patient))
    seg_result = _run_stage(
        4, total, "CT segmentation", lambda: segment_ct(cfg, patient)
    )

    # Stage 5: Extract (custom display for per-ROI statuses)
    console.print(f"  [5/{total}] ROI extraction")
    try:
        extract_results = extract_rois(cfg, patient, seg_result)
        _print_extract_statuses(extract_results.get("_roi_statuses", []))
    except Exception as exc:
        _print_stage(5, total, "ROI extraction", "error")
        raise RuntimeError(f"ROI extraction failed: {exc}") from exc

    scores = _run_stage(
        6, total, "Deauville scoring", lambda: score_deauville(cfg, extract_results)
    )

    write_patient_suv_csv(extract_results, patient.suv_csv_path)
    write_patient_deauville_csv(scores, patient.deauville_csv_path)

    display_results = {
        k: v for k, v in extract_results.items() if not k.startswith("_")
    }
    _print_results_table(display_results, scores)

    return {
        "patient_id": patient_id,
        "scores": scores,
        "extract_results": {
            k: {"stats": _extract_roi_stats(v)}
            for k, v in extract_results.items()
            if not k.startswith("_")
        },
    }


def _build_ds_dataframe(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a DataFrame with Deauville Scores (one row per patient)."""
    import pandas as pd

    from autods_pet.pipeline import DS_COLUMN_ORDER

    rows = []
    for r in results:
        if r.get("error"):
            continue
        row: dict[str, Any] = {"patient_id": r["patient_id"]}
        for ds_key, ds_val in r.get("scores", {}).items():
            row[ds_key] = ds_val
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure fixed column order; add missing DS columns as NaN.
    cols = ["patient_id"] + list(DS_COLUMN_ORDER)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    # DS values are integers (1-5); use nullable Int64 to avoid "4.0".
    # BLR is a float ratio - leave as-is.
    for c in DS_COLUMN_ORDER:
        if c != "BLR" and c in df.columns:
            df[c] = df[c].astype("Int64")
    return df[cols]


def _build_suv_dataframe(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a DataFrame with SUV statistics (one row per patient)."""
    import pandas as pd

    rows = []
    for r in results:
        if r.get("error"):
            continue
        row: dict[str, Any] = {"patient_id": r["patient_id"]}
        for roi_name, roi_data in r.get("extract_results", {}).items():
            for stat_name, value in roi_data.get("stats", {}).items():
                col = f"{roi_name.replace(' ', '_')}_{stat_name}"
                row[col] = value
        rows.append(row)

    return pd.DataFrame(rows)


def _build_errors_dataframe(
    results: list[dict[str, Any]],
) -> pd.DataFrame | None:
    """Build a DataFrame with errors (only if any exist)."""
    import pandas as pd

    rows = [
        {"patient_id": r["patient_id"], "error": r["error"]}
        for r in results
        if r.get("error")
    ]
    if not rows:
        return None
    return pd.DataFrame(rows)


@app.command()
def convert(
    config: Path = _opt_config,
    patients: Optional[str] = _opt_patients,
    log_file: Optional[Path] = _opt_log_file,
    verbose: bool = _opt_verbose,
    force: bool = _opt_force,
) -> None:
    """Convert DICOM/NIfTI/NRRD images to the standard NIfTI layout."""
    from autods_pet.patient import PatientCase
    from autods_pet.pipeline import convert_images, generate_metadata_template

    _init_logging(verbose, log_file)
    cfg, patient_ids = _load_and_resolve(config, patients)
    if force:
        cfg.setdefault("pipeline", {})["force"] = True

    for pid in patient_ids:
        p = PatientCase(cfg, pid)
        _print_patient_header(pid)
        try:
            convert_images(cfg, p)
            _print_stage(1, 1, "Converting images", "ok")
        except Exception as exc:
            _print_stage(1, 1, "Converting images", "error")
            console.print(f"    [red]{exc}[/red]")

    # Generate a ready-to-fill CSV if any patient has incomplete metadata.
    csv_path = generate_metadata_template(cfg, patient_ids)
    if csv_path:
        console.print(
            f"\n[yellow]Some patients have incomplete metadata.\n"
            f"Fill in [bold]{csv_path}[/bold] then run normalize.[/yellow]"
        )


@app.command()
def normalize(
    config: Path = _opt_config,
    patients: Optional[str] = _opt_patients,
    log_file: Optional[Path] = _opt_log_file,
    verbose: bool = _opt_verbose,
    force: bool = _opt_force,
) -> None:
    """Compute SUV body-weight from raw PET images."""
    from autods_pet.patient import PatientCase
    from autods_pet.pipeline import generate_metadata_template, normalize_pet

    _init_logging(verbose, log_file)
    cfg, patient_ids = _load_and_resolve(config, patients)
    if force:
        cfg.setdefault("pipeline", {})["force"] = True

    had_errors = False
    for pid in patient_ids:
        p = PatientCase(cfg, pid)
        _print_patient_header(pid)
        try:
            normalize_pet(cfg, p)
            _print_stage(1, 1, "SUV normalization", "ok")
        except Exception as exc:
            had_errors = True
            _print_stage(1, 1, "SUV normalization", "error")
            console.print(f"    [red]{exc}[/red]")

    if had_errors:
        csv_path = generate_metadata_template(cfg, patient_ids)
        if csv_path:
            console.print(
                f"\n[yellow]Some patients have incomplete metadata.\n"
                f"Fill in [bold]{csv_path}[/bold] and re-run normalize.[/yellow]"
            )


@app.command()
def register(
    config: Path = _opt_config,
    patients: Optional[str] = _opt_patients,
    log_file: Optional[Path] = _opt_log_file,
    verbose: bool = _opt_verbose,
    force: bool = _opt_force,
) -> None:
    """Rigidly register PET SUV onto the CT grid."""
    from autods_pet.patient import PatientCase
    from autods_pet.pipeline import register_pet

    _init_logging(verbose, log_file)
    cfg, patient_ids = _load_and_resolve(config, patients)
    if force:
        cfg.setdefault("pipeline", {})["force"] = True

    for pid in patient_ids:
        p = PatientCase(cfg, pid)
        _print_patient_header(pid)
        try:
            register_pet(cfg, p)
            _print_stage(1, 1, "PET-to-CT registration", "ok")
        except Exception as exc:
            _print_stage(1, 1, "PET-to-CT registration", "error")
            console.print(f"    [red]{exc}[/red]")


@app.command()
def segment(
    config: Path = _opt_config,
    patients: Optional[str] = _opt_patients,
    log_file: Optional[Path] = _opt_log_file,
    verbose: bool = _opt_verbose,
    fast: bool = typer.Option(False, "--fast", help="Use TotalSegmentator fast mode."),
    force: bool = _opt_force,
) -> None:
    """Run TotalSegmentator on CT images."""
    from autods_pet.patient import PatientCase
    from autods_pet.pipeline import segment_ct

    _init_logging(verbose, log_file)
    cfg, patient_ids = _load_and_resolve(config, patients)
    if force:
        cfg.setdefault("pipeline", {})["force"] = True

    if fast:
        cfg.setdefault("totalsegmentator", {})["fast"] = True

    for pid in patient_ids:
        p = PatientCase(cfg, pid)
        _print_patient_header(pid)
        try:
            segment_ct(cfg, p)
            _print_stage(1, 1, "CT segmentation", "ok")
        except Exception as exc:
            _print_stage(1, 1, "CT segmentation", "error")
            console.print(f"    [red]{exc}[/red]")


_opt_explain_masks = typer.Option(
    False,
    "--explain-masks",
    help="Print a per-patient mask discovery report (every SEG/file found, "
    "every segment, which targets matched, which came up empty) before "
    "running the rest of the command.",
)


def _print_mask_discovery_report(cfg: dict[str, Any], patient_id: str) -> None:
    """Print a Rich-formatted discovery report for one patient."""
    from rich.markup import escape

    from autods_pet.patient import PatientCase
    from autods_pet.pipeline import _discover_patient_masks

    p = PatientCase(cfg, patient_id)
    discovered, warnings = _discover_patient_masks(cfg, p)

    pet_uid = p.pet_series_uid
    uid_str = (
        pet_uid
        if pet_uid
        else "[yellow]<unknown - PET_metadata.json missing or pre-UID>[/yellow]"
    )
    console.print(
        f"  [bold]Mask discovery for {patient_id}[/bold]"
        f"  (PET SeriesInstanceUID: {uid_str})"
    )

    if not discovered:
        console.print("    [yellow]No target masks resolved.[/yellow]")
    else:
        for tname, mask in discovered.items():
            if mask.format == "dicom_seg":
                detail = (
                    f"DICOM SEG segment '{mask.segment_label}' "
                    f"(#{mask.segment_number}) in {mask.path}"
                )
            else:
                detail = f"{mask.format.upper()} file {mask.path}"
            console.print(
                f"    [green]✓[/green] \\[{escape(tname)}] → {escape(detail)}"
            )

    if warnings:
        console.print("    [bold]Notes[/bold]:")
        for w in warnings:
            console.print(f"      • {escape(w)}")


@app.command()
def extract(
    config: Path = _opt_config,
    patients: Optional[str] = _opt_patients,
    log_file: Optional[Path] = _opt_log_file,
    verbose: bool = _opt_verbose,
    save_masks: bool = typer.Option(
        False, "--save-masks", help="Save both raw and refined masks as NIfTI."
    ),
    save_raw_masks: bool = typer.Option(
        False, "--save-raw-masks", help="Save individual raw label masks as NIfTI."
    ),
    save_refined_masks: bool = typer.Option(
        False, "--save-refined-masks", help="Save refined ROI masks as NIfTI."
    ),
    subtract_lesions: bool = typer.Option(
        False,
        "--subtract-lesions",
        help="Subtract target lesion masks from marrow ROIs (BM, LB).",
    ),
    explain_masks: bool = _opt_explain_masks,
    force: bool = _opt_force,
) -> None:
    """Extract ROI statistics from registered PET images."""
    from autods_pet.patient import PatientCase
    from autods_pet.pipeline import (
        _has_new_target_masks,
        extract_rois,
        write_patient_suv_csv,
    )

    _init_logging(verbose, log_file)
    cfg, patient_ids = _load_and_resolve(config, patients)
    _apply_mask_flags(
        cfg, save_masks, save_raw_masks, save_refined_masks, subtract_lesions
    )

    batch_results: list[dict[str, Any]] = []

    for pid in patient_ids:
        p = PatientCase(cfg, pid)
        _print_patient_header(pid)
        if explain_masks:
            _print_mask_discovery_report(cfg, pid)
        try:
            if not force and not _has_new_target_masks(cfg, p):
                log.info("No new target masks for %s, skipping.", pid)
                continue

            console.print("  [1/1] ROI extraction")
            results = extract_rois(cfg, p)
            _print_extract_statuses(results.get("_roi_statuses", []))
            write_patient_suv_csv(results, p.suv_csv_path)

            batch_results.append(
                {
                    "patient_id": pid,
                    "scores": {},
                    "extract_results": {
                        k: {"stats": _extract_roi_stats(v)}
                        for k, v in results.items()
                        if not k.startswith("_")
                    },
                }
            )
        except Exception as exc:
            console.print(f"    [red]{exc}[/red]")
            batch_results.append({"patient_id": pid, "error": str(exc)})

    _save_batch_results(cfg, batch_results, patient_ids, "csv")


@app.command()
def score(
    config: Path = _opt_config,
    patients: Optional[str] = _opt_patients,
    log_file: Optional[Path] = _opt_log_file,
    verbose: bool = _opt_verbose,
    subtract_lesions: bool = typer.Option(
        False,
        "--subtract-lesions",
        help="Subtract target lesion masks from marrow ROIs (BM, LB).",
    ),
    explain_masks: bool = _opt_explain_masks,
    force: bool = _opt_force,
) -> None:
    """Assign Deauville Scores from pre-extracted ROI statistics."""
    from autods_pet.patient import PatientCase
    from autods_pet.pipeline import (
        _has_new_target_masks,
        extract_rois,
        score_deauville,
        write_patient_deauville_csv,
        write_patient_suv_csv,
    )

    _init_logging(verbose, log_file)
    cfg, patient_ids = _load_and_resolve(config, patients)

    if subtract_lesions:
        cfg.setdefault("output", {})["subtract_lesions_from_marrow"] = True

    batch_results: list[dict[str, Any]] = []

    for pid in patient_ids:
        p = PatientCase(cfg, pid)
        _print_patient_header(pid)
        if explain_masks:
            _print_mask_discovery_report(cfg, pid)
        try:
            if not force and not _has_new_target_masks(cfg, p):
                log.info("No new target masks for %s, skipping.", pid)
                continue

            console.print("  [1/2] ROI extraction")
            extract_results = extract_rois(cfg, p)
            _print_extract_statuses(extract_results.get("_roi_statuses", []))
            scores = score_deauville(cfg, extract_results)
            _print_stage(2, 2, "Deauville scoring", "ok")

            write_patient_suv_csv(extract_results, p.suv_csv_path)
            write_patient_deauville_csv(scores, p.deauville_csv_path)

            display_results = {
                k: v for k, v in extract_results.items() if not k.startswith("_")
            }
            _print_results_table(display_results, scores)

            batch_results.append(
                {
                    "patient_id": pid,
                    "scores": scores,
                    "extract_results": {
                        k: {"stats": _extract_roi_stats(v)}
                        for k, v in extract_results.items()
                        if not k.startswith("_")
                    },
                }
            )
        except Exception as exc:
            console.print(f"    [red]{exc}[/red]")
            batch_results.append({"patient_id": pid, "error": str(exc)})

    _save_batch_results(cfg, batch_results, patient_ids, "csv")


@app.command()
def run(
    config: Path = _opt_config,
    patients: Optional[str] = _opt_patients,
    log_file: Optional[Path] = _opt_log_file,
    verbose: bool = _opt_verbose,
    fast: bool = typer.Option(False, "--fast", help="Use TotalSegmentator fast mode."),
    output_format: str = typer.Option(
        "csv", "--format", "-f", help="Output format: csv or xlsx."
    ),
    save_masks: bool = typer.Option(
        False, "--save-masks", help="Save both raw and refined masks as NIfTI."
    ),
    save_raw_masks: bool = typer.Option(
        False, "--save-raw-masks", help="Save individual raw label masks as NIfTI."
    ),
    save_refined_masks: bool = typer.Option(
        False, "--save-refined-masks", help="Save refined ROI masks as NIfTI."
    ),
    subtract_lesions: bool = typer.Option(
        False,
        "--subtract-lesions",
        help="Subtract target lesion masks from marrow ROIs (BM, LB).",
    ),
    explain_masks: bool = _opt_explain_masks,
    force: bool = _opt_force,
) -> None:
    """Run the full Deauville Score pipeline end-to-end."""
    _init_logging(verbose, log_file)
    cfg, patient_ids = _load_and_resolve(config, patients)

    if explain_masks:
        for pid in patient_ids:
            _print_mask_discovery_report(cfg, pid)

    if fast:
        cfg.setdefault("totalsegmentator", {})["fast"] = True
    if force:
        cfg.setdefault("pipeline", {})["force"] = True
    _apply_mask_flags(
        cfg, save_masks, save_raw_masks, save_refined_masks, subtract_lesions
    )

    batch_results: list[dict[str, Any]] = []

    if len(patient_ids) > 1:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing patients", total=len(patient_ids))
            for pid in patient_ids:
                progress.update(task, description=f"[bold blue]{pid}")
                try:
                    row = _run_pipeline_for_patient(cfg, pid, fast=fast)
                    batch_results.append(row)
                except Exception as exc:
                    log.error("Patient %s failed: %s", pid, exc)
                    batch_results.append({"patient_id": pid, "error": str(exc)})
                progress.advance(task)
    else:
        pid = patient_ids[0]
        try:
            row = _run_pipeline_for_patient(cfg, pid, fast=fast)
            batch_results.append(row)
        except Exception as exc:
            log.error("Patient %s failed: %s", pid, exc)
            batch_results.append({"patient_id": pid, "error": str(exc)})

    _save_batch_results(cfg, batch_results, patient_ids, output_format)


def _save_batch_results(
    cfg: dict[str, Any],
    batch_results: list[dict[str, Any]],
    patient_ids: list[str],
    output_format: str,
) -> None:
    """Save batch results (DS, SUV, errors) and print summary."""
    from autods_pet.config import resolve_output_dir
    from autods_pet.manifest import write_manifest
    from autods_pet.pipeline import DS_COLUMN_ORDER, save_batch_csv

    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_df = _build_ds_dataframe(batch_results)
    ds_path = output_dir / f"batch_results_DS.{output_format}"
    ds_int_cols = [c for c in DS_COLUMN_ORDER if c != "BLR"]
    save_batch_csv(ds_df, ds_path, output_format, int_columns=ds_int_cols)

    suv_df = _build_suv_dataframe(batch_results)
    suv_path = output_dir / f"batch_results_SUV.{output_format}"
    save_batch_csv(suv_df, suv_path, output_format)

    errors_df = _build_errors_dataframe(batch_results)
    if errors_df is not None:
        errors_path = output_dir / "batch_errors.csv"
        save_batch_csv(errors_df, errors_path, "csv")
        console.print(f"  Errors saved to: [bold]{errors_path}[/bold]")

    manifest_path = write_manifest(cfg, output_dir)

    if len(patient_ids) > 1:
        _print_batch_summary(batch_results)
    console.print(f"\n  Results saved to: [bold]{ds_path}[/bold]")
    console.print(f"  SUV stats saved to: [bold]{suv_path}[/bold]")
    if manifest_path:
        console.print(f"  Manifest saved to: [bold]{manifest_path}[/bold]")


@app.callback(invoke_without_command=True)
def version_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit."
    ),
) -> None:
    """Deauville Score computation from PET/CT images."""
    if version:
        _print_banner()
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


_output_option = typer.Option(
    "config.ini",
    "--output",
    "-o",
    help="Output path for the generated config file.",
)
_force_option = typer.Option(False, "--force", help="Overwrite an existing file.")
_profile_option = typer.Option(
    "standard",
    "--profile",
    "-p",
    help="Config profile: quick, standard, advanced, full, brain.",
)
_config_argument = typer.Argument(..., help="Path to the config file to validate.")


@app.command("create-config")
def create_config(
    output: Path = _output_option,
    force: bool = _force_option,
    profile: str = _profile_option,
) -> None:
    """Generate a config.ini from a named profile with default values and comments."""
    from autods_pet.config import PROFILE_NAMES, create_default_config

    if profile not in PROFILE_NAMES:
        console.print(
            f"[red]Unknown profile:[/red] {profile!r}\n"
            f"Choose from: {', '.join(PROFILE_NAMES)}",
        )
        raise typer.Exit(1)

    if output.exists() and not force:
        console.print(
            f"[red]File already exists:[/red] {output}\n"
            "Use [bold]--force[/bold] to overwrite.",
        )
        raise typer.Exit(1)
    create_default_config(output, profile=profile)
    console.print(
        Panel(
            f"Config written to [bold]{output}[/bold] (profile: [cyan]{profile}[/cyan])\n"
            "Edit it to match your data layout, then run:\n"
            f"  [cyan]autods-pet validate-config {output}[/cyan]",
            title="[green]Done[/green]",
            border_style="green",
        )
    )


@app.command("validate-config")
def validate_config(
    config: Path = _config_argument,
    patients: Optional[str] = typer.Option(
        None,
        "--patients",
        "-p",
        help="Comma-separated patient IDs.  When given, also runs a "
        "dry-run mask-discovery preview for each patient.",
    ),
) -> None:
    """Validate a config file and report all issues.

    With ``--patients``, also previews which manual lesion masks would
    be discovered for each named patient - useful before launching a
    full ``extract``/``score`` run.
    """
    from autods_pet.config import ConfigValidator, load_config

    if not config.exists():
        console.print(f"[red]File not found:[/red] {config}")
        raise typer.Exit(1)

    # Parse the INI (may raise on malformed syntax or unknown sections)
    try:
        cfg = load_config(config, validate=False)
    except Exception as exc:
        console.print(f"[red]Parse error:[/red] {exc}")
        raise typer.Exit(1) from None

    validator = ConfigValidator(cfg)
    validator.validate()

    if validator.is_valid:
        console.print(
            Panel(
                f"[bold]{config}[/bold] is valid.",
                title="[green]OK[/green]",
                border_style="green",
            )
        )
    else:
        console.print(
            f"\n[red bold]Found {len(validator.errors)} error(s)[/red bold] "
            f"in [bold]{config}[/bold]:\n"
        )
        for i, issue in enumerate(validator.errors, 1):
            loc = f"[{issue.section}]"
            if issue.key:
                loc += f" {issue.key}"
            console.print(f"  {i}. {loc}: {issue.message}")
        console.print()
        raise typer.Exit(1)

    if patients:
        patient_ids = [p.strip() for p in patients.split(",") if p.strip()]
        if patient_ids:
            console.print("\n[bold]Mask discovery preview[/bold]\n")
            for pid in patient_ids:
                _print_mask_discovery_report(cfg, pid)
                console.print()


@app.command("list-segments")
def list_segments_cmd(
    path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to a DICOM SEG (.dcm) file.",
        exists=True,
        readable=True,
    ),
) -> None:
    """List segments contained in a DICOM SEG file.

    Useful for finding the correct ``segment_label`` value to put in
    the configuration file when working with multi-segment DICOM SEG
    masks.
    """
    from autods_pet.ops.dicom_seg import list_segments

    segments = list_segments(path)
    if not segments:
        console.print("[yellow]No segments found.[/yellow]")
        raise typer.Exit(1)

    table = Table(title=f"Segments in {path.name}")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Label", style="bold")
    table.add_column("Description")
    for seg in segments:
        table.add_row(str(seg["number"]), seg["label"], seg["description"])
    console.print(table)


def main() -> None:
    """Entry point for the ``autods-pet`` CLI."""
    app()
