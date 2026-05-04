"""I/O utilities for patient lists and data export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_patient_list(path: Path) -> list[str]:
    """Read a text file of patient IDs (one per line, ``#`` comments allowed).

    Parameters
    ----------
    path : Path
        Path to the patient list text file.

    Returns
    -------
    list[str]
        Patient IDs with whitespace stripped; blank lines and
        ``#``-comments are excluded.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Patient list file not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    output_format: str = "csv",
) -> Path:
    """Save a DataFrame as CSV or Excel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str or Path
        Output file path.  If *output_format* is ``"xlsx"``, the extension
        is changed to ``.xlsx`` automatically.
    output_format : str
        ``"csv"`` (default) or ``"xlsx"``.

    Returns
    -------
    Path
        The actual path written (may differ from *path* if extension changed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "xlsx":
        path = path.with_suffix(".xlsx")
        df.to_excel(path, index=False, engine="openpyxl")
    else:
        df.to_csv(path, index=False)
    return path
