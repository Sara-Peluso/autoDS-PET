"""Tests for autods_pet.io - I/O utilities."""

import pytest

from autods_pet.io import read_patient_list, save_dataframe


def test_read_patient_list_valid(tmp_path):
    f = tmp_path / "patients.txt"
    f.write_text("PAT001\n# comment\nPAT002\n\nPAT003\n")
    result = read_patient_list(f)
    assert result == ["PAT001", "PAT002", "PAT003"]


def test_read_patient_list_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_patient_list(tmp_path / "nonexistent.txt")


def test_read_patient_list_empty(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("# only comments\n\n")
    assert read_patient_list(f) == []


def test_save_dataframe_csv(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"PatientID": ["P1", "P2"], "value": [1.0, 2.0]})
    out = save_dataframe(df, tmp_path / "results.csv", "csv")
    assert out.suffix == ".csv"
    assert out.exists()
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == ["PatientID", "value"]
    assert len(loaded) == 2


def test_save_dataframe_xlsx(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"PatientID": ["P1"], "value": [3.14]})
    out = save_dataframe(df, tmp_path / "results.csv", "xlsx")
    assert out.suffix == ".xlsx"
    assert out.exists()
    loaded = pd.read_excel(out, engine="openpyxl")
    assert list(loaded.columns) == ["PatientID", "value"]
    assert len(loaded) == 1


def test_save_dataframe_xlsx_changes_extension(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": [1]})
    out = save_dataframe(df, tmp_path / "data.csv", "xlsx")
    assert out.name == "data.xlsx"


def test_save_dataframe_creates_parent_dirs(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": [1]})
    out = save_dataframe(df, tmp_path / "sub" / "dir" / "out.csv", "csv")
    assert out.exists()


def test_read_patient_list_crlf_line_endings(tmp_path):
    """Windows-style CRLF line endings are handled correctly."""
    f = tmp_path / "patients.txt"
    f.write_bytes(b"PAT001\r\nPAT002\r\n")
    result = read_patient_list(f)
    assert result == ["PAT001", "PAT002"]


def test_read_patient_list_whitespace_only_lines(tmp_path):
    """Lines with only whitespace are skipped."""
    f = tmp_path / "patients.txt"
    f.write_text("PAT001\n   \n  \t  \nPAT002\n")
    result = read_patient_list(f)
    assert result == ["PAT001", "PAT002"]
