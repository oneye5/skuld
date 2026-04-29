from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/raw_data_analysis.py", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=30,
    )


def test_raw_data_analysis_cli_writes_expected_artifacts(
    tmp_path: Path,
    raw_analysis_csv_path: Path,
) -> None:
    python_root = Path(__file__).parent.parent
    out_dir = tmp_path / "reports"
    result = _run_cli(
        "--data",
        str(raw_analysis_csv_path),
        "--out",
        str(out_dir),
        "--run-date",
        "2026-04-29",
        cwd=python_root,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "2026-04-29" / "report.md").exists()
    assert (out_dir / "2026-04-29" / "summary.json").exists()
    assert (out_dir / "2026-04-29" / "tables" / "source_inventory.csv").exists()
    assert "report.md" in result.stdout
    assert "summary.json" in result.stdout


def test_raw_data_analysis_cli_rejects_invalid_run_date(
    tmp_path: Path,
    raw_analysis_csv_path: Path,
) -> None:
    python_root = Path(__file__).parent.parent
    out_dir = tmp_path / "reports"

    result = _run_cli(
        "--data",
        str(raw_analysis_csv_path),
        "--out",
        str(out_dir),
        "--run-date",
        "..\\outside",
        cwd=python_root,
    )

    assert result.returncode != 0
    assert "run-date" in result.stderr
    assert not out_dir.exists()


def test_raw_data_analysis_cli_rejects_non_file_data_path(tmp_path: Path) -> None:
    python_root = Path(__file__).parent.parent
    data_dir = tmp_path / "not_a_file"
    data_dir.mkdir()
    out_dir = tmp_path / "reports"

    result = _run_cli(
        "--data",
        str(data_dir),
        "--out",
        str(out_dir),
        "--run-date",
        "2026-04-29",
        cwd=python_root,
    )

    assert result.returncode != 0
    assert result.stderr == f"ERROR: data path is not a file: {data_dir}\n"
    assert not out_dir.exists()


def test_raw_data_analysis_cli_reports_missing_data_path(tmp_path: Path) -> None:
    python_root = Path(__file__).parent.parent
    data_path = tmp_path / "missing.csv"
    out_dir = tmp_path / "reports"

    result = _run_cli(
        "--data",
        str(data_path),
        "--out",
        str(out_dir),
        "--run-date",
        "2026-04-29",
        cwd=python_root,
    )

    assert result.returncode != 0
    assert result.stderr == f"ERROR: data file not found: {data_path}\n"
    assert not out_dir.exists()


def test_raw_data_analysis_cli_reports_missing_source_legend(tmp_path: Path) -> None:
    python_root = Path(__file__).parent.parent
    out_dir = tmp_path / "reports"
    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n1706659200000,ANZ.NZ,adj_close,50.0,6\n",
        encoding="utf-8",
    )

    result = _run_cli(
        "--data",
        str(data_path),
        "--out",
        str(out_dir),
        "--run-date",
        "2026-04-29",
        cwd=python_root,
    )

    assert result.returncode != 0
    expected_error = f"ERROR: source legend file not found: {tmp_path / 'source_legend.csv'}\n"
    assert result.stderr == expected_error
    assert not out_dir.exists()


def test_raw_data_analysis_cli_rejects_non_file_source_legend(tmp_path: Path) -> None:
    python_root = Path(__file__).parent.parent
    out_dir = tmp_path / "reports"
    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n1706659200000,ANZ.NZ,adj_close,50.0,6\n",
        encoding="utf-8",
    )
    legend_dir = tmp_path / "source_legend.csv"
    legend_dir.mkdir()

    result = _run_cli(
        "--data",
        str(data_path),
        "--out",
        str(out_dir),
        "--run-date",
        "2026-04-29",
        cwd=python_root,
    )

    assert result.returncode != 0
    assert result.stderr == f"ERROR: source legend path is not a file: {legend_dir}\n"
    assert not out_dir.exists()
