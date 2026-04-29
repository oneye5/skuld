"""Tests for skuld_portfolio.inputs.sharesies_holdings."""
from pathlib import Path

import pandas as pd
import pytest

from skuld_portfolio.inputs.sharesies_holdings import (
    SharesiesSchemaError,
    parse_sharesies_csv,
)


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def valid_sharesies_csv(fixture_dir: Path) -> Path:
    return fixture_dir / "sharesies_export_2026-04-26.csv"


def test_parse_sharesies_csv_happy_path(valid_sharesies_csv: Path):
    """Parse valid Sharesies CSV with expected schema."""
    portfolio = parse_sharesies_csv(valid_sharesies_csv, cash_nzd=2500.0)
    
    assert portfolio.cash_nzd == 2500.0
    assert len(portfolio.holdings) == 7
    assert portfolio.holdings["AIR"] == 250
    assert portfolio.holdings["FBU"] == 180
    assert portfolio.prices["AIR"] == pytest.approx(2.58)
    assert portfolio.prices["MFT"] == pytest.approx(71.20)


def test_parse_sharesies_csv_missing_column(tmp_path: Path):
    """Raise SharesiesSchemaError if CSV is missing required columns."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "Instrument code,Market code,Name\n"
        "AIR,NZ,Air New Zealand\n"
    )
    
    with pytest.raises(SharesiesSchemaError, match="Missing columns"):
        parse_sharesies_csv(bad_csv)


def test_parse_sharesies_csv_extra_column(tmp_path: Path):
    """Raise SharesiesSchemaError if CSV has extra columns."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "Instrument code,Market code,Name,Shares held,Average cost per share NZD,"
        "Latest market price NZD,Total cost NZD,Market value NZD,Extra Column\n"
        "AIR,NZ,Air New Zealand,100,2.50,2.58,250.00,258.00,999\n"
    )
    
    with pytest.raises(SharesiesSchemaError, match="Extra columns"):
        parse_sharesies_csv(bad_csv)


def test_parse_sharesies_csv_empty_file(tmp_path: Path):
    """Raise SharesiesSchemaError if CSV has only header, no data rows."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text(
        "Instrument code,Market code,Name,Shares held,Average cost per share NZD,"
        "Latest market price NZD,Total cost NZD,Market value NZD\n"
    )
    
    with pytest.raises(SharesiesSchemaError, match="empty"):
        parse_sharesies_csv(empty_csv)


def test_parse_sharesies_csv_file_not_found():
    """Raise FileNotFoundError if path does not exist."""
    with pytest.raises(FileNotFoundError):
        parse_sharesies_csv(Path("/nonexistent/file.csv"))
