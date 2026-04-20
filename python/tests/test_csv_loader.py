"""Tests for the raw CSV loader."""

from pathlib import Path

import pandas as pd

from skuld_research.data.csv_loader import RawData, load_raw_csv


def test_load_returns_raw_data(synthetic_csv_path: Path):
    """load_raw_csv returns a RawData with prices, volumes, etc."""
    raw = load_raw_csv(synthetic_csv_path)
    assert isinstance(raw, RawData)


def test_prices_extracted(synthetic_csv_path: Path):
    """Prices DataFrame has adj_close pivoted to ticker columns."""
    raw = load_raw_csv(synthetic_csv_path)
    assert "ANZ.NZ" in raw.prices.columns
    assert "SPK.NZ" in raw.prices.columns
    assert raw.prices.index.name == "date"


def test_volumes_extracted(synthetic_csv_path: Path):
    """Volumes DataFrame has volume pivoted to ticker columns."""
    raw = load_raw_csv(synthetic_csv_path)
    assert "ANZ.NZ" in raw.volumes.columns
    assert raw.volumes["ANZ.NZ"].iloc[0] > 0


def test_fundamentals_extracted(synthetic_csv_path: Path):
    """Fundamentals have a MultiIndex of (ticker, publication_date)."""
    raw = load_raw_csv(synthetic_csv_path)
    assert raw.fundamentals.index.names == ["ticker", "publication_date"]
    assert "annual_net_income_common_stockholders" in raw.fundamentals.columns


def test_macro_extracted(synthetic_csv_path: Path):
    """Macro data has date index and feature columns."""
    raw = load_raw_csv(synthetic_csv_path)
    assert "oecd_bcicp" in raw.macro.columns


def test_corporate_actions_extracted(synthetic_csv_path: Path):
    """Corporate actions include dividends and splits."""
    raw = load_raw_csv(synthetic_csv_path)
    assert len(raw.corporate_actions) == 2  # 1 dividend + 1 split
    types = set(raw.corporate_actions["type"])
    assert types == {"dividend", "split"}
