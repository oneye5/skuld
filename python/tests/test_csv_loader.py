"""Tests for the raw CSV loader."""

from skuld_research.data.csv_loader import RawData


def test_load_returns_raw_data(synthetic_raw: RawData):
    """load_raw_csv returns a RawData with prices, volumes, etc."""
    assert isinstance(synthetic_raw, RawData)


def test_prices_extracted(synthetic_raw: RawData):
    """Prices DataFrame has adj_close pivoted to ticker columns."""
    assert "ANZ.NZ" in synthetic_raw.prices.columns
    assert "SPK.NZ" in synthetic_raw.prices.columns
    assert synthetic_raw.prices.index.name == "date"


def test_volumes_extracted(synthetic_raw: RawData):
    """Volumes DataFrame has volume pivoted to ticker columns."""
    assert "ANZ.NZ" in synthetic_raw.volumes.columns
    assert synthetic_raw.volumes["ANZ.NZ"].iloc[0] > 0


def test_fundamentals_extracted(synthetic_raw: RawData):
    """Fundamentals have a MultiIndex of (ticker, publication_date)."""
    assert synthetic_raw.fundamentals.index.names == ["ticker", "publication_date"]
    assert "annual_net_income_common_stockholders" in synthetic_raw.fundamentals.columns


def test_macro_extracted(synthetic_raw: RawData):
    """Macro data has date index and feature columns."""
    assert "oecd_bcicp" in synthetic_raw.macro.columns


def test_corporate_actions_extracted(synthetic_raw: RawData):
    """Corporate actions include dividends and splits."""
    assert len(synthetic_raw.corporate_actions) == 2  # 1 dividend + 1 split
    types = set(synthetic_raw.corporate_actions["type"])
    assert types == {"dividend", "split"}
