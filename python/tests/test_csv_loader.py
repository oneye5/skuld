"""Tests for the raw CSV loader."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skuld_research.data.csv_loader import RawData, load_raw_csv, load_raw_ohlc
from skuld_research.data.scrubber import ScrubReport


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


def test_scrub_report_present_when_disabled(synthetic_raw: RawData):
    """When no scrubbing is requested the report exists but is empty."""
    assert isinstance(synthetic_raw.scrub_report, ScrubReport)
    assert synthetic_raw.scrub_report.events.empty


def _write_corrupt_csv(tmp_path: Path) -> Path:
    """Create a tiny long-format CSV with a single round-trip anomaly."""
    dates = pd.date_range("2010-01-04", periods=5, freq="B")
    rows = []
    # Two tickers, BAD.NZ has a 5.05 print between two ~32 prints.
    bad_close = [32.10, 5.05, 32.61, 33.06, 33.57]
    good_close = [10.0, 10.1, 10.2, 10.15, 10.3]
    for d, c in zip(dates, bad_close, strict=True):
        rows.append({"timestamp": int(d.value // 10**6), "ticker": "BAD.NZ",
                     "feature": "adj_close", "value": c, "src": "test"})
        rows.append({"timestamp": int(d.value // 10**6), "ticker": "BAD.NZ",
                     "feature": "close", "value": c, "src": "test"})
        rows.append({"timestamp": int(d.value // 10**6), "ticker": "BAD.NZ",
                     "feature": "high", "value": c * 1.01, "src": "test"})
        rows.append({"timestamp": int(d.value // 10**6), "ticker": "BAD.NZ",
                     "feature": "low", "value": c * 0.99, "src": "test"})
        rows.append({"timestamp": int(d.value // 10**6), "ticker": "BAD.NZ",
                     "feature": "volume", "value": 1000, "src": "test"})
    for d, c in zip(dates, good_close, strict=True):
        rows.append({"timestamp": int(d.value // 10**6), "ticker": "GOOD.NZ",
                     "feature": "adj_close", "value": c, "src": "test"})
        rows.append({"timestamp": int(d.value // 10**6), "ticker": "GOOD.NZ",
                     "feature": "close", "value": c, "src": "test"})
        rows.append({"timestamp": int(d.value // 10**6), "ticker": "GOOD.NZ",
                     "feature": "volume", "value": 1000, "src": "test"})
    csv_path = tmp_path / "data_long.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_load_raw_csv_scrubs_round_trip_when_requested(tmp_path: Path):
    """When scrub is provided, anomalous adj_close prints are replaced."""
    from skuld_research.config.spec import ScrubbingSpec

    csv_path = _write_corrupt_csv(tmp_path)
    raw = load_raw_csv(
        csv_path,
        scrub=ScrubbingSpec(kind="round_trip", threshold=0.30, reversal_tolerance=0.10),
    )

    assert len(raw.scrub_report.events) == 1
    event = raw.scrub_report.events.iloc[0]
    assert event["ticker"] == "BAD.NZ"
    assert pytest.approx(event["original"], rel=1e-9) == 5.05
    cleaned = raw.prices.at[pd.Timestamp("2010-01-05"), "BAD.NZ"]
    assert pytest.approx(cleaned, rel=1e-3) == np.sqrt(32.10 * 32.61)
    # Untouched ticker is unchanged.
    assert raw.prices["GOOD.NZ"].iloc[0] == 10.0


def test_load_raw_csv_no_scrub_leaves_prices_intact(tmp_path: Path):
    """Without a scrub spec the loader returns raw values."""
    csv_path = _write_corrupt_csv(tmp_path)
    raw = load_raw_csv(csv_path)

    assert raw.prices.at[pd.Timestamp("2010-01-05"), "BAD.NZ"] == 5.05
    assert raw.scrub_report.events.empty


def test_load_raw_ohlc_scrubs_close_when_requested(tmp_path: Path):
    """OHLC loader scrubs the close series too when scrub is provided."""
    from skuld_research.config.spec import ScrubbingSpec

    csv_path = _write_corrupt_csv(tmp_path)
    high, low, close = load_raw_ohlc(
        csv_path,
        scrub=ScrubbingSpec(kind="round_trip", threshold=0.30, reversal_tolerance=0.10),
    )

    assert pytest.approx(close.at[pd.Timestamp("2010-01-05"), "BAD.NZ"], rel=1e-3) == (
        np.sqrt(32.10 * 32.61)
    )
    # high/low are NOT scrubbed (microstructure only used for spread estimation).
    assert pytest.approx(high.at[pd.Timestamp("2010-01-05"), "BAD.NZ"], rel=1e-9) == 5.05 * 1.01
    assert pytest.approx(low.at[pd.Timestamp("2010-01-05"), "BAD.NZ"], rel=1e-9) == 5.05 * 0.99


def _write_missed_split_csv(tmp_path: Path) -> Path:
    """CSV with a 2-for-1 missed-split shaped jump (no split row in actions)."""
    dates = pd.date_range("2010-01-04", periods=10, freq="B")
    # Pre-split prices around 100, post-split around 50 (clean 0.5 ratio at idx 5).
    closes = [100.0, 100.5, 99.8, 100.2, 100.0, 50.0, 50.1, 49.9, 50.2, 50.05]
    rows = []
    for d, c in zip(dates, closes, strict=True):
        ts = int(d.value // 10**6)
        for feat in ("adj_close", "close"):
            rows.append({"timestamp": ts, "ticker": "MIS.NZ",
                         "feature": feat, "value": c, "src": "test"})
        rows.append({"timestamp": ts, "ticker": "MIS.NZ",
                     "feature": "volume", "value": 1000, "src": "test"})
    csv_path = tmp_path / "data_long_missed_split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_load_raw_csv_with_adjustments_attaches_report(tmp_path: Path):
    """Audit-mode integration: report attached, missed_split detected."""
    from skuld_research.config.spec import AdjustmentSpec

    csv_path = _write_missed_split_csv(tmp_path)
    raw = load_raw_csv(csv_path, adjustments=AdjustmentSpec(kind="audit"))

    assert raw.adjustment_report is not None
    events = raw.adjustment_report.events
    assert (events["kind"] == "missed_split").any()
    # Audit must NOT mutate prices.
    assert raw.prices.at[pd.Timestamp("2010-01-04"), "MIS.NZ"] == 100.0


def test_load_raw_csv_no_adjustments_leaves_report_none(synthetic_raw: RawData):
    """Default path keeps adjustment_report as None."""
    assert synthetic_raw.adjustment_report is None
