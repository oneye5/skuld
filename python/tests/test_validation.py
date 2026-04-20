"""Tests for the data validation layer."""

import pandas as pd

from skuld_research.data.validation import (
    ValidationReport,
    detect_negative_prices,
    detect_gaps,
    detect_stale_sources,
)


def test_detect_negative_prices_finds_them():
    """Negative prices are detected and reported."""
    prices = pd.DataFrame(
        {"ANZ.NZ": [50.0, -5.0, 51.0], "SPK.NZ": [4.8, 4.9, -1.0]},
        index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
    )
    report = detect_negative_prices(prices)
    assert report.issue_count == 2
    assert "ANZ.NZ" in report.details
    assert "SPK.NZ" in report.details


def test_detect_negative_prices_clean_data():
    """No negatives → zero issues."""
    prices = pd.DataFrame(
        {"ANZ.NZ": [50.0, 51.0]},
        index=pd.to_datetime(["2025-01-13", "2025-01-14"]),
    )
    report = detect_negative_prices(prices)
    assert report.issue_count == 0


def test_detect_gaps_finds_large_gap():
    """A gap of >5 trading days is detected."""
    # 10 trading days ending Jan 15, then resume Jan 27 — 7 bdays gap
    dates_before = pd.bdate_range("2025-01-02", periods=10)
    dates_after = pd.bdate_range("2025-01-27", periods=5)
    all_dates = dates_before.append(dates_after)
    prices = pd.DataFrame(
        {"ANZ.NZ": range(len(all_dates))},
        index=all_dates,
        dtype=float,
    )
    report = detect_gaps(prices, max_gap_days=5)
    assert report.issue_count >= 1
    assert "ANZ.NZ" in report.details


def test_detect_gaps_no_gap():
    """Consecutive trading days → no gap."""
    dates = pd.bdate_range("2025-01-02", periods=20)
    prices = pd.DataFrame({"ANZ.NZ": range(20)}, index=dates, dtype=float)
    report = detect_gaps(prices, max_gap_days=5)
    assert report.issue_count == 0


def test_detect_stale_sources():
    """Source whose latest timestamp is older than threshold is flagged."""
    now = pd.Timestamp("2025-03-01")
    source_latest = {"yf_prices": pd.Timestamp("2025-01-01"), "nz_gdp": pd.Timestamp("2025-02-25")}
    report = detect_stale_sources(source_latest, as_of=now, max_age_days=7)
    assert report.issue_count == 1
    assert "yf_prices" in report.details


def test_detect_stale_sources_all_fresh():
    """All sources within threshold → no issues."""
    now = pd.Timestamp("2025-03-01")
    source_latest = {"yf_prices": pd.Timestamp("2025-02-28"), "nz_gdp": pd.Timestamp("2025-02-25")}
    report = detect_stale_sources(source_latest, as_of=now, max_age_days=7)
    assert report.issue_count == 0
