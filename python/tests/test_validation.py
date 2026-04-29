"""Tests for the data validation layer."""

import pandas as pd

from skuld_common.validation import (
    ValidationReport,
    detect_duplicate_observations,
    detect_gaps,
    detect_invalid_corporate_actions,
    detect_nan_density,
    detect_negative_prices,
    detect_ohlc_inconsistencies,
    detect_stale_fundamentals,
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


def test_detect_duplicate_observations_finds_them():
    """Repeated (date, ticker, feature) rows are flagged."""
    long_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-13", "2025-01-13", "2025-01-13", "2025-01-14"]
            ),
            "ticker": ["ANZ.NZ", "ANZ.NZ", "ANZ.NZ", "ANZ.NZ"],
            "feature": ["adj_close", "adj_close", "volume", "adj_close"],
            "value": [50.0, 50.1, 1000.0, 51.0],
        }
    )
    report = detect_duplicate_observations(long_df)
    assert report.issue_count == 1  # one extra row beyond the first
    assert "ANZ.NZ" in report.details


def test_detect_duplicate_observations_clean():
    """Distinct (date, ticker, feature) tuples → no duplicates."""
    long_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-13", "2025-01-14"]),
            "ticker": ["ANZ.NZ", "ANZ.NZ"],
            "feature": ["adj_close", "adj_close"],
            "value": [50.0, 51.0],
        }
    )
    report = detect_duplicate_observations(long_df)
    assert report.issue_count == 0


def test_detect_invalid_corporate_actions_flags_zero_and_negative():
    """Zero or negative split/dividend factors are flagged."""
    ca = pd.DataFrame(
        {
            "ticker": ["ANZ.NZ", "ANZ.NZ", "SPK.NZ", "SPK.NZ"],
            "ex_date": pd.to_datetime(
                ["2025-01-15", "2025-02-15", "2025-03-15", "2025-04-15"]
            ),
            "type": ["dividend", "split", "split", "dividend"],
            "factor": [0.5, 0.0, -2.0, 0.25],
        }
    )
    report = detect_invalid_corporate_actions(ca)
    assert report.issue_count == 2
    assert "ANZ.NZ" in report.details
    assert "SPK.NZ" in report.details


def test_detect_invalid_corporate_actions_clean():
    ca = pd.DataFrame(
        {
            "ticker": ["ANZ.NZ"],
            "ex_date": pd.to_datetime(["2025-01-15"]),
            "type": ["dividend"],
            "factor": [0.5],
        }
    )
    assert detect_invalid_corporate_actions(ca).issue_count == 0


def test_detect_stale_fundamentals_flags_old_publications():
    """Tickers whose latest fundamental publication exceeds the age cap are flagged."""
    fund = pd.DataFrame(
        {"trailing_basic_average_shares": [1.0e9, 5.0e8, 2.0e9]},
        index=pd.MultiIndex.from_tuples(
            [
                ("ANZ.NZ", pd.Timestamp("2024-12-01")),
                ("OLD.NZ", pd.Timestamp("2022-01-01")),
                ("SPK.NZ", pd.Timestamp("2025-02-01")),
            ],
            names=["ticker", "publication_date"],
        ),
    )
    report = detect_stale_fundamentals(
        fund, as_of=pd.Timestamp("2025-03-01"), max_age_days=540
    )
    assert "OLD.NZ" in report.details
    assert "ANZ.NZ" not in report.details
    assert "SPK.NZ" not in report.details


def test_detect_stale_fundamentals_empty():
    empty = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"]),
    )
    report = detect_stale_fundamentals(empty, as_of=pd.Timestamp("2025-03-01"))
    assert report.issue_count == 0


# ---------------------------------------------------------------------------
# OHLC consistency
# ---------------------------------------------------------------------------

def test_detect_ohlc_inconsistencies_finds_violations():
    """High < Low is an obvious OHLC violation that should be caught."""
    dates = pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"])
    open_ = pd.DataFrame({"ANZ.NZ": [50.0, 50.0, 50.0]}, index=dates)
    high  = pd.DataFrame({"ANZ.NZ": [52.0, 48.0, 52.0]}, index=dates)  # row 1: high < low
    low   = pd.DataFrame({"ANZ.NZ": [49.0, 51.0, 49.0]}, index=dates)  # row 1: low > high
    close = pd.DataFrame({"ANZ.NZ": [51.0, 49.0, 51.0]}, index=dates)

    report = detect_ohlc_inconsistencies(open_, high, low, close)
    assert report.issue_count >= 1
    assert "ANZ.NZ" in report.details


def test_detect_ohlc_inconsistencies_clean():
    dates = pd.to_datetime(["2025-01-13", "2025-01-14"])
    open_ = pd.DataFrame({"ANZ.NZ": [50.0, 51.0]}, index=dates)
    high  = pd.DataFrame({"ANZ.NZ": [53.0, 54.0]}, index=dates)
    low   = pd.DataFrame({"ANZ.NZ": [49.0, 50.0]}, index=dates)
    close = pd.DataFrame({"ANZ.NZ": [51.5, 52.0]}, index=dates)

    assert detect_ohlc_inconsistencies(open_, high, low, close).issue_count == 0


# ---------------------------------------------------------------------------
# NaN density
# ---------------------------------------------------------------------------

def test_detect_nan_density_flags_sparse_tickers():
    """A ticker with >50% NaN is flagged; a dense ticker is not."""
    dates = pd.bdate_range("2024-01-02", periods=40)
    prices = pd.DataFrame(index=dates)
    prices["DENSE.NZ"] = 50.0
    prices["SPARSE.NZ"] = float("nan")
    prices.loc[dates[-5:], "SPARSE.NZ"] = 10.0  # only 5/40 rows non-null

    report = detect_nan_density(prices, max_nan_fraction=0.5)
    assert "SPARSE.NZ" in report.details
    assert "DENSE.NZ" not in report.details


def test_detect_nan_density_clean():
    dates = pd.bdate_range("2024-01-02", periods=40)
    prices = pd.DataFrame({"ANZ.NZ": 50.0}, index=dates)
    assert detect_nan_density(prices).issue_count == 0
