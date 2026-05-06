"""Tests for EpsMomentumFactor and VolumeTrendFactor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_research.factors.eps_momentum import EpsMomentumFactor
from skuld_research.factors.volume_trend import VolumeTrendFactor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fundamentals(
    tickers: list[str],
    eps_by_ticker: dict[str, list[tuple[str, float]]],
) -> pd.DataFrame:
    """Build a fundamentals MultiIndex DataFrame with trailing_diluted_eps."""
    records = []
    for ticker in tickers:
        for pub_date_str, eps in eps_by_ticker.get(ticker, []):
            records.append({
                "ticker": ticker,
                "publication_date": pd.Timestamp(pub_date_str),
                "trailing_diluted_eps": eps,
            })
    if not records:
        return pd.DataFrame(
            columns=["trailing_diluted_eps"],
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"]),
        )
    df = pd.DataFrame(records)
    df = df.set_index(["ticker", "publication_date"])
    return df


def _make_volumes(
    tickers: list[str],
    n_days: int = 90,
    base_vol: float = 1000.0,
    end_date: str = "2024-01-31",
) -> pd.DataFrame:
    """Build a calendar-daily volume DataFrame."""
    idx = pd.date_range(end=end_date, periods=n_days, freq="D")
    data = {t: np.full(n_days, base_vol) for t in tickers}
    return pd.DataFrame(data, index=idx)


def _make_minimal_panel(fundamentals=None, volumes=None):
    """Build a minimal PreparedPanel with just the fields we need."""
    from skuld_common.contracts import PreparedPanel

    tickers = ["A.NZ", "B.NZ", "C.NZ"]
    dates = pd.date_range("2020-01-01", periods=24, freq="BME")

    returns_daily = pd.DataFrame(
        np.zeros((len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    returns_monthly = returns_daily.copy()
    market_cap = pd.DataFrame(
        np.ones((len(dates), len(tickers))) * 1e6,
        index=dates,
        columns=tickers,
    )
    sector = pd.Series(["Finance", "Finance", "Energy"], index=tickers, name="sector")
    universe_mask = pd.DataFrame(
        True,
        index=dates,
        columns=tickers,
    )
    macro = pd.DataFrame(index=dates)

    kwargs: dict = dict(
        returns_daily=returns_daily,
        returns_monthly=returns_monthly,
        market_cap=market_cap,
        sector=sector,
        universe_mask=universe_mask,
        macro=macro,
        asof=pd.Timestamp("2024-02-01"),
    )
    if fundamentals is not None:
        kwargs["fundamentals"] = fundamentals
    if volumes is not None:
        kwargs["volumes"] = volumes

    return PreparedPanel(**kwargs)


# ---------------------------------------------------------------------------
# EpsMomentumFactor tests
# ---------------------------------------------------------------------------

class TestEpsMomentumFactor:
    def test_basic_positive_growth(self):
        """Positive YoY EPS growth → positive score."""
        tickers = ["A.NZ", "B.NZ"]
        fundamentals = _make_fundamentals(tickers, {
            "A.NZ": [("2022-12-01", 1.0), ("2023-12-01", 2.0)],
            "B.NZ": [("2022-12-01", 2.0), ("2023-12-01", 1.0)],
        })
        panel = _make_minimal_panel(fundamentals=fundamentals)
        factor = EpsMomentumFactor()
        t = pd.Timestamp("2024-02-01")
        scores = factor.score(panel, t, tickers)
        assert scores["A.NZ"] > 0, "Positive EPS growth should yield positive score"
        assert scores["B.NZ"] < 0, "Negative EPS growth should yield negative score"
        assert scores["A.NZ"] > scores["B.NZ"], "A grew faster than B"

    def test_missing_ticker_returns_nan(self):
        """Ticker not in fundamentals → NaN."""
        tickers = ["A.NZ", "X.NZ"]
        fundamentals = _make_fundamentals(tickers, {
            "A.NZ": [("2022-12-01", 1.0), ("2023-12-01", 2.0)],
        })
        panel = _make_minimal_panel(fundamentals=fundamentals)
        factor = EpsMomentumFactor()
        scores = factor.score(panel, pd.Timestamp("2024-02-01"), tickers)
        assert pd.isna(scores["X.NZ"])

    def test_no_year_ago_data_returns_nan(self):
        """Only one EPS observation (no 12-month lag) → NaN."""
        tickers = ["A.NZ"]
        fundamentals = _make_fundamentals(tickers, {
            "A.NZ": [("2024-01-01", 1.5)],  # only one point, nothing a year ago
        })
        panel = _make_minimal_panel(fundamentals=fundamentals)
        factor = EpsMomentumFactor()
        scores = factor.score(panel, pd.Timestamp("2024-02-01"), tickers)
        assert pd.isna(scores["A.NZ"])

    def test_pit_safe_excludes_future_publications(self):
        """Publications on or after rebalance date must not be used."""
        tickers = ["A.NZ"]
        # Latest publication is AFTER t; only the 2022 obs should count as base
        fundamentals = _make_fundamentals(tickers, {
            "A.NZ": [
                ("2022-12-01", 1.0),
                ("2023-12-01", 2.0),
                ("2024-02-15", 99.0),  # future — must be ignored
            ],
        })
        panel = _make_minimal_panel(fundamentals=fundamentals)
        factor = EpsMomentumFactor()
        t = pd.Timestamp("2024-02-01")
        scores_excl = factor.score(panel, t, tickers)
        # With 2024-02-15 excluded: growth = (2.0 - 1.0) / 1.0 = 1.0
        assert abs(scores_excl["A.NZ"] - 1.0) < 1e-9

    def test_near_zero_base_returns_nan(self):
        """EPS base near zero → NaN (unstable ratio)."""
        tickers = ["A.NZ"]
        fundamentals = _make_fundamentals(tickers, {
            "A.NZ": [("2022-12-01", 1e-10), ("2023-12-01", 1.0)],
        })
        panel = _make_minimal_panel(fundamentals=fundamentals)
        factor = EpsMomentumFactor()
        scores = factor.score(panel, pd.Timestamp("2024-02-01"), tickers)
        assert pd.isna(scores["A.NZ"])

    def test_growth_capped_at_max(self):
        """Extreme growth ratio is clamped at _MAX_ABS_GROWTH."""
        from skuld_research.factors.eps_momentum import _MAX_ABS_GROWTH
        tickers = ["A.NZ"]
        fundamentals = _make_fundamentals(tickers, {
            "A.NZ": [("2022-12-01", 1.0), ("2023-12-01", 1000.0)],
        })
        panel = _make_minimal_panel(fundamentals=fundamentals)
        factor = EpsMomentumFactor()
        scores = factor.score(panel, pd.Timestamp("2024-02-01"), tickers)
        assert scores["A.NZ"] == pytest.approx(_MAX_ABS_GROWTH)

    def test_empty_fundamentals_all_nan(self):
        """Empty fundamentals → all NaN."""
        tickers = ["A.NZ", "B.NZ"]
        panel = _make_minimal_panel()
        factor = EpsMomentumFactor()
        scores = factor.score(panel, pd.Timestamp("2024-02-01"), tickers)
        assert scores.isna().all()

    def test_returns_series_with_correct_index(self):
        """Output index matches universe list exactly."""
        tickers = ["A.NZ", "B.NZ", "C.NZ"]
        fundamentals = _make_fundamentals(tickers, {
            "A.NZ": [("2022-12-01", 1.0), ("2023-12-01", 2.0)],
        })
        panel = _make_minimal_panel(fundamentals=fundamentals)
        factor = EpsMomentumFactor()
        scores = factor.score(panel, pd.Timestamp("2024-02-01"), tickers)
        assert list(scores.index) == tickers


# ---------------------------------------------------------------------------
# VolumeTrendFactor tests
# ---------------------------------------------------------------------------

class TestVolumeTrendFactor:
    def test_accelerating_volume_positive(self):
        """Recent volume above baseline → positive score."""
        tickers = ["A.NZ", "B.NZ"]
        # A: flat then spike; B: flat then decline
        n = 90
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        vol_a = np.concatenate([np.full(70, 1000.0), np.full(20, 3000.0)])
        vol_b = np.concatenate([np.full(70, 1000.0), np.full(20, 200.0)])
        volumes = pd.DataFrame({"A.NZ": vol_a, "B.NZ": vol_b}, index=idx)
        panel = _make_minimal_panel(volumes=volumes)
        factor = VolumeTrendFactor()
        t = pd.Timestamp("2023-04-01")
        scores = factor.score(panel, t, tickers)
        assert scores["A.NZ"] > 0
        assert scores["B.NZ"] < 0
        assert scores["A.NZ"] > scores["B.NZ"]

    def test_flat_volume_near_zero(self):
        """Perfectly flat volume → log-ratio ≈ 0."""
        tickers = ["A.NZ"]
        volumes = _make_volumes(tickers, n_days=90, base_vol=500.0)
        panel = _make_minimal_panel(volumes=volumes)
        factor = VolumeTrendFactor()
        t = pd.Timestamp("2024-02-01")
        scores = factor.score(panel, t, tickers)
        assert abs(scores["A.NZ"]) < 0.01

    def test_missing_ticker_returns_nan(self):
        """Ticker absent from volumes → NaN."""
        tickers = ["A.NZ", "X.NZ"]
        volumes = _make_volumes(["A.NZ"], n_days=90)
        panel = _make_minimal_panel(volumes=volumes)
        factor = VolumeTrendFactor()
        scores = factor.score(panel, pd.Timestamp("2024-02-01"), tickers)
        assert pd.isna(scores["X.NZ"])

    def test_insufficient_history_returns_nan(self):
        """Fewer trading days than min_trading_days → NaN."""
        tickers = ["A.NZ"]
        volumes = _make_volumes(tickers, n_days=10)  # only 10 days
        panel = _make_minimal_panel(volumes=volumes)
        factor = VolumeTrendFactor(min_trading_days=30)
        t = pd.Timestamp("2024-02-01")
        scores = factor.score(panel, t, tickers)
        assert pd.isna(scores["A.NZ"])

    def test_empty_volumes_all_nan(self):
        """Empty volumes panel → all NaN."""
        tickers = ["A.NZ", "B.NZ"]
        panel = _make_minimal_panel()
        factor = VolumeTrendFactor()
        scores = factor.score(panel, pd.Timestamp("2024-02-01"), tickers)
        assert scores.isna().all()

    def test_pit_safe(self):
        """Data on or after t must not influence the score."""
        tickers = ["A.NZ"]
        idx = pd.date_range("2023-01-01", periods=120, freq="D")
        # Volumes BEFORE t are flat 1000; on/after t are huge
        vol = np.where(idx >= pd.Timestamp("2023-04-01"), 1_000_000.0, 1000.0)
        volumes = pd.DataFrame({"A.NZ": vol}, index=idx)
        panel = _make_minimal_panel(volumes=volumes)
        factor = VolumeTrendFactor()
        t = pd.Timestamp("2023-04-01")
        scores = factor.score(panel, t, tickers)
        # Should be approximately 0 (flat history before t)
        assert abs(scores["A.NZ"]) < 0.5

    def test_invalid_constructor_raises(self):
        """short_days >= long_days must raise ValueError."""
        with pytest.raises(ValueError, match="short_days"):
            VolumeTrendFactor(short_days=60, long_days=20)

    def test_returns_series_with_correct_index(self):
        """Output index matches universe list exactly."""
        tickers = ["A.NZ", "B.NZ", "C.NZ"]
        volumes = _make_volumes(tickers, n_days=90)
        panel = _make_minimal_panel(volumes=volumes)
        factor = VolumeTrendFactor()
        scores = factor.score(panel, pd.Timestamp("2024-02-01"), tickers)
        assert list(scores.index) == tickers
