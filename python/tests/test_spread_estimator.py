"""Tests for Abdi-Ranaldo OHLC spread estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_research.costs.spread_estimator import (
    DEFAULT_MIN_BPS_PER_SIDE,
    compute_abdi_ranaldo_spread_panel,
)


def _ohlc_with_known_spread(
    n_days: int,
    n_tickers: int,
    *,
    spread_frac: float,
    vol_frac: float,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic OHLC where mid follows a random walk and H/L/C
    are perturbed by a known effective half-spread.

    The Abdi-Ranaldo estimator should recover an effective spread close to
    `spread_frac` (within sampling noise) from this data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    tickers = [f"T{i}.NZ" for i in range(n_tickers)]

    log_returns = rng.normal(scale=vol_frac, size=(n_days, n_tickers))
    log_mid = np.cumsum(log_returns, axis=0) + np.log(50.0)
    mid = np.exp(log_mid)

    half_spread = spread_frac / 2.0

    # Close = mid +/- half_spread (random buy/sell)
    bs = rng.choice([-1.0, 1.0], size=(n_days, n_tickers))
    close = mid * (1.0 + bs * half_spread)

    # High/Low: intraday vol expanded by 2x daily vol, plus half-spread on each side
    intraday_log_range = np.abs(rng.normal(scale=2.0 * vol_frac, size=(n_days, n_tickers)))
    high = mid * np.exp(intraday_log_range / 2.0) * (1.0 + half_spread)
    low = mid * np.exp(-intraday_log_range / 2.0) * (1.0 - half_spread)

    return (
        pd.DataFrame(high, index=dates, columns=tickers),
        pd.DataFrame(low, index=dates, columns=tickers),
        pd.DataFrame(close, index=dates, columns=tickers),
    )


def test_estimator_recovers_known_spread_within_tolerance():
    """On synthetic data with known 50-bps effective spread, AR should
    estimate ~25 bps per side within order-of-magnitude tolerance."""
    high, low, close = _ohlc_with_known_spread(
        n_days=300, n_tickers=5, spread_frac=0.005, vol_frac=0.01, seed=0
    )
    bps = compute_abdi_ranaldo_spread_panel(high, low, close, window=60, min_obs=20)
    # Take the last row (longest history) and average across tickers
    last = bps.iloc[-1].mean()
    # True per-side bps = (0.005 / 2) * 1e4 = 25 bps.
    # AR is biased upward in noisy estimates; allow 5 .. 100 bps.
    assert 5.0 < last < 100.0, f"Estimator gave {last} bps; expected ~25"


def test_estimator_orders_tickers_by_spread():
    """Ticker with wider true spread should get higher estimated bps."""
    rng_seed = 42
    h_narrow, l_narrow, c_narrow = _ohlc_with_known_spread(
        n_days=300, n_tickers=1, spread_frac=0.001, vol_frac=0.01, seed=rng_seed
    )
    h_wide, l_wide, c_wide = _ohlc_with_known_spread(
        n_days=300, n_tickers=1, spread_frac=0.020, vol_frac=0.01, seed=rng_seed
    )
    h_narrow.columns = ["NARROW.NZ"]
    l_narrow.columns = ["NARROW.NZ"]
    c_narrow.columns = ["NARROW.NZ"]
    h_wide.columns = ["WIDE.NZ"]
    l_wide.columns = ["WIDE.NZ"]
    c_wide.columns = ["WIDE.NZ"]

    high = pd.concat([h_narrow, h_wide], axis=1)
    low = pd.concat([l_narrow, l_wide], axis=1)
    close = pd.concat([c_narrow, c_wide], axis=1)

    bps = compute_abdi_ranaldo_spread_panel(high, low, close, window=60, min_obs=20)
    last_narrow = bps["NARROW.NZ"].iloc[-1]
    last_wide = bps["WIDE.NZ"].iloc[-1]
    assert last_wide > last_narrow, (
        f"Wide-spread ticker estimated lower ({last_wide}) than narrow ({last_narrow})"
    )


def test_estimator_floors_at_min_bps():
    """Estimator never returns less than the configured floor."""
    # Identical H/L/C series → daily estimator collapses to 0
    dates = pd.bdate_range("2024-01-02", periods=100)
    flat = pd.DataFrame(
        np.full((100, 1), 50.0), index=dates, columns=["FLAT.NZ"]
    )
    bps = compute_abdi_ranaldo_spread_panel(flat, flat, flat, window=60, min_obs=20)
    assert (bps >= DEFAULT_MIN_BPS_PER_SIDE - 1e-9).all().all()


def test_estimator_ignores_future_for_each_date():
    """At date t, the estimator at t depends only on data up to and including
    that date (the rolling mean uses a trailing window). Truncating the data
    after t must not change the value at t."""
    high, low, close = _ohlc_with_known_spread(
        n_days=300, n_tickers=2, spread_frac=0.005, vol_frac=0.01, seed=7
    )
    bps_full = compute_abdi_ranaldo_spread_panel(
        high, low, close, window=60, min_obs=20
    )
    cutoff = 200
    bps_truncated = compute_abdi_ranaldo_spread_panel(
        high.iloc[:cutoff],
        low.iloc[:cutoff],
        close.iloc[:cutoff],
        window=60,
        min_obs=20,
    )
    # Last point in truncated series excludes one term (eta_{t+1} unavailable),
    # so compare up to penultimate point.
    common_end = bps_truncated.index[-2]
    pd.testing.assert_frame_equal(
        bps_full.loc[:common_end], bps_truncated.loc[:common_end],
        check_freq=False,
    )


def test_estimator_handles_empty_input():
    empty = pd.DataFrame()
    out = compute_abdi_ranaldo_spread_panel(empty, empty, empty)
    assert out.empty


def test_estimator_handles_short_history():
    """With fewer days than min_obs, output should be entirely the floor
    (after the NaN-fill applied by the estimator)."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    h = pd.DataFrame(np.full((10, 1), 51.0), index=dates, columns=["A.NZ"])
    l = pd.DataFrame(np.full((10, 1), 49.0), index=dates, columns=["A.NZ"])
    c = pd.DataFrame(np.full((10, 1), 50.0), index=dates, columns=["A.NZ"])
    bps = compute_abdi_ranaldo_spread_panel(h, l, c, window=60, min_obs=20)
    assert (bps == DEFAULT_MIN_BPS_PER_SIDE).all().all()
