"""Tests for survivorship bias adjustment module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skuld_research.survivorship import (
    SurvivorshipAdjuster,
    compute_drawdown_series,
    compute_max_drawdown,
)

CSV_PATH = Path(__file__).parent.parent / "src" / "survivorship" / "nzx_delistings.csv"


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------

def test_compute_max_drawdown_flat_returns():
    """Flat 0% returns every month → no drawdown."""
    returns = pd.Series([0.0] * 24)
    assert compute_max_drawdown(returns) == pytest.approx(0.0)


def test_compute_max_drawdown_positive_then_halved():
    """NAV goes 1 → 2 → 1: known −50% drawdown."""
    # +100% then −50%
    returns = pd.Series([1.0, -0.5])
    mdd = compute_max_drawdown(returns)
    assert mdd == pytest.approx(-0.5, abs=1e-9)


def test_compute_max_drawdown_empty():
    assert compute_max_drawdown(pd.Series(dtype=float)) == pytest.approx(0.0)


def test_compute_max_drawdown_always_non_positive():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.005, 0.05, 60))
    assert compute_max_drawdown(returns) <= 0.0


# ---------------------------------------------------------------------------
# compute_drawdown_series
# ---------------------------------------------------------------------------

def test_compute_drawdown_series_non_positive():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.005, 0.05, 60))
    dd = compute_drawdown_series(returns)
    assert (dd <= 0.0).all()


def test_compute_drawdown_series_empty():
    dd = compute_drawdown_series(pd.Series(dtype=float))
    assert dd.empty


def test_compute_drawdown_series_name():
    returns = pd.Series([0.01, -0.02, 0.03])
    dd = compute_drawdown_series(returns)
    assert dd.name == "drawdown"


# ---------------------------------------------------------------------------
# flat_haircut_sharpe
# ---------------------------------------------------------------------------

def test_flat_haircut_sharpe_subtracts_400bps():
    """400 bps = 0.04; Sharpe = (ret − 0.04) / vol."""
    adjuster = SurvivorshipAdjuster()
    raw_sharpe = 1.0
    ann_ret = 0.12
    ann_vol = 0.10
    expected = (0.12 - 0.04) / 0.10
    result = adjuster.flat_haircut_sharpe(raw_sharpe, ann_ret, ann_vol)
    assert result == pytest.approx(expected, rel=1e-9)


def test_flat_haircut_sharpe_zero_vol():
    adjuster = SurvivorshipAdjuster()
    assert adjuster.flat_haircut_sharpe(1.0, 0.10, 0.0) == pytest.approx(0.0)


def test_flat_haircut_sharpe_custom_bps():
    adjuster = SurvivorshipAdjuster(flat_haircut_bps=200.0)
    result = adjuster.flat_haircut_sharpe(1.0, 0.10, 0.10)
    expected = (0.10 - 0.02) / 0.10
    assert result == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# delisting_adjusted_sharpe — no CSV (fallback to flat haircut)
# ---------------------------------------------------------------------------

def test_delisting_adjusted_sharpe_no_csv_equals_flat_haircut():
    adjuster = SurvivorshipAdjuster()
    raw_sharpe = 1.5
    ann_ret = 0.15
    ann_vol = 0.10
    assert adjuster.delisting_adjusted_sharpe(raw_sharpe, ann_ret, ann_vol) == pytest.approx(
        adjuster.flat_haircut_sharpe(raw_sharpe, ann_ret, ann_vol)
    )


# ---------------------------------------------------------------------------
# DelistingStats from real CSV
# ---------------------------------------------------------------------------

def test_load_real_csv_produces_plausible_stats():
    adjuster = SurvivorshipAdjuster(delisting_csv_path=CSV_PATH)
    stats = adjuster._stats
    assert stats is not None
    assert 0.0 < stats.annual_loss_rate < 0.05
    assert stats.mean_terminal_return < 0.0
    assert stats.n_loss_delistings >= 8


def test_load_real_csv_n_loss_delistings_count():
    """CSV has 8 involuntary + several voluntary with negative returns = ≥ 8."""
    adjuster = SurvivorshipAdjuster(delisting_csv_path=CSV_PATH)
    stats = adjuster._stats
    assert stats is not None
    assert stats.n_loss_delistings >= 8


def test_load_real_csv_years_positive():
    adjuster = SurvivorshipAdjuster(delisting_csv_path=CSV_PATH)
    stats = adjuster._stats
    assert stats is not None
    assert stats.years_observed > 0.0


def test_unconditional_annual_drag_negative():
    adjuster = SurvivorshipAdjuster(delisting_csv_path=CSV_PATH)
    stats = adjuster._stats
    assert stats is not None
    # drag = p × μ_d; μ_d < 0, p > 0 → drag < 0
    assert stats.unconditional_annual_drag < 0.0


# ---------------------------------------------------------------------------
# delisting_adjusted_sharpe with real CSV
# ---------------------------------------------------------------------------

def test_delisting_adjusted_sharpe_with_csv_le_raw():
    """Adjusted Sharpe must be ≤ raw Sharpe (we only subtract drag)."""
    adjuster = SurvivorshipAdjuster(delisting_csv_path=CSV_PATH)
    raw_sharpe = 1.0
    ann_ret = 0.12
    ann_vol = 0.10
    adj = adjuster.delisting_adjusted_sharpe(raw_sharpe, ann_ret, ann_vol)
    assert adj <= raw_sharpe


def test_delisting_adjusted_sharpe_uses_conservative_drag():
    """Conservative drag = max(flat haircut, probabilistic drag)."""
    adjuster = SurvivorshipAdjuster(delisting_csv_path=CSV_PATH)
    stats = adjuster._stats
    assert stats is not None
    prob_drag = abs(stats.unconditional_annual_drag)
    flat_drag = adjuster.flat_haircut_annual
    conservative_drag = max(prob_drag, flat_drag)

    ann_ret = 0.12
    ann_vol = 0.10
    expected = (ann_ret - conservative_drag) / ann_vol
    result = adjuster.delisting_adjusted_sharpe(1.0, ann_ret, ann_vol)
    assert result == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# augmented_max_drawdown
# ---------------------------------------------------------------------------

def _make_monthly_returns(seed: int = 42, n: int = 60) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.007, 0.04, n))


def test_augmented_max_drawdown_with_csv_le_observed():
    """Augmented MDD should be ≤ (more negative or equal to) the observed MDD
    in expectation — the median over simulations is a softer test."""
    adjuster = SurvivorshipAdjuster(delisting_csv_path=CSV_PATH, rng_seed=42)
    returns = _make_monthly_returns()
    observed = compute_max_drawdown(returns)
    med_mdd, p90_mdd = adjuster.augmented_max_drawdown(returns, n_names_avg=20, n_simulations=200)
    # Both should be ≤ 0
    assert med_mdd <= 0.0
    assert p90_mdd <= 0.0
    # Median augmented drawdown ≤ observed (injected delistings worsen things)
    assert med_mdd <= observed + 1e-9


def test_augmented_max_drawdown_p90_more_extreme_than_median():
    """10th percentile (tail) must be ≤ median (both negative → more negative = worse)."""
    adjuster = SurvivorshipAdjuster(delisting_csv_path=CSV_PATH, rng_seed=42)
    returns = _make_monthly_returns()
    med_mdd, p90_mdd = adjuster.augmented_max_drawdown(returns, n_names_avg=20, n_simulations=500)
    assert p90_mdd <= med_mdd


def test_augmented_max_drawdown_no_stats_returns_observed():
    """Without CSV, both outputs equal observed MDD."""
    adjuster = SurvivorshipAdjuster()
    returns = _make_monthly_returns()
    observed = compute_max_drawdown(returns)
    med, p90 = adjuster.augmented_max_drawdown(returns, n_names_avg=20)
    assert med == pytest.approx(observed)
    assert p90 == pytest.approx(observed)


def test_augmented_max_drawdown_empty_returns_fallback():
    adjuster = SurvivorshipAdjuster(delisting_csv_path=CSV_PATH)
    med, p90 = adjuster.augmented_max_drawdown(pd.Series(dtype=float), n_names_avg=20)
    assert med == pytest.approx(0.0)
    assert p90 == pytest.approx(0.0)
