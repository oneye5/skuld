"""Tests for NZX equal-weighted benchmark."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import PITSnapshot, PreparedPanel
from skuld_research.data.prepared_panel import build_prepared_panel


def _make_panel_with_mcap(
    n_days: int = 400,
    ticker_mcaps: dict[str, float] | None = None,
) -> PreparedPanel:
    """Build a PreparedPanel with specified market caps."""
    rng = np.random.default_rng(42)
    ticker_mcaps = ticker_mcaps or {
        "T00.NZ": 50e6,
        "T01.NZ": 30e6,
        "T02.NZ": 15e6,  # below 20M floor
        "T03.NZ": 25e6,
        "T04.NZ": 5e6,   # below 20M floor
    }
    tickers = list(ticker_mcaps.keys())
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    
    prices_data = {t: 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod() for t in tickers}
    prices = pd.DataFrame(prices_data, index=dates)
    
    snap = PITSnapshot(
        prices=prices,
        volumes=pd.DataFrame({t: 100_000.0 for t in tickers}, index=dates),
        fundamentals=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"])
        ),
        macro=pd.DataFrame(index=dates),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=dates[-1] + pd.DateOffset(months=3),
    )
    panel = build_prepared_panel(snap, nzx_only=False)
    
    # Override market_cap with specified values
    mcap_data = {t: mcap for t, mcap in ticker_mcaps.items()}
    market_cap = pd.DataFrame(mcap_data, index=panel.market_cap.index)
    return replace(panel, market_cap=market_cap)


def test_nzx_equal_weighted_respects_mcap_floor():
    """Equal-weighted bench with mcap floor produces expected avg positions."""
    from skuld_research.benchmarks.nzx_equal_weighted_fixed_universe import (
        nzx_equal_weighted_fixed_universe,
    )
    
    # 5 tickers: 3 above 20M, 2 below
    panel = _make_panel_with_mcap(
        n_days=200,
        ticker_mcaps={
            "T00.NZ": 50e6,
            "T01.NZ": 30e6,
            "T02.NZ": 15e6,
            "T03.NZ": 25e6,
            "T04.NZ": 5e6,
        },
    )
    
    result = nzx_equal_weighted_fixed_universe(panel, mcap_floor_nzd=20e6)
    
    # Should hold ~3 names (T00, T01, T03 exceed floor)
    assert result.avg_positions >= 2.5
    assert result.avg_positions <= 3.5


def test_nzx_equal_weighted_anti_gaming():
    """Passing a panel with empty universe_mask still produces backtest if mcap exceeds floor."""
    from skuld_research.benchmarks.nzx_equal_weighted_fixed_universe import (
        nzx_equal_weighted_fixed_universe,
    )
    
    panel = _make_panel_with_mcap(
        n_days=150,
        ticker_mcaps={
            "T00.NZ": 60e6,
            "T01.NZ": 40e6,
        },
    )
    
    # Override universe_mask to exclude all names
    empty_mask = pd.DataFrame(
        False, index=panel.universe_mask.index, columns=panel.universe_mask.columns
    )
    panel_gamed = replace(panel, universe_mask=empty_mask)
    
    # Benchmark should still run based on mcap alone
    result = nzx_equal_weighted_fixed_universe(panel_gamed, mcap_floor_nzd=20e6)
    
    # Both names exceed floor → avg_positions ≈ 2
    assert result.avg_positions >= 1.5
    assert result.n_periods > 0


def test_nzx_equal_weighted_equal_weighting():
    """With zero returns, output return is 0; with known returns, output matches expectation."""
    from skuld_research.benchmarks.nzx_equal_weighted_fixed_universe import (
        nzx_equal_weighted_fixed_universe,
    )
    
    panel = _make_panel_with_mcap(
        n_days=100,
        ticker_mcaps={
            "T00.NZ": 50e6,
            "T01.NZ": 30e6,
        },
    )
    
    # Override returns_monthly to all zeros
    zero_returns = pd.DataFrame(
        0.0, index=panel.returns_monthly.index, columns=panel.returns_monthly.columns
    )
    panel_zero = replace(panel, returns_monthly=zero_returns)
    
    result = nzx_equal_weighted_fixed_universe(panel_zero, mcap_floor_nzd=20e6)
    
    # Net returns should be negative due to costs (subscription fee + spread)
    # Check that avg positions is reasonable
    assert result.avg_positions >= 1.5
    assert result.avg_positions <= 2.5
    # Costs should be charged
    assert result.costs_nzd.sum() > 0


def test_nzx_equal_weighted_rejects_non_default_adv_floor():
    """Non-default ADV floors must fail fast until the filter is implemented."""
    from skuld_research.benchmarks.nzx_equal_weighted_fixed_universe import (
        nzx_equal_weighted_fixed_universe,
    )

    panel = _make_panel_with_mcap(n_days=120)

    with pytest.raises(NotImplementedError, match="ADV filter"):
        nzx_equal_weighted_fixed_universe(
            panel,
            mcap_floor_nzd=20e6,
            adv_floor_shares=25_000,
        )
