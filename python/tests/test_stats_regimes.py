"""Tests for market regime labelling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import PreparedPanel
from skuld_research.stats.regimes import label_regimes


def _make_synthetic_panel_with_proxy(regime_returns: list[float]) -> PreparedPanel:
    """Build a PreparedPanel with known monthly proxy returns."""
    n_months = len(regime_returns)
    dates = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    
    tickers = ["T01.NZ", "T02.NZ", "T03.NZ"]
    
    # Market proxy = equal-weighted average (regimes.py will compute this)
    # Give all tickers the same returns → proxy = ticker returns
    returns_monthly = pd.DataFrame(
        {tk: regime_returns for tk in tickers},
        index=dates,
    )
    
    returns_daily = pd.DataFrame(
        {tk: np.zeros(len(dates)) for tk in tickers},
        index=dates,
    )
    
    market_cap = pd.DataFrame(
        {tk: 1e6 for tk in tickers},
        index=dates,
    )
    
    sector = pd.Series({tk: "Tech" for tk in tickers})
    
    universe_mask = pd.DataFrame(
        {tk: True for tk in tickers},
        index=dates,
    )
    
    return PreparedPanel(
        returns_daily=returns_daily,
        returns_monthly=returns_monthly,
        market_cap=market_cap,
        sector=sector,
        universe_mask=universe_mask,
        macro=pd.DataFrame(),
        asof=dates[-1],
    )


def test_first_12_months_are_chop():
    """First 12 months have insufficient history → labelled 'chop'."""
    # 24 months of returns
    returns = [0.02] * 24
    panel = _make_synthetic_panel_with_proxy(returns)
    
    labels = label_regimes(panel)
    
    # First 12 labels should be 'chop'
    assert all(labels.iloc[:12] == "chop")


def test_bull_regime_after_strong_12m():
    """Trailing 12m return > +0.10 → 'bull'."""
    # First 12 months: small positive (sum < 0.10)
    # Month 13: add return that makes trailing-12m > 0.10
    # Each month +1% → 12 months ≈ +12.7% (compounded slightly more)
    returns = [0.01] * 12 + [0.01] * 12
    panel = _make_synthetic_panel_with_proxy(returns)
    
    labels = label_regimes(panel)
    
    # After 12 months, trailing 12m is (1.01)^12 - 1 ≈ 0.1268 > 0.10 → bull
    assert labels.iloc[12] == "bull"


def test_bear_regime_after_strong_12m_loss():
    """Trailing 12m return < -0.10 → 'bear'."""
    # Each month -1.5% → trailing 12m ≈ -16.6% < -0.10
    # Need stronger loss to ensure < -10%
    returns = [0.005] * 12 + [-0.015] * 12
    panel = _make_synthetic_panel_with_proxy(returns)
    
    labels = label_regimes(panel)
    
    # After first 12 flat, months 13-24 with -1.5% each
    # At month 13, trailing 12m is from month 1-12 (all +0.5%) → chop
    # At month 24, trailing 12m is from month 12-23 (-1.5% each) → bear
    assert labels.iloc[-1] == "bear"  # Last month should be bear


def test_chop_regime_in_range():
    """Trailing 12m return in [-0.10, +0.10] → 'chop'."""
    # Returns that keep trailing-12m in the chop range
    returns = [0.005] * 24  # +0.5%/mo → trailing 12m ≈ +6.2% → chop
    panel = _make_synthetic_panel_with_proxy(returns)
    
    labels = label_regimes(panel)
    
    assert labels.iloc[12] == "chop"
    assert labels.iloc[-1] == "chop"


def test_pit_respected():
    """Label at month t does not depend on returns at t or later."""
    # First 12: flat, then month 13 has huge gain
    returns = [0.0] * 12 + [0.50]  # +50% in month 13
    panel = _make_synthetic_panel_with_proxy(returns)
    
    labels = label_regimes(panel)
    
    # At month 12 (index 12), the label is based on months 0..11 → all zero → chop
    assert labels.iloc[12] == "chop"
