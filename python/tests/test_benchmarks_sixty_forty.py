"""Tests for 60/40 benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PITSnapshot, PreparedPanel
from skuld_research.data.prepared_panel import build_prepared_panel


def _make_panel_with_bond_macro(
    n_days: int = 400,
    eq_ticker: str = "FNZ.NZ",
    bond_rate_annual: float = 0.05,
) -> PreparedPanel:
    """Build a PreparedPanel with equity returns and bond macro data."""
    rng = np.random.default_rng(42)
    tickers = [eq_ticker, "T01.NZ"]
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    
    prices_data = {t: 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod() for t in tickers}
    prices = pd.DataFrame(prices_data, index=dates)
    
    # Bond rates as decimal (0.05 for 5%)
    macro = pd.DataFrame({"long_term_interest_rates": bond_rate_annual}, index=dates)
    
    snap = PITSnapshot(
        prices=prices,
        volumes=pd.DataFrame({t: 100_000.0 for t in tickers}, index=dates),
        fundamentals=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"])
        ),
        macro=macro,
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=dates[-1] + pd.DateOffset(months=3),
    )
    return build_prepared_panel(snap, nzx_only=False)


def test_sixty_forty_constant_legs():
    """60/40 with constant equity and bond returns produces expected output."""
    from skuld_research.benchmarks.sixty_forty import sixty_forty
    
    panel = _make_panel_with_bond_macro(n_days=200, bond_rate_annual=0.04)
    
    # Override returns_monthly to constant 0.01 (1% per month)
    const_returns = pd.DataFrame(
        0.01, index=panel.returns_monthly.index, columns=panel.returns_monthly.columns
    )
    from dataclasses import replace
    panel = replace(panel, returns_monthly=const_returns)
    
    returns, start, end, notes = sixty_forty(panel, equity_proxy="FNZ.NZ", flat_haircut_bps=400.0)
    
    # Expected monthly return:
    # - equity: 0.01
    # - bond: (1.04)^(1/12) - 1 ≈ 0.00327
    # - 60/40: 0.6 * 0.01 + 0.4 * 0.00327 ≈ 0.00731
    # - haircut: 400 bps / 10000 / 12 ≈ 0.00333
    # - net: 0.00731 - 0.00333 ≈ 0.00398
    
    expected_bond_monthly = (1.04 ** (1.0 / 12.0)) - 1.0
    expected_combined = 0.6 * 0.01 + 0.4 * expected_bond_monthly
    expected_net = expected_combined - (400.0 / 10_000.0 / 12.0)
    
    assert len(returns) > 0
    assert abs(returns.mean() - expected_net) < 1e-4
    assert any("yield-only" in note.lower() for note in notes)
    assert any("400bps" in note or "haircut" in note.lower() for note in notes)


def test_sixty_forty_missing_equity_months_skipped():
    """When equity has missing months, coverage_start is later than panel start."""
    from skuld_research.benchmarks.sixty_forty import sixty_forty
    
    panel = _make_panel_with_bond_macro(n_days=250, bond_rate_annual=0.03)
    
    # Set first 3 months of equity returns to NaN
    from dataclasses import replace
    returns_mod = panel.returns_monthly.copy()
    returns_mod.iloc[:3, returns_mod.columns.get_loc("FNZ.NZ")] = float("nan")
    panel = replace(panel, returns_monthly=returns_mod)
    
    returns, start, end, notes = sixty_forty(panel, equity_proxy="FNZ.NZ")
    
    # coverage_start should be after the first 3 months
    assert start > panel.returns_monthly.index[2]


def test_sixty_forty_notes_contain_caveats():
    """Notes tuple contains expected caveats."""
    from skuld_research.benchmarks.sixty_forty import sixty_forty
    
    panel = _make_panel_with_bond_macro(n_days=150)
    
    returns, start, end, notes = sixty_forty(panel, equity_proxy="FNZ.NZ", flat_haircut_bps=200.0)
    
    assert len(notes) > 0
    # Check for key phrases
    notes_str = " ".join(notes)
    assert "yield-only" in notes_str.lower()
    assert "200bps" in notes_str or "haircut" in notes_str.lower()


def test_sixty_forty_bond_rates_percentage_auto_convert():
    """Bond rates in percentage form (>1.0) are auto-converted to decimal."""
    from skuld_research.benchmarks.sixty_forty import sixty_forty
    
    panel = _make_panel_with_bond_macro(n_days=200, bond_rate_annual=5.0)  # 5.0 (percentage)
    
    returns, start, end, notes = sixty_forty(panel, equity_proxy="FNZ.NZ")
    
    # Should auto-detect and convert
    assert "auto-converted from percentage" in " ".join(notes).lower()
    
    # Result should be sane (not absurdly large from treating 5.0 as 500%)
    assert returns.mean() < 0.1  # less than 10% per month
