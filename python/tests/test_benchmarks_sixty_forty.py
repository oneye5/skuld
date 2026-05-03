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

    prices_data = {
        t: 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod()
        for t in tickers
    }
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


def _make_explicit_panel(
    *,
    equity_returns: list[float],
    bond_yields: list[float],
    ticker: str = "FNZ.NZ",
) -> PreparedPanel:
    """Build a minimal panel with explicit month-end equity returns and bond yields."""
    month_ends = pd.to_datetime(["2021-01-31", "2021-02-28", "2021-03-31"])
    daily_dates = pd.to_datetime(["2021-01-29", "2021-02-26", "2021-03-31"])

    return PreparedPanel(
        returns_daily=pd.DataFrame({ticker: [0.0, 0.0, 0.0]}, index=daily_dates),
        returns_monthly=pd.DataFrame({ticker: equity_returns}, index=month_ends),
        market_cap=pd.DataFrame(
            {ticker: [1_000_000.0, 1_000_000.0, 1_000_000.0]},
            index=daily_dates,
        ),
        sector=pd.Series({ticker: "Unknown"}),
        universe_mask=pd.DataFrame({ticker: [True, True, True]}, index=month_ends),
        macro=pd.DataFrame({"long_term_interest_rates": bond_yields}, index=month_ends),
        asof=pd.Timestamp("2021-04-15"),
        prices=pd.DataFrame({ticker: [10.0, 10.0, 10.0]}, index=daily_dates),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
    )


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

    returns, start, end, notes = sixty_forty(
        panel,
        equity_proxy="FNZ.NZ",
        flat_haircut_bps=400.0,
    )

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
    assert returns.index[0] == start
    assert returns.index[-1] == end
    expected = pd.Series(expected_net, index=returns.index)
    pd.testing.assert_series_equal(returns, expected, check_names=False, rtol=1e-4)
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


def test_sixty_forty_bond_rates_mixed_units_auto_convert_elementwise():
    """Mixed decimal/percentage month-end yields should be normalized elementwise."""
    from skuld_research.benchmarks.sixty_forty import sixty_forty

    panel = _make_explicit_panel(
        equity_returns=[0.02, 0.02, 0.02],
        bond_yields=[0.05, 5.0, 0.06],
    )

    returns, start, end, notes = sixty_forty(
        panel,
        equity_proxy="FNZ.NZ",
        duration_years=0.0,
        flat_haircut_bps=0.0,
    )

    yields = pd.Series([0.05, 0.05, 0.06], index=panel.macro.index)
    expected_bond_returns = (1.0 + yields) ** (1.0 / 12.0) - 1.0
    expected = 0.60 * 0.02 + 0.40 * expected_bond_returns

    pd.testing.assert_series_equal(returns, expected, check_names=False)
    assert start == expected.index[0]
    assert end == expected.index[-1]
    assert any("auto-converted from percentage" in note.lower() for note in notes)


def test_sixty_forty_duration_mode_adds_price_return_from_yield_changes():
    """Falling yields should add positive bond price return in duration-aware mode."""
    from skuld_research.benchmarks.sixty_forty import sixty_forty

    panel = _make_explicit_panel(
        equity_returns=[0.02, 0.02, 0.02],
        bond_yields=[0.06, 0.05, 0.04],
    )

    returns, start, end, notes = sixty_forty(
        panel,
        equity_proxy="FNZ.NZ",
        duration_years=5.0,
        flat_haircut_bps=0.0,
    )

    yields = pd.Series([0.06, 0.05, 0.04], index=panel.macro.index)
    carry = (1.0 + yields) ** (1.0 / 12.0) - 1.0
    expected_bond_returns = carry - 5.0 * yields.diff()
    expected = 0.60 * 0.02 + 0.40 * expected_bond_returns.dropna()

    assert list(returns.index) == list(expected.index)
    pd.testing.assert_series_equal(returns, expected, check_names=False)
    assert start == expected.index[0]
    assert end == expected.index[-1]
    assert any("duration-aware" in note.lower() for note in notes)
    assert any("5.0" in note for note in notes)


def test_sixty_forty_yield_only_mode_preserves_existing_behavior():
    """Zero duration should preserve the pre-existing yield-only bond proxy."""
    from skuld_research.benchmarks.sixty_forty import sixty_forty

    panel = _make_explicit_panel(
        equity_returns=[0.02, 0.02, 0.02],
        bond_yields=[0.06, 0.05, 0.04],
    )

    returns, start, end, notes = sixty_forty(
        panel,
        equity_proxy="FNZ.NZ",
        duration_years=0.0,
        flat_haircut_bps=0.0,
    )

    yields = pd.Series([0.06, 0.05, 0.04], index=panel.macro.index)
    carry = (1.0 + yields) ** (1.0 / 12.0) - 1.0
    expected = 0.60 * 0.02 + 0.40 * carry

    pd.testing.assert_series_equal(returns, expected, check_names=False)
    assert start == expected.index[0]
    assert end == expected.index[-1]
    assert any("yield-only" in note.lower() for note in notes)
    assert not any("duration-aware" in note.lower() for note in notes)


def test_sixty_forty_negative_duration_preserves_existing_behavior():
    """Negative duration follows the same yield-only path as zero duration."""
    from skuld_research.benchmarks.sixty_forty import sixty_forty

    panel = _make_explicit_panel(
        equity_returns=[0.02, 0.02, 0.02],
        bond_yields=[0.06, 0.05, 0.04],
    )

    returns, start, end, notes = sixty_forty(
        panel,
        equity_proxy="FNZ.NZ",
        duration_years=-1.0,
        flat_haircut_bps=0.0,
    )

    yields = pd.Series([0.06, 0.05, 0.04], index=panel.macro.index)
    carry = (1.0 + yields) ** (1.0 / 12.0) - 1.0
    expected = 0.60 * 0.02 + 0.40 * carry

    pd.testing.assert_series_equal(returns, expected, check_names=False)
    assert start == expected.index[0]
    assert end == expected.index[-1]
    assert any("yield-only" in note.lower() for note in notes)
    assert not any("duration-aware" in note.lower() for note in notes)
