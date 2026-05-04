"""Tests for NZ TD floor benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import PITSnapshot, PreparedPanel
from skuld_research.data.prepared_panel import build_prepared_panel


def _make_panel_with_macro(
    n_days: int = 400,
    macro_field: str = "short_term_interest_rates",
    rate_annual_decimal: float = 0.04,
) -> PreparedPanel:
    """Build a PreparedPanel with a constant macro rate (decimal form)."""
    rng = np.random.default_rng(42)
    tickers = ["T00.NZ", "T01.NZ"]
    dates = pd.bdate_range("2021-01-01", periods=n_days)

    prices_data = {t: 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod() for t in tickers}
    prices = pd.DataFrame(prices_data, index=dates)

    # Macro data: constant rate in decimal form (e.g., 0.04 for 4%)
    macro = pd.DataFrame({macro_field: rate_annual_decimal}, index=dates)

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


def test_td_floor_constant_rate():
    """TD floor with constant 4% rate produces monthly return ≈ (1.04)**(1/12) - 1."""
    from skuld_research.benchmarks.nz_td_floor import nz_td_floor

    panel = _make_panel_with_macro(n_days=400, rate_annual_decimal=0.04)

    result = nz_td_floor(panel, panel.asof, default_floor=0.04)

    # Monthly return should be approx (1.04)^(1/12) - 1 ≈ 0.00327
    expected_monthly = (1.04 ** (1.0 / 12.0)) - 1.0

    assert len(result.returns) > 0
    assert abs(result.returns.mean() - expected_monthly) < 1e-4
    # All returns should be identical (constant rate)
    assert result.returns.std() < 1e-9
    # Costs and turnover should be zero
    assert result.costs_nzd.sum() == 0.0
    assert result.turnover.sum() == 0.0


def test_td_floor_percentage_rate_auto_converted():
    """Ingested rates may arrive as percentages; 4.0 should mean 4%, not 400%."""
    from skuld_research.benchmarks.nz_td_floor import nz_td_floor

    panel = _make_panel_with_macro(n_days=400, rate_annual_decimal=4.0)

    result = nz_td_floor(panel, panel.asof, default_floor=0.04)

    expected_monthly = (1.04 ** (1.0 / 12.0)) - 1.0
    assert abs(result.returns.mean() - expected_monthly) < 1e-4


def test_td_floor_mixed_decimal_and_percentage_rates():
    """Rate scale conversion is per observation, not based on the series mean."""
    from skuld_research.benchmarks.nz_td_floor import nz_td_floor

    panel = _make_panel_with_macro(n_days=400, rate_annual_decimal=0.04)
    rate_col = "short_term_interest_rates"
    panel.macro.loc[panel.macro.index[-20:], rate_col] = 4.0

    result = nz_td_floor(panel, panel.asof, default_floor=0.04)

    expected_monthly = (1.04 ** (1.0 / 12.0)) - 1.0
    assert result.returns.max() == pytest.approx(expected_monthly)


def test_td_floor_sub_one_percent_values_are_treated_as_percentage_points():
    """Values like 0.33 in the feed mean 0.33%, not 33%."""
    from skuld_research.benchmarks.nz_td_floor import nz_td_floor

    panel = _make_panel_with_macro(n_days=400, rate_annual_decimal=0.04)
    rate_col = "short_term_interest_rates"
    panel.macro.loc[:, rate_col] = 0.33

    result = nz_td_floor(panel, panel.asof, default_floor=0.04)

    expected_monthly = (1.0033 ** (1.0 / 12.0)) - 1.0
    assert abs(result.returns.iloc[1:].mean() - expected_monthly) < 1e-6


def test_td_floor_missing_rate_uses_default():
    """When macro field is missing, default_floor is used."""
    from skuld_research.benchmarks.nz_td_floor import nz_td_floor

    panel = _make_panel_with_macro(n_days=100, macro_field="other_field", rate_annual_decimal=0.05)

    result = nz_td_floor(panel, panel.asof, default_floor=0.03)

    # Should use default 3%
    expected_monthly = (1.03 ** (1.0 / 12.0)) - 1.0
    assert abs(result.returns.mean() - expected_monthly) < 1e-4


def test_td_floor_zero_rate():
    """TD floor with 0% rate produces zero returns."""
    from skuld_research.benchmarks.nz_td_floor import nz_td_floor

    panel = _make_panel_with_macro(n_days=200, rate_annual_decimal=0.0)

    result = nz_td_floor(panel, panel.asof, default_floor=0.0)

    assert result.returns.sum() == 0.0


def test_td_floor_drawdown_non_negative_for_positive_rate():
    """For a non-negative rate path, drawdown series should be ≥ 0 (no losses)."""
    from skuld_research.benchmarks.nz_td_floor import nz_td_floor

    panel = _make_panel_with_macro(n_days=300, rate_annual_decimal=0.02)

    result = nz_td_floor(panel, panel.asof, default_floor=0.02)

    # Drawdown should be zero for a monotonically increasing return path
    assert (result.drawdown >= -1e-9).all()
