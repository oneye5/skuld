"""Tests for cash overlay rules and application."""
import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import PreparedPanel, TargetPortfolio
from skuld_research.overlay import (
    NoOverlay,
    NzxMA200AndAggMomentumRule,
    apply_cash_overlay,
)


@pytest.fixture
def minimal_panel() -> PreparedPanel:
    """Minimal PreparedPanel for overlay tests."""
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    tickers = ["A.NZ", "B.NZ", "C.NZ"]

    # Synthetic daily returns: slightly positive drift
    np.random.seed(42)
    returns_daily = pd.DataFrame(
        np.random.normal(0.0005, 0.02, (len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )

    # Monthly returns (month-end)
    returns_monthly = (1.0 + returns_daily).resample("BME").prod() - 1.0

    # Market cap (constant for simplicity)
    market_cap = pd.DataFrame(
        1e7, index=dates, columns=tickers
    )

    # Sector (all Unknown)
    sector = pd.Series("Unknown", index=tickers)

    # Universe mask: all tickers in universe at month-ends
    rebal_dates = returns_monthly.index
    universe_mask = pd.DataFrame(True, index=rebal_dates, columns=tickers)

    # Empty macro
    macro = pd.DataFrame(index=dates)

    return PreparedPanel(
        returns_daily=returns_daily,
        returns_monthly=returns_monthly,
        market_cap=market_cap,
        sector=sector,
        universe_mask=universe_mask,
        macro=macro,
        asof=pd.Timestamp("2020-12-31"),
    )


@pytest.fixture
def minimal_target() -> TargetPortfolio:
    """Minimal target portfolio."""
    weights = pd.Series(
        [0.30, 0.40, 0.25],
        index=["A.NZ", "B.NZ", "C.NZ"],
    )
    return TargetPortfolio(
        weights=weights,
        cash_weight=0.05,
        method="HRP",
        asof=pd.Timestamp("2020-10-31"),
    )


def test_no_overlay_returns_zero(minimal_panel):
    """NoOverlay always returns 0.0."""
    rule = NoOverlay()
    asof = pd.Timestamp("2020-10-31")
    result = rule.evaluate(minimal_panel, asof)
    assert result == 0.0


def test_no_overlay_leaves_target_unchanged(minimal_panel, minimal_target):
    """apply_cash_overlay with NoOverlay returns target unchanged."""
    rule = NoOverlay()
    asof = minimal_target.asof

    result = apply_cash_overlay(minimal_target, minimal_panel, rule, asof)

    # Should be the same object
    assert result is minimal_target
    assert result.cash_weight == minimal_target.cash_weight
    pd.testing.assert_series_equal(result.weights, minimal_target.weights)


def test_nzx_ma200_rule_constructor_validation():
    """NzxMA200AndAggMomentumRule validates constructor args."""
    # Valid
    NzxMA200AndAggMomentumRule(defensive_cash_fraction=0.30, momentum_aggregate_lookback_months=12)

    # Invalid defensive_cash_fraction
    with pytest.raises(ValueError, match="defensive_cash_fraction must be in"):
        NzxMA200AndAggMomentumRule(defensive_cash_fraction=-0.1)

    with pytest.raises(ValueError, match="defensive_cash_fraction must be in"):
        NzxMA200AndAggMomentumRule(defensive_cash_fraction=1.5)

    # Invalid lookback
    with pytest.raises(ValueError, match="momentum_aggregate_lookback_months must be >= 1"):
        NzxMA200AndAggMomentumRule(momentum_aggregate_lookback_months=0)


def test_nzx_ma200_rule_insufficient_history(minimal_panel):
    """Rule returns 0.0 when insufficient history for 200-day MA."""
    rule = NzxMA200AndAggMomentumRule()

    # Ask at a date very early in the panel
    asof = minimal_panel.returns_daily.index[50]
    result = rule.evaluate(minimal_panel, pd.Timestamp(asof))

    # Insufficient history → should return 0.0
    assert result == 0.0


def test_nzx_ma200_rule_aggregate_momentum_negative_uses_raw_scores():
    """Aggregate momentum should be negative when eligible names have negative 12-1 returns."""
    monthly_dates = pd.date_range("2018-01-31", periods=15, freq="BME")
    tickers = ["A.NZ", "B.NZ", "C.NZ"]
    monthly = pd.DataFrame(0.0, index=monthly_dates, columns=tickers)
    # The score at 2019-03-29 uses 12 months ending 2019-01-31 and skips 2019-02-28.
    monthly.iloc[1:13, :] = -0.02

    daily_dates = pd.date_range("2018-01-01", "2019-04-30", freq="B")
    daily = pd.DataFrame(0.0, index=daily_dates, columns=tickers)
    universe_mask = pd.DataFrame(True, index=monthly_dates, columns=tickers)

    panel = PreparedPanel(
        returns_daily=daily,
        returns_monthly=monthly,
        market_cap=pd.DataFrame(1e7, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=universe_mask,
        macro=pd.DataFrame(index=daily_dates),
        asof=pd.Timestamp("2019-03-29"),
    )
    rule = NzxMA200AndAggMomentumRule(momentum_aggregate_lookback_months=12)

    assert rule._aggregate_momentum_negative(panel, pd.Timestamp("2019-03-29"))


def _overlay_panel_with_market_downtrend(monthly_return: float) -> PreparedPanel:
    tickers = ["A.NZ", "B.NZ", "C.NZ"]
    asof = pd.Timestamp("2019-04-30")
    daily_dates = pd.bdate_range("2018-05-01", "2019-04-29")
    returns_daily = pd.DataFrame(0.0, index=daily_dates, columns=tickers)
    returns_daily.iloc[-50:, :] = -0.01

    monthly_dates = pd.date_range("2018-01-31", periods=15, freq="BME")
    returns_monthly = pd.DataFrame(monthly_return, index=monthly_dates, columns=tickers)
    universe_mask = pd.DataFrame(True, index=[asof], columns=tickers)

    return PreparedPanel(
        returns_daily=returns_daily,
        returns_monthly=returns_monthly,
        market_cap=pd.DataFrame(1e7, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=universe_mask,
        macro=pd.DataFrame(index=daily_dates),
        asof=asof,
    )


def test_nzx_ma200_rule_triggers_when_market_below_ma_and_momentum_negative():
    """Rule raises cash when both defensive conditions are met."""
    panel = _overlay_panel_with_market_downtrend(monthly_return=-0.02)
    rule = NzxMA200AndAggMomentumRule(defensive_cash_fraction=0.30)

    assert rule.evaluate(panel, panel.asof) == 0.30


def test_nzx_ma200_rule_does_not_trigger_when_market_below_ma_but_momentum_positive():
    """Rule stays inactive unless both defensive conditions are met."""
    panel = _overlay_panel_with_market_downtrend(monthly_return=0.02)
    rule = NzxMA200AndAggMomentumRule(defensive_cash_fraction=0.30)

    assert rule.evaluate(panel, panel.asof) == 0.0


def test_apply_cash_overlay_raises_cash_and_renormalizes(minimal_panel, minimal_target):
    """apply_cash_overlay raises cash and preserves equity weight proportions."""
    # Mock rule that always returns 0.25 (25% cash)
    class FixedCashRule:
        def evaluate(self, panel, asof):
            return 0.25

    rule = FixedCashRule()
    asof = minimal_target.asof

    result = apply_cash_overlay(minimal_target, minimal_panel, rule, asof)

    # Cash should be raised to 0.25
    assert abs(result.cash_weight - 0.25) < 1e-9

    # Equity weights should be re-normalised to sum to 0.75
    assert abs(result.weights.sum() - 0.75) < 1e-6

    # Relative proportions should be preserved
    # Original equity: [0.30, 0.40, 0.25] summing to 0.95
    # Original proportions: 0.30/0.95 ≈ 0.316, 0.40/0.95 ≈ 0.421, 0.25/0.95 ≈ 0.263
    # New equity sum: 0.75
    # New weights should be: 0.75 * [0.316, 0.421, 0.263] ≈ [0.237, 0.316, 0.197]
    expected_props = minimal_target.weights / minimal_target.weights.sum()
    actual_props = result.weights / result.weights.sum()

    pd.testing.assert_series_equal(actual_props, expected_props, atol=1e-6)


def test_apply_cash_overlay_does_nothing_when_rule_returns_less_than_floor(
    minimal_panel, minimal_target
):
    """apply_cash_overlay returns original target when rule returns < current cash."""
    # Mock rule that returns 0.03 (less than target's 0.05 floor)
    class LowCashRule:
        def evaluate(self, panel, asof):
            return 0.03

    rule = LowCashRule()
    asof = minimal_target.asof

    result = apply_cash_overlay(minimal_target, minimal_panel, rule, asof)

    # Should return the same object unchanged
    assert result is minimal_target


def test_overlay_with_real_momentum_factor(minimal_panel):
    """Integration test: NzxMA200AndAggMomentumRule uses real MomentumFactor."""
    rule = NzxMA200AndAggMomentumRule(
        defensive_cash_fraction=0.30,
        momentum_aggregate_lookback_months=12,
    )

    # Pick a date with enough history
    asof = minimal_panel.returns_monthly.index[-1]

    # Should not raise
    result = rule.evaluate(minimal_panel, asof)

    # Result should be either 0.0 or 0.30
    assert result in {0.0, 0.30}
