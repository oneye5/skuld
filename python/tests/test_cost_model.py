"""Tests for skuld_research.costs.model (CostModel, CostConfig, CostBreakdown)."""
from __future__ import annotations

import pytest
import pandas as pd

from skuld_research.costs import CostBreakdown, CostConfig, CostModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(config: CostConfig | None = None) -> CostModel:
    return CostModel(config)


def _series(*values: float) -> pd.Series:
    return pd.Series(list(values), dtype=float)


# ---------------------------------------------------------------------------
# 1. Zero trades → subscription still charged, no spread
# ---------------------------------------------------------------------------

def test_zero_trades_empty_series():
    result = _model().compute_period_costs(_series())
    assert result.spread_cost_nzd == pytest.approx(0.0)
    assert result.sharesies_fee_nzd == pytest.approx(15.0)
    assert result.total_cost_nzd == pytest.approx(15.0)
    assert result.sharesies_fee_band == "subscription_only"


def test_zero_trades_all_zeros():
    result = _model().compute_period_costs(_series(0.0, 0.0, 0.0))
    assert result.spread_cost_nzd == pytest.approx(0.0)
    assert result.sharesies_fee_nzd == pytest.approx(15.0)
    assert result.total_cost_nzd == pytest.approx(15.0)
    assert result.sharesies_fee_band == "subscription_only"


# ---------------------------------------------------------------------------
# 2. Small volume (< $5000) → subscription only, spread proportional to volume
# ---------------------------------------------------------------------------

def test_small_volume_flat_fee():
    # volume = 1000 NZD; spread = 200 bps → spread_cost = 20; fee = 15 subscription
    result = _model().compute_period_costs(_series(1_000.0))
    assert result.sharesies_fee_band == "subscription_only"
    assert result.sharesies_fee_nzd == pytest.approx(15.0)
    assert result.spread_cost_nzd == pytest.approx(20.0)   # 1000 * 200/10000
    assert result.total_cost_nzd == pytest.approx(35.0)


def test_small_volume_multiple_trades():
    # 500 + 500 = 1000 total
    result = _model().compute_period_costs(_series(500.0, 500.0))
    assert result.sharesies_fee_band == "subscription_only"
    assert result.spread_cost_nzd == pytest.approx(20.0)
    assert result.total_cost_nzd == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# 3. Exact cap amount ($5000) → subscription only (coverage fully used)
# ---------------------------------------------------------------------------

def test_exact_cap_uses_flat_fee():
    result = _model().compute_period_costs(_series(5_000.0))
    assert result.sharesies_fee_band == "subscription_only"
    assert result.sharesies_fee_nzd == pytest.approx(15.0)
    assert result.spread_cost_nzd == pytest.approx(100.0)  # 5000 * 200/10000
    assert result.total_cost_nzd == pytest.approx(115.0)


# ---------------------------------------------------------------------------
# 4. Volume above coverage → subscription + 1.9% on EXCESS only
# ---------------------------------------------------------------------------

def test_above_cap_uses_percent_fee():
    # volume = 10_000 NZD; excess = 5_000
    # spread = 10000 * 200/10000 = 200
    # fee    = 15 (subscription) + 5000 * 190/10000 = 15 + 95 = 110
    result = _model().compute_period_costs(_series(10_000.0))
    assert result.sharesies_fee_band == "subscription_plus_excess"
    assert result.spread_cost_nzd == pytest.approx(200.0)
    assert result.sharesies_fee_nzd == pytest.approx(110.0)
    assert result.total_cost_nzd == pytest.approx(310.0)


def test_just_above_cap():
    result = _model().compute_period_costs(_series(5_000.01))
    assert result.sharesies_fee_band == "subscription_plus_excess"


# ---------------------------------------------------------------------------
# 5. total_cost == spread_cost + sharesies_fee (identity check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("volume", [100.0, 4_999.99, 5_000.0, 5_000.01, 50_000.0])
def test_total_equals_sum_of_parts(volume: float):
    result = _model().compute_period_costs(_series(volume))
    assert result.total_cost_nzd == pytest.approx(
        result.spread_cost_nzd + result.sharesies_fee_nzd
    )


# ---------------------------------------------------------------------------
# 6. Custom config works
# ---------------------------------------------------------------------------

def test_custom_config_spread():
    config = CostConfig(spread_bps=100.0)  # halved spread
    result = CostModel(config).compute_period_costs(_series(1_000.0))
    assert result.spread_cost_nzd == pytest.approx(10.0)   # 1000 * 100/10000
    assert result.sharesies_fee_nzd == pytest.approx(15.0)


def test_custom_config_cap_and_flat_fee():
    config = CostConfig(sharesies_coverage_nzd=2_000.0, sharesies_monthly_fee_nzd=10.0)
    # volume = 1500 → below custom coverage → subscription only
    result = CostModel(config).compute_period_costs(_series(1_500.0))
    assert result.sharesies_fee_band == "subscription_only"
    assert result.sharesies_fee_nzd == pytest.approx(10.0)


def test_custom_config_above_custom_cap():
    config = CostConfig(sharesies_coverage_nzd=2_000.0, sharesies_excess_bps=10.0)
    # volume = 3000, excess = 1000; fee = 15 + 1000 * 10/10000 = 15 + 1 = 16
    result = CostModel(config).compute_period_costs(_series(3_000.0))
    assert result.sharesies_fee_band == "subscription_plus_excess"
    assert result.sharesies_fee_nzd == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# 7. Negative trade values → abs() applied
# ---------------------------------------------------------------------------

def test_negative_trade_values_use_abs():
    # Sells represented as negative dollar amounts
    result_neg = _model().compute_period_costs(_series(-500.0, -500.0))
    result_pos = _model().compute_period_costs(_series(500.0, 500.0))
    assert result_neg.spread_cost_nzd == pytest.approx(result_pos.spread_cost_nzd)
    assert result_neg.sharesies_fee_nzd == pytest.approx(result_pos.sharesies_fee_nzd)
    assert result_neg.total_cost_nzd == pytest.approx(result_pos.total_cost_nzd)
    assert result_neg.sharesies_fee_band == result_pos.sharesies_fee_band


def test_mixed_sign_trades():
    # 1000 buy + (-1000) sell = 2000 abs volume; all within coverage
    result = _model().compute_period_costs(_series(1_000.0, -1_000.0))
    assert result.spread_cost_nzd == pytest.approx(40.0)   # 2000 * 200/10000
    assert result.sharesies_fee_band == "subscription_only"
    assert result.total_cost_nzd == pytest.approx(55.0)
