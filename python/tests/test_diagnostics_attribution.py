"""Tests for return attribution diagnostics module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import PreparedPanel
from skuld_research.diagnostics.attribution import AttributionReport, attribute_returns


def make_panel(
    tickers=("A", "B", "C", "D", "E"),
    n_months=24,
    asof=pd.Timestamp("2026-01-01"),
) -> PreparedPanel:
    dates_monthly = pd.date_range("2024-01-31", periods=n_months, freq="ME")
    dates_daily = pd.date_range("2024-01-01", periods=n_months * 21, freq="B")
    rng = np.random.default_rng(42)
    returns_m = pd.DataFrame(
        rng.normal(0.01, 0.05, (n_months, len(tickers))),
        index=dates_monthly,
        columns=list(tickers),
    )
    returns_d = pd.DataFrame(
        rng.normal(0.0005, 0.01, (len(dates_daily), len(tickers))),
        index=dates_daily,
        columns=list(tickers),
    )
    mc = pd.DataFrame(
        np.ones((n_months, len(tickers))) * 1e8,
        index=dates_monthly,
        columns=list(tickers),
    )
    sec = pd.Series({t: "Unknown" for t in tickers})
    mask = pd.DataFrame(True, index=dates_monthly, columns=list(tickers))
    return PreparedPanel(
        returns_daily=returns_d,
        returns_monthly=returns_m,
        market_cap=mc,
        sector=sec,
        universe_mask=mask,
        macro=pd.DataFrame(),
        asof=asof,
    )


def make_scores_panel(panel: PreparedPanel, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 1, (len(panel.universe_mask.index), len(panel.universe_mask.columns))),
        index=panel.universe_mask.index,
        columns=panel.universe_mask.columns,
    )


def make_production_returns(panel: PreparedPanel, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.01, 0.05, len(panel.returns_monthly.index)),
        index=panel.returns_monthly.index,
    )


def test_basic_attribution_runs():
    panel = make_panel()
    scores = make_scores_panel(panel)
    prod = make_production_returns(panel)
    result = attribute_returns(scores, panel, prod)
    assert isinstance(result, AttributionReport)


def test_market_proxy_length():
    panel = make_panel()
    scores = make_scores_panel(panel)
    prod = make_production_returns(panel)
    result = attribute_returns(scores, panel, prod)
    assert len(result.market_proxy_monthly) == len(panel.returns_monthly)


def test_signal_ew_positive_score():
    panel = make_panel()
    # Always give ticker "A" the highest score
    scores_data = np.zeros((len(panel.universe_mask.index), len(panel.universe_mask.columns)))
    scores_df = pd.DataFrame(
        scores_data,
        index=panel.universe_mask.index,
        columns=panel.universe_mask.columns,
    )
    scores_df["A"] = 10.0  # highest score always
    prod = make_production_returns(panel)
    result = attribute_returns(scores_df, panel, prod, top_frac=0.2)
    assert len(result.signal_ew_monthly) > 0
    assert not result.signal_ew_monthly.isin([float("inf"), float("-inf")]).any()


def test_total_alpha_equals_sum():
    panel = make_panel()
    scores = make_scores_panel(panel)
    prod = make_production_returns(panel)
    result = attribute_returns(scores, panel, prod)
    assert math.isclose(
        result.total_alpha_ann,
        result.signal_contribution_ann + result.construction_cost_drag_ann,
        abs_tol=1e-6,
    )


def test_cumulative_starts_at_one():
    """Cumulative series should start near 1.0 (first value = 1 + first_return)."""
    panel = make_panel()
    scores = make_scores_panel(panel)
    prod = make_production_returns(panel)
    result = attribute_returns(scores, panel, prod)
    # First value = (1 + first_return), should be close to 1.0 given small returns
    assert 0.5 < result.market_proxy_cumulative.iloc[0] < 2.0
    assert 0.5 < result.signal_ew_cumulative.iloc[0] < 2.0
    assert 0.5 < result.production_cumulative.iloc[0] < 2.0
    # Series should be non-decreasing in total (final > first roughly)
    assert result.market_proxy_cumulative.iloc[-1] > 0


def test_empty_production_returns():
    panel = make_panel()
    scores = make_scores_panel(panel)
    prod = pd.Series([], dtype=float)
    result = attribute_returns(scores, panel, prod)
    assert isinstance(result, AttributionReport)
    assert math.isnan(result.signal_contribution_ann)
    assert math.isnan(result.construction_cost_drag_ann)
    assert math.isnan(result.total_alpha_ann)


def test_top_frac_one():
    """With top_frac=1.0 and all tickers always in universe, signal_ew should equal market_proxy."""
    tickers = ("A", "B", "C", "D", "E")
    n_months = 24
    dates_monthly = pd.date_range("2024-01-31", periods=n_months, freq="ME")
    dates_daily = pd.date_range("2024-01-01", periods=n_months * 21, freq="B")
    rng = np.random.default_rng(42)
    returns_m = pd.DataFrame(
        rng.normal(0.01, 0.05, (n_months, len(tickers))),
        index=dates_monthly,
        columns=list(tickers),
    )
    returns_d = pd.DataFrame(
        rng.normal(0.0005, 0.01, (len(dates_daily), len(tickers))),
        index=dates_daily,
        columns=list(tickers),
    )
    mc = pd.DataFrame(np.ones((n_months, len(tickers))) * 1e8, index=dates_monthly, columns=list(tickers))
    sec = pd.Series({t: "Unknown" for t in tickers})
    mask = pd.DataFrame(True, index=dates_monthly, columns=list(tickers))
    panel = PreparedPanel(
        returns_daily=returns_d,
        returns_monthly=returns_m,
        market_cap=mc,
        sector=sec,
        universe_mask=mask,
        macro=pd.DataFrame(),
        asof=pd.Timestamp("2026-01-01"),
    )
    # All scores non-NaN for all tickers and all dates
    scores = pd.DataFrame(
        rng.normal(0, 1, (n_months, len(tickers))),
        index=dates_monthly,
        columns=list(tickers),
    )
    prod = make_production_returns(panel)
    result = attribute_returns(scores, panel, prod, top_frac=1.0)

    common = result.signal_ew_monthly.index.intersection(result.market_proxy_monthly.index)
    sig = result.signal_ew_monthly.reindex(common)
    mkt = result.market_proxy_monthly.reindex(common)
    assert abs(sig.mean() - mkt.mean()) < 1e-4


def test_scores_panel_all_nan():
    panel = make_panel()
    scores = pd.DataFrame(
        np.nan,
        index=panel.universe_mask.index,
        columns=panel.universe_mask.columns,
    )
    prod = make_production_returns(panel)
    result = attribute_returns(scores, panel, prod)
    assert isinstance(result, AttributionReport)
    # signal_ew_monthly should be empty or all NaN
    assert len(result.signal_ew_monthly) == 0 or result.signal_ew_monthly.isna().all()


def test_fewer_than_2_valid_scores():
    """When most rebalance dates have only 1 non-NaN score, signal_ew should be sparse/NaN."""
    panel = make_panel()
    # Build scores where only 1 ticker has a score (rest NaN)
    scores = pd.DataFrame(
        np.nan,
        index=panel.universe_mask.index,
        columns=panel.universe_mask.columns,
    )
    scores["A"] = 1.0  # only ticker A has a score — fewer than 2 valid scores per date
    prod = make_production_returns(panel)
    result = attribute_returns(scores, panel, prod)
    assert isinstance(result, AttributionReport)
    # signal_ew_monthly should be empty or very sparse (all skipped due to <2 valid scores)
    assert len(result.signal_ew_monthly) == 0 or result.signal_ew_monthly.isna().all()
