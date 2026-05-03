"""Tests for factor IC comparison module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import PreparedPanel
from skuld_research.diagnostics.factor_comparison import (
    FactorComparisonReport,
    compare_factors,
)


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


class ConstantFactor:
    """Always returns a fixed score series (for testing)."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def score(self, panel, date, universe):
        return pd.Series({t: self._scores.get(t, float("nan")) for t in universe})


def test_basic_comparison_runs():
    panel = make_panel()
    tickers = list(panel.universe_mask.columns)
    scores_a = {t: float(i) for i, t in enumerate(tickers)}
    scores_b = {t: float(len(tickers) - i) for i, t in enumerate(tickers)}
    factors = {"alpha": ConstantFactor(scores_a), "beta": ConstantFactor(scores_b)}
    result = compare_factors(factors, panel, min_cross_section=3)
    assert isinstance(result, FactorComparisonReport)


def test_factor_names_preserved():
    panel = make_panel()
    tickers = list(panel.universe_mask.columns)
    scores = {t: float(i) for i, t in enumerate(tickers)}
    factors = {"z_factor": ConstantFactor(scores), "a_factor": ConstantFactor(scores)}
    result = compare_factors(factors, panel, min_cross_section=3)
    assert result.factor_names == ("z_factor", "a_factor")


def test_ic_series_corr_shape():
    panel = make_panel()
    tickers = list(panel.universe_mask.columns)
    scores = {t: float(i) for i, t in enumerate(tickers)}
    factors = {"f1": ConstantFactor(scores), "f2": ConstantFactor(scores)}
    result = compare_factors(factors, panel, min_cross_section=3)
    corr = result.ic_series_corr
    assert corr.shape == (2, 2)
    assert list(corr.index) == ["f1", "f2"]
    assert list(corr.columns) == ["f1", "f2"]
    # Diagonal should be 1.0 (or NaN if no obs)
    for name in ["f1", "f2"]:
        val = corr.loc[name, name]
        assert val == 1.0 or np.isnan(val)


def test_no_redundant_pairs_uncorrelated():
    panel = make_panel()
    tickers = list(panel.universe_mask.columns)
    rng = np.random.default_rng(0)
    scores_a = {t: rng.standard_normal() for t in tickers}
    scores_b = {t: rng.standard_normal() for t in tickers}
    factors = {"f1": ConstantFactor(scores_a), "f2": ConstantFactor(scores_b)}
    result = compare_factors(factors, panel, min_cross_section=3, redundancy_threshold=0.99)
    assert result.redundant_pairs == ()


def test_redundant_pair_detected():
    panel = make_panel()
    tickers = list(panel.universe_mask.columns)
    scores = {t: float(i) for i, t in enumerate(tickers)}
    # Identical factors → IC series should be perfectly correlated
    factors = {"f1": ConstantFactor(scores), "f2": ConstantFactor(scores)}
    result = compare_factors(factors, panel, min_cross_section=3, redundancy_threshold=0.5)
    assert len(result.redundant_pairs) >= 1
    assert ("f1", "f2") in result.redundant_pairs


def test_single_factor():
    panel = make_panel()
    tickers = list(panel.universe_mask.columns)
    scores = {t: float(i) for i, t in enumerate(tickers)}
    factors = {"only": ConstantFactor(scores)}
    result = compare_factors(factors, panel, min_cross_section=3)
    assert result.redundant_pairs == ()
    assert result.ic_series_corr.shape == (1, 1)


def test_empty_factors():
    panel = make_panel()
    result = compare_factors({}, panel, min_cross_section=3)
    assert isinstance(result, FactorComparisonReport)
    assert result.factor_names == ()
    assert result.ic_reports == {}
    assert result.decay_reports == {}
    assert result.redundant_pairs == ()


def test_min_cross_section_respected():
    # Only 2 tickers, min_cross_section=3 → most IC obs dropped
    panel = make_panel(tickers=("A", "B"))
    scores = {"A": 1.0, "B": 2.0}
    factors = {"f1": ConstantFactor(scores)}
    result = compare_factors(factors, panel, min_cross_section=3)
    ic_rep = result.ic_reports["f1"]
    # With only 2 tickers, cross-section < min_cross_section → n_obs=0 or ic_mean is NaN
    assert ic_rep.n_obs == 0 or np.isnan(ic_rep.ic_mean)
