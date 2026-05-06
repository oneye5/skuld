"""Tests for Phase 2 momentum-extension factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PITSnapshot, PreparedPanel


def _panel() -> PreparedPanel:
    dates = pd.bdate_range("2022-01-03", periods=420)
    rng = np.random.default_rng(42)
    market_ret = 0.0004 + 0.004 * rng.standard_normal(len(dates))
    market = 10.0 * (1.0 + market_ret).cumprod()
    trend = 10.0 * (1.0 + market_ret + 0.001).cumprod()
    choppy = 10.0 * (1.0 + market_ret).cumprod()
    choppy[120:180] *= 1.25
    choppy[180:240] *= 0.80
    prices = pd.DataFrame(
        {"FNZ.NZ": market, "TREND.NZ": trend, "CHOP.NZ": choppy},
        index=dates,
    )
    prices.index.name = "date"
    volumes = pd.DataFrame(1_000_000.0, index=dates, columns=prices.columns)
    volumes.index.name = "date"
    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"])
        ),
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(columns=["ticker", "ex_date", "type", "factor"]),
        asof=pd.Timestamp("2024-01-01"),
    )
    from skuld_research.data.prepared_panel import build_prepared_panel

    return build_prepared_panel(snap, nzx_only=True, rebalance_start="2022-01-01")


def test_high_52_week_uses_only_prices_before_t():
    from skuld_research.factors.phase2_momentum import High52WeekFactor

    panel = _panel()
    factor = High52WeekFactor(lookback_days=252, min_days=126)
    t = pd.Timestamp("2023-08-01")
    base = factor.score(panel, t, ["TREND.NZ", "CHOP.NZ"])

    changed_prices = panel.prices.copy()
    changed_prices.loc[changed_prices.index >= t, "CHOP.NZ"] *= 100.0
    changed = PreparedPanel(
        returns_daily=panel.returns_daily,
        returns_monthly=panel.returns_monthly,
        market_cap=panel.market_cap,
        sector=panel.sector,
        universe_mask=panel.universe_mask,
        macro=panel.macro,
        fundamentals=panel.fundamentals,
        asof=panel.asof,
        prices=changed_prices,
        corporate_actions=panel.corporate_actions,
        market_cap_proxy=panel.market_cap_proxy,
    )

    pd.testing.assert_series_equal(base, factor.score(changed, t, ["TREND.NZ", "CHOP.NZ"]))


def test_residual_momentum_scores_idiosyncratic_strength():
    from skuld_research.factors.phase2_momentum import ResidualMomentumFactor

    panel = _panel()
    scores = ResidualMomentumFactor(min_months=6).score(
        panel,
        pd.Timestamp("2023-08-01"),
        ["TREND.NZ", "CHOP.NZ"],
    )

    assert not scores.isna().any()
    assert scores["TREND.NZ"] > scores["CHOP.NZ"]


def test_momentum_vol_penalized_penalizes_choppy_path():
    from skuld_research.factors.phase2_momentum import MomentumVolPenalizedFactor

    panel = _panel()
    scores = MomentumVolPenalizedFactor(min_months=6, vol_penalty=1.0).score(
        panel,
        pd.Timestamp("2023-08-01"),
        ["TREND.NZ", "CHOP.NZ"],
    )

    assert not scores.isna().any()
    assert scores["TREND.NZ"] > scores["CHOP.NZ"]


def test_max_daily_return_avoidance_penalizes_lottery_spike():
    from skuld_research.factors.phase2_momentum import MaxDailyReturnAvoidanceFactor

    panel = _panel()
    changed_returns = panel.returns_daily.copy()
    spike_date = changed_returns.index[changed_returns.index < pd.Timestamp("2023-08-01")][-20]
    changed_returns.loc[spike_date, "CHOP.NZ"] = 0.25
    panel = PreparedPanel(
        returns_daily=changed_returns,
        returns_monthly=panel.returns_monthly,
        market_cap=panel.market_cap,
        sector=panel.sector,
        universe_mask=panel.universe_mask,
        macro=panel.macro,
        fundamentals=panel.fundamentals,
        asof=panel.asof,
        prices=panel.prices,
        corporate_actions=panel.corporate_actions,
        market_cap_proxy=panel.market_cap_proxy,
    )
    scores = MaxDailyReturnAvoidanceFactor(lookback_days=252, min_days=126).score(
        panel,
        pd.Timestamp("2023-08-01"),
        ["TREND.NZ", "CHOP.NZ"],
    )

    assert not scores.isna().any()
    assert scores["TREND.NZ"] > scores["CHOP.NZ"]


def test_reversal_adjusted_momentum_penalizes_positive_skip_month():
    from skuld_research.factors.phase2_momentum import ReversalAdjustedMomentumFactor

    panel = _panel()
    factor = ReversalAdjustedMomentumFactor(min_months=6, reversal_penalty=1.0)
    t = pd.Timestamp("2023-08-01")
    base = factor.score(panel, t, ["TREND.NZ", "CHOP.NZ"])

    changed_returns = panel.returns_monthly.copy()
    skip_month = changed_returns.index[changed_returns.index < t][-1]
    changed_returns.loc[skip_month, "CHOP.NZ"] = 0.5
    changed = PreparedPanel(
        returns_daily=panel.returns_daily,
        returns_monthly=changed_returns,
        market_cap=panel.market_cap,
        sector=panel.sector,
        universe_mask=panel.universe_mask,
        macro=panel.macro,
        fundamentals=panel.fundamentals,
        asof=panel.asof,
        prices=panel.prices,
        corporate_actions=panel.corporate_actions,
        market_cap_proxy=panel.market_cap_proxy,
    )

    after = factor.score(changed, t, ["TREND.NZ", "CHOP.NZ"])
    assert after["CHOP.NZ"] < base["CHOP.NZ"]
