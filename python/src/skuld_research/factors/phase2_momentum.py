"""PIT-safe momentum extensions for the Phase 2 exploration funnel."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel


class ResidualMomentumFactor:
    """Momentum of returns residualised against a market proxy."""

    name: str = "residual_momentum"

    def __init__(self, min_months: int = 11, market_ticker: str = "FNZ.NZ") -> None:
        self.min_months = min_months
        self.market_ticker = market_ticker

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        monthly = panel.returns_monthly
        t_naive = _naive(t)
        signal_idx = _monthly_signal_index(monthly, t_naive, 12)
        if len(signal_idx) < self.min_months or self.market_ticker not in monthly.columns:
            return _nan_series(universe, self.name)

        available = [ticker for ticker in universe if ticker in monthly.columns]
        if not available:
            return _nan_series(universe, self.name)

        market = monthly.loc[signal_idx, self.market_ticker]
        out: dict[str, float] = {}
        for ticker in available:
            y = monthly.loc[signal_idx, ticker]
            aligned = pd.concat([y, market], axis=1, keys=["asset", "market"]).dropna()
            if len(aligned) < self.min_months or aligned["market"].var(ddof=1) <= 1e-12:
                out[ticker] = np.nan
                continue
            beta = aligned["asset"].cov(aligned["market"]) / aligned["market"].var(ddof=1)
            resid = aligned["asset"] - float(beta) * aligned["market"]
            out[ticker] = float((1.0 + resid).prod() - 1.0)
        return pd.Series(out, dtype=float, name=self.name).reindex(universe)


class BetaAdjustedMomentumFactor:
    """Raw momentum minus the market-beta component over the same window."""

    name: str = "beta_adjusted_momentum"

    def __init__(self, min_months: int = 11, market_ticker: str = "FNZ.NZ") -> None:
        self.min_months = min_months
        self.market_ticker = market_ticker

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        monthly = panel.returns_monthly
        t_naive = _naive(t)
        signal_idx = _monthly_signal_index(monthly, t_naive, 12)
        if len(signal_idx) < self.min_months or self.market_ticker not in monthly.columns:
            return _nan_series(universe, self.name)

        available = [ticker for ticker in universe if ticker in monthly.columns]
        if not available:
            return _nan_series(universe, self.name)

        market = monthly.loc[signal_idx, self.market_ticker]
        out: dict[str, float] = {}
        for ticker in available:
            asset = monthly.loc[signal_idx, ticker]
            aligned = pd.concat([asset, market], axis=1, keys=["asset", "market"]).dropna()
            if len(aligned) < self.min_months or aligned["market"].var(ddof=1) <= 1e-12:
                out[ticker] = np.nan
                continue
            beta = aligned["asset"].cov(aligned["market"]) / aligned["market"].var(ddof=1)
            raw_cum = float((1.0 + aligned["asset"]).prod() - 1.0)
            market_cum = float((1.0 + aligned["market"]).prod() - 1.0)
            out[ticker] = raw_cum - float(beta) * market_cum
        return pd.Series(out, dtype=float, name=self.name).reindex(universe)


class MomentumVolPenalizedFactor:
    """Momentum score penalised by realised daily volatility."""

    name: str = "momentum_vol_penalized"

    def __init__(
        self,
        min_months: int = 11,
        vol_lookback_months: int = 12,
        vol_penalty: float = 1.0,
    ) -> None:
        self.min_months = min_months
        self.vol_lookback_months = vol_lookback_months
        self.vol_penalty = vol_penalty

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        momentum = _raw_momentum(panel, t, universe, lookback_months=12, min_months=self.min_months)
        vol = _daily_vol(panel, t, universe, self.vol_lookback_months, self.min_months)
        scores = momentum - self.vol_penalty * vol
        scores.name = self.name
        return scores.reindex(universe)


class High52WeekFactor:
    """Current adjusted price proximity to the trailing 52-week high."""

    name: str = "high_52_week"

    def __init__(self, lookback_days: int = 252, min_days: int = 126) -> None:
        self.lookback_days = lookback_days
        self.min_days = min_days

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        prices = panel.prices
        t_naive = _naive(t)
        avail_idx = prices.index[prices.index < t_naive]
        if len(avail_idx) < self.min_days:
            return _nan_series(universe, self.name)

        cols = [c for c in universe if c in prices.columns]
        window = prices.loc[avail_idx[-self.lookback_days:], cols]
        valid_counts = window.notna().sum()
        highs = window.max()
        last = window.ffill().iloc[-1]
        scores = (last / highs).where((valid_counts >= self.min_days) & (highs > 0.0))
        scores.name = self.name
        return scores.reindex(universe)


class MomentumConsistencyFactor:
    """Path-quality score using monthly return information ratio or hit rate."""

    name: str = "momentum_consistency"

    def __init__(self, min_months: int = 11, variant: Literal["ir", "hitrate"] = "ir") -> None:
        self.min_months = min_months
        self.variant = variant

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        monthly = panel.returns_monthly
        t_naive = _naive(t)
        signal_idx = _monthly_signal_index(monthly, t_naive, 12)
        if len(signal_idx) < self.min_months:
            return _nan_series(universe, self.name)

        window = monthly.loc[signal_idx, [c for c in universe if c in monthly.columns]]
        counts = window.notna().sum()
        if self.variant == "hitrate":
            scores = (window > 0.0).sum() / counts.replace(0, np.nan)
        else:
            scores = window.mean() / window.std(ddof=1).where(window.std(ddof=1) > 1e-12)
        scores = scores.where(counts >= self.min_months)
        scores.name = self.name
        return scores.reindex(universe)


class MomentumDrawdownAwareFactor:
    """Momentum with a penalty for maximum drawdown during the lookback path."""

    name: str = "momentum_drawdown_aware"

    def __init__(self, min_months: int = 11, drawdown_penalty: float = 1.0) -> None:
        self.min_months = min_months
        self.drawdown_penalty = drawdown_penalty

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        momentum = _raw_momentum(panel, t, universe, lookback_months=12, min_months=self.min_months)
        drawdown = _daily_max_drawdown(
            panel,
            t,
            universe,
            lookback_days=252,
            min_days=self.min_months * 21,
        )
        scores = momentum - self.drawdown_penalty * drawdown.abs()
        scores.name = self.name
        return scores.reindex(universe)


class DualHorizonMomentumFactor:
    """Equal-weight blend of medium- and long-horizon momentum."""

    name: str = "dual_horizon_momentum"

    def __init__(self, short_months: int = 6, long_months: int = 12, min_months: int = 6) -> None:
        self.short_months = short_months
        self.long_months = long_months
        self.min_months = min_months

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        short = _raw_momentum(
            panel,
            t,
            universe,
            lookback_months=self.short_months,
            min_months=min(self.min_months, self.short_months),
        )
        long = _raw_momentum(
            panel,
            t,
            universe,
            lookback_months=self.long_months,
            min_months=self.min_months,
        )
        scores = (short + long) / 2.0
        scores.name = self.name
        return scores.reindex(universe)


class MomentumExShortSpikeFactor:
    """Momentum that penalises dependence on the latest non-skip three months."""

    name: str = "momentum_ex_short_spike"

    def __init__(
        self,
        min_months: int = 11,
        recent_months: int = 3,
        recent_penalty: float = 1.0,
    ) -> None:
        self.min_months = min_months
        self.recent_months = recent_months
        self.recent_penalty = recent_penalty

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        monthly = panel.returns_monthly
        t_naive = _naive(t)
        signal_idx = _monthly_signal_index(monthly, t_naive, 12)
        if len(signal_idx) < self.min_months:
            return _nan_series(universe, self.name)

        cols = [c for c in universe if c in monthly.columns]
        window = monthly.loc[signal_idx, cols]
        counts = window.notna().sum()
        raw = (1.0 + window.fillna(0.0)).prod() - 1.0
        recent = (1.0 + window.tail(self.recent_months).fillna(0.0)).prod() - 1.0
        scores = (raw - self.recent_penalty * recent.clip(lower=0.0)).where(
            counts >= self.min_months
        )
        scores.name = self.name
        return scores.reindex(universe)


class TimeSeriesFilteredMomentumFactor:
    """Momentum discounted when the stock is below its own moving average."""

    name: str = "time_series_filtered_momentum"

    def __init__(
        self,
        min_months: int = 11,
        ma_days: int = 252,
        downtrend_discount: float = 0.0,
    ) -> None:
        self.min_months = min_months
        self.ma_days = ma_days
        self.downtrend_discount = downtrend_discount

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        momentum = _raw_momentum(panel, t, universe, lookback_months=12, min_months=self.min_months)
        prices = panel.prices
        t_naive = _naive(t)
        avail_idx = prices.index[prices.index < t_naive]
        if len(avail_idx) < self.ma_days:
            return _nan_series(universe, self.name)
        window = prices.loc[avail_idx[-self.ma_days:], [c for c in universe if c in prices.columns]]
        last = window.ffill().iloc[-1]
        ma = window.mean()
        discount = pd.Series(1.0, index=window.columns).where(last >= ma, self.downtrend_discount)
        scores = momentum * discount.reindex(universe)
        scores.name = self.name
        return scores.reindex(universe)


class ReversalAdjustedMomentumFactor:
    """Momentum penalised by the skipped one-month return when it is positive."""

    name: str = "reversal_adjusted_momentum"

    def __init__(self, min_months: int = 11, reversal_penalty: float = 0.5) -> None:
        self.min_months = min_months
        self.reversal_penalty = reversal_penalty

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        momentum = _raw_momentum(
            panel,
            t,
            universe,
            lookback_months=12,
            min_months=self.min_months,
        )
        skip_return = _skip_month_return(panel, t, universe)
        scores = momentum - self.reversal_penalty * skip_return.clip(lower=0.0)
        scores.name = self.name
        return scores.reindex(universe)


class MaxDailyReturnAvoidanceFactor:
    """Lottery/MAX avoidance score: lower recent maximum daily return is better."""

    name: str = "max_daily_return_avoidance"

    def __init__(self, lookback_days: int = 63, min_days: int = 42) -> None:
        self.lookback_days = lookback_days
        self.min_days = min_days

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        daily = panel.returns_daily
        t_naive = _naive(t)
        avail_idx = daily.index[daily.index < t_naive]
        if len(avail_idx) < self.min_days:
            return _nan_series(universe, self.name)

        cols = [c for c in universe if c in daily.columns]
        window = daily.loc[avail_idx[-self.lookback_days:], cols]
        counts = window.notna().sum()
        scores = -window.max().where(counts >= self.min_days)
        scores.name = self.name
        return scores.reindex(universe)


class MomentumAccelerationFactor:
    """Formation-shape score: recent six-month momentum minus older six-month momentum."""

    name: str = "momentum_acceleration"

    def __init__(self, min_months: int = 10) -> None:
        self.min_months = min_months

    def score(self, panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
        monthly = panel.returns_monthly
        t_naive = _naive(t)
        signal_idx = _monthly_signal_index(monthly, t_naive, 12)
        if len(signal_idx) < self.min_months:
            return _nan_series(universe, self.name)

        cols = [c for c in universe if c in monthly.columns]
        window = monthly.loc[signal_idx, cols]
        counts = window.notna().sum()
        older = (1.0 + window.iloc[:6].fillna(0.0)).prod() - 1.0
        recent = (1.0 + window.iloc[-6:].fillna(0.0)).prod() - 1.0
        scores = (recent - older).where(counts >= self.min_months)
        scores.name = self.name
        return scores.reindex(universe)


def _raw_momentum(
    panel: PreparedPanel,
    t: pd.Timestamp,
    universe: list[str],
    *,
    lookback_months: int,
    min_months: int,
) -> pd.Series:
    monthly = panel.returns_monthly
    t_naive = _naive(t)
    signal_idx = _monthly_signal_index(monthly, t_naive, lookback_months)
    if len(signal_idx) < min_months:
        return _nan_series(universe, "raw_momentum")

    cols = [c for c in universe if c in monthly.columns]
    if not cols:
        return _nan_series(universe, "raw_momentum")
    window = monthly.loc[signal_idx, cols]
    counts = window.notna().sum()
    scores = ((1.0 + window.fillna(0.0)).prod() - 1.0).where(counts >= min_months)
    scores.name = "raw_momentum"
    return scores.reindex(universe)


def _skip_month_return(panel: PreparedPanel, t: pd.Timestamp, universe: list[str]) -> pd.Series:
    monthly = panel.returns_monthly
    t_naive = _naive(t)
    avail_idx = monthly.index[monthly.index < t_naive]
    if len(avail_idx) < 1:
        return _nan_series(universe, "skip_month_return")
    cols = [c for c in universe if c in monthly.columns]
    scores = monthly.loc[avail_idx[-1], cols]
    scores.name = "skip_month_return"
    return scores.reindex(universe)


def _daily_vol(
    panel: PreparedPanel,
    t: pd.Timestamp,
    universe: list[str],
    lookback_months: int,
    min_months: int,
) -> pd.Series:
    daily = panel.returns_daily
    t_naive = _naive(t)
    avail_idx = daily.index[daily.index < t_naive]
    min_days = min_months * 21
    if len(avail_idx) < min_days:
        return _nan_series(universe, "daily_vol")
    cols = [c for c in universe if c in daily.columns]
    window = daily.loc[avail_idx[-lookback_months * 21:], cols]
    counts = window.notna().sum()
    vol = window.std(ddof=1) * np.sqrt(252.0)
    vol = vol.where(counts >= min_days)
    vol.name = "daily_vol"
    return vol.reindex(universe)


def _daily_max_drawdown(
    panel: PreparedPanel,
    t: pd.Timestamp,
    universe: list[str],
    lookback_days: int,
    min_days: int,
) -> pd.Series:
    prices = panel.prices
    t_naive = _naive(t)
    avail_idx = prices.index[prices.index < t_naive]
    if len(avail_idx) < min_days:
        return _nan_series(universe, "max_drawdown")
    cols = [c for c in universe if c in prices.columns]
    window = prices.loc[avail_idx[-lookback_days:], cols]
    counts = window.notna().sum()
    running_max = window.ffill().cummax()
    drawdowns = window / running_max - 1.0
    max_dd = drawdowns.min().where(counts >= min_days)
    max_dd.name = "max_drawdown"
    return max_dd.reindex(universe)


def _monthly_signal_index(
    monthly: pd.DataFrame,
    t_naive: pd.Timestamp,
    lookback_months: int,
) -> pd.DatetimeIndex:
    avail_idx = monthly.index[monthly.index < t_naive]
    if len(avail_idx) < 2:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(avail_idx[:-1][-lookback_months:])


def _naive(t: pd.Timestamp) -> pd.Timestamp:
    return t.tz_localize(None) if t.tzinfo else t


def _nan_series(universe: list[str], name: str) -> pd.Series:
    return pd.Series(np.nan, index=universe, dtype=float, name=name)
