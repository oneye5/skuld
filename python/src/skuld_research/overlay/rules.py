"""Overlay rules: evaluate market conditions and return desired cash fraction."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from skuld_common.contracts import PreparedPanel


class OverlayRule(Protocol):
    """Protocol for overlay rules.
    
    An overlay rule evaluates market conditions at a point in time and returns
    the desired cash fraction (0.0 to 1.0). The `apply_cash_overlay` function
    uses this to increase cash beyond the configured floor when conditions
    warrant defensive positioning.
    """
    
    def evaluate(self, panel: PreparedPanel, asof: pd.Timestamp) -> float:
        """Evaluate market conditions and return desired cash fraction.
        
        Args:
            panel: PreparedPanel with returns and macro data.
            asof: Rebalance date. Only data strictly before this date is used.
        
        Returns:
            Desired cash fraction in [0.0, 1.0].
        """
        ...


class NoOverlay:
    """Always returns 0.0 — cash stays at configured floor."""
    
    def evaluate(self, panel: PreparedPanel, asof: pd.Timestamp) -> float:
        """Return 0.0 (no overlay increase)."""
        return 0.0


class NzxMA200AndAggMomentumRule:
    """Dual-condition defensive trigger.
    
    Raises cash to `defensive_cash_fraction` when BOTH conditions are met:
    1. Equal-weighted market proxy (liquid universe) is below its 200-trading-day MA
    2. Cross-sectional aggregate momentum z-score < 0
    
    Otherwise returns 0.0 (cash stays at floor).
    
    The dual condition reduces false positives vs. a single-MA rule per
    Clare/Seaton/Smith/Thomas (2013) findings on small markets.
    """
    
    def __init__(
        self,
        defensive_cash_fraction: float = 0.30,
        momentum_aggregate_lookback_months: int = 12,
    ):
        """
        Args:
            defensive_cash_fraction: Cash fraction when both conditions trigger.
                Default 0.30 (30% cash).
            momentum_aggregate_lookback_months: Lookback for momentum z-score
                aggregation. Default 12 (matches the momentum factor).
        """
        if not (0.0 <= defensive_cash_fraction <= 1.0):
            raise ValueError(
                f"defensive_cash_fraction must be in [0, 1], got {defensive_cash_fraction}"
            )
        if momentum_aggregate_lookback_months < 1:
            raise ValueError(
                f"momentum_aggregate_lookback_months must be >= 1, "
                f"got {momentum_aggregate_lookback_months}"
            )
        
        self.defensive_cash_fraction = defensive_cash_fraction
        self.momentum_aggregate_lookback_months = momentum_aggregate_lookback_months
    
    def evaluate(self, panel: PreparedPanel, asof: pd.Timestamp) -> float:
        """Evaluate dual trigger and return cash fraction.
        
        Args:
            panel: PreparedPanel with returns_daily and returns_monthly.
            asof: Rebalance date.
        
        Returns:
            `defensive_cash_fraction` if both conditions met, else 0.0.
        """
        asof_naive = asof.tz_localize(None) if asof.tzinfo else asof
        
        # Condition 1: Market proxy below 200-day MA
        market_below_ma = self._market_below_200d_ma(panel, asof_naive)
        
        # Condition 2: Aggregate cross-sectional momentum z-score < 0
        momentum_negative = self._aggregate_momentum_negative(panel, asof_naive)
        
        if market_below_ma and momentum_negative:
            return self.defensive_cash_fraction
        else:
            return 0.0
    
    def _market_below_200d_ma(
        self, panel: PreparedPanel, asof_naive: pd.Timestamp
    ) -> bool:
        """Check if equal-weighted market proxy is below its 200-day MA.
        
        Uses liquid universe (tickers that were ever in universe_mask) as proxy,
        matching the regime labelling approach in stats/regimes.py.
        """
        # Build equal-weighted market proxy from liquid universe
        ever_in_universe = panel.universe_mask.any(axis=0)
        liquid_tickers = ever_in_universe[ever_in_universe].index.tolist()
        
        if not liquid_tickers:
            # No liquid universe → cannot evaluate → conservatively return False
            return False
        
        daily_ret = panel.returns_daily
        avail_dates = daily_ret.index[daily_ret.index < asof_naive]
        
        if len(avail_dates) < 200:
            # Insufficient history for 200-day MA → return False
            return False
        
        # Equal-weighted daily returns
        market_ret = daily_ret.loc[avail_dates, liquid_tickers].mean(axis=1)
        
        # Cumulative index (start = 100)
        market_idx = 100.0 * (1.0 + market_ret).cumprod()
        
        # 200-day MA of the index
        ma_200 = market_idx.rolling(window=200, min_periods=200).mean()
        
        # Latest available values strictly before asof
        latest_idx = market_idx.iloc[-1]
        latest_ma = ma_200.iloc[-1]
        
        if pd.isna(latest_ma):
            return False
        
        return latest_idx < latest_ma
    
    def _aggregate_momentum_negative(
        self, panel: PreparedPanel, asof_naive: pd.Timestamp
    ) -> bool:
        """Check if cross-sectional mean of momentum z-scores < 0.
        
        Computes 12-1 month momentum for the liquid universe, z-scores them,
        and returns True if the mean z-score < 0.
        """
        from skuld_research.factors.momentum import MomentumFactor
        
        # Get liquid universe at this rebalance date
        if asof_naive not in panel.universe_mask.index:
            # No universe defined for this date → conservatively return False
            return False
        
        universe_at_t = panel.universe_mask.loc[asof_naive]
        liquid_tickers = universe_at_t[universe_at_t].index.tolist()
        
        if len(liquid_tickers) < 2:
            # Too few tickers to compute meaningful cross-sectional stats
            return False
        
        # Compute momentum scores using the same factor as the strategy
        momentum = MomentumFactor(min_months=self.momentum_aggregate_lookback_months - 1)
        scores = momentum.score(panel, pd.Timestamp(asof_naive), liquid_tickers)
        
        # Drop NaN scores
        valid_scores = scores.dropna()
        
        if len(valid_scores) < 2:
            return False
        
        # Cross-sectional z-score (mean should be ~0, std ~1 for large enough sample)
        mean = valid_scores.mean()
        std = valid_scores.std(ddof=1)
        
        if std < 1e-12:
            # Degenerate — all scores identical
            return False
        
        z_scores = (valid_scores - mean) / std
        
        # Aggregate = mean of the z-scores
        aggregate_z = z_scores.mean()
        
        return aggregate_z < 0.0
