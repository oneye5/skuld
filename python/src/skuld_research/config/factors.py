"""Construct configured factor signal generators."""

from __future__ import annotations

from skuld_research.config.spec import FactorSpec
from skuld_research.factors.momentum import MomentumFactor
from skuld_research.factors.protocols import SignalGenerator


def build_factors_from_specs(factor_specs: list[FactorSpec]) -> list[SignalGenerator]:
    """Build factor generators from pre-registered factor specs."""
    factors: list[SignalGenerator] = []
    for factor_spec in factor_specs:
        if factor_spec.kind == "momentum":
            factors.append(
                MomentumFactor(
                    min_months=factor_spec.min_months,
                    smoothing_months=factor_spec.smoothing_months,
                )
            )
        elif factor_spec.kind == "low_vol":
            from skuld_research.factors.low_volatility import LowVolatilityFactor

            factors.append(
                LowVolatilityFactor(
                    lookback_months=factor_spec.lookback_months,
                    min_months=factor_spec.min_months,
                )
            )
        elif factor_spec.kind == "size":
            from skuld_research.factors.size import SizeFactor

            factors.append(SizeFactor())
        elif factor_spec.kind == "dividend_yield":
            from skuld_research.factors.dividend_yield import DividendYieldFactor

            factors.append(
                DividendYieldFactor(
                    lookback_months=factor_spec.lookback_months,
                    min_dividends=factor_spec.min_dividends,
                )
            )
        elif factor_spec.kind == "return_on_risk":
            from skuld_research.factors.return_on_risk import ReturnOnRiskFactor

            factors.append(
                ReturnOnRiskFactor(
                    lookback_months=factor_spec.lookback_months,
                    min_months=factor_spec.min_months,
                )
            )
        elif factor_spec.kind == "book_to_market":
            from skuld_research.factors.book_to_market import BookToMarketFactor

            factors.append(BookToMarketFactor())
        elif factor_spec.kind == "ocf_to_assets":
            from skuld_research.factors.ocf_to_assets import OcfToAssetsFactor

            factors.append(OcfToAssetsFactor())
        elif factor_spec.kind == "residual_momentum":
            from skuld_research.factors.phase2_momentum import ResidualMomentumFactor

            factors.append(
                ResidualMomentumFactor(
                    min_months=factor_spec.min_months,
                    market_ticker=factor_spec.market_ticker,
                )
            )
        elif factor_spec.kind == "beta_adjusted_momentum":
            from skuld_research.factors.phase2_momentum import BetaAdjustedMomentumFactor

            factors.append(
                BetaAdjustedMomentumFactor(
                    min_months=factor_spec.min_months,
                    market_ticker=factor_spec.market_ticker,
                )
            )
        elif factor_spec.kind == "momentum_vol_penalized":
            from skuld_research.factors.phase2_momentum import MomentumVolPenalizedFactor

            factors.append(
                MomentumVolPenalizedFactor(
                    min_months=factor_spec.min_months,
                    vol_lookback_months=factor_spec.vol_lookback_months,
                    vol_penalty=factor_spec.vol_penalty,
                )
            )
        elif factor_spec.kind == "high_52_week":
            from skuld_research.factors.phase2_momentum import High52WeekFactor

            factors.append(
                High52WeekFactor(
                    lookback_days=factor_spec.lookback_days,
                    min_days=factor_spec.min_days,
                )
            )
        elif factor_spec.kind == "momentum_consistency":
            from skuld_research.factors.phase2_momentum import MomentumConsistencyFactor

            factors.append(
                MomentumConsistencyFactor(
                    min_months=factor_spec.min_months,
                    variant=factor_spec.variant,
                )
            )
        elif factor_spec.kind == "momentum_drawdown_aware":
            from skuld_research.factors.phase2_momentum import MomentumDrawdownAwareFactor

            factors.append(
                MomentumDrawdownAwareFactor(
                    min_months=factor_spec.min_months,
                    drawdown_penalty=factor_spec.drawdown_penalty,
                )
            )
        elif factor_spec.kind == "dual_horizon_momentum":
            from skuld_research.factors.phase2_momentum import DualHorizonMomentumFactor

            factors.append(
                DualHorizonMomentumFactor(
                    short_months=factor_spec.short_months,
                    long_months=factor_spec.long_months,
                    min_months=factor_spec.min_months,
                )
            )
        elif factor_spec.kind == "momentum_ex_short_spike":
            from skuld_research.factors.phase2_momentum import MomentumExShortSpikeFactor

            factors.append(
                MomentumExShortSpikeFactor(
                    min_months=factor_spec.min_months,
                    recent_months=factor_spec.recent_months,
                    recent_penalty=factor_spec.recent_penalty,
                )
            )
        elif factor_spec.kind == "time_series_filtered_momentum":
            from skuld_research.factors.phase2_momentum import TimeSeriesFilteredMomentumFactor

            factors.append(
                TimeSeriesFilteredMomentumFactor(
                    min_months=factor_spec.min_months,
                    ma_days=factor_spec.ma_days,
                    downtrend_discount=factor_spec.downtrend_discount,
                )
            )
        elif factor_spec.kind == "reversal_adjusted_momentum":
            from skuld_research.factors.phase2_momentum import ReversalAdjustedMomentumFactor

            factors.append(
                ReversalAdjustedMomentumFactor(
                    min_months=factor_spec.min_months,
                    reversal_penalty=factor_spec.reversal_penalty,
                )
            )
        elif factor_spec.kind == "max_daily_return_avoidance":
            from skuld_research.factors.phase2_momentum import MaxDailyReturnAvoidanceFactor

            factors.append(
                MaxDailyReturnAvoidanceFactor(
                    lookback_days=factor_spec.lookback_days,
                    min_days=factor_spec.min_days,
                )
            )
        elif factor_spec.kind == "momentum_acceleration":
            from skuld_research.factors.phase2_momentum import MomentumAccelerationFactor

            factors.append(MomentumAccelerationFactor(min_months=factor_spec.min_months))
        elif factor_spec.kind == "eps_momentum":
            from skuld_research.factors.eps_momentum import EpsMomentumFactor

            factors.append(EpsMomentumFactor())
        elif factor_spec.kind == "volume_trend":
            from skuld_research.factors.volume_trend import VolumeTrendFactor

            factors.append(
                VolumeTrendFactor(
                    short_days=factor_spec.short_days,
                    long_days=factor_spec.long_days,
                    min_trading_days=factor_spec.min_trading_days,
                )
            )
        else:
            raise ValueError(f"Unknown factor kind: {factor_spec.kind}")
    return factors
