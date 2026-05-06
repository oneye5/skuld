"""Tests for constructing factor generators from BacktestSpec factor configs."""

from __future__ import annotations

from skuld_research.config.spec import (
    BetaAdjustedMomentumFactorSpec,
    BookToMarketFactorSpec,
    DividendYieldFactorSpec,
    DualHorizonMomentumFactorSpec,
    High52WeekFactorSpec,
    LowVolatilityFactorSpec,
    MaxDailyReturnAvoidanceFactorSpec,
    MomentumAccelerationFactorSpec,
    MomentumConsistencyFactorSpec,
    MomentumDrawdownAwareFactorSpec,
    MomentumExShortSpikeFactorSpec,
    MomentumFactorSpec,
    MomentumVolPenalizedFactorSpec,
    OcfToAssetsFactorSpec,
    ResidualMomentumFactorSpec,
    ReversalAdjustedMomentumFactorSpec,
    SizeFactorSpec,
    TimeSeriesFilteredMomentumFactorSpec,
)


def test_build_factors_from_specs_preserves_configured_order_and_parameters():
    """Diagnostics and backtests should use the same configured factor set."""
    from skuld_research.config.factors import build_factors_from_specs
    from skuld_research.factors.book_to_market import BookToMarketFactor
    from skuld_research.factors.dividend_yield import DividendYieldFactor
    from skuld_research.factors.low_volatility import LowVolatilityFactor
    from skuld_research.factors.momentum import MomentumFactor
    from skuld_research.factors.ocf_to_assets import OcfToAssetsFactor
    from skuld_research.factors.size import SizeFactor

    factors = build_factors_from_specs(
        [
            MomentumFactorSpec(min_months=9, smoothing_months=3),
            DividendYieldFactorSpec(lookback_months=18, min_dividends=2),
            LowVolatilityFactorSpec(lookback_months=6, min_months=4),
            SizeFactorSpec(),
            BookToMarketFactorSpec(),
            OcfToAssetsFactorSpec(),
            ResidualMomentumFactorSpec(min_months=10, market_ticker="FNZ.NZ"),
            BetaAdjustedMomentumFactorSpec(min_months=10, market_ticker="FNZ.NZ"),
            MomentumVolPenalizedFactorSpec(min_months=10, vol_penalty=0.5),
            High52WeekFactorSpec(lookback_days=200, min_days=100),
            MomentumConsistencyFactorSpec(min_months=10, variant="hitrate"),
            MomentumDrawdownAwareFactorSpec(min_months=10, drawdown_penalty=0.5),
            DualHorizonMomentumFactorSpec(short_months=4, long_months=10, min_months=4),
            MomentumExShortSpikeFactorSpec(min_months=10, recent_months=2),
            TimeSeriesFilteredMomentumFactorSpec(min_months=10, ma_days=200),
            ReversalAdjustedMomentumFactorSpec(min_months=10, reversal_penalty=0.25),
            MaxDailyReturnAvoidanceFactorSpec(lookback_days=42, min_days=21),
            MomentumAccelerationFactorSpec(min_months=10),
        ]
    )

    assert [factor.name for factor in factors] == [
        "momentum",
        "dividend_yield",
        "low_volatility",
        "size",
        "book_to_market",
        "ocf_to_assets",
        "residual_momentum",
        "beta_adjusted_momentum",
        "momentum_vol_penalized",
        "high_52_week",
        "momentum_consistency",
        "momentum_drawdown_aware",
        "dual_horizon_momentum",
        "momentum_ex_short_spike",
        "time_series_filtered_momentum",
        "reversal_adjusted_momentum",
        "max_daily_return_avoidance",
        "momentum_acceleration",
    ]
    assert isinstance(factors[0], MomentumFactor)
    assert factors[0].min_months == 9
    assert factors[0].smoothing_months == 3
    assert isinstance(factors[1], DividendYieldFactor)
    assert factors[1].lookback_months == 18
    assert factors[1].min_dividends == 2
    assert isinstance(factors[2], LowVolatilityFactor)
    assert factors[2].lookback_months == 6
    assert factors[2].min_months == 4
    assert isinstance(factors[3], SizeFactor)
    assert isinstance(factors[4], BookToMarketFactor)
    assert isinstance(factors[5], OcfToAssetsFactor)
    assert factors[6].min_months == 10
    assert factors[7].min_months == 10
    assert factors[8].vol_penalty == 0.5
    assert factors[9].lookback_days == 200
    assert factors[10].variant == "hitrate"
    assert factors[11].drawdown_penalty == 0.5
    assert factors[12].short_months == 4
    assert factors[13].recent_months == 2
    assert factors[14].ma_days == 200
    assert factors[15].reversal_penalty == 0.25
    assert factors[16].lookback_days == 42
    assert factors[17].min_months == 10


def test_production_recommend_pipeline_imports_shared_factor_factory():
    """Production must share spec factor construction with research backtests."""
    import inspect

    import skuld_portfolio.pipeline.recommend as recommend_module

    source = inspect.getsource(recommend_module)

    assert "build_factors_from_specs" in source
    assert "MomentumFactor(min_months=factor_spec.min_months)" not in source
