"""Tests for constructing factor generators from BacktestSpec factor configs."""

from __future__ import annotations

from skuld_research.config.spec import (
    DividendYieldFactorSpec,
    LowVolatilityFactorSpec,
    MomentumFactorSpec,
    SizeFactorSpec,
)


def test_build_factors_from_specs_preserves_configured_order_and_parameters():
    """Diagnostics and backtests should use the same configured factor set."""
    from skuld_research.config.factors import build_factors_from_specs
    from skuld_research.factors.dividend_yield import DividendYieldFactor
    from skuld_research.factors.low_volatility import LowVolatilityFactor
    from skuld_research.factors.momentum import MomentumFactor
    from skuld_research.factors.size import SizeFactor

    factors = build_factors_from_specs(
        [
            MomentumFactorSpec(min_months=9, smoothing_months=3),
            DividendYieldFactorSpec(lookback_months=18, min_dividends=2),
            LowVolatilityFactorSpec(lookback_months=6, min_months=4),
            SizeFactorSpec(),
        ]
    )

    assert [factor.name for factor in factors] == [
        "momentum",
        "dividend_yield",
        "low_volatility",
        "size",
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


def test_production_recommend_pipeline_imports_shared_factor_factory():
    """Production must share spec factor construction with research backtests."""
    import inspect

    import skuld_portfolio.pipeline.recommend as recommend_module

    source = inspect.getsource(recommend_module)

    assert "build_factors_from_specs" in source
    assert "MomentumFactor(min_months=factor_spec.min_months)" not in source
