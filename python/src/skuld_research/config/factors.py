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
        else:
            raise ValueError(f"Unknown factor kind: {factor_spec.kind}")
    return factors
