"""Strategy spec config system for reproducible backtests."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skuld_research.config.hashing import short_hash, spec_hash
    from skuld_research.config.loader import (
        SpecValidationError,
        find_python_root,
        iter_preregistered_specs,
        iter_strategy_specs,
        load_spec,
    )
    from skuld_research.config.runner import RunResult, run_from_spec
    from skuld_research.config.spec import (
        BacktestEngineSpec,
        BacktestSpec,
        BenchmarksSpec,
        BetaAdjustedMomentumFactorSpec,
        BookToMarketFactorSpec,
        CostSpec,
        DividendYieldFactorSpec,
        DualHorizonMomentumFactorSpec,
        FactorSpec,
        GatingSpec,
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
        OutputSpec,
        ResidualMomentumFactorSpec,
        ReturnOnRiskFactorSpec,
        ReversalAdjustedMomentumFactorSpec,
        RollingDriverSpec,
        ScrubbingSpec,
        SizeFactorSpec,
        SurvivorshipSpec,
        TimeSeriesFilteredMomentumFactorSpec,
        UniverseSpec,
        WalkForwardSpec,
    )

_EXPORTS = {
    "BacktestSpec": ("skuld_research.config.spec", "BacktestSpec"),
    "UniverseSpec": ("skuld_research.config.spec", "UniverseSpec"),
    "FactorSpec": ("skuld_research.config.spec", "FactorSpec"),
    "BookToMarketFactorSpec": ("skuld_research.config.spec", "BookToMarketFactorSpec"),
    "OcfToAssetsFactorSpec": ("skuld_research.config.spec", "OcfToAssetsFactorSpec"),
    "MomentumFactorSpec": ("skuld_research.config.spec", "MomentumFactorSpec"),
    "ResidualMomentumFactorSpec": (
        "skuld_research.config.spec",
        "ResidualMomentumFactorSpec",
    ),
    "BetaAdjustedMomentumFactorSpec": (
        "skuld_research.config.spec",
        "BetaAdjustedMomentumFactorSpec",
    ),
    "MomentumVolPenalizedFactorSpec": (
        "skuld_research.config.spec",
        "MomentumVolPenalizedFactorSpec",
    ),
    "High52WeekFactorSpec": ("skuld_research.config.spec", "High52WeekFactorSpec"),
    "MomentumConsistencyFactorSpec": (
        "skuld_research.config.spec",
        "MomentumConsistencyFactorSpec",
    ),
    "MomentumDrawdownAwareFactorSpec": (
        "skuld_research.config.spec",
        "MomentumDrawdownAwareFactorSpec",
    ),
    "DualHorizonMomentumFactorSpec": (
        "skuld_research.config.spec",
        "DualHorizonMomentumFactorSpec",
    ),
    "MomentumExShortSpikeFactorSpec": (
        "skuld_research.config.spec",
        "MomentumExShortSpikeFactorSpec",
    ),
    "TimeSeriesFilteredMomentumFactorSpec": (
        "skuld_research.config.spec",
        "TimeSeriesFilteredMomentumFactorSpec",
    ),
    "ReversalAdjustedMomentumFactorSpec": (
        "skuld_research.config.spec",
        "ReversalAdjustedMomentumFactorSpec",
    ),
    "MaxDailyReturnAvoidanceFactorSpec": (
        "skuld_research.config.spec",
        "MaxDailyReturnAvoidanceFactorSpec",
    ),
    "MomentumAccelerationFactorSpec": (
        "skuld_research.config.spec",
        "MomentumAccelerationFactorSpec",
    ),
    "LowVolatilityFactorSpec": (
        "skuld_research.config.spec",
        "LowVolatilityFactorSpec",
    ),
    "SizeFactorSpec": ("skuld_research.config.spec", "SizeFactorSpec"),
    "DividendYieldFactorSpec": (
        "skuld_research.config.spec",
        "DividendYieldFactorSpec",
    ),
    "ReturnOnRiskFactorSpec": (
        "skuld_research.config.spec",
        "ReturnOnRiskFactorSpec",
    ),
    "CostSpec": ("skuld_research.config.spec", "CostSpec"),
    "BacktestEngineSpec": ("skuld_research.config.spec", "BacktestEngineSpec"),
    "WalkForwardSpec": ("skuld_research.config.spec", "WalkForwardSpec"),
    "RollingDriverSpec": ("skuld_research.config.spec", "RollingDriverSpec"),
    "SurvivorshipSpec": ("skuld_research.config.spec", "SurvivorshipSpec"),
    "GatingSpec": ("skuld_research.config.spec", "GatingSpec"),
    "BenchmarksSpec": ("skuld_research.config.spec", "BenchmarksSpec"),
    "OutputSpec": ("skuld_research.config.spec", "OutputSpec"),
    "ScrubbingSpec": ("skuld_research.config.spec", "ScrubbingSpec"),
    "spec_hash": ("skuld_research.config.hashing", "spec_hash"),
    "short_hash": ("skuld_research.config.hashing", "short_hash"),
    "load_spec": ("skuld_research.config.loader", "load_spec"),
    "SpecValidationError": (
        "skuld_research.config.loader",
        "SpecValidationError",
    ),
    "iter_preregistered_specs": (
        "skuld_research.config.loader",
        "iter_preregistered_specs",
    ),
    "iter_strategy_specs": (
        "skuld_research.config.loader",
        "iter_strategy_specs",
    ),
    "find_python_root": ("skuld_research.config.loader", "find_python_root"),
    "RunResult": ("skuld_research.config.runner", "RunResult"),
    "run_from_spec": ("skuld_research.config.runner", "run_from_spec"),
}

__all__ = [
    "BacktestSpec",
    "UniverseSpec",
    "FactorSpec",
    "BookToMarketFactorSpec",
    "OcfToAssetsFactorSpec",
    "MomentumFactorSpec",
    "ResidualMomentumFactorSpec",
    "BetaAdjustedMomentumFactorSpec",
    "MomentumVolPenalizedFactorSpec",
    "High52WeekFactorSpec",
    "MomentumConsistencyFactorSpec",
    "MomentumDrawdownAwareFactorSpec",
    "DualHorizonMomentumFactorSpec",
    "MomentumExShortSpikeFactorSpec",
    "TimeSeriesFilteredMomentumFactorSpec",
    "ReversalAdjustedMomentumFactorSpec",
    "MaxDailyReturnAvoidanceFactorSpec",
    "MomentumAccelerationFactorSpec",
    "LowVolatilityFactorSpec",
    "SizeFactorSpec",
    "DividendYieldFactorSpec",
    "ReturnOnRiskFactorSpec",
    "CostSpec",
    "BacktestEngineSpec",
    "WalkForwardSpec",
    "RollingDriverSpec",
    "SurvivorshipSpec",
    "GatingSpec",
    "BenchmarksSpec",
    "OutputSpec",
    "ScrubbingSpec",
    "spec_hash",
    "short_hash",
    "load_spec",
    "SpecValidationError",
    "iter_strategy_specs",
    "iter_preregistered_specs",
    "find_python_root",
    "RunResult",
    "run_from_spec",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
