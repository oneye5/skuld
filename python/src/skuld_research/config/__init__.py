"""Pre-registration config system for reproducible backtests."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skuld_research.config.hashing import short_hash, spec_hash
    from skuld_research.config.loader import (
        SpecValidationError,
        find_python_root,
        iter_preregistered_specs,
        load_spec,
    )
    from skuld_research.config.runner import RunResult, run_from_spec
    from skuld_research.config.spec import (
        BacktestEngineSpec,
        BacktestSpec,
        BenchmarksSpec,
        CostSpec,
        FactorSpec,
        GatingSpec,
        LowVolatilityFactorSpec,
        MomentumFactorSpec,
        OutputSpec,
        RollingDriverSpec,
        SizeFactorSpec,
        SurvivorshipSpec,
        UniverseSpec,
        WalkForwardSpec,
    )

_EXPORTS = {
    "BacktestSpec": ("skuld_research.config.spec", "BacktestSpec"),
    "UniverseSpec": ("skuld_research.config.spec", "UniverseSpec"),
    "FactorSpec": ("skuld_research.config.spec", "FactorSpec"),
    "MomentumFactorSpec": ("skuld_research.config.spec", "MomentumFactorSpec"),
    "LowVolatilityFactorSpec": (
        "skuld_research.config.spec",
        "LowVolatilityFactorSpec",
    ),
    "SizeFactorSpec": ("skuld_research.config.spec", "SizeFactorSpec"),
    "CostSpec": ("skuld_research.config.spec", "CostSpec"),
    "BacktestEngineSpec": ("skuld_research.config.spec", "BacktestEngineSpec"),
    "WalkForwardSpec": ("skuld_research.config.spec", "WalkForwardSpec"),
    "RollingDriverSpec": ("skuld_research.config.spec", "RollingDriverSpec"),
    "SurvivorshipSpec": ("skuld_research.config.spec", "SurvivorshipSpec"),
    "GatingSpec": ("skuld_research.config.spec", "GatingSpec"),
    "BenchmarksSpec": ("skuld_research.config.spec", "BenchmarksSpec"),
    "OutputSpec": ("skuld_research.config.spec", "OutputSpec"),
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
    "find_python_root": ("skuld_research.config.loader", "find_python_root"),
    "RunResult": ("skuld_research.config.runner", "RunResult"),
    "run_from_spec": ("skuld_research.config.runner", "run_from_spec"),
}

__all__ = [
    "BacktestSpec",
    "UniverseSpec",
    "FactorSpec",
    "MomentumFactorSpec",
    "LowVolatilityFactorSpec",
    "SizeFactorSpec",
    "CostSpec",
    "BacktestEngineSpec",
    "WalkForwardSpec",
    "RollingDriverSpec",
    "SurvivorshipSpec",
    "GatingSpec",
    "BenchmarksSpec",
    "OutputSpec",
    "spec_hash",
    "short_hash",
    "load_spec",
    "SpecValidationError",
    "iter_preregistered_specs",
    "find_python_root",
    "RunResult",
    "run_from_spec",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
