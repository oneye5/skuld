"""Skuld common types and contracts."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skuld_common.contracts import (
        BacktestResult,
        BenchmarkResult,
        BootstrapResult,
        CombinedScores,
        DecayReport,
        DecompositionReport,
        DeflatedSharpeResult,
        DominanceResult,
        FoldResult,
        GatingDecision,
        ICReport,
        MethodologyReport,
        PITSnapshot,
        PreparedPanel,
        TargetPortfolio,
        WalkForwardResult,
    )
    from skuld_common.validation import (
        ValidationReport,
        detect_duplicate_observations,
        detect_gaps,
        detect_invalid_corporate_actions,
        detect_nan_density,
        detect_negative_prices,
        detect_ohlc_inconsistencies,
        detect_stale_fundamentals,
        detect_stale_sources,
    )

_EXPORTS = {
    "PITSnapshot": ("skuld_common.contracts", "PITSnapshot"),
    "PreparedPanel": ("skuld_common.contracts", "PreparedPanel"),
    "CombinedScores": ("skuld_common.contracts", "CombinedScores"),
    "TargetPortfolio": ("skuld_common.contracts", "TargetPortfolio"),
    "BacktestResult": ("skuld_common.contracts", "BacktestResult"),
    "BenchmarkResult": ("skuld_common.contracts", "BenchmarkResult"),
    "FoldResult": ("skuld_common.contracts", "FoldResult"),
    "WalkForwardResult": ("skuld_common.contracts", "WalkForwardResult"),
    "ICReport": ("skuld_common.contracts", "ICReport"),
    "DecayReport": ("skuld_common.contracts", "DecayReport"),
    "DecompositionReport": ("skuld_common.contracts", "DecompositionReport"),
    "BootstrapResult": ("skuld_common.contracts", "BootstrapResult"),
    "DeflatedSharpeResult": ("skuld_common.contracts", "DeflatedSharpeResult"),
    "DominanceResult": ("skuld_common.contracts", "DominanceResult"),
    "GatingDecision": ("skuld_common.contracts", "GatingDecision"),
    "MethodologyReport": ("skuld_common.contracts", "MethodologyReport"),
    "ValidationReport": ("skuld_common.validation", "ValidationReport"),
    "detect_duplicate_observations": (
        "skuld_common.validation",
        "detect_duplicate_observations",
    ),
    "detect_gaps": ("skuld_common.validation", "detect_gaps"),
    "detect_invalid_corporate_actions": (
        "skuld_common.validation",
        "detect_invalid_corporate_actions",
    ),
    "detect_nan_density": ("skuld_common.validation", "detect_nan_density"),
    "detect_negative_prices": ("skuld_common.validation", "detect_negative_prices"),
    "detect_ohlc_inconsistencies": (
        "skuld_common.validation",
        "detect_ohlc_inconsistencies",
    ),
    "detect_stale_fundamentals": (
        "skuld_common.validation",
        "detect_stale_fundamentals",
    ),
    "detect_stale_sources": ("skuld_common.validation", "detect_stale_sources"),
}

__all__ = [
    "PITSnapshot",
    "PreparedPanel",
    "CombinedScores",
    "TargetPortfolio",
    "BacktestResult",
    "BenchmarkResult",
    "FoldResult",
    "WalkForwardResult",
    "ICReport",
    "DecayReport",
    "DecompositionReport",
    "BootstrapResult",
    "DeflatedSharpeResult",
    "DominanceResult",
    "GatingDecision",
    "MethodologyReport",
    "ValidationReport",
    "detect_duplicate_observations",
    "detect_gaps",
    "detect_invalid_corporate_actions",
    "detect_nan_density",
    "detect_negative_prices",
    "detect_ohlc_inconsistencies",
    "detect_stale_fundamentals",
    "detect_stale_sources",
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
