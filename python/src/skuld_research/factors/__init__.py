"""Factors package for Skuld research pipeline."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skuld_research.factors.book_to_market import BookToMarketFactor
    from skuld_research.factors.combiner import combine_signals
    from skuld_research.factors.low_volatility import LowVolatilityFactor
    from skuld_research.factors.momentum import MomentumFactor
    from skuld_research.factors.ocf_to_assets import OcfToAssetsFactor
    from skuld_research.factors.protocols import SignalGenerator
    from skuld_research.factors.size import SizeFactor

_EXPORTS = {
    "SignalGenerator": ("skuld_research.factors.protocols", "SignalGenerator"),
    "BookToMarketFactor": ("skuld_research.factors.book_to_market", "BookToMarketFactor"),
    "MomentumFactor": ("skuld_research.factors.momentum", "MomentumFactor"),
    "OcfToAssetsFactor": ("skuld_research.factors.ocf_to_assets", "OcfToAssetsFactor"),
    "LowVolatilityFactor": (
        "skuld_research.factors.low_volatility",
        "LowVolatilityFactor",
    ),
    "SizeFactor": ("skuld_research.factors.size", "SizeFactor"),
    "combine_signals": ("skuld_research.factors.combiner", "combine_signals"),
}

__all__ = [
    "SignalGenerator",
    "BookToMarketFactor",
    "MomentumFactor",
    "OcfToAssetsFactor",
    "LowVolatilityFactor",
    "SizeFactor",
    "combine_signals",
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
