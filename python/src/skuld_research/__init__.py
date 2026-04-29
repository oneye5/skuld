"""Skuld research — backtesting, walk-forward, factor models."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skuld_research.factors.combiner import combine_signals
    from skuld_research.factors.momentum import MomentumFactor
    from skuld_research.factors.protocols import SignalGenerator
    from skuld_research.portfolio.optimizer import build_target_portfolio

_EXPORTS = {
    "SignalGenerator": ("skuld_research.factors.protocols", "SignalGenerator"),
    "MomentumFactor": ("skuld_research.factors.momentum", "MomentumFactor"),
    "combine_signals": ("skuld_research.factors.combiner", "combine_signals"),
    "build_target_portfolio": (
        "skuld_research.portfolio.optimizer",
        "build_target_portfolio",
    ),
}

__all__ = [
    "SignalGenerator",
    "MomentumFactor",
    "combine_signals",
    "build_target_portfolio",
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
