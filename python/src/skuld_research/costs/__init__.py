from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skuld_research.costs.model import CostBreakdown, CostConfig, CostModel
    from skuld_research.costs.spread_estimator import (
        DEFAULT_MIN_BPS_PER_SIDE,
        compute_abdi_ranaldo_spread_panel,
    )

_EXPORTS = {
    "CostBreakdown": ("skuld_research.costs.model", "CostBreakdown"),
    "CostConfig": ("skuld_research.costs.model", "CostConfig"),
    "CostModel": ("skuld_research.costs.model", "CostModel"),
    "DEFAULT_MIN_BPS_PER_SIDE": (
        "skuld_research.costs.spread_estimator",
        "DEFAULT_MIN_BPS_PER_SIDE",
    ),
    "compute_abdi_ranaldo_spread_panel": (
        "skuld_research.costs.spread_estimator",
        "compute_abdi_ranaldo_spread_panel",
    ),
}

__all__ = [
    "CostBreakdown",
    "CostConfig",
    "CostModel",
    "DEFAULT_MIN_BPS_PER_SIDE",
    "compute_abdi_ranaldo_spread_panel",
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
