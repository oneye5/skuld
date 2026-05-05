"""Benchmark strategies for M6."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skuld_research.benchmarks.nz_td_floor import nz_td_floor
    from skuld_research.benchmarks.nzx_equal_weighted_fixed_universe import (
        nzx_equal_weighted_fixed_universe,
    )
    from skuld_research.benchmarks.sixty_forty import sixty_forty

_EXPORTS = {
    "nz_td_floor": ("skuld_research.benchmarks.nz_td_floor", "nz_td_floor"),
    "nzx_equal_weighted_fixed_universe": (
        "skuld_research.benchmarks.nzx_equal_weighted_fixed_universe",
        "nzx_equal_weighted_fixed_universe",
    ),
    "sixty_forty": ("skuld_research.benchmarks.sixty_forty", "sixty_forty"),
}

__all__ = [
    "nz_td_floor",
    "nzx_equal_weighted_fixed_universe",
    "sixty_forty",
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
