"""Stage 6 — Cash overlay.

Rules-based cash allocation that raises cash fraction beyond the configured
floor when market conditions trigger defensive positioning.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skuld_research.overlay.apply import apply_cash_overlay
    from skuld_research.overlay.rules import (
        NoOverlay,
        NzxMA200AndAggMomentumRule,
        OverlayRule,
    )

_EXPORTS = {
    "apply_cash_overlay": ("skuld_research.overlay.apply", "apply_cash_overlay"),
    "NoOverlay": ("skuld_research.overlay.rules", "NoOverlay"),
    "NzxMA200AndAggMomentumRule": (
        "skuld_research.overlay.rules",
        "NzxMA200AndAggMomentumRule",
    ),
    "OverlayRule": ("skuld_research.overlay.rules", "OverlayRule"),
}

__all__ = [
    "apply_cash_overlay",
    "NoOverlay",
    "NzxMA200AndAggMomentumRule",
    "OverlayRule",
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
