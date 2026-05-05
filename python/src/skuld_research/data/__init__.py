"""Data loading and point-in-time filtering.

Validation utilities live in `skuld_common.validation` and are re-exported
here for convenience.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    from skuld_research.data.csv_loader import RawData, load_raw_csv, load_raw_ohlc
    from skuld_research.data.pit_loader import PITLoader
    from skuld_research.data.prepared_panel import build_prepared_panel

_EXPORTS = {
    "RawData": ("skuld_research.data.csv_loader", "RawData"),
    "load_raw_csv": ("skuld_research.data.csv_loader", "load_raw_csv"),
    "load_raw_ohlc": ("skuld_research.data.csv_loader", "load_raw_ohlc"),
    "PITLoader": ("skuld_research.data.pit_loader", "PITLoader"),
    "build_prepared_panel": (
        "skuld_research.data.prepared_panel",
        "build_prepared_panel",
    ),
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
    "RawData",
    "load_raw_csv",
    "load_raw_ohlc",
    "PITLoader",
    "build_prepared_panel",
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
