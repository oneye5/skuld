"""Public reporting API."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skuld_research.reporting.markdown_writer import write_methodology_report
    from skuld_research.reporting.report_builder import build_methodology_report

_EXPORTS = {
    "build_methodology_report": (
        "skuld_research.reporting.report_builder",
        "build_methodology_report",
    ),
    "write_methodology_report": (
        "skuld_research.reporting.markdown_writer",
        "write_methodology_report",
    ),
}

__all__ = [
    "build_methodology_report",
    "write_methodology_report",
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
