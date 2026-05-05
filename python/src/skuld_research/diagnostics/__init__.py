"""Signal-level diagnostics for factor research."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skuld_research.diagnostics.attribution import AttributionReport, attribute_returns
    from skuld_research.diagnostics.factor_comparison import (
        FactorComparisonReport,
        compare_factors,
    )
    from skuld_research.diagnostics.audit import PipelineAuditReport, audit_pipeline
    from skuld_research.diagnostics.decay import alpha_decay
    from skuld_research.diagnostics.decomposition import factor_decomposition
    from skuld_research.diagnostics.ic import ranking_ic
    from skuld_research.diagnostics.report import write_diagnostics_report

_EXPORTS = {
    "ranking_ic": ("skuld_research.diagnostics.ic", "ranking_ic"),
    "alpha_decay": ("skuld_research.diagnostics.decay", "alpha_decay"),
    "factor_decomposition": (
        "skuld_research.diagnostics.decomposition",
        "factor_decomposition",
    ),
    "write_diagnostics_report": (
        "skuld_research.diagnostics.report",
        "write_diagnostics_report",
    ),
    "audit_pipeline": ("skuld_research.diagnostics.audit", "audit_pipeline"),
    "PipelineAuditReport": ("skuld_research.diagnostics.audit", "PipelineAuditReport"),
    "attribute_returns": ("skuld_research.diagnostics.attribution", "attribute_returns"),
    "AttributionReport": ("skuld_research.diagnostics.attribution", "AttributionReport"),
    "compare_factors": ("skuld_research.diagnostics.factor_comparison", "compare_factors"),
    "FactorComparisonReport": (
        "skuld_research.diagnostics.factor_comparison",
        "FactorComparisonReport",
    ),
}

__all__ = [
    "ranking_ic",
    "alpha_decay",
    "factor_decomposition",
    "write_diagnostics_report",
    "audit_pipeline",
    "PipelineAuditReport",
    "attribute_returns",
    "AttributionReport",
    "compare_factors",
    "FactorComparisonReport",
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
