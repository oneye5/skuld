"""Evaluation package - metrics, trading simulation, and reporting."""

from evaluation.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
)
from evaluation.simulator import (
    TradingMetrics,
    Trade,
    run_trading_simulation,
)
from evaluation.reporter import generate_report

__all__ = [
    "ClassificationMetrics",
    "compute_classification_metrics",
    "TradingMetrics",
    "Trade",
    "run_trading_simulation",
    "generate_report",
]
