"""Evaluation module exports."""

from .model_evaluation import (
    evaluate_predictions,
    ClassificationMetrics,
    metrics_to_dict as classification_metrics_to_dict,
    aggregate_metrics as aggregate_classification_metrics,
)
from .trade_simulation import (
    run_trading_simulation,
    run_baseline_simulation,
    TradingMetrics,
    Trade,
    Position,
    metrics_to_dict as trading_metrics_to_dict,
    aggregate_trading_metrics,
)

__all__ = [
    "evaluate_predictions",
    "ClassificationMetrics",
    "classification_metrics_to_dict",
    "aggregate_classification_metrics",
    "run_trading_simulation",
    "run_baseline_simulation",
    "TradingMetrics",
    "Trade",
    "Position",
    "trading_metrics_to_dict",
    "aggregate_trading_metrics",
]
