"""Trade simulation module exports."""

from .simulator import (
    run_trading_simulation,
    run_baseline_simulation,
    TradingMetrics,
    Trade,
    Position,
    metrics_to_dict,
    aggregate_trading_metrics,
)

__all__ = [
    "run_trading_simulation",
    "run_baseline_simulation",
    "TradingMetrics",
    "Trade",
    "Position",
    "metrics_to_dict",
    "aggregate_trading_metrics",
]
