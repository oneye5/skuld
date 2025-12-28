"""Evaluation package - ranking metrics and portfolio simulation."""

from evaluation.ranking_metrics import (
    RankingMetrics,
    compute_ic,
    compute_rank_ic,
    compute_icir,
    compute_quintile_returns,
)
from evaluation.portfolio_simulator import (
    PortfolioConfig,
    BacktestResult,
    run_portfolio_backtest,
    compute_sharpe_ratio,
    infer_periods_per_year,
)

__all__ = [
    "RankingMetrics",
    "compute_ic",
    "compute_rank_ic",
    "compute_icir",
    "compute_quintile_returns",
    "PortfolioConfig",
    "BacktestResult",
    "run_portfolio_backtest",
    "compute_sharpe_ratio",
    "infer_periods_per_year",
]
