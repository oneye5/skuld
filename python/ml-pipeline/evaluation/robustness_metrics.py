"""Robustness metrics for evaluating model stability across time.

This module provides metrics that penalize instability and reward consistency,
complementing the standard ranking metrics (IC, Sharpe).

Key Concepts:
- A high Sharpe ratio from a few lucky windows is less valuable than
  consistent positive performance across many windows
- Max drawdown matters for real-world portfolio management
- ICIR (IC consistency) is as important as mean IC
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class RobustnessMetrics:
    """Metrics evaluating stability and robustness across time windows.
    
    Attributes:
        sharpe_ratio: Overall Sharpe ratio
        icir: Information Coefficient Information Ratio
        max_drawdown: Maximum drawdown (as positive value, e.g., 0.25 = 25%)
        pct_windows_positive_sharpe: % of windows with positive Sharpe
        pct_windows_positive_return: % of windows with positive return
        sharpe_std: Standard deviation of per-window Sharpe ratios
        return_std: Standard deviation of per-window returns
        worst_window_return: Worst single window return
        best_window_return: Best single window return
        robustness_score: Composite score (higher = better)
    """
    sharpe_ratio: float
    icir: float
    max_drawdown: float
    pct_windows_positive_sharpe: float
    pct_windows_positive_return: float
    sharpe_std: float
    return_std: float
    worst_window_return: float
    best_window_return: float
    robustness_score: float
    
    # Per-window data for analysis
    window_sharpes: Optional[list[float]] = None
    window_returns: Optional[list[float]] = None


def compute_robustness_score(
    sharpe_ratio: float,
    icir: float,
    max_drawdown: float,
    pct_windows_positive_sharpe: float,
    sharpe_std: float,
    weights: Optional[dict] = None,
) -> float:
    """Compute composite robustness score.
    
    This score balances return (Sharpe), consistency (ICIR, window positivity),
    and risk (drawdown, Sharpe volatility).
    
    Higher score = better. Range is typically 0 to ~1.5 for good strategies.
    
    Args:
        sharpe_ratio: Overall Sharpe ratio
        icir: IC Information Ratio (annualized)
        max_drawdown: Maximum drawdown as positive decimal (e.g., 0.25 for 25%)
        pct_windows_positive_sharpe: Fraction of windows with positive Sharpe
        sharpe_std: Standard deviation of per-window Sharpe ratios
        weights: Optional custom weights dict
        
    Returns:
        Robustness score (higher is better)
    """
    # Default weights emphasizing consistency over raw performance
    if weights is None:
        weights = {
            "sharpe": 0.30,           # Base performance
            "icir": 0.25,             # Signal consistency  
            "consistency": 0.25,       # % positive windows
            "drawdown_penalty": 0.10,  # Tail risk
            "volatility_penalty": 0.10,  # Sharpe stability
        }
    
    # Normalize components to roughly similar scales
    
    # Sharpe contribution (cap at 1.5 to avoid domination by outliers)
    sharpe_contrib = min(sharpe_ratio, 1.5) * weights["sharpe"]
    
    # ICIR contribution (cap at 1.5)
    icir_contrib = min(icir, 1.5) * weights["icir"]
    
    # Consistency contribution (already 0-1 scale)
    consistency_contrib = pct_windows_positive_sharpe * weights["consistency"]
    
    # Drawdown penalty (starts at 15% drawdown, maxes out penalty at 40%)
    drawdown_penalty_factor = max(0, (max_drawdown - 0.15) / 0.25)
    drawdown_penalty_factor = min(drawdown_penalty_factor, 1.0)
    drawdown_penalty = drawdown_penalty_factor * weights["drawdown_penalty"]
    
    # Sharpe volatility penalty (high std = unstable)
    # Sharpe std of 0.5 is moderate, > 1.0 is high
    vol_penalty_factor = min(sharpe_std / 1.0, 1.0)
    vol_penalty = vol_penalty_factor * weights["volatility_penalty"]
    
    # Compute final score
    score = (
        sharpe_contrib +
        icir_contrib +
        consistency_contrib -
        drawdown_penalty -
        vol_penalty
    )
    
    return score


def compute_window_statistics(
    window_returns: list[float],
    periods_per_year: int = 252,
) -> dict:
    """Compute statistics across rolling windows.
    
    Args:
        window_returns: List of returns per window
        periods_per_year: For annualization (unused if returns are already period returns)
        
    Returns:
        Dictionary with window statistics
    """
    if not window_returns or len(window_returns) == 0:
        return {
            "pct_positive": 0.0,
            "mean_return": 0.0,
            "std_return": 0.0,
            "worst_return": 0.0,
            "best_return": 0.0,
        }
    
    returns = np.array(window_returns)
    
    return {
        "pct_positive": float(np.mean(returns > 0)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "worst_return": float(np.min(returns)),
        "best_return": float(np.max(returns)),
    }


def compute_window_sharpe_statistics(
    daily_returns: pd.Series,
    window_boundaries: list[tuple[int, int]],
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute Sharpe ratio for each window separately.
    
    Args:
        daily_returns: Series of daily returns indexed by timestamp
        window_boundaries: List of (start_ts, end_ts) tuples for each window
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary with per-window Sharpe ratios and statistics
    """
    window_sharpes = []
    window_returns = []
    
    daily_rf = risk_free_rate / 252
    
    for start_ts, end_ts in window_boundaries:
        window_rets = daily_returns[
            (daily_returns.index >= start_ts) & 
            (daily_returns.index <= end_ts)
        ]
        
        if len(window_rets) < 5:
            continue
            
        total_return = (1 + window_rets).prod() - 1
        window_returns.append(float(total_return))
        
        excess_returns = window_rets - daily_rf
        if excess_returns.std() > 0:
            sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        window_sharpes.append(float(sharpe))
    
    if not window_sharpes:
        return {
            "window_sharpes": [],
            "window_returns": [],
            "pct_positive_sharpe": 0.0,
            "pct_positive_return": 0.0,
            "mean_sharpe": 0.0,
            "std_sharpe": 0.0,
        }
    
    sharpes = np.array(window_sharpes)
    returns = np.array(window_returns)
    
    return {
        "window_sharpes": window_sharpes,
        "window_returns": window_returns,
        "pct_positive_sharpe": float(np.mean(sharpes > 0)),
        "pct_positive_return": float(np.mean(returns > 0)),
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
    }


@dataclass
class ExperimentComparison:
    """Comparison of multiple experiment configurations.
    
    Helps identify which configs are genuinely better vs lucky.
    """
    configs: list[dict]
    sharpe_ratios: list[float]
    robustness_scores: list[float]
    icirs: list[float]
    max_drawdowns: list[float]
    
    def rank_by_robustness(self) -> list[tuple[int, float, dict]]:
        """Rank configs by robustness score.
        
        Returns:
            List of (rank, score, config) tuples sorted by robustness score descending.
        """
        indexed = list(enumerate(zip(self.robustness_scores, self.configs)))
        sorted_results = sorted(indexed, key=lambda x: x[1][0], reverse=True)
        
        return [
            (rank + 1, score, config)
            for rank, (idx, (score, config)) in enumerate(sorted_results)
        ]
    
    def summarize(self) -> str:
        """Generate summary comparison."""
        lines = [
            "=== Experiment Comparison Summary ===",
            f"Total configs: {len(self.configs)}",
            "",
            "Top 5 by Robustness Score:",
        ]
        
        ranked = self.rank_by_robustness()
        for rank, score, config in ranked[:5]:
            lines.append(f"  #{rank}: Score={score:.4f}")
            lines.append(f"       Config: {config}")
        
        return "\n".join(lines)


def compute_robustness_metrics(
    sharpe_ratio: float,
    icir: float,
    max_drawdown: float,
    daily_returns: Optional[pd.Series] = None,
    window_boundaries: Optional[list[tuple[int, int]]] = None,
) -> RobustnessMetrics:
    """Compute all robustness metrics from pipeline results.
    
    Args:
        sharpe_ratio: Overall Sharpe ratio
        icir: IC Information Ratio
        max_drawdown: Maximum drawdown (positive decimal)
        daily_returns: Optional daily returns series for per-window analysis
        window_boundaries: Optional window boundaries for per-window analysis
        
    Returns:
        RobustnessMetrics dataclass with all computed metrics
    """
    # Defaults for when per-window data is not available
    pct_positive_sharpe = 0.5  # Assume neutral
    pct_positive_return = 0.5
    sharpe_std = 0.5  # Assume moderate
    window_sharpes = None
    window_returns = None
    worst_return = -max_drawdown
    best_return = 0.0
    
    # Compute per-window stats if we have the data
    if daily_returns is not None and window_boundaries is not None:
        window_stats = compute_window_sharpe_statistics(
            daily_returns, window_boundaries
        )
        pct_positive_sharpe = window_stats["pct_positive_sharpe"]
        pct_positive_return = window_stats["pct_positive_return"]
        sharpe_std = window_stats["std_sharpe"]
        window_sharpes = window_stats["window_sharpes"]
        window_returns = window_stats["window_returns"]
        
        if window_returns:
            worst_return = min(window_returns)
            best_return = max(window_returns)
    
    # Compute robustness score
    robustness_score = compute_robustness_score(
        sharpe_ratio=sharpe_ratio,
        icir=icir,
        max_drawdown=max_drawdown,
        pct_windows_positive_sharpe=pct_positive_sharpe,
        sharpe_std=sharpe_std,
    )
    
    return RobustnessMetrics(
        sharpe_ratio=sharpe_ratio,
        icir=icir,
        max_drawdown=max_drawdown,
        pct_windows_positive_sharpe=pct_positive_sharpe,
        pct_windows_positive_return=pct_positive_return,
        sharpe_std=sharpe_std,
        return_std=np.std(window_returns) if window_returns else 0.0,
        worst_window_return=worst_return,
        best_window_return=best_return,
        robustness_score=robustness_score,
        window_sharpes=window_sharpes,
        window_returns=window_returns,
    )
