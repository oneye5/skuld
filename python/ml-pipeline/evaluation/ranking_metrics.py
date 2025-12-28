"""Metrics for evaluating ranking-based stock prediction models.

This module provides cross-sectional evaluation metrics used in quantitative
finance: Information Coefficient (IC), Rank IC, ICIR, hit rate, and quintile
analysis.

Key Metrics:
- IC (Information Coefficient): Pearson correlation between predictions and returns
- Rank IC: Spearman rank correlation (more robust to outliers)
- ICIR (IC Information Ratio): IC consistency measure = mean(IC) / std(IC)
- Hit Rate: % of top-N predictions with positive returns
- Quintile Spread: Return difference between top and bottom quintiles
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional

from config.columns import TIMESTAMP, TICKER
from config.settings import FORWARD_RETURN_DAYS


# =============================================================================
# BASIC IC CALCULATIONS
# =============================================================================

def compute_ic(
    predicted: pd.Series, 
    actual: pd.Series,
) -> float:
    """Compute Information Coefficient (Pearson correlation).
    
    Args:
        predicted: Predicted scores/rankings.
        actual: Actual returns.
    
    Returns:
        Pearson correlation coefficient between predicted and actual.
        Returns NaN if insufficient data or constant values.
    """
    if len(predicted) < 3:
        return np.nan
    
    # Remove any NaN values
    mask = ~(predicted.isna() | actual.isna())
    pred_clean = predicted[mask]
    actual_clean = actual[mask]
    
    if len(pred_clean) < 3:
        return np.nan
    
    # Check for constant values
    if pred_clean.std() == 0 or actual_clean.std() == 0:
        return np.nan
    
    corr, _ = stats.pearsonr(pred_clean, actual_clean)
    return corr


def compute_rank_ic(
    predicted: pd.Series, 
    actual: pd.Series,
) -> float:
    """Compute Rank Information Coefficient (Spearman correlation).
    
    Rank IC is more robust to outliers than Pearson IC because it operates
    on ranks rather than raw values.
    
    Args:
        predicted: Predicted scores/rankings.
        actual: Actual returns.
    
    Returns:
        Spearman rank correlation coefficient.
        Returns NaN if insufficient data.
    """
    if len(predicted) < 3:
        return np.nan
    
    # Remove any NaN values
    mask = ~(predicted.isna() | actual.isna())
    pred_clean = predicted[mask]
    actual_clean = actual[mask]
    
    if len(pred_clean) < 3:
        return np.nan
    
    corr, _ = stats.spearmanr(pred_clean, actual_clean)
    return corr


def compute_icir(
    ic_series: pd.Series,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> float:
    """Compute IC Information Ratio (ICIR).
    
    ICIR = mean(IC) / std(IC) measures the consistency of the IC.
    A higher ICIR indicates more stable predictive power.
    
    Args:
        ic_series: Series of IC values over time.
        annualize: If True, multiply by sqrt(periods_per_year).
        periods_per_year: Number of independent periods per year. For overlapping
                         returns (e.g., 5-day returns observed daily), use
                         252/forward_return_days (e.g., 252/5=50 for 5-day returns)
                         to avoid inflating ICIR due to autocorrelation.
    
    Returns:
        ICIR value. Higher is better.
    """
    ic_clean = ic_series.dropna()
    
    if len(ic_clean) < 2:
        return np.nan
    
    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()
    
    if std_ic == 0:
        return np.nan
    
    icir = mean_ic / std_ic
    
    if annualize:
        icir *= np.sqrt(periods_per_year)
    
    return icir


# =============================================================================
# QUINTILE ANALYSIS
# =============================================================================

def assign_quintiles(
    scores: pd.Series,
    n_quantiles: int = 5,
) -> pd.Series:
    """Assign quintile labels based on scores.
    
    Q5 (quintile 5) contains the highest scores, Q1 the lowest.
    
    Args:
        scores: Series of scores to rank.
        n_quantiles: Number of quantile groups (default 5 for quintiles).
    
    Returns:
        Series with quintile labels (1 = lowest, n_quantiles = highest).
    """
    # qcut with labels gives 0-indexed, add 1 for 1-indexed quintiles
    try:
        quintiles = pd.qcut(scores, q=n_quantiles, labels=False, duplicates='drop') + 1
    except ValueError:
        # Too few unique values for n_quantiles buckets
        # Fall back to rank-based assignment
        ranks = scores.rank(method='first')
        quintiles = pd.cut(ranks, bins=n_quantiles, labels=False) + 1
    
    return quintiles


def compute_quintile_returns(
    predicted: pd.Series,
    actual: pd.Series,
    n_quantiles: int = 5,
) -> Dict[int, float]:
    """Compute average return for each quintile.
    
    Args:
        predicted: Predicted scores.
        actual: Actual returns.
        n_quantiles: Number of quantile groups.
    
    Returns:
        Dictionary mapping quintile number to average return.
        {1: avg_return_Q1, ..., 5: avg_return_Q5}
    """
    quintiles = assign_quintiles(predicted, n_quantiles)
    
    df = pd.DataFrame({
        'quintile': quintiles,
        'return': actual.values,
    })
    
    result = {}
    for q in range(1, n_quantiles + 1):
        q_returns = df[df['quintile'] == q]['return']
        result[q] = q_returns.mean() if len(q_returns) > 0 else np.nan
    
    return result


def compute_quintile_spread(quintile_returns: Dict[int, float]) -> float:
    """Compute spread between top and bottom quintiles.
    
    Args:
        quintile_returns: Dictionary from compute_quintile_returns.
    
    Returns:
        Q5 return - Q1 return (top minus bottom).
    """
    q5 = quintile_returns.get(5, np.nan)
    q1 = quintile_returns.get(1, np.nan)
    
    if pd.isna(q5) or pd.isna(q1):
        return np.nan
    
    return q5 - q1


# =============================================================================
# HIT RATE
# =============================================================================

def compute_hit_rate(
    predicted: pd.Series,
    actual: pd.Series,
    top_n: int = 10,
) -> float:
    """Compute hit rate for top-N predictions.
    
    Hit rate = percentage of top-N predicted stocks with positive actual returns.
    
    Args:
        predicted: Predicted scores (higher = better predicted).
        actual: Actual returns.
        top_n: Number of top predictions to consider.
    
    Returns:
        Hit rate as a fraction (0.0 to 1.0).
    """
    if len(predicted) < top_n:
        top_n = len(predicted)
    
    if top_n == 0:
        return np.nan
    
    # Get indices of top-N predictions
    df = pd.DataFrame({
        'predicted': predicted.values,
        'actual': actual.values,
    })
    
    top_indices = df['predicted'].nlargest(top_n).index
    top_actual = df.loc[top_indices, 'actual']
    
    # Count positive returns
    n_positive = (top_actual > 0).sum()
    
    return n_positive / top_n


# =============================================================================
# CROSS-SECTIONAL (PER-TIMESTAMP) METRICS
# =============================================================================

def compute_cross_sectional_ic_series(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP,
    predicted_col: str = "predicted_score",
    actual_col: str = "actual_return",
    min_stocks: int = 5,
    use_rank: bool = False,
) -> pd.Series:
    """Compute IC for each timestamp (cross-sectionally).
    
    This is the standard way to evaluate ranking models in finance:
    compute IC within each time period, then aggregate.
    
    Args:
        df: DataFrame with timestamp, predicted scores, and actual returns.
        timestamp_col: Column name for timestamp.
        predicted_col: Column name for predicted scores.
        actual_col: Column name for actual returns.
        min_stocks: Minimum stocks per timestamp to compute IC.
        use_rank: If True, compute Rank IC instead of IC.
    
    Returns:
        Series with IC values, indexed by timestamp.
    """
    ic_fn = compute_rank_ic if use_rank else compute_ic
    
    ic_dict = {}
    
    for ts, group in df.groupby(timestamp_col):
        if len(group) < min_stocks:
            continue
        
        predicted = group[predicted_col]
        actual = group[actual_col]
        
        ic = ic_fn(predicted, actual)
        if not pd.isna(ic):
            ic_dict[ts] = ic
    
    return pd.Series(ic_dict)


def compute_cross_sectional_quintile_returns(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP,
    predicted_col: str = "predicted_score",
    actual_col: str = "actual_return",
    min_stocks: int = 10,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Compute quintile returns for each timestamp.
    
    Args:
        df: DataFrame with timestamp, predicted scores, and actual returns.
        timestamp_col: Column name for timestamp.
        predicted_col: Column name for predicted scores.
        actual_col: Column name for actual returns.
        min_stocks: Minimum stocks per timestamp.
        n_quantiles: Number of quantile groups.
    
    Returns:
        DataFrame with columns [Q1, Q2, ..., Qn] and index = timestamps.
    """
    results = []
    
    for ts, group in df.groupby(timestamp_col):
        if len(group) < min_stocks:
            continue
        
        quintile_returns = compute_quintile_returns(
            group[predicted_col],
            group[actual_col],
            n_quantiles=n_quantiles,
        )
        
        row = {"timestamp": ts}
        for q, ret in quintile_returns.items():
            row[f"Q{q}"] = ret
        
        results.append(row)
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    result_df = result_df.set_index("timestamp")
    
    return result_df


# =============================================================================
# RANKING METRICS DATACLASS
# =============================================================================

@dataclass
class RankingMetrics:
    """Container for all ranking evaluation metrics.
    
    Attributes:
        mean_ic: Average Information Coefficient across timestamps.
        std_ic: Standard deviation of IC.
        icir: IC Information Ratio (annualized).
        mean_rank_ic: Average Rank IC (Spearman) across timestamps.
        std_rank_ic: Standard deviation of Rank IC.
        rank_icir: Rank IC Information Ratio (annualized).
        hit_rate_top_n: Hit rate for top-N predictions.
        quintile_returns: Average return for each quintile.
        quintile_spread: Q5 return - Q1 return.
        num_timestamps: Number of timestamps with valid metrics.
        avg_stocks_per_timestamp: Average number of stocks per timestamp.
    """
    mean_ic: float
    std_ic: float
    icir: float
    mean_rank_ic: float
    std_rank_ic: float
    rank_icir: float
    hit_rate_top_n: float
    quintile_returns: Dict[int, float]
    quintile_spread: float
    num_timestamps: int
    avg_stocks_per_timestamp: float
    
    # Optional: store the full IC series for plotting
    ic_series: pd.Series = field(default_factory=pd.Series, repr=False)
    rank_ic_series: pd.Series = field(default_factory=pd.Series, repr=False)
    
    @classmethod
    def from_predictions(
        cls,
        df: pd.DataFrame,
        timestamp_col: str = TIMESTAMP,
        predicted_col: str = "predicted_score",
        actual_col: str = "actual_return",
        min_stocks: int = 5,
        top_n_for_hit_rate: int = 10,
        forward_return_days: int = FORWARD_RETURN_DAYS,
    ) -> "RankingMetrics":
        """Create RankingMetrics from a predictions DataFrame.
        
        Args:
            df: DataFrame with timestamps, predicted scores, and actual returns.
            timestamp_col: Column name for timestamp grouping.
            predicted_col: Column name for predicted scores.
            actual_col: Column name for actual returns.
            min_stocks: Minimum stocks per timestamp to include.
            top_n_for_hit_rate: N for hit rate calculation.
            forward_return_days: The horizon of forward returns in days (e.g., 5 for
                                5-day returns). Used to compute periods_per_year for
                                ICIR annualization as 252/forward_return_days.
        
        Returns:
            RankingMetrics instance with all computed metrics.
        """
        # Compute periods_per_year from forward return horizon
        # E.g., 5-day returns = 252/5 ≈ 50 independent periods per year
        periods_per_year = int(252 / forward_return_days)
        # Compute IC series
        ic_series = compute_cross_sectional_ic_series(
            df, timestamp_col, predicted_col, actual_col, 
            min_stocks=min_stocks, use_rank=False
        )
        
        rank_ic_series = compute_cross_sectional_ic_series(
            df, timestamp_col, predicted_col, actual_col,
            min_stocks=min_stocks, use_rank=True
        )
        
        # Aggregate IC metrics
        mean_ic = ic_series.mean() if len(ic_series) > 0 else np.nan
        std_ic = ic_series.std() if len(ic_series) > 0 else np.nan
        icir = compute_icir(ic_series, annualize=True, periods_per_year=periods_per_year)
        
        mean_rank_ic = rank_ic_series.mean() if len(rank_ic_series) > 0 else np.nan
        std_rank_ic = rank_ic_series.std() if len(rank_ic_series) > 0 else np.nan
        rank_icir = compute_icir(rank_ic_series, annualize=True, periods_per_year=periods_per_year)
        
        # Compute quintile returns (aggregate across all data)
        quintile_returns = compute_quintile_returns(
            df[predicted_col], df[actual_col]
        )
        quintile_spread = compute_quintile_spread(quintile_returns)
        
        # Compute hit rate (aggregate)
        hit_rate = compute_hit_rate(
            df[predicted_col], df[actual_col], top_n=top_n_for_hit_rate
        )
        
        # Count timestamps and stocks
        timestamp_counts = df.groupby(timestamp_col).size()
        valid_timestamps = timestamp_counts[timestamp_counts >= min_stocks]
        num_timestamps = len(valid_timestamps)
        avg_stocks = valid_timestamps.mean() if len(valid_timestamps) > 0 else 0
        
        return cls(
            mean_ic=mean_ic,
            std_ic=std_ic,
            icir=icir,
            mean_rank_ic=mean_rank_ic,
            std_rank_ic=std_rank_ic,
            rank_icir=rank_icir,
            hit_rate_top_n=hit_rate,
            quintile_returns=quintile_returns,
            quintile_spread=quintile_spread,
            num_timestamps=num_timestamps,
            avg_stocks_per_timestamp=avg_stocks,
            ic_series=ic_series,
            rank_ic_series=rank_ic_series,
        )
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "mean_ic": float(self.mean_ic) if not pd.isna(self.mean_ic) else None,
            "std_ic": float(self.std_ic) if not pd.isna(self.std_ic) else None,
            "icir": float(self.icir) if not pd.isna(self.icir) else None,
            "mean_rank_ic": float(self.mean_rank_ic) if not pd.isna(self.mean_rank_ic) else None,
            "std_rank_ic": float(self.std_rank_ic) if not pd.isna(self.std_rank_ic) else None,
            "rank_icir": float(self.rank_icir) if not pd.isna(self.rank_icir) else None,
            "hit_rate_top_n": float(self.hit_rate_top_n) if not pd.isna(self.hit_rate_top_n) else None,
            "quintile_returns": {k: float(v) if not pd.isna(v) else None 
                                for k, v in self.quintile_returns.items()},
            "quintile_spread": float(self.quintile_spread) if not pd.isna(self.quintile_spread) else None,
            "num_timestamps": self.num_timestamps,
            "avg_stocks_per_timestamp": float(self.avg_stocks_per_timestamp),
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary of metrics."""
        lines = [
            "=== Ranking Metrics Summary ===",
            f"Mean IC:        {self.mean_ic:.4f}" if not pd.isna(self.mean_ic) else "Mean IC:        N/A",
            f"ICIR:           {self.icir:.4f}" if not pd.isna(self.icir) else "ICIR:           N/A",
            f"Mean Rank IC:   {self.mean_rank_ic:.4f}" if not pd.isna(self.mean_rank_ic) else "Mean Rank IC:   N/A",
            f"Rank ICIR:      {self.rank_icir:.4f}" if not pd.isna(self.rank_icir) else "Rank ICIR:      N/A",
            f"Hit Rate:       {self.hit_rate_top_n:.2%}" if not pd.isna(self.hit_rate_top_n) else "Hit Rate:       N/A",
            f"Quintile Spread:{self.quintile_spread:.4f}" if not pd.isna(self.quintile_spread) else "Quintile Spread:N/A",
            f"Timestamps:     {self.num_timestamps}",
            f"Avg Stocks:     {self.avg_stocks_per_timestamp:.1f}",
        ]
        
        if self.quintile_returns:
            lines.append("Quintile Returns:")
            for q in sorted(self.quintile_returns.keys()):
                ret = self.quintile_returns[q]
                if not pd.isna(ret):
                    lines.append(f"  Q{q}: {ret:.4f}")
        
        return "\n".join(lines)
