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

Advanced Metrics:
- Sortino Ratio: Risk-adjusted return using only downside deviation
- Calmar Ratio: Return over maximum drawdown
- Tail Ratio: Ratio of right tail to left tail
- Win/Loss Analysis: Win rate, average win/loss, profit factor
- Stability Score: Composite score for model robustness
- IC Decay: Autocorrelation of IC series
- Decile Analysis: 10-group quantile analysis for finer granularity
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, List, Tuple, Any

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
        # NOTE: Use max(1, ...) to avoid periods_per_year < 1 for long horizons
        # which would cause ICIR to be incorrectly scaled down
        periods_per_year = max(1, 252 / forward_return_days)
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
        
        # Compute hit rate (per-timestamp, then average)
        # This is the correct approach for cross-sectional evaluation
        hit_rates_per_ts = []
        for ts, group in df.groupby(timestamp_col):
            if len(group) >= top_n_for_hit_rate:
                hr = compute_hit_rate(
                    group[predicted_col], group[actual_col], top_n=top_n_for_hit_rate
                )
                if not np.isnan(hr):
                    hit_rates_per_ts.append(hr)
        hit_rate = np.mean(hit_rates_per_ts) if hit_rates_per_ts else np.nan
        
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


# =============================================================================
# ADVANCED RISK METRICS
# =============================================================================

def compute_sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute Sortino ratio (risk-adjusted return using downside deviation).
    
    Unlike Sharpe which penalizes all volatility, Sortino only penalizes
    downside volatility, which is more relevant for investors.
    
    Args:
        returns: Series of returns.
        target_return: Minimum acceptable return (default 0).
        periods_per_year: For annualization.
    
    Returns:
        Annualized Sortino ratio.
    """
    if len(returns) < 2:
        return np.nan
    
    excess_returns = returns - target_return / periods_per_year
    
    # Downside returns only
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf  # No downside periods
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_std == 0:
        return np.inf
    
    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)


def compute_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute Calmar ratio (annual return / max drawdown).
    
    Higher is better. Shows return relative to worst-case loss.
    
    Args:
        returns: Series of returns.
        periods_per_year: For annualization.
    
    Returns:
        Calmar ratio.
    """
    if len(returns) < 2:
        return np.nan
    
    # Annualized return
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (running_max - cumulative) / running_max
    max_drawdown = drawdown.max()
    
    if max_drawdown == 0:
        return np.inf
    
    return annual_return / max_drawdown


def compute_tail_ratio(
    returns: pd.Series,
    percentile: float = 95,
) -> float:
    """Compute tail ratio (right tail / left tail).
    
    Values > 1 indicate positive skew (larger wins than losses).
    
    Args:
        returns: Series of returns.
        percentile: Percentile for tail measurement.
    
    Returns:
        Tail ratio (right tail magnitude / left tail magnitude).
    """
    if len(returns) < 10:
        return np.nan
    
    right_tail = np.percentile(returns, percentile)
    left_tail = np.percentile(returns, 100 - percentile)
    
    if abs(left_tail) < 1e-10:
        return np.inf if right_tail > 0 else 0.0
    
    return abs(right_tail) / abs(left_tail)


def compute_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
) -> float:
    """Compute Omega ratio.
    
    Omega = sum of gains above threshold / sum of losses below threshold.
    Values > 1 indicate profitable strategy.
    
    Args:
        returns: Series of returns.
        threshold: Return threshold (default 0).
    
    Returns:
        Omega ratio.
    """
    if len(returns) < 2:
        return np.nan
    
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    
    total_gains = gains.sum()
    total_losses = losses.sum()
    
    if total_losses == 0:
        return np.inf if total_gains > 0 else 1.0
    
    return total_gains / total_losses


# =============================================================================
# WIN/LOSS ANALYSIS
# =============================================================================

@dataclass
class WinLossMetrics:
    """Win/loss analysis metrics."""
    win_rate: float  # % of positive periods
    loss_rate: float  # % of negative periods
    avg_win: float  # Average winning return
    avg_loss: float  # Average losing return
    win_loss_ratio: float  # avg_win / abs(avg_loss)
    profit_factor: float  # sum(wins) / abs(sum(losses))
    max_consecutive_wins: int
    max_consecutive_losses: int
    expectancy: float  # Expected return per trade
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "win_rate": float(self.win_rate) if not pd.isna(self.win_rate) else None,
            "loss_rate": float(self.loss_rate) if not pd.isna(self.loss_rate) else None,
            "avg_win": float(self.avg_win) if not pd.isna(self.avg_win) else None,
            "avg_loss": float(self.avg_loss) if not pd.isna(self.avg_loss) else None,
            "win_loss_ratio": float(self.win_loss_ratio) if not pd.isna(self.win_loss_ratio) else None,
            "profit_factor": float(self.profit_factor) if not pd.isna(self.profit_factor) else None,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "expectancy": float(self.expectancy) if not pd.isna(self.expectancy) else None,
        }


def compute_consecutive_streak(series: pd.Series, condition: bool) -> int:
    """Compute maximum consecutive streak of positive or negative values."""
    if len(series) == 0:
        return 0
    
    if condition:
        mask = series > 0
    else:
        mask = series <= 0
    
    max_streak = 0
    current_streak = 0
    
    for val in mask:
        if val:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak


def compute_win_loss_metrics(returns: pd.Series) -> WinLossMetrics:
    """Compute comprehensive win/loss analysis.
    
    Args:
        returns: Series of returns.
    
    Returns:
        WinLossMetrics with all win/loss statistics.
    """
    if len(returns) < 2:
        return WinLossMetrics(
            win_rate=np.nan, loss_rate=np.nan, avg_win=np.nan, avg_loss=np.nan,
            win_loss_ratio=np.nan, profit_factor=np.nan,
            max_consecutive_wins=0, max_consecutive_losses=0, expectancy=np.nan
        )
    
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    
    n_total = len(returns)
    n_wins = len(wins)
    n_losses = len(losses)
    
    win_rate = n_wins / n_total
    loss_rate = n_losses / n_total
    
    avg_win = wins.mean() if n_wins > 0 else 0.0
    avg_loss = losses.mean() if n_losses > 0 else 0.0
    
    # Win/loss ratio (average win / average loss)
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    
    # Profit factor (sum of wins / sum of losses)
    sum_wins = wins.sum() if n_wins > 0 else 0.0
    sum_losses = abs(losses.sum()) if n_losses > 0 else 0.0
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else np.inf
    
    # Consecutive streaks
    max_wins = compute_consecutive_streak(returns, True)
    max_losses = compute_consecutive_streak(returns, False)
    
    # Expectancy: expected return per trade
    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    
    return WinLossMetrics(
        win_rate=win_rate,
        loss_rate=loss_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        profit_factor=profit_factor,
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses,
        expectancy=expectancy,
    )


# =============================================================================
# IC ANALYSIS METRICS
# =============================================================================

def compute_ic_decay(
    ic_series: pd.Series,
    max_lag: int = 10,
) -> Dict[int, float]:
    """Compute autocorrelation of IC series at different lags.
    
    Higher autocorrelation suggests IC is persistent (predictable).
    
    Args:
        ic_series: Series of IC values.
        max_lag: Maximum lag to compute.
    
    Returns:
        Dictionary mapping lag to autocorrelation.
    """
    if len(ic_series) < max_lag + 5:
        return {}
    
    ic_clean = ic_series.dropna()
    
    decay = {}
    for lag in range(1, max_lag + 1):
        if len(ic_clean) > lag:
            autocorr = ic_clean.autocorr(lag=lag)
            decay[lag] = float(autocorr) if not pd.isna(autocorr) else 0.0
    
    return decay


def compute_ic_stability_metrics(ic_series: pd.Series) -> Dict[str, float]:
    """Compute stability metrics for IC series.
    
    Args:
        ic_series: Series of IC values.
    
    Returns:
        Dictionary with stability metrics.
    """
    if len(ic_series) < 5:
        return {
            "ic_positive_rate": np.nan,
            "ic_skewness": np.nan,
            "ic_kurtosis": np.nan,
            "ic_stability_score": np.nan,
        }
    
    ic_clean = ic_series.dropna()
    
    positive_rate = (ic_clean > 0).mean()
    skewness = stats.skew(ic_clean)
    kurtosis = stats.kurtosis(ic_clean)
    
    # Stability score: combines positive rate and low volatility
    # Higher is better
    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()
    
    if std_ic > 0:
        stability_score = positive_rate * (mean_ic / std_ic)
    else:
        stability_score = positive_rate if mean_ic > 0 else -positive_rate
    
    return {
        "ic_positive_rate": float(positive_rate),
        "ic_skewness": float(skewness),
        "ic_kurtosis": float(kurtosis),
        "ic_stability_score": float(stability_score),
    }


# =============================================================================
# DECILE ANALYSIS (10-group for finer granularity)
# =============================================================================

def compute_decile_returns(
    predicted: pd.Series,
    actual: pd.Series,
) -> Dict[int, float]:
    """Compute average return for each decile (10 groups).
    
    More granular than quintiles for detailed analysis.
    
    Args:
        predicted: Predicted scores.
        actual: Actual returns.
    
    Returns:
        Dictionary mapping decile number (1-10) to average return.
    """
    return compute_quintile_returns(predicted, actual, n_quantiles=10)


def compute_decile_spread(decile_returns: Dict[int, float]) -> float:
    """Compute spread between top and bottom deciles.
    
    Args:
        decile_returns: Dictionary from compute_decile_returns.
    
    Returns:
        D10 return - D1 return.
    """
    d10 = decile_returns.get(10, np.nan)
    d1 = decile_returns.get(1, np.nan)
    
    if pd.isna(d10) or pd.isna(d1):
        return np.nan
    
    return d10 - d1


def check_quintile_monotonicity(quintile_returns: Dict[int, float]) -> Dict[str, Any]:
    """Check if quintile returns are monotonically increasing.
    
    A good ranking model should show Q1 < Q2 < Q3 < Q4 < Q5.
    
    Args:
        quintile_returns: Dictionary of quintile returns.
    
    Returns:
        Dictionary with monotonicity analysis.
    """
    n_quintiles = len(quintile_returns)
    if n_quintiles < 2:
        return {
            "is_monotonic": False,
            "violations": 0,
            "monotonicity_score": 0.0,
        }
    
    returns = [quintile_returns.get(q, np.nan) for q in range(1, n_quintiles + 1)]
    
    # Count violations (where Q_i+1 <= Q_i)
    violations = 0
    for i in range(len(returns) - 1):
        if not pd.isna(returns[i]) and not pd.isna(returns[i+1]):
            if returns[i+1] <= returns[i]:
                violations += 1
    
    is_monotonic = violations == 0
    max_violations = n_quintiles - 1
    monotonicity_score = 1.0 - (violations / max_violations) if max_violations > 0 else 1.0
    
    return {
        "is_monotonic": is_monotonic,
        "violations": violations,
        "monotonicity_score": monotonicity_score,
    }


# =============================================================================
# COMPREHENSIVE METRICS DATACLASS
# =============================================================================

@dataclass
class ComprehensiveMetrics:
    """All metrics for comprehensive model evaluation.
    
    Combines ranking metrics, risk metrics, win/loss analysis, and stability metrics.
    """
    # Core ranking metrics
    ranking_metrics: RankingMetrics
    
    # Risk-adjusted metrics
    sortino_ratio: float
    calmar_ratio: float
    tail_ratio: float
    omega_ratio: float
    
    # Win/loss analysis
    win_loss_metrics: WinLossMetrics
    
    # IC stability
    ic_positive_rate: float
    ic_skewness: float
    ic_kurtosis: float
    ic_stability_score: float
    ic_decay: Dict[int, float]
    
    # Quintile/decile analysis
    decile_returns: Dict[int, float]
    decile_spread: float
    quintile_monotonicity: Dict[str, Any]
    
    # Statistical tests
    ic_ttest_pvalue: float  # p-value for H0: mean_ic = 0
    returns_ttest_pvalue: float  # p-value for H0: mean_return = 0
    
    @classmethod
    def from_predictions_and_returns(
        cls,
        predictions_df: pd.DataFrame,
        returns_series: pd.Series,
        timestamp_col: str = TIMESTAMP,
        predicted_col: str = "predicted_score",
        actual_col: str = "actual_return",
        min_stocks: int = 5,
        top_n_for_hit_rate: int = 10,
        forward_return_days: int = FORWARD_RETURN_DAYS,
    ) -> "ComprehensiveMetrics":
        """Create ComprehensiveMetrics from predictions and returns.
        
        Args:
            predictions_df: DataFrame with timestamp, predicted scores, actual returns.
            returns_series: Portfolio returns series.
            timestamp_col: Column name for timestamp.
            predicted_col: Column name for predicted scores.
            actual_col: Column name for actual returns.
            min_stocks: Minimum stocks per timestamp.
            top_n_for_hit_rate: N for hit rate calculation.
            forward_return_days: Forward return horizon in days.
        
        Returns:
            ComprehensiveMetrics instance.
        """
        # Core ranking metrics
        ranking_metrics = RankingMetrics.from_predictions(
            predictions_df, timestamp_col, predicted_col, actual_col,
            min_stocks, top_n_for_hit_rate, forward_return_days
        )
        
        # Periods per year for risk metrics
        periods_per_year = max(1, 252 // forward_return_days)
        
        # Risk-adjusted metrics
        sortino = compute_sortino_ratio(returns_series, periods_per_year=periods_per_year)
        calmar = compute_calmar_ratio(returns_series, periods_per_year=periods_per_year)
        tail = compute_tail_ratio(returns_series)
        omega = compute_omega_ratio(returns_series)
        
        # Win/loss analysis
        win_loss = compute_win_loss_metrics(returns_series)
        
        # IC stability
        ic_stability = compute_ic_stability_metrics(ranking_metrics.ic_series)
        ic_decay = compute_ic_decay(ranking_metrics.ic_series)
        
        # Decile analysis
        decile_rets = compute_decile_returns(
            predictions_df[predicted_col], 
            predictions_df[actual_col]
        )
        decile_sprd = compute_decile_spread(decile_rets)
        
        # Quintile monotonicity
        monotonicity = check_quintile_monotonicity(ranking_metrics.quintile_returns)
        
        # Statistical tests
        ic_clean = ranking_metrics.ic_series.dropna()
        if len(ic_clean) > 2:
            _, ic_pvalue = stats.ttest_1samp(ic_clean, 0)
        else:
            ic_pvalue = np.nan
        
        returns_clean = returns_series.dropna()
        if len(returns_clean) > 2:
            _, returns_pvalue = stats.ttest_1samp(returns_clean, 0)
        else:
            returns_pvalue = np.nan
        
        return cls(
            ranking_metrics=ranking_metrics,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            tail_ratio=tail,
            omega_ratio=omega,
            win_loss_metrics=win_loss,
            ic_positive_rate=ic_stability["ic_positive_rate"],
            ic_skewness=ic_stability["ic_skewness"],
            ic_kurtosis=ic_stability["ic_kurtosis"],
            ic_stability_score=ic_stability["ic_stability_score"],
            ic_decay=ic_decay,
            decile_returns=decile_rets,
            decile_spread=decile_sprd,
            quintile_monotonicity=monotonicity,
            ic_ttest_pvalue=ic_pvalue,
            returns_ttest_pvalue=returns_pvalue,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary for JSON serialization."""
        result = self.ranking_metrics.to_dict()
        
        # Add risk metrics
        result.update({
            "sortino_ratio": float(self.sortino_ratio) if not pd.isna(self.sortino_ratio) else None,
            "calmar_ratio": float(self.calmar_ratio) if not pd.isna(self.calmar_ratio) else None,
            "tail_ratio": float(self.tail_ratio) if not pd.isna(self.tail_ratio) else None,
            "omega_ratio": float(self.omega_ratio) if not pd.isna(self.omega_ratio) else None,
        })
        
        # Add win/loss metrics
        result["win_loss"] = self.win_loss_metrics.to_dict()
        
        # Add IC stability metrics
        result.update({
            "ic_positive_rate": float(self.ic_positive_rate) if not pd.isna(self.ic_positive_rate) else None,
            "ic_skewness": float(self.ic_skewness) if not pd.isna(self.ic_skewness) else None,
            "ic_kurtosis": float(self.ic_kurtosis) if not pd.isna(self.ic_kurtosis) else None,
            "ic_stability_score": float(self.ic_stability_score) if not pd.isna(self.ic_stability_score) else None,
            "ic_decay": {str(k): float(v) for k, v in self.ic_decay.items()},
        })
        
        # Add decile analysis
        result.update({
            "decile_returns": {str(k): float(v) if not pd.isna(v) else None 
                              for k, v in self.decile_returns.items()},
            "decile_spread": float(self.decile_spread) if not pd.isna(self.decile_spread) else None,
            "quintile_monotonicity": self.quintile_monotonicity,
        })
        
        # Add statistical tests
        result.update({
            "ic_ttest_pvalue": float(self.ic_ttest_pvalue) if not pd.isna(self.ic_ttest_pvalue) else None,
            "returns_ttest_pvalue": float(self.returns_ttest_pvalue) if not pd.isna(self.returns_ttest_pvalue) else None,
        })
        
        return result
    
    def summary(self) -> str:
        """Generate comprehensive summary."""
        lines = [
            "=" * 60,
            "COMPREHENSIVE RANKING MODEL EVALUATION",
            "=" * 60,
            "",
            "--- Core Ranking Metrics ---",
            f"Mean IC:           {self.ranking_metrics.mean_ic:.4f}" if not pd.isna(self.ranking_metrics.mean_ic) else "Mean IC:           N/A",
            f"ICIR:              {self.ranking_metrics.icir:.4f}" if not pd.isna(self.ranking_metrics.icir) else "ICIR:              N/A",
            f"Rank IC:           {self.ranking_metrics.mean_rank_ic:.4f}" if not pd.isna(self.ranking_metrics.mean_rank_ic) else "Rank IC:           N/A",
            f"Hit Rate:          {self.ranking_metrics.hit_rate_top_n:.2%}" if not pd.isna(self.ranking_metrics.hit_rate_top_n) else "Hit Rate:          N/A",
            f"Quintile Spread:   {self.ranking_metrics.quintile_spread:.4f}" if not pd.isna(self.ranking_metrics.quintile_spread) else "Quintile Spread:   N/A",
            "",
            "--- Risk-Adjusted Metrics ---",
            f"Sortino Ratio:     {self.sortino_ratio:.2f}" if not pd.isna(self.sortino_ratio) else "Sortino Ratio:     N/A",
            f"Calmar Ratio:      {self.calmar_ratio:.2f}" if not pd.isna(self.calmar_ratio) else "Calmar Ratio:      N/A",
            f"Tail Ratio:        {self.tail_ratio:.2f}" if not pd.isna(self.tail_ratio) else "Tail Ratio:        N/A",
            f"Omega Ratio:       {self.omega_ratio:.2f}" if not pd.isna(self.omega_ratio) else "Omega Ratio:       N/A",
            "",
            "--- Win/Loss Analysis ---",
            f"Win Rate:          {self.win_loss_metrics.win_rate:.2%}" if not pd.isna(self.win_loss_metrics.win_rate) else "Win Rate:          N/A",
            f"Profit Factor:     {self.win_loss_metrics.profit_factor:.2f}" if not pd.isna(self.win_loss_metrics.profit_factor) else "Profit Factor:     N/A",
            f"Max Win Streak:    {self.win_loss_metrics.max_consecutive_wins}",
            f"Max Loss Streak:   {self.win_loss_metrics.max_consecutive_losses}",
            f"Expectancy:        {self.win_loss_metrics.expectancy:.4f}" if not pd.isna(self.win_loss_metrics.expectancy) else "Expectancy:        N/A",
            "",
            "--- IC Stability ---",
            f"IC Positive Rate:  {self.ic_positive_rate:.2%}" if not pd.isna(self.ic_positive_rate) else "IC Positive Rate:  N/A",
            f"IC Stability Score:{self.ic_stability_score:.4f}" if not pd.isna(self.ic_stability_score) else "IC Stability Score:N/A",
            "",
            "--- Quintile Monotonicity ---",
            f"Is Monotonic:      {self.quintile_monotonicity.get('is_monotonic', 'N/A')}",
            f"Violations:        {self.quintile_monotonicity.get('violations', 'N/A')}",
            f"Monotonicity Score:{self.quintile_monotonicity.get('monotonicity_score', 0):.2f}",
            "",
            "--- Statistical Significance ---",
            f"IC t-test p-value: {self.ic_ttest_pvalue:.4f}" if not pd.isna(self.ic_ttest_pvalue) else "IC t-test p-value: N/A",
            f"Returns p-value:   {self.returns_ttest_pvalue:.4f}" if not pd.isna(self.returns_ttest_pvalue) else "Returns p-value:   N/A",
            "=" * 60,
        ]
        
        return "\n".join(lines)
