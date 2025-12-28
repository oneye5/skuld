"""Portfolio simulator for ranking-based stock prediction backtesting.

This module implements long-short portfolio construction and backtesting
based on ranking model predictions. It supports:
- Long-short and long-only strategies
- Equal-weight and score-weighted portfolios
- Transaction cost modeling
- Sharpe ratio, max drawdown, and other performance metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction and backtesting.
    
    Attributes:
        top_n: Number of top-ranked stocks for long portfolio.
        bottom_n: Number of bottom-ranked stocks for short portfolio.
        weighting: Portfolio weighting scheme ('equal' or 'score_weighted').
        transaction_cost_bps: Round-trip transaction cost in basis points.
        slippage_bps: Slippage in basis points per trade (market impact, bid-ask spread).
        long_only: If True, only take long positions (no shorting).
        initial_capital: Starting capital for simulation.
    """
    top_n: int = 10
    bottom_n: int = 10
    weighting: str = "equal"
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 0.0
    long_only: bool = False
    initial_capital: float = 100_000.0
    
    @property
    def total_cost_bps(self) -> float:
        """Total trading cost (transaction + slippage) in basis points."""
        return self.transaction_cost_bps + self.slippage_bps


@dataclass
class BacktestResult:
    """Results from portfolio backtest.
    
    Attributes:
        daily_returns: Series of daily portfolio returns (index = timestamp).
        cumulative_returns: Series of cumulative returns.
        sharpe_ratio: Annualized Sharpe ratio.
        max_drawdown: Maximum drawdown as a fraction.
        total_return: Total return over the period.
        avg_turnover: Average portfolio turnover per rebalance.
        quintile_returns: DataFrame with returns by quintile.
        holdings_history: DataFrame with timestamp, ticker, weight history.
    """
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    avg_turnover: float
    quintile_returns: Optional[pd.DataFrame] = None
    holdings_history: Optional[pd.DataFrame] = None
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Backtest Results ===",
            f"Total Return:   {self.total_return:.2%}",
            f"Sharpe Ratio:   {self.sharpe_ratio:.2f}",
            f"Max Drawdown:   {self.max_drawdown:.2%}",
            f"Avg Turnover:   {self.avg_turnover:.2%}",
            f"Num Periods:    {len(self.daily_returns)}",
        ]
        return "\n".join(lines)


# =============================================================================
# STOCK SELECTION
# =============================================================================

def select_top_n_stocks(
    df: pd.DataFrame,
    n: int,
    score_col: str = "predicted_score",
) -> pd.DataFrame:
    """Select top-N stocks by predicted score.
    
    Args:
        df: DataFrame with ticker and score columns.
        n: Number of top stocks to select.
        score_col: Column name for prediction scores.
    
    Returns:
        DataFrame with top-N stocks.
    """
    return df.nlargest(n, score_col)


def select_bottom_n_stocks(
    df: pd.DataFrame,
    n: int,
    score_col: str = "predicted_score",
) -> pd.DataFrame:
    """Select bottom-N stocks by predicted score.
    
    Args:
        df: DataFrame with ticker and score columns.
        n: Number of bottom stocks to select.
        score_col: Column name for prediction scores.
    
    Returns:
        DataFrame with bottom-N stocks.
    """
    return df.nsmallest(n, score_col)


# =============================================================================
# PORTFOLIO RETURN CALCULATION
# =============================================================================

def compute_portfolio_return(
    long_returns: pd.Series,
    short_returns: pd.Series,
    weighting: str = "equal",
    long_weights: Optional[pd.Series] = None,
    short_weights: Optional[pd.Series] = None,
) -> float:
    """Compute portfolio return from long and short positions.
    
    For equal weighting:
        Long-short return = mean(long_returns) - mean(short_returns)
    
    For weighted:
        Long-short return = sum(long_weights * long_returns) - sum(short_weights * short_returns)
    
    Args:
        long_returns: Returns of long positions.
        short_returns: Returns of short positions (can be empty for long-only).
        weighting: 'equal' or 'score_weighted'.
        long_weights: Weights for long positions (required if weighting='score_weighted').
        short_weights: Weights for short positions.
    
    Returns:
        Portfolio return for the period.
    """
    if weighting == "equal":
        long_return = long_returns.mean() if len(long_returns) > 0 else 0.0
        short_return = short_returns.mean() if len(short_returns) > 0 else 0.0
    else:
        if long_weights is None:
            raise ValueError("long_weights required for non-equal weighting")
        long_return = (long_returns * long_weights).sum() if len(long_returns) > 0 else 0.0
        short_return = (short_returns * short_weights).sum() if len(short_returns) > 0 else 0.0
    
    # Long-short: we gain from long positions and from shorts declining
    return long_return - short_return


def apply_transaction_costs(
    gross_return: float,
    turnover: float,
    cost_bps: float,
) -> float:
    """Apply transaction costs to gross return.
    
    Args:
        gross_return: Return before costs.
        turnover: Portfolio turnover as a fraction (0.5 = 50% of positions changed).
        cost_bps: Round-trip transaction cost in basis points.
    
    Returns:
        Net return after transaction costs.
    """
    cost = turnover * cost_bps / 10_000
    return gross_return - cost


# =============================================================================
# TURNOVER CALCULATION
# =============================================================================

def compute_turnover(
    prev_holdings: Dict[str, float],
    curr_holdings: Dict[str, float],
) -> float:
    """Compute portfolio turnover between two periods.
    
    Turnover = sum of absolute weight changes / 2
    (divide by 2 because we count both buys and sells)
    
    Args:
        prev_holdings: Previous period holdings {ticker: weight}.
        curr_holdings: Current period holdings {ticker: weight}.
    
    Returns:
        Turnover as a fraction (1.0 = 100% turnover).
    """
    all_tickers = set(prev_holdings.keys()) | set(curr_holdings.keys())
    
    total_change = 0.0
    for ticker in all_tickers:
        prev_weight = prev_holdings.get(ticker, 0.0)
        curr_weight = curr_holdings.get(ticker, 0.0)
        total_change += abs(curr_weight - prev_weight)
    
    # Divide by 2 because a complete replacement counts as 1.0 turnover
    return total_change / 2


def build_equal_weight_holdings(
    long_tickers: List[str],
    short_tickers: List[str],
) -> Dict[str, float]:
    """Build equal-weight holdings dictionary.
    
    Args:
        long_tickers: Tickers in long portfolio.
        short_tickers: Tickers in short portfolio.
    
    Returns:
        Dictionary mapping ticker to weight (positive for long, negative for short).
    """
    holdings = {}
    
    n_long = len(long_tickers)
    n_short = len(short_tickers)
    
    if n_long > 0:
        long_weight = 1.0 / n_long
        for ticker in long_tickers:
            holdings[ticker] = long_weight
    
    if n_short > 0:
        short_weight = -1.0 / n_short
        for ticker in short_tickers:
            holdings[ticker] = short_weight
    
    return holdings


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def infer_periods_per_year(timestamps: pd.Series) -> int:
    """Infer the number of periods per year from timestamp data.
    
    Analyzes the median spacing between consecutive timestamps to determine
    the appropriate annualization factor.
    
    Args:
        timestamps: Series of timestamps (in milliseconds).
    
    Returns:
        Estimated number of periods per year.
    """
    if len(timestamps) < 2:
        return 252  # Default to daily
    
    # Calculate median spacing in days
    # Using MS_PER_DAY from config
    MS_PER_DAY_LOCAL = 86_400_000
    ts_sorted = sorted(timestamps)
    spacings = [(ts_sorted[i+1] - ts_sorted[i]) / MS_PER_DAY_LOCAL 
                for i in range(len(ts_sorted) - 1)]
    
    median_spacing_days = np.median(spacings)
    
    if median_spacing_days < 0.5:
        # Intraday - assume 252 trading days
        return 252
    elif median_spacing_days < 3:
        # Daily
        return 252
    elif median_spacing_days < 8:
        # Weekly (5-7 days)
        return 52
    elif median_spacing_days < 25:
        # Monthly
        return 12
    else:
        # Longer period - compute directly
        return max(1, int(365 / median_spacing_days))


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
) -> float:
    """Compute annualized Sharpe ratio.
    
    Args:
        returns: Series of periodic returns. Index should be timestamps.
        risk_free_rate: Annual risk-free rate (default 0).
        periods_per_year: Number of periods per year. If None, inferred from
                         timestamps in the returns index.
    
    Returns:
        Annualized Sharpe ratio.
    
    Note:
        If periods_per_year is not provided and cannot be inferred,
        defaults to 252 (daily).
    """
    if len(returns) < 2:
        return np.nan
    
    # Infer periods_per_year if not provided
    if periods_per_year is None:
        try:
            periods_per_year = infer_periods_per_year(pd.Series(returns.index))
        except:
            periods_per_year = 252  # Fallback to daily
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    
    if std_return == 0 or np.isnan(std_return):
        return np.inf if mean_return > 0 else (-np.inf if mean_return < 0 else 0.0)
    
    return (mean_return / std_return) * np.sqrt(periods_per_year)


def compute_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Compute maximum drawdown.
    
    Max drawdown = largest peak-to-trough decline in cumulative return.
    
    Args:
        cumulative_returns: Series of cumulative returns.
    
    Returns:
        Maximum drawdown as a positive fraction.
    """
    if len(cumulative_returns) < 2:
        return 0.0
    
    # Convert to wealth (1 + cumulative return)
    wealth = 1 + cumulative_returns
    
    # Running maximum
    running_max = wealth.cummax()
    
    # Drawdown at each point
    drawdown = (running_max - wealth) / running_max
    
    return drawdown.max()


# =============================================================================
# MAIN BACKTEST FUNCTION
# =============================================================================

def run_portfolio_backtest(
    df: pd.DataFrame,
    config: PortfolioConfig,
    timestamp_col: str = TIMESTAMP,
    ticker_col: str = TICKER,
    score_col: str = "predicted_score",
    return_col: str = "actual_return",
    return_horizon_days: int = 1,
) -> BacktestResult:
    """Run long-short portfolio backtest.
    
    At each timestamp:
    1. Rank stocks by predicted_score.
    2. Long top-N, short bottom-N (equal weight or score-weighted).
    3. Compute portfolio return = mean(long returns) - mean(short returns).
    4. Apply transaction costs based on turnover.
    
    Args:
        df: DataFrame with timestamp, ticker, predicted_score, actual_return.
        config: Portfolio configuration.
        timestamp_col: Column name for timestamp.
        ticker_col: Column name for ticker.
        score_col: Column name for predicted scores.
        return_col: Column name for actual returns.
        return_horizon_days: The horizon of the returns in days (e.g., 5 for 5-day returns).
                            Used for proper Sharpe ratio annualization.
    
    Returns:
        BacktestResult with performance metrics.
    """
    timestamps = sorted(df[timestamp_col].unique())
    
    # Sample timestamps at intervals matching the return horizon to avoid overlapping periods
    # For example, with 126-day returns, we should only rebalance every ~126 days
    # to avoid compounding overlapping returns
    if return_horizon_days > 1:
        # Sample every return_horizon_days timestamps
        timestamps = timestamps[::return_horizon_days]
    
    daily_returns = []
    turnovers = []
    prev_holdings: Dict[str, float] = {}
    holdings_records = []
    
    for ts in timestamps:
        ts_df = df[df[timestamp_col] == ts].copy()
        
        if len(ts_df) < config.top_n + config.bottom_n:
            continue
        
        # Select long and short positions
        long_df = select_top_n_stocks(ts_df, config.top_n, score_col)
        long_tickers = long_df[ticker_col].tolist()
        long_returns = long_df[return_col]
        
        if config.long_only:
            short_tickers = []
            short_returns = pd.Series(dtype=float)
        else:
            short_df = select_bottom_n_stocks(ts_df, config.bottom_n, score_col)
            short_tickers = short_df[ticker_col].tolist()
            short_returns = short_df[return_col]
        
        # Build current holdings
        curr_holdings = build_equal_weight_holdings(long_tickers, short_tickers)
        
        # Record holdings
        for ticker, weight in curr_holdings.items():
            holdings_records.append({
                timestamp_col: ts,
                ticker_col: ticker,
                "weight": weight,
            })
        
        # Compute turnover
        turnover = compute_turnover(prev_holdings, curr_holdings)
        turnovers.append(turnover)
        
        # Compute gross return
        gross_return = compute_portfolio_return(
            long_returns, short_returns, config.weighting
        )
        
        # Apply transaction costs (including slippage)
        net_return = apply_transaction_costs(
            gross_return, turnover, config.total_cost_bps
        )
        
        daily_returns.append({"timestamp": ts, "return": net_return})
        prev_holdings = curr_holdings
    
    if not daily_returns:
        return BacktestResult(
            daily_returns=pd.Series(dtype=float),
            cumulative_returns=pd.Series(dtype=float),
            sharpe_ratio=np.nan,
            max_drawdown=np.nan,
            total_return=np.nan,
            avg_turnover=np.nan,
        )
    
    # Convert to Series
    returns_df = pd.DataFrame(daily_returns)
    returns_series = returns_df.set_index("timestamp")["return"]
    
    # Compute cumulative returns
    cumulative = (1 + returns_series).cumprod() - 1
    
    # Compute metrics
    # For proper annualization, use the return horizon to determine periods per year
    # E.g., 5-day returns = 252/5 ≈ 50 periods per year
    # Ensure at least 1 period per year to avoid division by zero
    periods_per_year = max(1, 252 // return_horizon_days)
    sharpe = compute_sharpe_ratio(returns_series, periods_per_year=periods_per_year)
    max_dd = compute_max_drawdown(cumulative)
    total_return = cumulative.iloc[-1] if len(cumulative) > 0 else 0.0
    avg_turnover = np.mean(turnovers) if turnovers else 0.0
    
    # Build holdings history
    holdings_df = pd.DataFrame(holdings_records) if holdings_records else None
    
    return BacktestResult(
        daily_returns=returns_series,
        cumulative_returns=cumulative,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        total_return=total_return,
        avg_turnover=avg_turnover,
        holdings_history=holdings_df,
    )


# =============================================================================
# QUINTILE ANALYSIS
# =============================================================================

def compute_quintile_portfolio_returns(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP,
    score_col: str = "predicted_score",
    return_col: str = "actual_return",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Compute returns for each quintile portfolio at each timestamp.
    
    This is useful for analyzing whether the model properly separates
    winners from losers. Q5 should have highest returns if model is good.
    
    Args:
        df: DataFrame with timestamp, predicted_score, actual_return.
        timestamp_col: Column name for timestamp.
        score_col: Column name for predicted scores.
        return_col: Column name for actual returns.
        n_quantiles: Number of quantile groups.
    
    Returns:
        DataFrame with columns [Q1, Q2, ..., Qn] and index = timestamps.
    """
    results = []
    
    for ts, ts_df in df.groupby(timestamp_col):
        if len(ts_df) < n_quantiles:
            continue
        
        # Assign quintiles based on predicted score
        try:
            quintiles = pd.qcut(
                ts_df[score_col], 
                q=n_quantiles, 
                labels=False, 
                duplicates='drop'
            ) + 1
        except ValueError:
            # Fall back to rank-based assignment
            ranks = ts_df[score_col].rank(method='first')
            quintiles = pd.cut(ranks, bins=n_quantiles, labels=False) + 1
        
        ts_df = ts_df.copy()
        ts_df["quintile"] = quintiles
        
        # Compute mean return per quintile
        row = {timestamp_col: ts}
        for q in range(1, n_quantiles + 1):
            q_returns = ts_df[ts_df["quintile"] == q][return_col]
            row[f"Q{q}"] = q_returns.mean() if len(q_returns) > 0 else np.nan
        
        results.append(row)
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    result_df = result_df.set_index(timestamp_col)
    
    return result_df
