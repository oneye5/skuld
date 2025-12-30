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
        daily_returns: Series of daily portfolio returns after fees (index = timestamp).
        cumulative_returns: Series of cumulative returns after fees.
        sharpe_ratio: Annualized Sharpe ratio (post-fee).
        max_drawdown: Maximum drawdown as a fraction.
        total_return: Total return over the period (post-fee).
        avg_turnover: Average portfolio turnover per rebalance.
        quintile_returns: DataFrame with returns by quintile.
        holdings_history: DataFrame with timestamp, ticker, weight history.
        pre_fee_sharpe_ratio: Annualized Sharpe ratio before transaction costs.
        pre_fee_daily_returns: Series of daily portfolio returns before fees.
        pre_fee_total_return: Total return over the period (pre-fee).
        # Implementation metrics
        annualized_return_post_fee: Annualized return (post-fee).
        annualized_return_pre_fee: Annualized return (pre-fee).
        annualized_volatility: Annualized volatility of returns.
        total_cost_drag: Total cost impact as negative percentage of gross return.
        avg_cost_per_rebalance: Average cost per rebalance (percentage).
        num_rebalances: Number of rebalance periods.
        avg_holding_period_years: Average holding period in years.
        return_per_unit_turnover: Return / turnover ratio.
        calmar_ratio: Return / max drawdown.
    """
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    avg_turnover: float
    quintile_returns: Optional[pd.DataFrame] = None
    holdings_history: Optional[pd.DataFrame] = None
    pre_fee_sharpe_ratio: float = np.nan
    pre_fee_daily_returns: Optional[pd.Series] = None
    pre_fee_total_return: float = np.nan
    turnover_series: Optional[pd.Series] = None
    # Implementation metrics
    annualized_return_post_fee: float = np.nan
    annualized_return_pre_fee: float = np.nan
    annualized_volatility: float = np.nan
    total_cost_drag: float = np.nan
    avg_cost_per_rebalance: float = np.nan
    num_rebalances: int = 0
    avg_holding_period_years: float = np.nan
    return_per_unit_turnover: float = np.nan
    calmar_ratio: float = np.nan
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Backtest Results ===",
            "",
            "--- Returns ---",
            f"Total Return (post-fee):      {self.total_return:.2%}",
            f"Total Return (pre-fee):       {self.pre_fee_total_return:.2%}" if not np.isnan(self.pre_fee_total_return) else "Total Return (pre-fee):       N/A",
            f"Annualized Return (post-fee): {self.annualized_return_post_fee:.2%}" if not np.isnan(self.annualized_return_post_fee) else "Annualized Return (post-fee): N/A",
            f"Annualized Return (pre-fee):  {self.annualized_return_pre_fee:.2%}" if not np.isnan(self.annualized_return_pre_fee) else "Annualized Return (pre-fee):  N/A",
            f"Annualized Volatility:        {self.annualized_volatility:.2%}" if not np.isnan(self.annualized_volatility) else "Annualized Volatility:        N/A",
            "",
            "--- Risk Metrics ---",
            f"Sharpe Ratio (post-fee):      {self.sharpe_ratio:.2f}",
            f"Sharpe Ratio (pre-fee):       {self.pre_fee_sharpe_ratio:.2f}" if not np.isnan(self.pre_fee_sharpe_ratio) else "Sharpe Ratio (pre-fee):       N/A",
            f"Calmar Ratio (ret/drawdown):  {self.calmar_ratio:.2f}" if not np.isnan(self.calmar_ratio) else "Calmar Ratio:                 N/A",
            f"Max Drawdown:                 {self.max_drawdown:.2%}",
            "",
            "--- Implementation Metrics ---",
            f"Avg Turnover per Rebalance:   {self.avg_turnover:.2%}",
            f"Avg Cost per Rebalance:       {self.avg_cost_per_rebalance:.2%}" if not np.isnan(self.avg_cost_per_rebalance) else "Avg Cost per Rebalance:       N/A",
            f"Total Cost Drag:              {self.total_cost_drag:.2%}" if not np.isnan(self.total_cost_drag) else "Total Cost Drag:              N/A",
            f"Return per Unit Turnover:     {self.return_per_unit_turnover:.2f}" if not np.isnan(self.return_per_unit_turnover) else "Return per Unit Turnover:     N/A",
            f"Avg Holding Period:           {self.avg_holding_period_years:.2f} years" if not np.isnan(self.avg_holding_period_years) else "Avg Holding Period:           N/A",
            f"Num Rebalances:               {self.num_rebalances}",
            f"Num Periods:                  {len(self.daily_returns)}",
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
        except Exception as e:
            import warnings
            warnings.warn(f"Could not infer periods_per_year, defaulting to 252: {e}")
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
        
    if len(timestamps) < 2:
        print(f"WARNING: Only {len(timestamps)} period(s) for backtest after sampling.")
        print(f"  Return horizon: {return_horizon_days} days")
        print(f"  Total timestamps: {len(df[timestamp_col].unique())}")
        print("  Sharpe ratio will be NaN.")
    
    daily_returns = []  # post-fee returns
    daily_returns_pre_fee = []  # gross returns (before fees)
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
        
        # Compute gross return (pre-fee)
        gross_return = compute_portfolio_return(
            long_returns, short_returns, config.weighting
        )
        daily_returns_pre_fee.append({"timestamp": ts, "return": gross_return})
        
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
            pre_fee_sharpe_ratio=np.nan,
            pre_fee_daily_returns=pd.Series(dtype=float),
            pre_fee_total_return=np.nan,
            turnover_series=pd.Series(dtype=float),
        )
    
    # Convert to Series - post-fee (net) returns
    returns_df = pd.DataFrame(daily_returns)
    returns_series = returns_df.set_index("timestamp")["return"]
    
    # Convert to Series - pre-fee (gross) returns
    returns_pre_fee_df = pd.DataFrame(daily_returns_pre_fee)
    returns_pre_fee_series = returns_pre_fee_df.set_index("timestamp")["return"]

    turnover_series = pd.Series(turnovers, index=returns_series.index)
    
    # Compute cumulative returns (post-fee)
    cumulative = (1 + returns_series).cumprod() - 1
    
    # Compute cumulative returns (pre-fee)
    cumulative_pre_fee = (1 + returns_pre_fee_series).cumprod() - 1
    
    # Compute metrics
    # For proper annualization, use the return horizon to determine periods per year
    # E.g., 5-day returns = 252/5 ≈ 50 periods per year
    # Ensure at least 1 period per year to avoid division by zero
    periods_per_year = max(1, 252 // return_horizon_days)
    sharpe = compute_sharpe_ratio(returns_series, periods_per_year=periods_per_year)
    sharpe_pre_fee = compute_sharpe_ratio(returns_pre_fee_series, periods_per_year=periods_per_year)
    max_dd = compute_max_drawdown(cumulative)
    total_return = cumulative.iloc[-1] if len(cumulative) > 0 else 0.0
    total_return_pre_fee = cumulative_pre_fee.iloc[-1] if len(cumulative_pre_fee) > 0 else 0.0
    avg_turnover = np.mean(turnovers) if turnovers else 0.0
    
    # Compute implementation metrics
    num_rebalances = len(returns_series)
    
    # Calculate annualized returns
    # Annualized Return = (1 + Total Return)^(365 / Total Days) - 1
    if num_rebalances >= 2 and len(returns_series.index) >= 2:
        # Calculate total time span in years
        # Timestamps are in milliseconds
        MS_PER_DAY_LOCAL = 86_400_000
        first_ts = returns_series.index[0]
        last_ts = returns_series.index[-1]
        total_days = (last_ts - first_ts) / MS_PER_DAY_LOCAL
        total_years = total_days / 365.0
        
        # Calculate annualized returns
        if total_years > 0:
            annualized_return_post_fee = (1 + total_return) ** (1 / total_years) - 1
            annualized_return_pre_fee = (1 + total_return_pre_fee) ** (1 / total_years) - 1
        else:
            annualized_return_post_fee = np.nan
            annualized_return_pre_fee = np.nan
            
        # Calculate annualized volatility
        annualized_volatility = returns_series.std() * np.sqrt(periods_per_year)
        
        # Cost analysis
        total_cost_impact = (returns_pre_fee_series - returns_series).sum()
        # Cost drag as percentage of gross return (should be negative, representing loss)
        total_cost_drag = -total_cost_impact / (1 + total_return_pre_fee) if total_return_pre_fee > 0 else np.nan
        avg_cost_per_rebalance = total_cost_impact / num_rebalances if num_rebalances > 0 else np.nan
        
        # Average holding period in years
        avg_holding_period_years = total_years / num_rebalances if num_rebalances > 0 else np.nan
        
        # Return per unit turnover
        return_per_unit_turnover = total_return / avg_turnover if avg_turnover > 0 else np.nan
        
        # Calmar ratio (handle edge case where drawdown is 0)
        calmar_ratio = annualized_return_post_fee / max_dd if max_dd > 0.0001 else np.nan
    else:
        annualized_return_post_fee = np.nan
        annualized_return_pre_fee = np.nan
        annualized_volatility = np.nan
        total_cost_drag = np.nan
        avg_cost_per_rebalance = np.nan
        avg_holding_period_years = np.nan
        return_per_unit_turnover = np.nan
        calmar_ratio = np.nan
    
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
        pre_fee_sharpe_ratio=sharpe_pre_fee,
        pre_fee_daily_returns=returns_pre_fee_series,
        pre_fee_total_return=total_return_pre_fee,
        turnover_series=turnover_series,
        # Implementation metrics
        annualized_return_post_fee=annualized_return_post_fee,
        annualized_return_pre_fee=annualized_return_pre_fee,
        annualized_volatility=annualized_volatility,
        total_cost_drag=total_cost_drag,
        avg_cost_per_rebalance=avg_cost_per_rebalance,
        num_rebalances=num_rebalances,
        avg_holding_period_years=avg_holding_period_years,
        return_per_unit_turnover=return_per_unit_turnover,
        calmar_ratio=calmar_ratio,
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


# =============================================================================
# RANDOM BASELINE
# =============================================================================

@dataclass
class RandomBaselineResult:
    """Results from random baseline comparison.
    
    Attributes:
        mean_sharpe_post_fee: Mean Sharpe ratio across random trials (post-fee).
        std_sharpe_post_fee: Std dev of Sharpe ratio across random trials (post-fee).
        mean_sharpe_pre_fee: Mean Sharpe ratio across random trials (pre-fee).
        std_sharpe_pre_fee: Std dev of Sharpe ratio across random trials (pre-fee).
        mean_total_return: Mean total return across random trials (post-fee).
        std_total_return: Std dev of total return across random trials (post-fee).
        n_trials: Number of random trials performed.
        all_sharpes_post_fee: List of all individual trial Sharpe ratios (post-fee).
        all_sharpes_pre_fee: List of all individual trial Sharpe ratios (pre-fee).
        percentile_of_model: Percentile rank of model Sharpe among random trials (if provided).
    """
    mean_sharpe_post_fee: float
    std_sharpe_post_fee: float
    mean_sharpe_pre_fee: float
    std_sharpe_pre_fee: float
    mean_total_return: float
    std_total_return: float
    n_trials: int
    all_sharpes_post_fee: List[float]
    all_sharpes_pre_fee: List[float]
    percentile_of_model: Optional[float] = None
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Random Baseline Results ===",
            f"Number of trials:          {self.n_trials}",
            f"Mean Sharpe (post-fee):    {self.mean_sharpe_post_fee:.2f} ± {self.std_sharpe_post_fee:.2f}",
            f"Mean Sharpe (pre-fee):     {self.mean_sharpe_pre_fee:.2f} ± {self.std_sharpe_pre_fee:.2f}",
            f"Mean Total Return:         {self.mean_total_return:.2%} ± {self.std_total_return:.2%}",
        ]
        if self.percentile_of_model is not None:
            lines.append(f"Model percentile vs random: {self.percentile_of_model:.1f}%")
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        def to_native(x):
            """Convert numpy types to native Python types."""
            if x is None:
                return None
            if isinstance(x, (np.floating, np.integer)):
                return float(x)
            return x
        
        return {
            "mean_sharpe_post_fee": to_native(self.mean_sharpe_post_fee),
            "std_sharpe_post_fee": to_native(self.std_sharpe_post_fee),
            "mean_sharpe_pre_fee": to_native(self.mean_sharpe_pre_fee),
            "std_sharpe_pre_fee": to_native(self.std_sharpe_pre_fee),
            "mean_total_return": to_native(self.mean_total_return),
            "std_total_return": to_native(self.std_total_return),
            "n_trials": self.n_trials,
            "percentile_of_model": to_native(self.percentile_of_model),
        }


def run_random_baseline(
    df: pd.DataFrame,
    config: PortfolioConfig,
    timestamp_col: str = TIMESTAMP,
    ticker_col: str = TICKER,
    return_col: str = "actual_return",
    return_horizon_days: int = 1,
    n_trials: int = 100,
    model_sharpe: Optional[float] = None,
    random_seed: int = 42,
) -> RandomBaselineResult:
    """Run random stock selection baseline for comparison.
    
    At each timestamp, instead of using model predictions, randomly select
    top_n and bottom_n stocks. Run multiple trials to get distribution.
    
    Args:
        df: DataFrame with timestamp, ticker, actual_return.
        config: Portfolio configuration (top_n, bottom_n, costs).
        timestamp_col: Column name for timestamp.
        ticker_col: Column name for ticker.
        return_col: Column name for actual returns.
        return_horizon_days: The horizon of the returns in days.
        n_trials: Number of random trials to run.
        model_sharpe: Optional model Sharpe ratio to compute percentile.
        random_seed: Random seed for reproducibility.
    
    Returns:
        RandomBaselineResult with statistics across random trials.
    """
    np.random.seed(random_seed)
    
    all_sharpes_post_fee = []
    all_sharpes_pre_fee = []
    all_total_returns = []
    
    for trial in range(n_trials):
        # Create random predictions for each trial
        trial_df = df.copy()
        trial_df["random_score"] = np.random.randn(len(trial_df))
        
        # Run backtest with random scores
        result = run_portfolio_backtest(
            trial_df,
            config,
            timestamp_col=timestamp_col,
            ticker_col=ticker_col,
            score_col="random_score",
            return_col=return_col,
            return_horizon_days=return_horizon_days,
        )
        
        if not np.isnan(result.sharpe_ratio):
            all_sharpes_post_fee.append(result.sharpe_ratio)
            all_sharpes_pre_fee.append(result.pre_fee_sharpe_ratio)
            all_total_returns.append(result.total_return)
    
    if not all_sharpes_post_fee:
        return RandomBaselineResult(
            mean_sharpe_post_fee=np.nan,
            std_sharpe_post_fee=np.nan,
            mean_sharpe_pre_fee=np.nan,
            std_sharpe_pre_fee=np.nan,
            mean_total_return=np.nan,
            std_total_return=np.nan,
            n_trials=0,
            all_sharpes_post_fee=[],
            all_sharpes_pre_fee=[],
        )
    
    # Compute statistics
    mean_sharpe_post = np.mean(all_sharpes_post_fee)
    std_sharpe_post = np.std(all_sharpes_post_fee)
    mean_sharpe_pre = np.mean(all_sharpes_pre_fee)
    std_sharpe_pre = np.std(all_sharpes_pre_fee)
    mean_return = np.mean(all_total_returns)
    std_return = np.std(all_total_returns)
    
    # Compute percentile of model if provided
    percentile = None
    if model_sharpe is not None and not np.isnan(model_sharpe):
        percentile = 100 * np.mean([s <= model_sharpe for s in all_sharpes_post_fee])
    
    return RandomBaselineResult(
        mean_sharpe_post_fee=mean_sharpe_post,
        std_sharpe_post_fee=std_sharpe_post,
        mean_sharpe_pre_fee=mean_sharpe_pre,
        std_sharpe_pre_fee=std_sharpe_pre,
        mean_total_return=mean_return,
        std_total_return=std_return,
        n_trials=len(all_sharpes_post_fee),
        all_sharpes_post_fee=all_sharpes_post_fee,
        all_sharpes_pre_fee=all_sharpes_pre_fee,
        percentile_of_model=percentile,
    )
