"""Portfolio simulator for ranking-based stock prediction backtesting.

This module implements long-short portfolio construction and backtesting
based on ranking model predictions. It supports:
- Long-short and long-only strategies
- Equal-weight and score-weighted portfolios
- Transaction cost modeling
- Sharpe ratio, max drawdown, and other performance metrics
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER


# Module-level warning tracking to avoid spam
_warned_long_holding_period = False
_warned_low_sample_size = False


# =============================================================================
# ANNUAL STATISTICS FOR REAL-WORLD IMPLEMENTATION
# =============================================================================

@dataclass
class AnnualStatistics:
    """Statistics computed from resampling daily returns into annual periods.
    
    These metrics help understand expected annual performance for real-world implementation
    and can be used for Monte Carlo simulations and risk modeling.
    
    Attributes:
        mean_annual_return: Mean of annual returns.
        median_annual_return: Median of annual returns.
        std_annual_return: Standard deviation of annual returns.
        min_annual_return: Worst annual return observed.
        max_annual_return: Best annual return observed.
        pct_5_annual_return: 5th percentile of annual returns.
        pct_25_annual_return: 25th percentile (Q1) of annual returns.
        pct_75_annual_return: 75th percentile (Q3) of annual returns.
        pct_95_annual_return: 95th percentile of annual returns.
        pct_positive_years: Percentage of years with positive returns.
        avg_positive_year: Average return in positive years.
        avg_negative_year: Average return in negative years.
        skewness_annual: Skewness of annual return distribution.
        kurtosis_annual: Excess kurtosis of annual return distribution.
        sharpe_annual_avg: Average annual Sharpe ratio.
        num_years: Number of complete years in sample.
        years_sampled: Actual years covered in the data.
    """
    mean_annual_return: float
    median_annual_return: float
    std_annual_return: float
    min_annual_return: float
    max_annual_return: float
    pct_5_annual_return: float
    pct_25_annual_return: float
    pct_75_annual_return: float
    pct_95_annual_return: float
    pct_positive_years: float
    avg_positive_year: float
    avg_negative_year: float
    skewness_annual: float
    kurtosis_annual: float
    sharpe_annual_avg: float
    num_years: int
    years_sampled: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            'mean_annual_return': self.mean_annual_return,
            'median_annual_return': self.median_annual_return,
            'std_annual_return': self.std_annual_return,
            'min_annual_return': self.min_annual_return,
            'max_annual_return': self.max_annual_return,
            'pct_5_annual_return': self.pct_5_annual_return,
            'pct_25_annual_return': self.pct_25_annual_return,
            'pct_75_annual_return': self.pct_75_annual_return,
            'pct_95_annual_return': self.pct_95_annual_return,
            'pct_positive_years': self.pct_positive_years,
            'avg_positive_year': self.avg_positive_year,
            'avg_negative_year': self.avg_negative_year,
            'skewness_annual': self.skewness_annual,
            'kurtosis_annual': self.kurtosis_annual,
            'sharpe_annual_avg': self.sharpe_annual_avg,
            'num_years': self.num_years,
            'years_sampled': self.years_sampled,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Annual Return Statistics ===",
            f"Years Covered:        {self.years_sampled:.1f}",
            f"Complete Years:       {self.num_years}",
            "",
            "--- Return Distribution ---",
            f"Mean Annual Return:   {self.mean_annual_return:>8.2%}",
            f"Median Annual Return: {self.median_annual_return:>8.2%}",
            f"Std Dev (Annual):     {self.std_annual_return:>8.2%}",
            "",
            "--- Percentiles ---",
            f"5th Percentile:       {self.pct_5_annual_return:>8.2%}",
            f"25th Percentile (Q1): {self.pct_25_annual_return:>8.2%}",
            f"75th Percentile (Q3): {self.pct_75_annual_return:>8.2%}",
            f"95th Percentile:      {self.pct_95_annual_return:>8.2%}",
            "",
            "--- Range ---",
            f"Best Year:            {self.max_annual_return:>8.2%}",
            f"Worst Year:           {self.min_annual_return:>8.2%}",
            "",
            "--- Win/Loss Profile ---",
            f"% Positive Years:     {self.pct_positive_years:>8.1%}",
            f"Avg Winning Year:     {self.avg_positive_year:>8.2%}",
            f"Avg Losing Year:      {self.avg_negative_year:>8.2%}",
            "",
            "--- Shape ---",
            f"Skewness:             {self.skewness_annual:>8.2f}",
            f"Excess Kurtosis:      {self.kurtosis_annual:>8.2f}",
            "",
            "--- Risk-Adjusted ---",
            f"Avg Annual Sharpe:    {self.sharpe_annual_avg:>8.2f}",
        ]
        return "\n".join(lines)


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
        daily_returns: Series of portfolio returns per rebalance period (index = timestamp).
                       NOTE: Despite the name, these are per-period returns, not daily returns.
                       With long holding periods (e.g., 365 days), each return represents
                       the full holding period return.
        cumulative_returns: Series of cumulative returns after fees.
        sharpe_ratio: Annualized Sharpe ratio (post-fee), properly adjusted for holding period.
        max_drawdown: Maximum drawdown as a fraction. NOTE: With long holding periods,
                      this only captures drawdown between rebalance points, not intra-period.
        total_return: Total return over the period (post-fee).
        avg_turnover: Average portfolio turnover per rebalance.
        quintile_returns: DataFrame with returns by quintile.
        holdings_history: DataFrame with timestamp, ticker, weight history.
        pre_fee_sharpe_ratio: Annualized Sharpe ratio before transaction costs.
        pre_fee_daily_returns: Series of portfolio returns per rebalance period (pre-fee).
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
    
    Warnings:
        When holding periods are long (e.g., annual rebalancing), be cautious interpreting:
        - max_drawdown: Only measures drawdown at rebalance points, not intra-period
        - Sharpe ratio: Based on few observations, may have high estimation error
        - Metrics like Sortino, Calmar may be undefined if all periods are positive
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
    true_daily_returns: Optional[pd.Series] = None  # Continuous daily returns (if computed)
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
    annual_stats: Optional[AnnualStatistics] = None  # Annual return distribution statistics
    
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
        
        # Add annual statistics if available
        if self.annual_stats is not None:
            lines.append("")
            lines.append(self.annual_stats.summary())
        
        # Add warnings for potential issues
        warnings = []
        if self.num_rebalances < 10:
            warnings.append(f"[!] Low sample size ({self.num_rebalances} periods) - metrics may be unreliable")
        if self.max_drawdown == 0 and self.num_rebalances > 1:
            warnings.append("[!] Max drawdown = 0 (all periods positive) - Calmar ratio undefined")
        if not np.isnan(self.avg_holding_period_years) and self.avg_holding_period_years > 0.5:
            warnings.append(f"[!] Long holding period ({self.avg_holding_period_years:.1f}y) - drawdown only at rebalance points")
        
        if warnings:
            lines.append("")
            lines.append("--- Warnings ---")
            lines.extend(warnings)
        
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
# DATA QUALITY UTILITIES
# =============================================================================

def detect_suspicious_tickers(
    price_data: pd.DataFrame,
    split_data: pd.DataFrame | None = None,
    timestamp_col: str = TIMESTAMP,
    ticker_col: str = TICKER,
    price_col: str = "Close",
    return_threshold: float = 2.0,
) -> dict:
    """Detect tickers with suspicious price behavior indicating data quality issues.
    
    This function identifies tickers that have:
    1. Extreme daily returns (>200% by default) suggesting unadjusted splits
    2. Split events in the data that may not be properly reflected in prices
    3. Price level discontinuities that don't match recorded splits
    
    Args:
        price_data: DataFrame with price data (timestamp, ticker, price columns).
        split_data: Optional DataFrame with split information (timestamp, ticker, split_ratio).
        timestamp_col: Column name for timestamp.
        ticker_col: Column name for ticker.
        price_col: Column name for price.
        return_threshold: Absolute return threshold to flag as suspicious (default 2.0 = 200%).
    
    Returns:
        Dictionary with:
            - 'suspicious_tickers': List of ticker symbols with data issues
            - 'extreme_moves': DataFrame of extreme price moves detected
            - 'unmatched_splits': DataFrame of splits without corresponding price adjustments
            - 'summary': Human-readable summary string
    """
    result = {
        'suspicious_tickers': [],
        'extreme_moves': pd.DataFrame(),
        'unmatched_splits': pd.DataFrame(),
        'summary': '',
    }
    
    if price_data is None or price_data.empty:
        return result
    
    df = price_data.copy()
    df = df.sort_values([ticker_col, timestamp_col])
    
    # Calculate daily returns
    df['_prev_price'] = df.groupby(ticker_col)[price_col].shift(1)
    df['_daily_ret'] = (df[price_col] - df['_prev_price']) / df['_prev_price']
    
    # Find extreme moves
    extreme_mask = abs(df['_daily_ret']) > return_threshold
    extreme_moves = df[extreme_mask][[timestamp_col, ticker_col, price_col, '_prev_price', '_daily_ret']].copy()
    extreme_moves.columns = [timestamp_col, ticker_col, 'price', 'prev_price', 'daily_return']
    
    if len(extreme_moves) > 0:
        extreme_moves['date'] = pd.to_datetime(extreme_moves[timestamp_col], unit='ms')
        result['extreme_moves'] = extreme_moves
        result['suspicious_tickers'] = list(extreme_moves[ticker_col].unique())
    
    # Check split data if provided
    if split_data is not None and not split_data.empty:
        # Find splits that aren't 1.0 (actual split events)
        actual_splits = split_data[split_data['value'] != 1.0].copy()
        if len(actual_splits) > 0:
            actual_splits['date'] = pd.to_datetime(actual_splits[timestamp_col], unit='ms')
            
            # Check which splits have extreme moves nearby (within 5 days)
            # This could indicate the split wasn't properly applied
            unmatched = []
            for _, split_row in actual_splits.iterrows():
                ticker = split_row[ticker_col]
                split_ts = split_row[timestamp_col]
                
                # Look for extreme moves in this ticker within 5 days
                nearby_extreme = extreme_moves[
                    (extreme_moves[ticker_col] == ticker) &
                    (abs(extreme_moves[timestamp_col] - split_ts) < 5 * 86_400_000)
                ]
                
                if len(nearby_extreme) == 0:
                    # Split recorded but no extreme move - might be correctly adjusted
                    pass
                else:
                    # Split with nearby extreme move - likely data issue
                    unmatched.append({
                        'ticker': ticker,
                        'split_date': split_row['date'],
                        'split_ratio': split_row['value'],
                        'extreme_return': nearby_extreme['daily_return'].iloc[0],
                    })
            
            if unmatched:
                result['unmatched_splits'] = pd.DataFrame(unmatched)
    
    # Generate summary
    n_suspicious = len(result['suspicious_tickers'])
    n_extreme = len(result['extreme_moves'])
    summary_lines = [
        f"Data Quality Check Results:",
        f"  - Tickers with extreme moves (>{return_threshold*100:.0f}%): {n_suspicious}",
        f"  - Total extreme move events: {n_extreme}",
    ]
    if n_suspicious > 0:
        summary_lines.append(f"  - Suspicious tickers: {', '.join(result['suspicious_tickers'][:10])}")
        if n_suspicious > 10:
            summary_lines.append(f"    ... and {n_suspicious - 10} more")
    
    result['summary'] = '\n'.join(summary_lines)
    
    return result


# =============================================================================
# TRUE DAILY RETURNS CALCULATION
# =============================================================================

def compute_daily_portfolio_returns(
    holdings_history: pd.DataFrame,
    price_data: pd.DataFrame,
    timestamp_col: str = TIMESTAMP,
    ticker_col: str = TICKER,
    price_col: str = "Close",
    cost_bps_per_rebalance: float = 0.0,
    suspect_return_threshold: float = 0.5,
) -> pd.Series:
    """Compute true daily portfolio returns from holdings and price data.
    
    This computes actual daily returns by tracking the value of held positions
    each day, providing accurate drawdown and volatility measurements.
    
    WARNING: This function assumes price data is split-adjusted. Unadjusted splits
    will cause extreme daily returns that distort drawdown calculations.
    
    Suspicious returns handling:
        When a stock's daily return exceeds `suspect_return_threshold`, the return
        is treated as a data quality issue (e.g., unadjusted split, ticker recycling).
        Instead of using the suspicious return, we assume 0% return for that stock
        on that day (equivalent to the position value staying flat). This is more
        conservative than clipping, as it doesn't inject artificial signal.
    
    Args:
        holdings_history: DataFrame with columns [timestamp, ticker, weight].
                         Each row is a holding at a rebalance point.
        price_data: DataFrame with columns [timestamp, ticker, Close].
                    Should contain daily prices for all tickers in holdings.
        timestamp_col: Column name for timestamp.
        ticker_col: Column name for ticker.
        price_col: Column name for price (default 'Close').
        cost_bps_per_rebalance: Transaction cost to apply at each rebalance (in bps).
        suspect_return_threshold: Absolute return threshold above which a daily
                                  return is treated as suspicious data (default 0.5 = 50%).
                                  Suspicious returns are treated as 0% (flat).
                                  Set to None to disable this check.
    
    Returns:
        Series of daily portfolio returns indexed by timestamp.
    """
    if holdings_history is None or holdings_history.empty:
        return pd.Series(dtype=float)
    
    if price_data is None or price_data.empty:
        return pd.Series(dtype=float)
    
    # Get all rebalance timestamps (sorted)
    rebalance_timestamps = sorted(holdings_history[timestamp_col].unique())
    
    if len(rebalance_timestamps) < 1:
        return pd.Series(dtype=float)
    
    # Get all daily timestamps from price data
    all_daily_timestamps = sorted(price_data[timestamp_col].unique())
    
    # Filter to timestamps >= first rebalance
    # Include all timestamps after first rebalance to track holdings performance
    first_rebalance = rebalance_timestamps[0]
    
    # Get daily timestamps starting from first rebalance
    daily_timestamps = [ts for ts in all_daily_timestamps if ts >= first_rebalance]
    
    if len(daily_timestamps) < 2:
        return pd.Series(dtype=float)
    
    # Detect anomalous prices (likely unadjusted splits) and build exclusion set
    # A split signature: >40% move followed by opposite >40% move
    anomalous_ticker_days = _detect_anomalous_prices(
        price_data, timestamp_col, ticker_col, price_col
    )
    if anomalous_ticker_days:
        warnings.warn(
            f"Detected {len(anomalous_ticker_days)} ticker-days with anomalous prices "
            f"(likely unadjusted splits). These will be excluded from daily returns. "
            f"Consider using split-adjusted price data for accurate results."
        )
    
    # Create price lookup: {(timestamp, ticker): price}
    price_lookup = {}
    for _, row in price_data.iterrows():
        key = (row[timestamp_col], row[ticker_col])
        if price_col in row and pd.notna(row[price_col]):
            price_lookup[key] = row[price_col]
    
    # Create holdings lookup: {rebalance_ts: {ticker: weight}}
    holdings_by_rebalance = {}
    for rebalance_ts in rebalance_timestamps:
        rebalance_holdings = holdings_history[
            holdings_history[timestamp_col] == rebalance_ts
        ]
        holdings_by_rebalance[rebalance_ts] = {
            row[ticker_col]: row['weight']
            for _, row in rebalance_holdings.iterrows()
        }
    
    # Compute daily returns
    daily_returns = []
    current_holdings = {}
    rebalance_idx = 0
    pending_rebalance_cost = False  # Track if we need to apply cost on next return
    
    for i, ts in enumerate(daily_timestamps):
        # Check if we need to rebalance
        if rebalance_idx < len(rebalance_timestamps):
            if ts >= rebalance_timestamps[rebalance_idx]:
                current_holdings = holdings_by_rebalance[rebalance_timestamps[rebalance_idx]]
                rebalance_idx += 1
                pending_rebalance_cost = True  # Apply cost on next recorded return
        
        if i == 0 or not current_holdings:
            # First day or no holdings yet
            continue
        
        prev_ts = daily_timestamps[i - 1]
        
        # Compute portfolio return for this day
        portfolio_return = 0.0
        total_weight_with_data = 0.0
        
        for ticker, weight in current_holdings.items():
            # Skip this ticker-day if it's flagged as anomalous
            if (ts, ticker) in anomalous_ticker_days or (prev_ts, ticker) in anomalous_ticker_days:
                continue
                
            prev_price = price_lookup.get((prev_ts, ticker))
            curr_price = price_lookup.get((ts, ticker))
            
            if prev_price is not None and curr_price is not None and prev_price > 0:
                stock_return = (curr_price - prev_price) / prev_price
                
                # Handle suspicious returns (likely data quality issues)
                # Instead of clipping, treat as 0% return (position value stays flat)
                # This is equivalent to "we don't trust this data point"
                if suspect_return_threshold is not None:
                    if abs(stock_return) > suspect_return_threshold:
                        # Suspicious return - treat as flat (0% return for this stock)
                        # The position's weight still counts toward total_weight_with_data
                        # so it doesn't get scaled up
                        stock_return = 0.0
                
                # Long positions: positive weight, gain when stock goes up
                # Short positions: negative weight, gain when stock goes down
                portfolio_return += weight * stock_return
                total_weight_with_data += abs(weight)
        
        # Skip days where we have data for very few holdings (likely data gap)
        total_weight = sum(abs(w) for w in current_holdings.values())
        if total_weight_with_data < total_weight * 0.5:
            # Less than 50% of portfolio weight has valid price data - skip this day
            continue
        
        # Scale return to account for missing weights (assume flat for missing)
        if total_weight_with_data > 0 and total_weight_with_data < total_weight:
            portfolio_return = portfolio_return * (total_weight / total_weight_with_data)
        
        # Apply transaction cost if we just rebalanced
        if pending_rebalance_cost and cost_bps_per_rebalance > 0:
            portfolio_return -= cost_bps_per_rebalance / 10_000
            pending_rebalance_cost = False
        
        daily_returns.append({'timestamp': ts, 'return': portfolio_return})
    
    if not daily_returns:
        return pd.Series(dtype=float)
    
    returns_df = pd.DataFrame(daily_returns)
    returns_series = returns_df.set_index('timestamp')['return']
    
    return returns_series


def _detect_anomalous_prices(
    price_data: pd.DataFrame,
    timestamp_col: str,
    ticker_col: str,
    price_col: str,
    threshold: float = 0.40,
) -> set:
    """Detect anomalous price moves (likely unadjusted splits or data errors).
    
    Identifies ticker-days where price moves >threshold% and then reverses
    by a similar magnitude the next day - the signature of an unadjusted split.
    
    Args:
        price_data: DataFrame with price data.
        timestamp_col: Column name for timestamp.
        ticker_col: Column name for ticker.
        price_col: Column name for price.
        threshold: Minimum absolute return to flag (default 40%).
    
    Returns:
        Set of (timestamp, ticker) tuples to exclude from calculations.
    """
    if price_data.empty:
        return set()
    
    df = price_data.copy()
    df = df.sort_values([ticker_col, timestamp_col])
    
    # Compute daily returns and next-day returns
    df['_prev_price'] = df.groupby(ticker_col)[price_col].shift(1)
    df['_next_price'] = df.groupby(ticker_col)[price_col].shift(-1)
    
    df['_ret'] = (df[price_col] - df['_prev_price']) / df['_prev_price']
    df['_next_ret'] = (df['_next_price'] - df[price_col]) / df[price_col]
    
    # Find reversal pattern: big move followed by opposite big move
    # This is the signature of an unadjusted stock split
    anomalous = df[
        (abs(df['_ret']) > threshold) & 
        (df['_ret'] * df['_next_ret'] < -(threshold * threshold))  # Opposite signs, both big
    ]
    
    # Return set of (timestamp, ticker) to exclude
    # Include both the anomalous day and the day after (reversal day)
    excluded = set()
    for _, row in anomalous.iterrows():
        excluded.add((row[timestamp_col], row[ticker_col]))
    
    # Also get the next day for each anomalous ticker
    for _, row in anomalous.iterrows():
        ticker = row[ticker_col]
        ts = row[timestamp_col]
        # Find next timestamp for this ticker
        next_rows = df[(df[ticker_col] == ticker) & (df[timestamp_col] > ts)]
        if not next_rows.empty:
            next_ts = next_rows[timestamp_col].min()
            excluded.add((next_ts, ticker))
    
    return excluded


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


def compute_sharpe_ratio_from_timestamps(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio using actual timestamp spacing.
    
    This version uses the actual calendar time between observations to 
    properly annualize, which is critical when holding periods are long
    (e.g., annual rebalancing with 365-day forward returns).
    
    Args:
        returns: Series of periodic returns with timestamp index (milliseconds).
        risk_free_rate: Annual risk-free rate (default 0).
    
    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return np.nan
    
    # Calculate actual time span
    MS_PER_DAY_LOCAL = 86_400_000
    timestamps = returns.index.values
    first_ts = timestamps[0]
    last_ts = timestamps[-1]
    total_days = (last_ts - first_ts) / MS_PER_DAY_LOCAL
    total_years = total_days / 365.0
    
    if total_years <= 0:
        return np.nan
    
    # Number of observations
    n_obs = len(returns)
    
    # Actual periods per year based on observation frequency
    periods_per_year = n_obs / total_years
    
    # Adjust for risk-free rate
    rf_per_period = risk_free_rate / periods_per_year if periods_per_year > 0 else 0
    excess_returns = returns - rf_per_period
    
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


def compute_annual_statistics(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> Optional[AnnualStatistics]:
    """Compute annual return statistics for Monte Carlo simulations and risk modeling.
    
    This function resamples daily portfolio returns into calendar years and computes
    distribution statistics useful for understanding real-world implementation expectations.
    
    Args:
        daily_returns: Series of daily portfolio returns (index must be timestamps in milliseconds).
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation.
    
    Returns:
        AnnualStatistics dataclass or None if insufficient data.
    """
    if len(daily_returns) < 252:  # Need at least ~1 year of daily data
        return None
    
    # Convert timestamp index to datetime for resampling
    daily_returns_dt = daily_returns.copy()
    daily_returns_dt.index = pd.to_datetime(daily_returns_dt.index, unit='ms')
    
    # Group by calendar year and compute annual returns
    # Annual return = (1 + r1) * (1 + r2) * ... - 1
    annual_returns = (1 + daily_returns_dt).groupby(daily_returns_dt.index.year).prod() - 1
    
    # Remove partial years at start/end if they have < 200 trading days
    days_per_year = daily_returns_dt.groupby(daily_returns_dt.index.year).size()
    complete_years = days_per_year[days_per_year >= 200].index
    annual_returns = annual_returns.loc[complete_years]
    
    if len(annual_returns) < 2:
        return None  # Need at least 2 complete years
    
    # Basic statistics
    mean_ret = annual_returns.mean()
    median_ret = annual_returns.median()
    std_ret = annual_returns.std()
    min_ret = annual_returns.min()
    max_ret = annual_returns.max()
    
    # Percentiles
    pct_5 = np.percentile(annual_returns, 5)
    pct_25 = np.percentile(annual_returns, 25)
    pct_75 = np.percentile(annual_returns, 75)
    pct_95 = np.percentile(annual_returns, 95)
    
    # Win/loss statistics
    positive_years = annual_returns[annual_returns > 0]
    negative_years = annual_returns[annual_returns <= 0]
    
    pct_positive = len(positive_years) / len(annual_returns) if len(annual_returns) > 0 else 0
    avg_positive = positive_years.mean() if len(positive_years) > 0 else 0
    avg_negative = negative_years.mean() if len(negative_years) > 0 else 0
    
    # Shape statistics
    from scipy import stats
    skewness = stats.skew(annual_returns)
    kurtosis = stats.kurtosis(annual_returns)  # Excess kurtosis (normal = 0)
    
    # Average annual Sharpe ratio
    # Compute Sharpe for each year's daily returns, then average
    sharpe_ratios = []
    for year in complete_years:
        year_daily = daily_returns_dt[daily_returns_dt.index.year == year]
        if len(year_daily) >= 50:  # Need reasonable sample size
            excess_ret = year_daily - risk_free_rate / 252
            if excess_ret.std() > 0:
                sharpe = (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)
                sharpe_ratios.append(sharpe)
    
    avg_sharpe = np.mean(sharpe_ratios) if len(sharpe_ratios) > 0 else np.nan
    
    # Calculate years covered
    first_date = daily_returns_dt.index.min()
    last_date = daily_returns_dt.index.max()
    years_covered = (last_date - first_date).days / 365.0
    
    return AnnualStatistics(
        mean_annual_return=mean_ret,
        median_annual_return=median_ret,
        std_annual_return=std_ret,
        min_annual_return=min_ret,
        max_annual_return=max_ret,
        pct_5_annual_return=pct_5,
        pct_25_annual_return=pct_25,
        pct_75_annual_return=pct_75,
        pct_95_annual_return=pct_95,
        pct_positive_years=pct_positive,
        avg_positive_year=avg_positive,
        avg_negative_year=avg_negative,
        skewness_annual=skewness,
        kurtosis_annual=kurtosis,
        sharpe_annual_avg=avg_sharpe,
        num_years=len(annual_returns),
        years_sampled=years_covered,
    )


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
    price_data: Optional[pd.DataFrame] = None,
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
        price_data: Optional DataFrame with daily prices [timestamp, ticker, Close].
                   If provided, computes true daily returns for accurate drawdown calculation.
    
    Returns:
        BacktestResult with performance metrics.
    """
    timestamps = sorted(df[timestamp_col].unique())
    
    # Sample timestamps at intervals matching the return horizon to avoid overlapping periods
    # For example, with 365-day returns, we should only rebalance every ~365 days
    # to avoid compounding overlapping returns
    if return_horizon_days > 1:
        # Sample timestamps that are at least return_horizon_days apart (in actual days)
        # Vectorized implementation for speed
        MS_PER_DAY_LOCAL = 86_400_000
        horizon_ms = return_horizon_days * MS_PER_DAY_LOCAL
        
        ts_array = np.array(timestamps)
        # Start with first timestamp, then find next that's >= horizon_ms away
        sampled_indices = [0]
        last_ts = ts_array[0]
        for i in range(1, len(ts_array)):
            if ts_array[i] - last_ts >= horizon_ms:
                sampled_indices.append(i)
                last_ts = ts_array[i]
        timestamps = ts_array[sampled_indices].tolist()
        
    if len(timestamps) < 2:
        print(f"WARNING: Only {len(timestamps)} period(s) for backtest after sampling.")
        print(f"  Return horizon: {return_horizon_days} days")
        print(f"  Total timestamps: {len(df[timestamp_col].unique())}")
        print("  Sharpe ratio will be NaN.")
    
    # Pre-group data by timestamp for faster lookups
    timestamp_groups = {ts: group for ts, group in df.groupby(timestamp_col) if ts in timestamps}
    
    daily_returns = []  # post-fee returns
    daily_returns_pre_fee = []  # gross returns (before fees)
    turnovers = []
    prev_holdings: Dict[str, float] = {}
    holdings_records = []
    
    for ts in timestamps:
        # Use pre-grouped data instead of filtering each iteration
        if ts not in timestamp_groups:
            continue
        ts_df = timestamp_groups[ts]
        
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
    
    # Calculate actual time span for proper annualization
    MS_PER_DAY_LOCAL = 86_400_000
    first_ts = returns_series.index[0]
    last_ts = returns_series.index[-1]
    total_days = (last_ts - first_ts) / MS_PER_DAY_LOCAL
    total_years = max(total_days / 365.0, 0.01)  # Avoid division by zero
    
    # Compute periods per year based on ACTUAL observation frequency
    # This is critical for proper Sharpe annualization with long holding periods
    num_rebalances = len(returns_series)
    actual_periods_per_year = num_rebalances / total_years if total_years > 0 else 1
    
    # Warn if sample size is too small for reliable statistics (only once)
    MIN_PERIODS_FOR_RELIABLE_STATS = 10
    global _warned_low_sample_size
    if num_rebalances < MIN_PERIODS_FOR_RELIABLE_STATS and not _warned_low_sample_size:
        _warned_low_sample_size = True
        warnings.warn(
            f"Backtest has only {num_rebalances} return periods over {total_years:.1f} years. "
            f"Statistics (Sharpe, drawdown) may not be reliable. "
            f"Consider using a shorter return horizon or more data. "
            f"(Minimum recommended: {MIN_PERIODS_FOR_RELIABLE_STATS})"
        )
    
    # Use timestamp-based Sharpe calculation for accuracy with long holding periods
    sharpe = compute_sharpe_ratio_from_timestamps(returns_series)
    sharpe_pre_fee = compute_sharpe_ratio_from_timestamps(returns_pre_fee_series)
    
    # Build holdings history first (needed for true daily returns)
    holdings_df = pd.DataFrame(holdings_records) if holdings_records else None
    
    # Compute max drawdown
    # CRITICAL: For long holding periods, we MUST use true daily returns to get
    # realistic drawdown. Period returns only measure drawdown at rebalance points.
    true_daily_returns = None
    use_true_daily = False
    
    if price_data is not None and holdings_df is not None:
        try:
            print(f"Computing continuous daily returns for accurate drawdown (holdings: {len(holdings_df)} rows)...")
            true_daily_returns = compute_daily_portfolio_returns(
                holdings_df,
                price_data,
                timestamp_col=timestamp_col,
                ticker_col=ticker_col,
                price_col="Close",
                cost_bps_per_rebalance=config.total_cost_bps,
            )
            if len(true_daily_returns) > 10:
                # Use true daily returns for drawdown calculation
                true_cumulative = (1 + true_daily_returns).cumprod() - 1
                max_dd = compute_max_drawdown(true_cumulative)
                # Also compute proper daily volatility for Sharpe
                annualized_volatility = true_daily_returns.std() * np.sqrt(252)
                use_true_daily = True
                print(f"[OK] Using {len(true_daily_returns)} daily returns for drawdown (max_dd={max_dd:.2%})")
            else:
                warnings.warn(
                    f"Computed only {len(true_daily_returns)} daily returns (need >10). "
                    f"Using period returns for drawdown."
                )
                max_dd = compute_max_drawdown(cumulative)
        except Exception as e:
            import traceback
            warnings.warn(
                f"Could not compute true daily returns: {e}\n"
                f"Traceback: {traceback.format_exc()}\n"
                f"Using period returns for drawdown (will only capture rebalance-point drawdown)."
            )
            max_dd = compute_max_drawdown(cumulative)
    else:
        # No price data provided - use period returns
        max_dd = compute_max_drawdown(cumulative)
        global _warned_long_holding_period
        if return_horizon_days > 20 and not _warned_long_holding_period:
            _warned_long_holding_period = True
            warnings.warn(
                f"Long holding period ({return_horizon_days} days) but no price_data provided. "
                f"Drawdown will only be measured at rebalance points, "
                f"not continuously. This may significantly underestimate true drawdown."
            )
    
    total_return = cumulative.iloc[-1] if len(cumulative) > 0 else 0.0
    total_return_pre_fee = cumulative_pre_fee.iloc[-1] if len(cumulative_pre_fee) > 0 else 0.0
    avg_turnover = np.mean(turnovers) if turnovers else 0.0
    
    # Calculate annualized returns using actual calendar time
    if num_rebalances >= 2 and total_years > 0:
        annualized_return_post_fee = (1 + total_return) ** (1 / total_years) - 1
        annualized_return_pre_fee = (1 + total_return_pre_fee) ** (1 / total_years) - 1
        
        # Calculate annualized volatility using actual periods per year
        # (only if not already set from true daily returns)
        if true_daily_returns is None or len(true_daily_returns) <= 10:
            annualized_volatility = returns_series.std() * np.sqrt(actual_periods_per_year)
        
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
        # When drawdown is 0 (all periods positive), Calmar is undefined
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
    
    # Compute annual statistics if we have true daily returns
    annual_stats = None
    if use_true_daily and true_daily_returns is not None and len(true_daily_returns) >= 252:
        annual_stats = compute_annual_statistics(true_daily_returns, risk_free_rate=0.0)
    
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
        true_daily_returns=true_daily_returns if use_true_daily else None,
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
        annual_stats=annual_stats,
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
            f"Mean Sharpe (post-fee):    {self.mean_sharpe_post_fee:.2f} +/- {self.std_sharpe_post_fee:.2f}",
            f"Mean Sharpe (pre-fee):     {self.mean_sharpe_pre_fee:.2f} +/- {self.std_sharpe_pre_fee:.2f}",
            f"Mean Total Return:         {self.mean_total_return:.2%} +/- {self.std_total_return:.2%}",
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
    
    # Suppress long-holding-period warning during random baseline
    # We know we're not passing price_data and don't need precise drawdown for baseline
    global _warned_long_holding_period
    original_warned_state = _warned_long_holding_period
    _warned_long_holding_period = True  # Suppress warning during random trials
    
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
    
    # Restore warning flag to original state
    _warned_long_holding_period = original_warned_state
    
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
