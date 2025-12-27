"""Trading simulation module."""

from dataclasses import dataclass
import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER, CLOSE, PREDICTION_PROB, PREDICTION
from config.settings import (
    PREDICTION_THRESHOLD,
    TOP_N_PREDICTIONS,
    INITIAL_CAPITAL,
    TRANSACTION_COST_PCT,
    LOOKAHEAD_DAYS,
    MS_PER_DAY,
    MAX_POSITION_SIZE_PCT,
)


@dataclass
class Trade:
    """Represents a completed trade."""
    ticker: str
    buy_timestamp: int
    sell_timestamp: int
    buy_price: float
    sell_price: float
    shares: float
    return_pct: float


@dataclass
class TradingMetrics:
    """Container for trading simulation metrics."""
    total_return_pct: float
    mean_return_pct: float
    median_return_pct: float
    std_return_pct: float
    sharpe_ratio: float
    num_trades: int
    final_capital: float
    lqr_return_pct: float  # 25th percentile
    uqr_return_pct: float  # 75th percentile
    min_return_pct: float
    max_return_pct: float


def run_trading_simulation(
    predictions_df: pd.DataFrame,
    price_data: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    threshold: float = PREDICTION_THRESHOLD,
    top_n: int | None = TOP_N_PREDICTIONS,
    lookahead_days: int = LOOKAHEAD_DAYS,
    transaction_cost_pct: float = TRANSACTION_COST_PCT,
    max_position_pct: float = MAX_POSITION_SIZE_PCT,
) -> tuple[TradingMetrics, list[Trade]]:
    """Simulate trading based on model predictions.
    
    Strategy:
    - Select top N predictions per timestamp (or threshold if top_n=None)
    - Buy selected stocks
    - Sell after lookahead_days regardless of performance
    - Position size limited to max_position_pct of initial capital
    
    Args:
        predictions_df: DataFrame with timestamp, ticker, prediction_probability.
        price_data: DataFrame with timestamp, ticker, Close prices.
        initial_capital: Starting capital amount.
        threshold: Minimum probability to trigger buy (legacy, used if top_n=None).
        top_n: Number of top predictions to select per timestamp. 
               If None, fall back to threshold-based selection.
        lookahead_days: Days to hold position before selling.
        transaction_cost_pct: Transaction cost as percentage.
        max_position_pct: Maximum position size as percentage of initial capital.
    
    Returns:
        Tuple of (TradingMetrics, list of completed Trades).
    """
    lookahead_ms = lookahead_days * MS_PER_DAY
    cost_multiplier = transaction_cost_pct / 100
    max_position_size = initial_capital * (max_position_pct / 100)
    
    # Sort predictions by timestamp
    predictions_df = predictions_df.sort_values(TIMESTAMP)
    
    # Build price lookup
    price_lookup = {}
    timestamps_by_ticker = {}
    
    for _, row in price_data.iterrows():
        ticker = row[TICKER]
        ts = row[TIMESTAMP]
        if CLOSE in row and pd.notna(row[CLOSE]):
            price_lookup[(ticker, ts)] = row[CLOSE]
            if ticker not in timestamps_by_ticker:
                timestamps_by_ticker[ticker] = []
            timestamps_by_ticker[ticker].append(ts)
    
    # Sort timestamps for each ticker
    for ticker in timestamps_by_ticker:
        timestamps_by_ticker[ticker] = sorted(timestamps_by_ticker[ticker])
    
    completed_trades: list[Trade] = []
    
    # Select buy signals using top N or threshold
    if top_n is not None and top_n > 0:
        # Top N selection per timestamp
        buy_signals = _select_top_n_signals(predictions_df, top_n)
    else:
        # Legacy threshold-based selection
        buy_signals = predictions_df[predictions_df[PREDICTION_PROB] >= threshold]
    
    for _, signal in buy_signals.iterrows():
        ticker = signal[TICKER]
        buy_ts = signal[TIMESTAMP]
        sell_ts = buy_ts + lookahead_ms
        
        # Get buy price
        buy_price = price_lookup.get((ticker, buy_ts))
        if buy_price is None or buy_price <= 0:
            continue
        
        # Find sell price at or after target sell date
        sell_price = _find_closest_price(
            price_lookup, ticker, sell_ts, timestamps_by_ticker.get(ticker, [])
        )
        
        if sell_price is None:
            continue
        
        # Calculate trade
        actual_buy_price = buy_price * (1 + cost_multiplier)
        actual_sell_price = sell_price * (1 - cost_multiplier)
        shares = max_position_size / actual_buy_price
        return_pct = ((actual_sell_price - actual_buy_price) / actual_buy_price) * 100
        
        completed_trades.append(Trade(
            ticker=ticker,
            buy_timestamp=buy_ts,
            sell_timestamp=sell_ts,
            buy_price=actual_buy_price,
            sell_price=actual_sell_price,
            shares=shares,
            return_pct=return_pct,
        ))
    
    # Calculate metrics
    metrics = _calculate_trading_metrics(completed_trades, initial_capital)
    
    return metrics, completed_trades


def _select_top_n_signals(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select top N predictions per unique timestamp.
    
    This is more realistic for trading - on each decision date, 
    pick the top N most confident predictions.
    
    Args:
        df: DataFrame with TIMESTAMP, TICKER, PREDICTION_PROB columns.
        top_n: Number of top predictions to select per timestamp.
    
    Returns:
        DataFrame containing only the top N rows per timestamp.
    """
    result_frames = []
    
    for ts in df[TIMESTAMP].unique():
        ts_df = df[df[TIMESTAMP] == ts]
        
        if len(ts_df) <= top_n:
            # If fewer than N samples, select all
            result_frames.append(ts_df)
        else:
            # Select top N by probability
            result_frames.append(ts_df.nlargest(top_n, PREDICTION_PROB))
    
    if result_frames:
        return pd.concat(result_frames, ignore_index=True)
    return df.head(0)  # Empty dataframe with same columns


def _find_closest_price(
    price_lookup: dict,
    ticker: str,
    target_ts: int,
    ticker_timestamps: list[int],
) -> float | None:
    """Find the price at or after the target timestamp."""
    for ts in ticker_timestamps:
        if ts >= target_ts:
            return price_lookup.get((ticker, ts))
    return None


def _calculate_trading_metrics(
    trades: list[Trade],
    initial_capital: float,
) -> TradingMetrics:
    """Calculate trading metrics from completed trades."""
    if not trades:
        return TradingMetrics(
            total_return_pct=0.0,
            mean_return_pct=0.0,
            median_return_pct=0.0,
            std_return_pct=0.0,
            sharpe_ratio=0.0,
            num_trades=0,
            final_capital=initial_capital,
            lqr_return_pct=0.0,
            uqr_return_pct=0.0,
            min_return_pct=0.0,
            max_return_pct=0.0,
        )
    
    returns = np.array([t.return_pct for t in trades])
    
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=0))
    sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
    
    # Calculate final capital (simplified: average return applied to all trades)
    total_invested = len(trades) * initial_capital * (MAX_POSITION_SIZE_PCT / 100)
    total_return = (returns.mean() / 100) * total_invested
    final_capital = initial_capital + total_return
    
    return TradingMetrics(
        total_return_pct=float(returns.sum()),
        mean_return_pct=mean_ret,
        median_return_pct=float(np.median(returns)),
        std_return_pct=std_ret,
        sharpe_ratio=sharpe,
        num_trades=len(trades),
        final_capital=final_capital,
        lqr_return_pct=float(np.percentile(returns, 25)),
        uqr_return_pct=float(np.percentile(returns, 75)),
        min_return_pct=float(returns.min()),
        max_return_pct=float(returns.max()),
    )


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert trades list to DataFrame."""
    if not trades:
        return pd.DataFrame(columns=[
            "ticker", "buy_timestamp", "sell_timestamp",
            "buy_price", "sell_price", "shares", "return_pct"
        ])
    
    return pd.DataFrame([
        {
            "ticker": t.ticker,
            "buy_timestamp": t.buy_timestamp,
            "sell_timestamp": t.sell_timestamp,
            "buy_price": t.buy_price,
            "sell_price": t.sell_price,
            "shares": t.shares,
            "return_pct": t.return_pct,
        }
        for t in trades
    ])


def metrics_to_dict(metrics: TradingMetrics) -> dict:
    """Convert TradingMetrics to dictionary."""
    return {
        "total_return_pct": metrics.total_return_pct,
        "mean_return_pct": metrics.mean_return_pct,
        "median_return_pct": metrics.median_return_pct,
        "std_return_pct": metrics.std_return_pct,
        "sharpe_ratio": metrics.sharpe_ratio,
        "num_trades": metrics.num_trades,
        "final_capital": metrics.final_capital,
        "lqr_return_pct": metrics.lqr_return_pct,
        "uqr_return_pct": metrics.uqr_return_pct,
        "min_return_pct": metrics.min_return_pct,
        "max_return_pct": metrics.max_return_pct,
    }
