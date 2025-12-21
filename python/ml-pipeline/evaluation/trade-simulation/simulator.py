"""Module for trading simulation based on model predictions."""

from dataclasses import dataclass
import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, CLOSE, PREDICTION_PROB
from config.model_config import (
    PREDICTION_THRESHOLD,
    INITIAL_CAPITAL,
    TRANSACTION_COST_PCT,
    LOOKAHEAD_DAYS,
    MS_PER_DAY,
    RISK_FREE_RATE,
    MAX_POSITION_SIZE_PCT,
    INVERT_PREDICTIONS,
)


@dataclass
class Position:
    """Represents an open trading position."""
    ticker: str
    buy_timestamp: int
    buy_price: float
    shares: float


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
    median_return_pct: float
    lqr_return_pct: float  # Lower quartile return
    uqr_return_pct: float  # Upper quartile return
    std_return_pct: float
    sharpe_ratio: float
    num_trades: int
    final_capital: float


def run_trading_simulation(
    predictions_df: pd.DataFrame,
    price_data: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    threshold: float = PREDICTION_THRESHOLD,
    lookahead_days: int = LOOKAHEAD_DAYS,
    transaction_cost_pct: float = TRANSACTION_COST_PCT,
    max_position_pct: float = MAX_POSITION_SIZE_PCT,
    invert_predictions: bool = INVERT_PREDICTIONS,
) -> tuple[TradingMetrics, list[Trade]]:
    """
    Simulate trading based on model predictions.
    
    Strategy:
    - Buy when prediction probability >= threshold (or <= if inverted)
    - Sell after lookahead_days regardless of performance
    - Position size limited to max_position_pct of initial capital
    
    Args:
        predictions_df: DataFrame with timestamp, ticker, prediction_probability.
        price_data: Wide format DataFrame with timestamp, ticker, Close prices.
        initial_capital: Starting capital amount.
        threshold: Minimum probability to trigger buy (or maximum if inverted).
        lookahead_days: Days to hold position before selling.
        transaction_cost_pct: Transaction cost as percentage.
        max_position_pct: Maximum position size as percentage of initial capital.
        invert_predictions: If True, buy when prob < threshold instead of >= threshold.
    
    Returns:
        Tuple of (TradingMetrics, list of completed Trades).
    """
    lookahead_ms = lookahead_days * MS_PER_DAY
    cost_multiplier = transaction_cost_pct / 100
    max_position_size = initial_capital * (max_position_pct / 100)
    
    # Sort predictions by timestamp
    predictions_df = predictions_df.sort_values(TIMESTAMP)
    
    # Create price lookup by ticker and timestamp
    price_lookup = {}
    all_timestamps_by_ticker = {}
    for _, row in price_data.iterrows():
        ticker = row[TICKER]
        ts = row[TIMESTAMP]
        if CLOSE in row and pd.notna(row[CLOSE]):
            price_lookup[(ticker, ts)] = row[CLOSE]
            if ticker not in all_timestamps_by_ticker:
                all_timestamps_by_ticker[ticker] = []
            all_timestamps_by_ticker[ticker].append(ts)
    
    # Sort timestamps for each ticker
    for ticker in all_timestamps_by_ticker:
        all_timestamps_by_ticker[ticker] = sorted(all_timestamps_by_ticker[ticker])
    
    capital = initial_capital
    open_positions: list[Position] = []
    completed_trades: list[Trade] = []
    
    # Get unique timestamps and process day by day
    unique_timestamps = sorted(predictions_df[TIMESTAMP].unique())
    
    for current_ts in unique_timestamps:
        # First, close any positions that have reached their sell date
        positions_to_close = [
            p for p in open_positions
            if current_ts >= p.buy_timestamp + lookahead_ms
        ]
        
        for pos in positions_to_close:
            # Find sell price at or after the target sell date
            sell_ts = pos.buy_timestamp + lookahead_ms
            ticker_timestamps = all_timestamps_by_ticker.get(pos.ticker, [])
            sell_price = _find_closest_price(
                price_lookup, pos.ticker, sell_ts, ticker_timestamps
            )
            
            if sell_price is not None:
                # Apply transaction cost to sell
                actual_sell_price = sell_price * (1 - cost_multiplier)
                proceeds = pos.shares * actual_sell_price
                return_pct = ((actual_sell_price - pos.buy_price) / pos.buy_price) * 100
                
                completed_trades.append(Trade(
                    ticker=pos.ticker,
                    buy_timestamp=pos.buy_timestamp,
                    sell_timestamp=current_ts,
                    buy_price=pos.buy_price,
                    sell_price=actual_sell_price,
                    shares=pos.shares,
                    return_pct=return_pct,
                ))
                
                capital += proceeds
            
            open_positions.remove(pos)
        
        # Get buy signals for current day
        day_predictions = predictions_df[predictions_df[TIMESTAMP] == current_ts]
        
        # Apply threshold - inverted or normal
        if invert_predictions:
            buy_signals = day_predictions[day_predictions[PREDICTION_PROB] <= threshold]
            # Sort by probability (lowest first when inverted)
            buy_signals = buy_signals.sort_values(PREDICTION_PROB, ascending=True)
        else:
            buy_signals = day_predictions[day_predictions[PREDICTION_PROB] >= threshold]
            # Sort by probability (highest first)
            buy_signals = buy_signals.sort_values(PREDICTION_PROB, ascending=False)
        
        for _, signal in buy_signals.iterrows():
            if capital < max_position_size:
                break  # Not enough capital for another position
            
            ticker = signal[TICKER]
            buy_price = price_lookup.get((ticker, current_ts))
            
            if buy_price is not None and buy_price > 0:
                # Use fixed position size (limited by max_position_size)
                position_size = min(max_position_size, capital)
                
                # Apply transaction cost to buy
                actual_buy_price = buy_price * (1 + cost_multiplier)
                shares = position_size / actual_buy_price
                
                open_positions.append(Position(
                    ticker=ticker,
                    buy_timestamp=current_ts,
                    buy_price=actual_buy_price,
                    shares=shares,
                ))
                
                capital -= position_size
    
    # Close all remaining open positions at their sell date
    for pos in open_positions:
        sell_ts = pos.buy_timestamp + lookahead_ms
        ticker_timestamps = all_timestamps_by_ticker.get(pos.ticker, [])
        sell_price = _find_closest_price(
            price_lookup, pos.ticker, sell_ts, ticker_timestamps
        )
        
        if sell_price is not None:
            actual_sell_price = sell_price * (1 - cost_multiplier)
            proceeds = pos.shares * actual_sell_price
            return_pct = ((actual_sell_price - pos.buy_price) / pos.buy_price) * 100
            
            completed_trades.append(Trade(
                ticker=pos.ticker,
                buy_timestamp=pos.buy_timestamp,
                sell_timestamp=sell_ts,
                buy_price=pos.buy_price,
                sell_price=actual_sell_price,
                shares=pos.shares,
                return_pct=return_pct,
            ))
            
            capital += proceeds
    
    # Calculate metrics
    return _calculate_metrics(completed_trades, initial_capital, capital), completed_trades


def _find_closest_price(
    price_lookup: dict,
    ticker: str,
    target_ts: int,
    available_timestamps: list[int],
) -> float | None:
    """Find the closest available price at or after target timestamp."""
    for ts in available_timestamps:
        if ts >= target_ts:
            price = price_lookup.get((ticker, ts))
            if price is not None:
                return price
    return None


def _calculate_metrics(
    trades: list[Trade],
    initial_capital: float,
    final_capital: float,
) -> TradingMetrics:
    """Calculate trading performance metrics."""
    if not trades:
        return TradingMetrics(
            total_return_pct=0.0,
            median_return_pct=0.0,
            lqr_return_pct=0.0,
            uqr_return_pct=0.0,
            std_return_pct=0.0,
            sharpe_ratio=0.0,
            num_trades=0,
            final_capital=final_capital,
        )
    
    returns = [t.return_pct for t in trades]
    
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    median_return = np.median(returns)
    lqr = np.percentile(returns, 25)
    uqr = np.percentile(returns, 75)
    std_return = np.std(returns)
    
    # Sharpe ratio (assuming risk-free rate = 0)
    mean_return = np.mean(returns)
    sharpe = mean_return / std_return if std_return > 0 else 0.0
    
    return TradingMetrics(
        total_return_pct=total_return_pct,
        median_return_pct=median_return,
        lqr_return_pct=lqr,
        uqr_return_pct=uqr,
        std_return_pct=std_return,
        sharpe_ratio=sharpe,
        num_trades=len(trades),
        final_capital=final_capital,
    )


def run_baseline_simulation(
    price_data: pd.DataFrame,
    start_ts: int,
    end_ts: int,
    initial_capital: float = INITIAL_CAPITAL,
    lookahead_days: int = LOOKAHEAD_DAYS,
    transaction_cost_pct: float = TRANSACTION_COST_PCT,
) -> tuple[TradingMetrics, list[Trade]]:
    """
    Run baseline simulation buying every ticker (for comparison).
    
    Buys all tickers at start_ts and sells after lookahead_days.
    """
    lookahead_ms = lookahead_days * MS_PER_DAY
    cost_multiplier = transaction_cost_pct / 100
    
    # Get prices at start time
    start_prices = price_data[price_data[TIMESTAMP] == start_ts]
    
    if start_prices.empty:
        # Find closest timestamp to start
        available_ts = price_data[TIMESTAMP].unique()
        closest_ts = min(available_ts, key=lambda x: abs(x - start_ts))
        start_prices = price_data[price_data[TIMESTAMP] == closest_ts]
        start_ts = closest_ts
    
    tickers = start_prices[TICKER].unique()
    position_size = initial_capital / len(tickers) if len(tickers) > 0 else 0
    
    # Create price lookup
    price_lookup = {}
    for _, row in price_data.iterrows():
        key = (row[TICKER], row[TIMESTAMP])
        if CLOSE in row and pd.notna(row[CLOSE]):
            price_lookup[key] = row[CLOSE]
    
    trades = []
    sell_ts = start_ts + lookahead_ms
    
    for ticker in tickers:
        buy_price = price_lookup.get((ticker, start_ts))
        if buy_price is None or buy_price <= 0:
            continue
        
        actual_buy_price = buy_price * (1 + cost_multiplier)
        shares = position_size / actual_buy_price
        
        # Find sell price
        sell_price = None
        for ts in sorted(price_data[TIMESTAMP].unique()):
            if ts >= sell_ts:
                sell_price = price_lookup.get((ticker, ts))
                if sell_price is not None:
                    sell_ts_actual = ts
                    break
        
        if sell_price is not None:
            actual_sell_price = sell_price * (1 - cost_multiplier)
            return_pct = ((actual_sell_price - actual_buy_price) / actual_buy_price) * 100
            
            trades.append(Trade(
                ticker=ticker,
                buy_timestamp=start_ts,
                sell_timestamp=sell_ts_actual,
                buy_price=actual_buy_price,
                sell_price=actual_sell_price,
                shares=shares,
                return_pct=return_pct,
            ))
    
    # Calculate final capital
    final_capital = sum(t.shares * t.sell_price for t in trades)
    
    return _calculate_metrics(trades, initial_capital, final_capital), trades


def metrics_to_dict(metrics: TradingMetrics) -> dict:
    """Convert TradingMetrics to dictionary for serialization."""
    return {
        "total_return_pct": metrics.total_return_pct,
        "median_return_pct": metrics.median_return_pct,
        "lqr_return_pct": metrics.lqr_return_pct,
        "uqr_return_pct": metrics.uqr_return_pct,
        "std_return_pct": metrics.std_return_pct,
        "sharpe_ratio": metrics.sharpe_ratio,
        "num_trades": metrics.num_trades,
        "final_capital": metrics.final_capital,
    }


def aggregate_trading_metrics(metrics_list: list[TradingMetrics]) -> dict:
    """Aggregate trading metrics across multiple windows."""
    return {
        "total_return_mean": np.mean([m.total_return_pct for m in metrics_list]),
        "total_return_std": np.std([m.total_return_pct for m in metrics_list]),
        "median_return_mean": np.mean([m.median_return_pct for m in metrics_list]),
        "sharpe_ratio_mean": np.mean([m.sharpe_ratio for m in metrics_list]),
        "sharpe_ratio_std": np.std([m.sharpe_ratio for m in metrics_list]),
        "total_trades": sum(m.num_trades for m in metrics_list),
    }
