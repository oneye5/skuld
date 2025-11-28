import numpy as np
from typing import Union, Tuple, Dict, List
from scipy.stats import skew, kurtosis

from src.evaluation.trading_metrics import TradingMetrics
from src.evaluation.utils import *
from src.preprocessing.preprocessing import restore_ticker_column

# =======================================================
# === TRADING SIMULATION ================================
# =======================================================

def simulate_trades(
        predictions_df: pd.DataFrame,
        price_df: pd.DataFrame,
        probability_threshold: float
) -> pd.DataFrame:
    """
    Simulate trades based on model predictions.
    """
    preds = restore_ticker_column(predictions_df.copy())
    prices = restore_ticker_column(price_df.copy())

    preds = preds.sort_values([TICKER_COL, TIMESTAMP_COL]).reset_index(drop=True)
    prices = prices.sort_values([TICKER_COL, TIMESTAMP_COL]).reset_index(drop=True)

    buy_signals = preds[preds[PREDICTION_COL] > probability_threshold]

    trades = _execute_trades(buy_signals, prices)
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("\n=== Trading Simulation Results ===")
        print("No trades executed (no signals above threshold).")
        return trades_df

    metrics = _calculate_trading_metrics(trades_df)
    _display_trading_metrics(metrics, probability_threshold)

    return trades_df


def _execute_trades(buy_signals: pd.DataFrame, prices: pd.DataFrame) -> List[Dict]:
    """Execute simulated trades based on buy signals."""
    trades = []

    for ticker, group in buy_signals.groupby(TICKER_COL):
        ticker_prices = prices[prices[TICKER_COL] == ticker].sort_values(TIMESTAMP_COL)

        for _, signal in group.iterrows():
            buy_time = signal[TIMESTAMP_COL]
            buy_price = signal[CLOSE_COL]

            if not buy_price or np.isnan(buy_price) or buy_price <= 0:
                continue

            sell_time = buy_time + LABEL_LOOKAHEAD_MILLIS
            future_prices = ticker_prices[ticker_prices[TIMESTAMP_COL] >= sell_time]

            if future_prices.empty:
                continue

            sell_row = future_prices.iloc[0]
            sell_price = sell_row[CLOSE_COL]

            if not sell_price or np.isnan(sell_price) or sell_price <= 0:
                continue

            pct_return = (sell_price - buy_price) / buy_price

            trades.append({
                "ticker": ticker,
                "buy_time": buy_time,
                "buy_price": buy_price,
                "sell_time": sell_row[TIMESTAMP_COL],
                "sell_price": sell_price,
                "return_pct": pct_return,
                "profit_loss": sell_price - buy_price,
            })

    return trades


def _calculate_trading_metrics(trades_df: pd.DataFrame) -> TradingMetrics:
    """Calculate comprehensive trading performance metrics."""
    trades_df = trades_df.sort_values("sell_time")
    returns = trades_df["return_pct"]

    # --- Basic Stats ---
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    n_trades = len(returns)
    n_wins = len(wins)
    n_losses = len(losses)

    # --- Return Metrics ---
    total_return = returns.sum()
    avg_return = returns.mean()
    median_return = returns.median()
    std_return = returns.std()

    # --- Distribution Metrics ---
    q1 = returns.quantile(0.25)
    q3 = returns.quantile(0.75)
    iqr = q3 - q1
    skew_val = skew(returns)
    kurt_val = kurtosis(returns)

    # --- Win/Loss Analysis ---
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    avg_win = wins.mean() if n_wins > 0 else 0
    avg_loss = abs(losses.mean()) if n_losses > 0 else 0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf

    # --- Profit Metrics ---
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # --- System Quality ---
    # SQN = SquareRoot(N) * (AvgProfit / StdDevProfit)
    sqn = (np.sqrt(n_trades) * (avg_return / std_return)) if std_return > 0 else 0

    # Kelly = W - (1-W)/R
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio) if win_loss_ratio > 0 else 0

    # --- Streak Analysis ---
    # Convert wins to boolean series
    win_series = returns > 0
    # Group by consecutive values
    streak_groups = (win_series != win_series.shift()).cumsum()
    streaks = win_series.groupby(streak_groups).agg(['count', 'first'])

    max_con_wins = streaks[streaks['first'] == True]['count'].max() if not streaks.empty else 0
    max_con_losses = streaks[streaks['first'] == False]['count'].max() if not streaks.empty else 0
    # Handle NaN cases if no wins or no losses ever occurred
    max_con_wins = int(max_con_wins) if not np.isnan(max_con_wins) else 0
    max_con_losses = int(max_con_losses) if not np.isnan(max_con_losses) else 0

    # --- Risk-Adjusted Metrics ---
    sharpe_ratio = avg_return / std_return if std_return > 0 else 0

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-10
    sortino_ratio = avg_return / downside_std if downside_std > 0 else 0

    # --- Drawdown & Tail Risk ---
    max_dd = _calculate_max_drawdown(returns)
    max_dd_duration = _calculate_max_drawdown_duration(returns)
    calmar_ratio = avg_return / abs(max_dd) if max_dd != 0 else 0
    recovery_factor = total_return / abs(max_dd) if max_dd != 0 else 0
    ulcer_index = _calculate_ulcer_index(returns)

    # VaR & CVaR (95% Confidence)
    var_95 = np.percentile(returns, 5)  # 5th percentile
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    return TradingMetrics(
        total_trades=n_trades,
        winning_trades=n_wins,
        losing_trades=n_losses,
        win_rate=win_rate,
        total_return=total_return,
        avg_return=avg_return,
        median_return=median_return,
        std_return=std_return,
        best_trade=returns.max(),
        worst_trade=returns.min(),
        return_25th=q1,
        return_75th=q3,
        return_iqr=iqr,
        skewness=skew_val,
        kurtosis=kurt_val,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        expectancy=expectancy,
        sqn=sqn,
        kelly_criterion=kelly,
        max_consecutive_wins=max_con_wins,
        max_consecutive_losses=max_con_losses,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        recovery_factor=recovery_factor,
        ulcer_index=ulcer_index,
        var_95=var_95,
        cvar_95=cvar_95
    )


def _calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from peak equity."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def _calculate_max_drawdown_duration(returns: pd.Series) -> int:
    """Calculate longest drawdown duration in number of trades."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()

    in_drawdown = cumulative < running_max
    if not in_drawdown.any():
        return 0

    drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()
    drawdown_lengths = in_drawdown.groupby(drawdown_periods).sum()

    return int(drawdown_lengths.max()) if not drawdown_lengths.empty else 0


def _calculate_ulcer_index(returns: pd.Series) -> float:
    """Calculate Ulcer Index - measures depth and duration of drawdowns."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown_pct = ((cumulative - running_max) / running_max) * 100
    squared_drawdowns = drawdown_pct ** 2
    mean_squared_dd = squared_drawdowns.mean()
    return np.sqrt(mean_squared_dd)


def _display_trading_metrics(metrics: TradingMetrics, threshold: float):
    """Display trading metrics in formatted table."""
    print(f"\n{'=' * 60}")
    print(f"TRADING SIMULATION RESULTS (Threshold: {threshold})")
    print(f"{'=' * 60}\n")

    print("TRADE STATISTICS")
    print("-" * 60)
    print(f"{'Total Trades':<30} {metrics.total_trades:>15,}")
    print(f"{'Win Rate':<30} {metrics.win_rate:>14.2%}")
    print(f"{'Max Consecutive Wins':<30} {metrics.max_consecutive_wins:>15}")
    print(f"{'Max Consecutive Losses':<30} {metrics.max_consecutive_losses:>15}")

    print(f"\nRETURN STATISTICS")
    print("-" * 60)
    print(f"{'Average Return':<30} {metrics.avg_return:>14.2%}")
    print(f"{'Median Return':<30} {metrics.median_return:>14.2%}")
    print(f"{'Std Dev Returns':<30} {metrics.std_return:>14.2%}")
    print(f"{'Skewness':<30} {metrics.skewness:>15.4f}")
    print(f"{'Kurtosis':<30} {metrics.kurtosis:>15.4f}")

    print(f"\nRETURN DISTRIBUTION")
    print("-" * 60)
    print(f"{'25th Percentile':<30} {metrics.return_25th:>14.2%}")
    print(f"{'75th Percentile':<30} {metrics.return_75th:>14.2%}")
    print(f"{'IQR':<30} {metrics.return_iqr:>14.2%}")
    print(f"{'Best Trade':<30} {metrics.best_trade:>14.2%}")
    print(f"{'Worst Trade':<30} {metrics.worst_trade:>14.2%}")

    print(f"\nSYSTEM EFFICIENCY")
    print("-" * 60)
    print(f"{'Profit Factor':<30} {metrics.profit_factor:>15.2f}")
    print(f"{'SQN (System Quality)':<30} {metrics.sqn:>15.2f}")
    print(f"{'Kelly Criterion':<30} {metrics.kelly_criterion:>14.2%}")
    print(f"{'Expectancy':<30} {metrics.expectancy:>14.4f}")
    print(f"{'Avg Win':<30} {metrics.avg_win:>14.2%}")
    print(f"{'Avg Loss':<30} {metrics.avg_loss:>14.2%}")
    print(f"{'Win/Loss Ratio':<30} {metrics.win_loss_ratio:>15.2f}")

    print(f"\nRISK & DRAWDOWN")
    print("-" * 60)
    print(f"{'Sharpe Ratio':<30} {metrics.sharpe_ratio:>15.2f}")
    print(f"{'Sortino Ratio':<30} {metrics.sortino_ratio:>15.2f}")
    print(f"{'Calmar Ratio':<30} {metrics.calmar_ratio:>15.2f}")
    print(f"{'Max Drawdown':<30} {metrics.max_drawdown:>14.2%}")
    print(f"{'Recovery Factor':<30} {metrics.recovery_factor:>15.2f}")
    print(f"{'Ulcer Index':<30} {metrics.ulcer_index:>15.2f}")
    print(f"{'VaR 95%':<30} {metrics.var_95:>14.2%}")
    print(f"{'CVaR 95% (Exp Shortfall)':<30} {metrics.cvar_95:>14.2%}")

    print(f"\n{'=' * 60}\n")