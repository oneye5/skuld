import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Dict, List
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, log_loss, balanced_accuracy_score,
    precision_recall_curve, average_precision_score, auc
)
from scipy.stats import skew, kurtosis

from src.config.config import *
from src.preprocessing.preprocessing import restore_ticker_column
from src.utils.csv_utils import load_csv, save_csv
from src.utils.path_utils import get_skuld_root


@dataclass
class TradingMetrics:
    """Container for trading performance metrics."""
    # --- Volume ---
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # --- Returns ---
    total_return: float
    avg_return: float
    median_return: float
    std_return: float
    best_trade: float
    worst_trade: float

    # --- Distribution ---
    return_25th: float
    return_75th: float
    return_iqr: float
    skewness: float
    kurtosis: float

    # --- Profitability ---
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float

    # --- System Quality ---
    sqn: float  # System Quality Number
    kelly_criterion: float

    # --- Streaks ---
    max_consecutive_wins: int
    max_consecutive_losses: int

    # --- Risk Adjusted ---
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # --- Drawdown & Tail Risk ---
    max_drawdown: float
    max_drawdown_duration: int
    recovery_factor: float
    ulcer_index: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR / Expected Shortfall (95%)

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {k: v for k, v in self.__dict__.items()}


# =======================================================
# === DATA LOADING ======================================
# =======================================================

def load_combined_predictions(
        directory: Path,
        pattern: str = "predictions*.csv"
) -> pd.DataFrame:
    """
    Load and combine multiple prediction CSV files.
    """
    import re

    files = sorted(directory.glob(pattern))
    if not files:
        print(f"Warning: No files found matching '{pattern}' in {directory}")
        return pd.DataFrame()

    numbered_pattern = re.compile(r'predictions\d+\.csv$')
    numbered_files = [f for f in files if numbered_pattern.search(f.name)]

    if not numbered_files:
        print(f"Warning: No numbered prediction files found in {directory}")
        return pd.DataFrame()

    print(f"Found {len(numbered_files)} prediction files. Combining...")
    dfs = []

    for f in numbered_files:
        try:
            df = load_csv(str(f))
            if not df.empty:
                dfs.append(df)
                print(f"  ✓ Loaded: {f.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = restore_ticker_column(combined_df)

    if TIMESTAMP_COL in combined_df.columns and TICKER_COL in combined_df.columns:
        initial_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=[TIMESTAMP_COL, TICKER_COL],
            keep='last'
        )
        dropped = initial_len - len(combined_df)
        if dropped > 0:
            print(f"Dropped {dropped} duplicate rows.")

    return combined_df


# =======================================================
# === MAIN EVALUATION FUNCTION ==========================
# =======================================================

def run_evaluation(
        predictions: Union[str, pd.DataFrame],
        labeled_csv_path: str,
        raw_price_csv_path: str,
        probability_threshold: float
):
    # Load predictions
    if isinstance(predictions, (str, Path)):
        preds_df = load_csv(str(predictions))
    elif isinstance(predictions, pd.DataFrame):
        preds_df = predictions
    else:
        raise ValueError("predictions must be a file path or DataFrame")

    if preds_df.empty:
        print("Warning: Predictions DataFrame is empty.")
        return pd.DataFrame(), pd.DataFrame()

    print("Loading preprocessed data...")
    preprocessed_data = load_csv(labeled_csv_path)
    preprocessed_data = restore_ticker_column(preprocessed_data)

    print("Using preprocessed data for simulation...")

    trade_results = simulate_trades(preds_df, preprocessed_data, probability_threshold)
    ml_metrics = calculate_ml_metrics(preds_df, probability_threshold)

    return trade_results, ml_metrics


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


# =======================================================
# === MACHINE LEARNING METRICS ==========================
# =======================================================

def calculate_ml_metrics(
        predictions_df: pd.DataFrame,
        probability_threshold: float
) -> pd.DataFrame:
    """Calculate comprehensive ML classification metrics."""
    df = predictions_df.copy()
    df["prediction"] = (df[PREDICTION_COL] > probability_threshold).astype(int)

    y_true = df[LABEL_COL]
    y_pred = df["prediction"]
    y_prob = df[PREDICTION_COL]

    metrics = _calculate_classification_metrics(y_true, y_pred, y_prob)
    _display_ml_metrics(metrics, probability_threshold)

    return pd.DataFrame([metrics])


def _calculate_classification_metrics(
        y_true: pd.Series,
        y_pred: pd.Series,
        y_prob: pd.Series
) -> Dict:
    """Calculate comprehensive classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Basic metrics
    n_samples = len(y_true)
    n_positive = y_true.sum()
    n_negative = n_samples - n_positive
    class_imbalance = n_positive / n_samples if n_samples > 0 else 0

    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Probability-based metrics
    has_both_classes = len(np.unique(y_true)) > 1

    roc_auc = np.nan
    avg_precision = np.nan
    pr_auc = np.nan

    if has_both_classes:
        roc_auc = roc_auc_score(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        # Calculate PR AUC manually
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)

    logloss = log_loss(y_true, y_prob)
    brier_score = np.mean((y_prob - y_true) ** 2)

    # Cohen's Kappa
    p_observed = accuracy
    p_expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n_samples ** 2)
    cohens_kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected != 1 else 0

    return {
        "n_samples": n_samples,
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "class_imbalance": class_imbalance,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "mcc": mcc,
        "cohens_kappa": cohens_kappa,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,  # Added
        "avg_precision": avg_precision,
        "log_loss": logloss,
        "brier_score": brier_score,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def _display_ml_metrics(metrics: Dict, threshold: float):
    """Display ML metrics in formatted table."""
    print(f"\n{'=' * 60}")
    print(f"CLASSIFICATION METRICS (Threshold: {threshold})")
    print(f"{'=' * 60}\n")

    print("DATASET COMPOSITION")
    print("-" * 60)
    print(f"{'Total Samples':<30} {metrics['n_samples']:>15,}")
    print(f"{'Positive Class':<30} {metrics['n_positive']:>15,}")
    print(f"{'Negative Class':<30} {metrics['n_negative']:>15,}")
    print(f"{'Class Imbalance Ratio':<30} {metrics['class_imbalance']:>14.2%}")

    print(f"\nCONFUSION MATRIX")
    print("-" * 60)
    print(f"{'True Positives':<30} {metrics['true_positives']:>15,}")
    print(f"{'False Positives':<30} {metrics['false_positives']:>15,}")
    print(f"{'True Negatives':<30} {metrics['true_negatives']:>15,}")
    print(f"{'False Negatives':<30} {metrics['false_negatives']:>15,}")

    print(f"\nCLASSIFICATION PERFORMANCE")
    print("-" * 60)
    print(f"{'Accuracy':<30} {metrics['accuracy']:>14.4f}")
    print(f"{'Balanced Accuracy':<30} {metrics['balanced_accuracy']:>14.4f}")
    print(f"{'Precision':<30} {metrics['precision']:>14.4f}")
    print(f"{'Recall (Sensitivity)':<30} {metrics['recall']:>14.4f}")
    print(f"{'Specificity':<30} {metrics['specificity']:>14.4f}")
    print(f"{'F1 Score':<30} {metrics['f1_score']:>14.4f}")

    print(f"\nADVANCED METRICS")
    print("-" * 60)
    print(f"{'Matthews Corr Coef (MCC)':<30} {metrics['mcc']:>14.4f}")
    print(f"{'Cohens Kappa':<30} {metrics['cohens_kappa']:>14.4f}")

    if not np.isnan(metrics['roc_auc']):
        print(f"{'ROC AUC':<30} {metrics['roc_auc']:>14.4f}")
        print(f"{'PR AUC':<30} {metrics['pr_auc']:>14.4f}")
        print(f"{'Avg Precision Score':<30} {metrics['avg_precision']:>14.4f}")

    print(f"{'Log Loss':<30} {metrics['log_loss']:>14.4f}")
    print(f"{'Brier Score':<30} {metrics['brier_score']:>14.4f}")

    print(f"\n{'=' * 60}\n")


# =======================================================
# === MAIN EXECUTION ====================================
# =======================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRADING MODEL EVALUATION")
    print("=" * 60)

    print("\nLoading and combining prediction files...")
    combined_preds = load_combined_predictions(
        PY_DATA_DIR_PATH,
        pattern="predictions*.csv"
    )

    if combined_preds.empty:
        print("Error: No predictions to evaluate.")
    else:
        print(f"\nTotal predictions loaded: {len(combined_preds):,}")

        # Save combined predictions
        save_csv(combined_preds, str(AGGREGATE_PREDICTIONS_CSV_PATH))
        print(f"Saved combined predictions to: {AGGREGATE_PREDICTIONS_CSV_PATH}")

        # Run evaluation
        trades, metrics = run_evaluation(
            combined_preds,
            str(PREPROCESSED_CSV_PATH),
            str(PREPROCESSED_CSV_PATH),
            EVAL_CLASSIFICATION_BOUNDARY
        )

        # Save results
        if not trades.empty:
            save_csv(trades, str(TRADE_SIMULATION_CSV_PATH))
            print(f"\nSaved trade results to: {TRADE_SIMULATION_CSV_PATH}")

        if not metrics.empty:
            save_csv(metrics, str(EVALUATION_RESULTS_CSV_PATH))
            print(f"Saved metrics to: {EVALUATION_RESULTS_CSV_PATH}")

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)