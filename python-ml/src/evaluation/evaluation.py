import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, log_loss, balanced_accuracy_score
)

from src.config.config import *
from src.utils.csv_utils import load_csv, save_csv
from src.utils.path_utils import get_skuld_root

# =======================================================
# === MULTI-FILE LOADER =================================
# =======================================================

def load_combined_predictions(directory: Path, pattern: str = "predictions*.csv") -> pd.DataFrame:
    files = sorted(directory.glob(pattern))
    if not files:
        print(f"Warning: No files found matching '{pattern}' in {directory}")
        return pd.DataFrame()

    print(f"Found {len(files)} prediction files. Combining...")
    dfs = []
    for f in files:
        try:
            df = load_csv(str(f))
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = restore_ticker_column(combined_df)

    if TIMESTAMP_COL in combined_df.columns and TICKER_COL in combined_df.columns:
        initial_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=[TIMESTAMP_COL, TICKER_COL], keep='last')
        if len(combined_df) < initial_len:
            print(f"Dropped {initial_len - len(combined_df)} duplicate rows.")

    return combined_df


# =======================================================
# === PUBLIC API ========================================
# =======================================================

def run_evaluation(predictions: Union[str, pd.DataFrame],
                   labeled_csv_path: str,
                   raw_price_csv_path: str,
                   probability_threshold: float):
    if isinstance(predictions, (str, Path)):
        preds_df = load_csv(str(predictions))
    elif isinstance(predictions, pd.DataFrame):
        preds_df = predictions
    else:
        raise ValueError("predictions must be a path or a DataFrame")

    if preds_df.empty:
        print("Predictions DataFrame is empty.")
        return pd.DataFrame(), pd.DataFrame()

    print("Loading labeled data for metrics...")
    labeled_data = load_csv(labeled_csv_path)

    print("Loading raw price data for simulation...")
    raw_price_data = load_csv(raw_price_csv_path)

    trade_results = simulate_trades(preds_df, raw_price_data, probability_threshold)
    metrics = prediction_metrics(preds_df, probability_threshold)

    return trade_results, metrics


# =======================================================
# === FINANCIAL METRICS & TRADING =======================
# =======================================================

def calculate_max_drawdown(returns_series):
    """Calculates the worst peak-to-valley loss in the equity curve."""
    # Create a cumulative wealth index (starting at 1)
    wealth_index = (1 + returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns.min()


def simulate_trades(predictions_df: pd.DataFrame, full_df: pd.DataFrame, probability_threshold: float) -> pd.DataFrame:
    preds = restore_ticker_column(predictions_df.copy())
    full = restore_ticker_column(full_df.copy())

    preds = preds.sort_values([TICKER_COL, TIMESTAMP_COL]).reset_index(drop=True)
    full = full.sort_values([TICKER_COL, TIMESTAMP_COL]).reset_index(drop=True)

    buys = preds[preds[PREDICTION_COL] > probability_threshold]
    trades = []

    # --- 1. Generate Trades ---
    for ticker, group in buys.groupby(TICKER_COL):
        ticker_full = full[full[TICKER_COL] == ticker].sort_values(TIMESTAMP_COL)
        for _, row in group.iterrows():
            buy_time = row[TIMESTAMP_COL]
            buy_price = row[CLOSE_COL]
            # If buy_price is 0 or NaN, skip
            if not buy_price or np.isnan(buy_price): continue

            sell_time = buy_time + LABEL_LOOKAHEAD_MILLIS
            future_rows = ticker_full[ticker_full[TIMESTAMP_COL] >= sell_time]

            if len(future_rows) == 0: continue

            sell_row = future_rows.iloc[0]
            sell_price = sell_row[CLOSE_COL]

            trades.append({
                "ticker": ticker,
                "buy_time": buy_time,
                "buy_price": buy_price,
                "sell_time": sell_row[TIMESTAMP_COL],
                "sell_price": sell_price,
                "return_pct": (sell_price - buy_price) / buy_price,
                "profit_loss": sell_price - buy_price,
            })

    trades_df = pd.DataFrame(trades)

    print(f"\n=== Trading Simulation (Threshold: {probability_threshold}) ===")
    if trades_df.empty:
        print("No trades executed.")
        return trades_df

    # --- 2. Calculate Advanced Financial Metrics ---

    # Sort by sell time to simulate a timeline
    trades_df = trades_df.sort_values("sell_time")
    returns = trades_df["return_pct"]

    # Basic Counts
    n_trades = len(trades_df)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / n_trades

    # Profit Factor (Gross Wins / Gross Losses)
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Risk metrics (Assuming per-trade returns)
    avg_return = returns.mean()
    std_dev = returns.std()

    # Sharpe Ratio (Simplified: Mean / StdDev)
    # Note: For annualized Sharpe, we'd need to normalize by time.
    # Here we present "Per Trade Information Ratio".
    sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0

    # Sortino Ratio (Mean / Downside StdDev)
    downside_std = losses.std()
    sortino_ratio = avg_return / downside_std if downside_std > 0 else 0

    # Win/Loss Ratio
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # Drawdown
    max_dd = calculate_max_drawdown(returns)

    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 40)
    print(f"{'Total Trades':<25} | {n_trades}")
    print(f"{'Win Rate':<25} | {win_rate:.2%}")
    print(f"{'Total Return (Simple)':<25} | {returns.sum():.2%}")
    print(f"{'Avg Return per Trade':<25} | {avg_return:.2%}")
    print(f"{'Profit Factor':<25} | {profit_factor:.2f}")
    print(f"{'Avg Win / Avg Loss':<25} | {wl_ratio:.2f}")
    print(f"{'Sharpe Ratio (Trade)':<25} | {sharpe_ratio:.2f}")
    print(f"{'Sortino Ratio':<25} | {sortino_ratio:.2f}")
    print(f"{'Max Drawdown':<25} | {max_dd:.2%}")
    print("-" * 40)

    return trades_df


# =======================================================
# === ML METRICS ========================================
# =======================================================

def prediction_metrics(predictions_df: pd.DataFrame, probability_threshold: float) -> pd.DataFrame:
    df = predictions_df.copy()
    df["prediction"] = (df[PREDICTION_COL] > probability_threshold).astype(int)

    y_true = df[LABEL_COL]
    y_pred = df["prediction"]
    y_prob = df[PREDICTION_COL]

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    print("\n=== Classification Model Metrics ===")
    print(pd.DataFrame([metrics]).transpose().to_string(header=False))
    return pd.DataFrame([metrics])


def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "N Samples": len(y_true),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "Log Loss": log_loss(y_true, y_prob),
        "Matthews Corr. Coef": matthews_corrcoef(y_true, y_pred),
        "True Positives": int(tp),
        "False Positives": int(fp),
        "True Negatives": int(tn),
        "False Negatives": int(fn),
    }


def restore_ticker_column(df: pd.DataFrame) -> pd.DataFrame:
    ticker_cols = [c for c in df.columns if c.startswith(f"{TICKER_PREFIX}_")]
    if not ticker_cols or TICKER_COL in df.columns:
        return df
    df[TICKER_COL] = df[ticker_cols].idxmax(axis=1)
    prefix_len = len(TICKER_PREFIX) + 1
    df[TICKER_COL] = df[TICKER_COL].str[prefix_len:]
    return df


if __name__ == "__main__":
    threshold = 0.55
    print("Loading and combining prediction files...")
    combined_preds = load_combined_predictions(PY_DATA_DIR_PATH, pattern="predictions*.csv")

    if not combined_preds.empty:
        save_csv(combined_preds, str(AGGREGATE_PREDICTIONS_CSV_PATH))
        trades, metrics = run_evaluation(
            combined_preds,
            str(PREPROCESSED_CSV_PATH),
            str(WIDE_CSV_PATH),
            threshold
        )

        if not trades.empty:
            save_csv(trades, str(TRADE_SIMULATION_CSV_PATH))

        save_csv(metrics, str(EVALUATION_RESULTS_CSV_PATH))