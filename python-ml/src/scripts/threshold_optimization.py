"""Threshold optimization to find optimal probability thresholds.

Run evaluation at multiple probability thresholds to find optimal balance of:
- Recall (want reasonable coverage)
- Drawdown (want <-50%)
- Sharpe ratio (want >1.0)
- Win rate (want >60%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
from scipy.stats import skew, kurtosis

from src.utils.io_utils import load_data, save_csv
from src.evaluation.trading_evaluation import simulate_trades
from src.evaluation.ml_evaluation import calculate_ml_metrics
from src.config.config import (
    AGGREGATE_PREDICTIONS_CSV_PATH,
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
    PREPROCESSED_CSV_PATH,
    PY_DATA_DIR_PATH
)
from src.preprocessing.pre_split_preprocessing import restore_ticker_column
from src.evaluation.utils import load_combined_predictions


def run_threshold_optimization(
        thresholds: List[float],
        predictions_df: Union[str, Path, pd.DataFrame] = None
) -> pd.DataFrame:
    """Test multiple probability thresholds and evaluate trading performance.
    
    Loads predictions (or accepts DataFrame) and evaluates performance at each
    probability threshold. Returns comprehensive metrics for comparison.
    
    Args:
        thresholds: List of probability thresholds to test (e.g., [0.50, 0.60, 0.70]).
        predictions_df: Path to predictions CSV, DataFrame, or None to load from AGGREGATE_PREDICTIONS_CSV_PATH.
    
    Returns:
        DataFrame with metrics for each threshold including ML and trading metrics.
    
    Raises:
        ValueError: If threshold values are not in [0, 1] or predictions are invalid.
        FileNotFoundError: If predictions file not found.
    """
    # Validate thresholds
    for t in thresholds:
        if not (0 <= t <= 1):
            raise ValueError(f"All thresholds must be in [0, 1], got {t}")
    
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)
    
    # Load predictions
    print("\nLoading predictions...")
    if predictions_df is None:
        if not AGGREGATE_PREDICTIONS_CSV_PATH.exists():
            raise FileNotFoundError(f"Predictions file not found: {AGGREGATE_PREDICTIONS_CSV_PATH}")
        # Aggregate predictions are external CSV files
        predictions = load_data(str(AGGREGATE_PREDICTIONS_CSV_PATH))
    elif isinstance(predictions_df, (str, Path)):
        # Auto-detect format from extension
        predictions = load_data(str(predictions_df))
    elif isinstance(predictions_df, pd.DataFrame):
        predictions = predictions_df.copy()
    else:
        raise ValueError("predictions_df must be a file path, DataFrame, or None")
    
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty")
    
    # Load price data for trading simulation
    print("Loading price data...")
    if PREPROCESSED_CSV_PATH.exists():
        # Preprocessed is now Parquet, auto-detect
        price_data = load_data(str(PREPROCESSED_CSV_PATH))
        price_data = restore_ticker_column(price_data)
    else:
        # Fallback: combine train and test (also Parquet now)
        train_df = load_data(str(TRAIN_CSV_PATH))
        test_df = load_data(str(TEST_CSV_PATH))
        price_data = pd.concat([train_df, test_df], ignore_index=True)
        price_data = restore_ticker_column(price_data)
    
    print(f"Predictions: {len(predictions)} rows")
    print(f"Price data: {len(price_data)} rows\n")
    
    results = []
    
    # Test each threshold
    for threshold in thresholds:
        print(f"\n{'=' * 80}")
        print(f"THRESHOLD: {threshold:.4f}")
        print(f"{'=' * 80}")
        
        try:
            # Run trading simulation
            trades_df = simulate_trades(
                predictions,
                price_data,
                threshold
            )
            
            # Calculate ML metrics
            ml_metrics_df = calculate_ml_metrics(predictions, threshold)
            ml_metrics = ml_metrics_df.iloc[0].to_dict() if not ml_metrics_df.empty else {}
            
            if trades_df.empty:
                print(f"  ⚠ No trades executed at this threshold")
                result = {
                    'threshold': threshold,
                    'n_trades': 0,
                    'win_rate': np.nan,
                    'avg_return': np.nan,
                    'total_return': np.nan,
                    'std_return': np.nan,
                    'max_dd': np.nan,
                    'sharpe': np.nan,
                    'sortino': np.nan,
                }
                result.update({k: ml_metrics.get(k, np.nan) for k in ml_metrics if k not in result})
                results.append(result)
                continue
            
            # Calculate trading metrics
            win_count = (trades_df['return_pct'] > 0).sum()
            win_rate = win_count / len(trades_df) if len(trades_df) > 0 else 0
            avg_return = trades_df['return_pct'].mean()
            total_return = trades_df['return_pct'].sum()
            std_return = trades_df['return_pct'].std()
            max_dd = _calculate_max_drawdown(trades_df['return_pct'])
            sharpe = _calculate_sharpe(trades_df['return_pct'])
            sortino = _calculate_sortino(trades_df['return_pct'])
            
            result = {
                'threshold': threshold,
                'n_trades': len(trades_df),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'total_return': total_return,
                'std_return': std_return,
                'max_dd': max_dd,
                'sharpe': sharpe,
                'sortino': sortino,
            }
            
            # Add ML metrics
            result.update({k: ml_metrics.get(k, np.nan) for k in ml_metrics if k not in result})
            results.append(result)
            
            print(f"  Trades: {len(trades_df)}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Avg Return: {avg_return:.4f}")
            print(f"  Sharpe: {sharpe:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Display summary
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80 + "\n")
    
    if not results_df.empty:
        # Format display columns
        display_df = results_df.copy()
        for col in ['win_rate', 'recall', 'balanced_accuracy']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                )
        for col in ['max_dd']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                )
        for col in ['avg_return', 'sharpe', 'sortino', 'f1', 'accuracy']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
        
        print(display_df.to_string(index=False))
    
    return results_df


def _calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def _calculate_sharpe(returns: pd.Series, rf=0.0) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - rf
    return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0


def _calculate_sortino(returns: pd.Series, rf=0.0) -> float:
    """Calculate Sortino ratio."""
    excess_returns = returns - rf
    downside = excess_returns[excess_returns < 0].std()
    return excess_returns.mean() / downside if downside > 0 else 0


def run(thresholds: List[float] = None, output_path: Path = None):
    """Execute threshold optimization and save results.
    
    Args:
        thresholds: List of probability thresholds to test. Defaults to standard set.
        output_path: Path to save results CSV. Defaults to data directory.
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.825]
    
    if output_path is None:
        output_path = Path(PY_DATA_DIR_PATH) / "threshold_optimization_results.csv"
    
    # Run optimization
    results_df = run_threshold_optimization(thresholds)
    
    # Save results
    if not results_df.empty:
        save_csv(results_df, str(output_path))
        print(f"\n✓ Results saved to {output_path}")
    else:
        print("\n⚠ No results to save")


if __name__ == "__main__":
    run()
