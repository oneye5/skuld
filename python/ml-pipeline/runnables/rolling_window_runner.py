"""Rolling window runner for backtesting across multiple time periods."""

import sys
from pathlib import Path
from dataclasses import dataclass
import json

import pandas as pd
import numpy as np

# Add paths for hyphenated directories
_ml_pipeline = Path(__file__).parent.parent
sys.path.insert(0, str(_ml_pipeline))
sys.path.insert(0, str(_ml_pipeline / "evaluation"))
sys.path.insert(0, str(_ml_pipeline / "evaluation" / "model-evaluation"))
sys.path.insert(0, str(_ml_pipeline / "evaluation" / "trade-simulation"))

from config.column_names import TIMESTAMP, TICKER, TARGET, CLOSE
from config.model_config import (
    NUM_ROLLING_WINDOWS,
    ROLLING_WINDOW_MOVEMENT_YEARS,
    LOOKAHEAD_DAYS,
    MS_PER_DAY,
)
from config.file_paths import OUTPUT_DIR, EVALUATION_DIR, ensure_output_dirs

from runnables.pipeline import prepare_wide_data, run_single_window, PipelineResult
from metrics import (
    evaluate_predictions,
    ClassificationMetrics,
    metrics_to_dict as classification_metrics_to_dict,
    aggregate_metrics as aggregate_classification_metrics,
)
from simulator import (
    run_trading_simulation,
    run_baseline_simulation,
    TradingMetrics,
    metrics_to_dict as trading_metrics_to_dict,
    aggregate_trading_metrics,
)


@dataclass
class WindowResult:
    """Results for a single rolling window."""
    window_id: int
    classification_metrics: ClassificationMetrics
    trading_metrics: TradingMetrics
    baseline_metrics: TradingMetrics
    train_start_ts: int
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int


@dataclass
class RollingWindowResults:
    """Aggregated results across all rolling windows."""
    window_results: list[WindowResult]
    aggregated_classification: dict
    aggregated_trading: dict
    aggregated_baseline: dict


def calculate_window_timestamps(
    data_max_ts: int,
    num_windows: int = NUM_ROLLING_WINDOWS,
    window_movement_years: float = ROLLING_WINDOW_MOVEMENT_YEARS,
    lookahead_days: int = LOOKAHEAD_DAYS,
) -> list[tuple[int, int]]:
    """
    Calculate train_end and test_end timestamps for each rolling window.
    
    Windows move backward in time from the most recent data.
    We need to ensure test data has at least lookahead_days of future data 
    available for labeling.
    
    Args:
        data_max_ts: Maximum timestamp in the dataset.
        num_windows: Number of rolling windows.
        window_movement_years: How far to move window back in time (in years).
        lookahead_days: Days needed for lookahead (affects test period end).
    
    Returns:
        List of (train_end_ts, test_end_ts) tuples.
    """
    window_movement_ms = int(window_movement_years * 365.25 * MS_PER_DAY)
    lookahead_ms = lookahead_days * MS_PER_DAY
    
    # The most recent test_end must leave room for lookahead
    # So test_end = data_max - lookahead_days (to allow labeling)
    latest_test_end = data_max_ts - lookahead_ms
    
    windows = []
    
    for i in range(num_windows):
        # Test end moves backward by window_movement for each window
        test_end_ts = latest_test_end - (i * window_movement_ms)
        
        # Train end is some time before test starts
        # We use test_end - 1 year as test_start, and train_end = test_start
        test_period_ms = int(1 * 365.25 * MS_PER_DAY)  # 1 year test period
        train_end_ts = test_end_ts - test_period_ms
        
        windows.append((train_end_ts, test_end_ts))
    
    return windows


def run_rolling_windows(
    long_df: pd.DataFrame,
    num_windows: int = NUM_ROLLING_WINDOWS,
    window_movement_years: float = ROLLING_WINDOW_MOVEMENT_YEARS,
) -> RollingWindowResults:
    """
    Run the pipeline across multiple rolling windows.
    
    Args:
        long_df: Long format input data.
        num_windows: Number of rolling windows.
        window_movement_years: How far to move window back in time.
    
    Returns:
        RollingWindowResults with metrics for all windows.
    """
    ensure_output_dirs()
    
    # Prepare wide format data
    wide_df = prepare_wide_data(long_df)
    
    if wide_df.empty:
        raise ValueError("No data after converting to wide format")
    
    # Get timestamp range
    data_max_ts = int(wide_df[TIMESTAMP].max())
    
    # Calculate window timestamps
    window_timestamps = calculate_window_timestamps(
        data_max_ts, num_windows, window_movement_years
    )
    
    window_results = []
    all_classification_metrics = []
    all_trading_metrics = []
    all_baseline_metrics = []
    
    for window_id, (train_end_ts, test_end_ts) in enumerate(window_timestamps):
        print(f"Processing window {window_id + 1}/{num_windows}...")
        
        # Run pipeline for this window
        result = run_single_window(
            wide_df, train_end_ts, test_end_ts, window_id
        )
        
        if result is None:
            print(f"  Skipping window {window_id}: insufficient data")
            continue
        
        # Evaluate classification metrics
        classification_metrics = evaluate_predictions(
            result.predictions, result.test_data_with_labels
        )
        
        # Run trading simulation
        trading_metrics, _ = run_trading_simulation(
            result.predictions, result.test_data_with_labels
        )
        
        # Run baseline simulation
        baseline_metrics, _ = run_baseline_simulation(
            result.test_data_with_labels,
            result.train_split.test_start_ts,
            result.train_split.test_end_ts,
        )
        
        window_result = WindowResult(
            window_id=window_id,
            classification_metrics=classification_metrics,
            trading_metrics=trading_metrics,
            baseline_metrics=baseline_metrics,
            train_start_ts=result.train_split.train_start_ts,
            train_end_ts=result.train_split.train_end_ts,
            test_start_ts=result.train_split.test_start_ts,
            test_end_ts=result.train_split.test_end_ts,
        )
        
        window_results.append(window_result)
        all_classification_metrics.append(classification_metrics)
        all_trading_metrics.append(trading_metrics)
        all_baseline_metrics.append(baseline_metrics)
        
        # Print window summary
        print(f"  Classification Accuracy: {classification_metrics.accuracy:.4f}")
        print(f"  Trading Return: {trading_metrics.total_return_pct:.2f}%")
        print(f"  Baseline Return: {baseline_metrics.total_return_pct:.2f}%")
    
    if not window_results:
        raise ValueError("No windows completed successfully")
    
    # Aggregate metrics
    aggregated_classification = aggregate_classification_metrics(all_classification_metrics)
    aggregated_trading = aggregate_trading_metrics(all_trading_metrics)
    aggregated_baseline = aggregate_trading_metrics(all_baseline_metrics)
    
    results = RollingWindowResults(
        window_results=window_results,
        aggregated_classification=aggregated_classification,
        aggregated_trading=aggregated_trading,
        aggregated_baseline=aggregated_baseline,
    )
    
    # Save results
    _save_results(results)
    
    return results


def _save_results(results: RollingWindowResults) -> None:
    """Save results to disk."""
    output = {
        "aggregated_classification": results.aggregated_classification,
        "aggregated_trading": results.aggregated_trading,
        "aggregated_baseline": results.aggregated_baseline,
        "windows": [
            {
                "window_id": w.window_id,
                "train_start_ts": w.train_start_ts,
                "train_end_ts": w.train_end_ts,
                "test_start_ts": w.test_start_ts,
                "test_end_ts": w.test_end_ts,
                "classification": classification_metrics_to_dict(w.classification_metrics),
                "trading": trading_metrics_to_dict(w.trading_metrics),
                "baseline": trading_metrics_to_dict(w.baseline_metrics),
            }
            for w in results.window_results
        ],
    }
    
    output_path = EVALUATION_DIR / "rolling_window_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def print_summary(results: RollingWindowResults) -> None:
    """Print a summary of the rolling window results."""
    print("\n" + "=" * 60)
    print("ROLLING WINDOW RESULTS SUMMARY")
    print("=" * 60)
    
    print("\nClassification Metrics (mean ± std):")
    agg = results.aggregated_classification
    print(f"  Accuracy:  {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
    print(f"  Precision: {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f}")
    print(f"  Recall:    {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f}")
    print(f"  F1 Score:  {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}")
    if 'roc_auc_mean' in agg:
        print(f"  ROC AUC:   {agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f}")
    
    print("\nTrading Metrics (mean ± std):")
    agg = results.aggregated_trading
    print(f"  Total Return:  {agg['total_return_mean']:.2f}% ± {agg['total_return_std']:.2f}%")
    print(f"  Sharpe Ratio:  {agg['sharpe_ratio_mean']:.4f} ± {agg['sharpe_ratio_std']:.4f}")
    print(f"  Total Trades:  {agg['total_trades']}")
    
    print("\nBaseline Metrics (buy all):")
    agg = results.aggregated_baseline
    print(f"  Total Return:  {agg['total_return_mean']:.2f}% ± {agg['total_return_std']:.2f}%")
    print(f"  Sharpe Ratio:  {agg['sharpe_ratio_mean']:.4f} ± {agg['sharpe_ratio_std']:.4f}")
    
    print("=" * 60)
