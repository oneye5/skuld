"""Rolling window runner for backtesting across multiple time periods."""

from dataclasses import dataclass
import json
import gc

import pandas as pd
import numpy as np

# Centralized path setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from path_setup import ML_PIPELINE_ROOT

from config.column_names import TIMESTAMP, TICKER, TARGET, CLOSE
from config.model_config import (
    NUM_ROLLING_WINDOWS,
    ROLLING_WINDOW_MOVEMENT_YEARS,
    LOOKAHEAD_DAYS,
    MS_PER_DAY,
    TEST_PERIOD_YEARS,
    get_config_dict,
)
from config.file_paths import (
    OUTPUT_DIR, EVALUATION_DIR, PREDICTIONS_DIR, ensure_output_dirs,
    get_run_dir, get_run_evaluation_dir, get_run_predictions_dir,
)

from runnables.pipeline import prepare_wide_data, run_single_window, PipelineResult
from metrics import (
    evaluate_predictions,
    ClassificationMetrics,
    metrics_to_dict as classification_metrics_to_dict,
)
from simulator import (
    run_trading_simulation,
    run_baseline_simulation,
    TradingMetrics,
    metrics_to_dict as trading_metrics_to_dict,
)
from visualization import generate_all_visualizations


@dataclass
class WindowData:
    """Data collected from a single window for combined evaluation."""
    window_id: int
    predictions: pd.DataFrame
    actuals: pd.DataFrame
    train_start_ts: int
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int


@dataclass
class CombinedResults:
    """Results from combined evaluation across all windows."""
    classification_metrics: ClassificationMetrics
    trading_metrics: TradingMetrics
    baseline_metrics: TradingMetrics
    trades: list[dict]
    window_summaries: list[dict]
    num_predictions: int
    num_windows: int


def calculate_window_timestamps(
    data_max_ts: int,
    num_windows: int = NUM_ROLLING_WINDOWS,
    window_movement_years: float = ROLLING_WINDOW_MOVEMENT_YEARS,
    lookahead_days: int = LOOKAHEAD_DAYS,
    test_period_years: float = TEST_PERIOD_YEARS,
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
        test_period_years: Length of test period in years.
    
    Returns:
        List of (train_end_ts, test_end_ts) tuples.
    """
    window_movement_ms = int(window_movement_years * 365.25 * MS_PER_DAY)
    lookahead_ms = lookahead_days * MS_PER_DAY
    test_period_ms = int(test_period_years * 365.25 * MS_PER_DAY)
    
    # The most recent test_end must leave room for lookahead
    # So test_end = data_max - lookahead_days (to allow labeling)
    latest_test_end = data_max_ts - lookahead_ms
    
    windows = []
    
    for i in range(num_windows):
        # Test end moves backward by window_movement for each window
        test_end_ts = latest_test_end - (i * window_movement_ms)
        
        # Train end is test_end minus test_period
        train_end_ts = test_end_ts - test_period_ms
        
        windows.append((train_end_ts, test_end_ts))
    
    return windows


def run_rolling_windows(
    long_df: pd.DataFrame,
    num_windows: int = NUM_ROLLING_WINDOWS,
    window_movement_years: float = ROLLING_WINDOW_MOVEMENT_YEARS,
) -> CombinedResults:
    """
    Run the pipeline across multiple rolling windows.
    
    Collects all predictions first, then evaluates them combined.
    
    Args:
        long_df: Long format input data.
        num_windows: Number of rolling windows.
        window_movement_years: How far to move window back in time.
    
    Returns:
        CombinedResults with metrics from combined evaluation.
    """
    ensure_output_dirs()
    
    # Prepare wide format data (keep_macro=False for faster processing initially)
    wide_df = prepare_wide_data(long_df, keep_macro=False)
    
    # Free memory from long_df immediately
    del long_df
    gc.collect()
    
    if wide_df.empty:
        raise ValueError("No data after converting to wide format")
    
    # Create memory-efficient price lookup (only timestamp, ticker, close)
    # This drastically reduces memory compared to keeping all features
    price_lookup_df = wide_df[[TIMESTAMP, TICKER, CLOSE]].copy()
    
    # Get timestamp range
    data_max_ts = int(wide_df[TIMESTAMP].max())
    
    # Calculate window timestamps
    window_timestamps = calculate_window_timestamps(
        data_max_ts, num_windows, window_movement_years
    )
    
    # Phase 1: Collect all predictions and actuals
    print("\n" + "=" * 60)
    print("PHASE 1: Collecting Predictions")
    print("=" * 60)
    
    all_window_data: list[WindowData] = []
    all_predictions: list[pd.DataFrame] = []
    all_actuals: list[pd.DataFrame] = []
    
    for window_id, (train_end_ts, test_end_ts) in enumerate(window_timestamps):
        print(f"\nProcessing window {window_id + 1}/{num_windows}...")
        
        # Run pipeline for this window
        result = run_single_window(
            wide_df, train_end_ts, test_end_ts, window_id
        )
        
        if result is None:
            print(f"  Skipping window {window_id}: insufficient data")
            continue
        
        # Add window_id to predictions for tracking (assign to avoid fragmentation)
        result.predictions = result.predictions.assign(window_id=window_id)
        result.test_data_with_labels = result.test_data_with_labels.assign(window_id=window_id)
        
        predictions = result.predictions
        actuals = result.test_data_with_labels
        
        window_data = WindowData(
            window_id=window_id,
            predictions=predictions,
            actuals=actuals,
            train_start_ts=result.train_split.train_start_ts,
            train_end_ts=result.train_split.train_end_ts,
            test_start_ts=result.train_split.test_start_ts,
            test_end_ts=result.train_split.test_end_ts,
        )
        
        all_window_data.append(window_data)
        all_predictions.append(predictions)
        all_actuals.append(actuals)
        
        # Save predictions for this window
        _save_predictions(result.predictions, window_id)
        print(f"  Collected {len(predictions):,} predictions")
        
        # Clean up memory after each window
        del result
        gc.collect()
    
    if not all_window_data:
        raise ValueError("No windows completed successfully")
    
    # Phase 2: Combine and Evaluate
    print("\n" + "=" * 60)
    print("PHASE 2: Combined Evaluation")
    print("=" * 60)
    
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    combined_actuals = pd.concat(all_actuals, ignore_index=True)
    
    print(f"\nTotal predictions: {len(combined_predictions):,}")
    print(f"Total actuals: {len(combined_actuals):,}")
    
    # Evaluate combined classification metrics
    print("\nEvaluating classification metrics...")
    classification_metrics = evaluate_predictions(combined_predictions, combined_actuals)
    
    # Run combined trading simulation
    print("Running trading simulation...")
    trading_metrics, trades = run_trading_simulation(combined_predictions, price_lookup_df)
    
    # Run baseline simulation (over entire test period)
    print("Running baseline simulation...")
    min_test_start = min(w.test_start_ts for w in all_window_data)
    max_test_end = max(w.test_end_ts for w in all_window_data)
    baseline_metrics, _ = run_baseline_simulation(price_lookup_df, min_test_start, max_test_end)
    
    # Create per-window summaries for visualization
    window_summaries = []
    for wd in all_window_data:
        # Evaluate per-window for comparison
        window_class_metrics = evaluate_predictions(wd.predictions, wd.actuals)
        window_trade_metrics, _ = run_trading_simulation(wd.predictions, price_lookup_df)
        
        window_summaries.append({
            'window_id': wd.window_id,
            'accuracy': window_class_metrics.accuracy,
            'precision': window_class_metrics.precision,
            'recall': window_class_metrics.recall,
            'f1': window_class_metrics.f1,
            'trading_return': window_trade_metrics.total_return_pct,
            'num_predictions': len(wd.predictions),
            'test_start': pd.to_datetime(wd.test_start_ts, unit='ms').strftime('%Y-%m-%d'),
            'test_end': pd.to_datetime(wd.test_end_ts, unit='ms').strftime('%Y-%m-%d'),
        })
    
    # Convert trades to dicts for serialization
    trades_dicts = [
        {
            'ticker': t.ticker,
            'buy_timestamp': t.buy_timestamp,
            'sell_timestamp': t.sell_timestamp,
            'buy_price': t.buy_price,
            'sell_price': t.sell_price,
            'shares': t.shares,
            'return_pct': t.return_pct,
        }
        for t in trades
    ]
    
    results = CombinedResults(
        classification_metrics=classification_metrics,
        trading_metrics=trading_metrics,
        baseline_metrics=baseline_metrics,
        trades=trades_dicts,
        window_summaries=window_summaries,
        num_predictions=len(combined_predictions),
        num_windows=len(all_window_data),
    )
    
    # Phase 3: Generate Visualizations
    print("\n" + "=" * 60)
    print("PHASE 3: Generating Visualizations")
    print("=" * 60)
    
    generate_all_visualizations(
        combined_predictions=combined_predictions,
        combined_actuals=combined_actuals,
        trades=trades_dicts,
        window_metrics=window_summaries,
    )
    
    # Save results
    _save_results(results)
    
    return results


def _save_predictions(predictions: pd.DataFrame, window_id: int) -> None:
    """Save predictions for a window to disk."""
    # Save to run-specific directory
    run_pred_dir = get_run_predictions_dir()
    output_path = run_pred_dir / f"predictions_window{window_id}.csv"
    predictions.to_csv(output_path, index=False)
    print(f"  Predictions saved to {output_path}")


def _save_results(results: CombinedResults) -> None:
    """Save results to disk."""
    output = {
        "config": get_config_dict(),  # Include config in results
        "combined_classification": classification_metrics_to_dict(results.classification_metrics),
        "combined_trading": trading_metrics_to_dict(results.trading_metrics),
        "baseline": trading_metrics_to_dict(results.baseline_metrics),
        "num_predictions": results.num_predictions,
        "num_windows": results.num_windows,
        "window_summaries": results.window_summaries,
        "num_trades": len(results.trades),
    }
    
    # Save to run-specific directory
    run_eval_dir = get_run_evaluation_dir()
    output_path = run_eval_dir / "rolling_window_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Also save trades to CSV
    if results.trades:
        trades_df = pd.DataFrame(results.trades)
        trades_path = run_eval_dir / "all_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"Trades saved to {trades_path}")
    
    # Save config separately for easy reference
    config_path = get_run_dir() / "config.json"
    with open(config_path, 'w') as f:
        json.dump(get_config_dict(), f, indent=2)
    print(f"Config saved to {config_path}")


def print_summary(results: CombinedResults) -> None:
    """Print a summary of the combined results."""
    print("\n" + "=" * 60)
    print("COMBINED EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nData Summary:")
    print(f"  Windows Evaluated: {results.num_windows}")
    print(f"  Total Predictions: {results.num_predictions:,}")
    print(f"  Total Trades: {len(results.trades)}")
    
    print("\nClassification Metrics (Combined):")
    cm = results.classification_metrics
    print(f"  Accuracy:  {cm.accuracy:.4f}")
    print(f"  Precision: {cm.precision:.4f}")
    print(f"  Recall:    {cm.recall:.4f}")
    print(f"  F1 Score:  {cm.f1:.4f}")
    if cm.roc_auc is not None:
        print(f"  ROC AUC:   {cm.roc_auc:.4f}")
    
    print("\nTrading Metrics (Strategy):")
    tm = results.trading_metrics
    print(f"  Total Return:   {tm.total_return_pct:.2f}%")
    print(f"  Sharpe Ratio:   {tm.sharpe_ratio:.4f}")
    print(f"  Median Return:  {tm.median_return_pct:.2f}%")
    print(f"  Return Std:     {tm.std_return_pct:.2f}%")
    print(f"  Num Trades:     {tm.num_trades}")
    print(f"  Final Capital:  ${tm.final_capital:,.2f}")
    
    print("\nBaseline Metrics (Buy All):")
    bm = results.baseline_metrics
    print(f"  Total Return:   {bm.total_return_pct:.2f}%")
    print(f"  Sharpe Ratio:   {bm.sharpe_ratio:.4f}")
    print(f"  Final Capital:  ${bm.final_capital:,.2f}")
    
    print("\nPer-Window Summary:")
    for ws in results.window_summaries:
        print(f"  Window {ws['window_id']}: {ws['test_start']} to {ws['test_end']}")
        print(f"    Predictions: {ws['num_predictions']:,}, F1: {ws['f1']:.4f}, Return: {ws['trading_return']:.2f}%")
    
    print("=" * 60)
