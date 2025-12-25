"""Rolling window pipeline - runs multiple train/test windows."""

from dataclasses import dataclass
from datetime import datetime
import gc

import pandas as pd

from config.columns import TIMESTAMP, TICKER, CLOSE, TARGET
from config.settings import (
    NUM_ROLLING_WINDOWS,
    ROLLING_WINDOW_MOVEMENT_YEARS,
    LOOKAHEAD_DAYS,
    TEST_PERIOD_YEARS,
)

from core.data_loader import load_long_data
from core.splitter import calculate_window_timestamps

from pipeline.single_window import prepare_wide_data, run_single_window, SingleWindowResult

from evaluation.metrics import compute_classification_metrics, ClassificationMetrics
from evaluation.simulator import run_trading_simulation, TradingMetrics, Trade
from evaluation.reporter import generate_report


@dataclass
class RollingWindowResult:
    """Combined results from all rolling windows."""
    classification: ClassificationMetrics
    trading: TradingMetrics
    trades: list[Trade]
    window_summaries: list[dict]
    num_windows: int


def run_rolling_windows(
    long_df: pd.DataFrame | None = None,
    num_windows: int = NUM_ROLLING_WINDOWS,
    window_movement_years: float = ROLLING_WINDOW_MOVEMENT_YEARS,
    test_period_years: float = TEST_PERIOD_YEARS,
    lookahead_days: int = LOOKAHEAD_DAYS,
) -> RollingWindowResult:
    """Run the pipeline across multiple rolling windows.
    
    Args:
        long_df: Long format input data. If None, loads from default path.
        num_windows: Number of rolling windows.
        window_movement_years: How far back each window moves.
        test_period_years: Length of test period per window.
        lookahead_days: Days to look ahead for labeling.
    
    Returns:
        RollingWindowResult with combined metrics.
    """
    # Load data if not provided
    if long_df is None:
        print("Loading data...")
        long_df = load_long_data()
        print(f"Loaded {len(long_df):,} rows")
    
    # Prepare wide format
    print("Converting to wide format...")
    wide_df = prepare_wide_data(long_df)
    print(f"Wide format: {len(wide_df):,} rows, {len(wide_df.columns)} columns")
    
    # Free memory
    del long_df
    gc.collect()
    
    if wide_df.empty:
        raise ValueError("No data after converting to wide format")
    
    # Get timestamp range for global time scaling
    data_min_ts = int(wide_df[TIMESTAMP].min())
    data_max_ts = int(wide_df[TIMESTAMP].max())
    
    # Calculate window timestamps
    window_timestamps = calculate_window_timestamps(
        data_max_ts,
        num_windows,
        window_movement_years,
        lookahead_days,
        test_period_years,
    )
    
    # Run each window
    print(f"\nRunning {num_windows} rolling windows...")
    
    all_predictions: list[pd.DataFrame] = []
    all_actuals: list[pd.DataFrame] = []
    window_summaries: list[dict] = []
    
    for window_id, (train_end_ts, test_end_ts) in enumerate(window_timestamps):
        print(f"\n--- Window {window_id + 1}/{num_windows} ---")
        
        result = run_single_window(
            wide_df, train_end_ts, test_end_ts, window_id, lookahead_days,
            global_time_min=data_min_ts,
            global_time_max=data_max_ts,
        )
        
        if result is None:
            print(f"  Skipping: insufficient data")
            continue
        
        # Add window_id for tracking
        result.predictions["window_id"] = window_id
        result.actuals["window_id"] = window_id
        
        all_predictions.append(result.predictions)
        all_actuals.append(result.actuals)
        
        # Create window summary
        summary = _create_window_summary(result)
        window_summaries.append(summary)
        
        print(f"  Train: {summary['train_period']}")
        print(f"  Test:  {summary['test_period']}")
        print(f"  Test samples: {summary['test_samples']}, Positive rate: {summary['positive_rate']}")
    
    if not all_predictions:
        raise ValueError("No windows completed successfully")
    
    # Combine all predictions and actuals
    print("\nCombining results from all windows...")
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    combined_actuals = pd.concat(all_actuals, ignore_index=True)
    
    # Calculate classification metrics
    classification = compute_classification_metrics(combined_actuals, combined_predictions)
    
    # Run trading simulation
    print("Running trading simulation...")
    price_data = wide_df[[TIMESTAMP, TICKER, CLOSE]].copy()
    trading, trades = run_trading_simulation(combined_predictions, price_data)
    
    # Generate report
    print("\nGenerating report...")
    run_dir = generate_report(classification, trading, trades, window_summaries)
    print(f"Results saved to: {run_dir}")
    
    return RollingWindowResult(
        classification=classification,
        trading=trading,
        trades=trades,
        window_summaries=window_summaries,
        num_windows=len(all_predictions),
    )


def _create_window_summary(result: SingleWindowResult) -> dict:
    """Create summary dict for a single window."""
    def ts_to_date(ts: int) -> str:
        return datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
    
    test_samples = len(result.actuals)
    positive_samples = int(result.actuals[TARGET].sum())
    positive_rate = positive_samples / test_samples if test_samples > 0 else 0
    
    return {
        "window_id": result.window_id,
        "train_period": f"{ts_to_date(result.split.train_start_ts)} to {ts_to_date(result.split.train_end_ts)}",
        "test_period": f"{ts_to_date(result.split.test_start_ts)} to {ts_to_date(result.split.test_end_ts)}",
        "test_samples": test_samples,
        "positive_samples": positive_samples,
        "positive_rate": f"{positive_rate:.1%}",
    }


def print_summary(result: RollingWindowResult) -> None:
    """Print a summary of rolling window results."""
    print("\n" + "=" * 60)
    print("ROLLING WINDOW RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nWindows completed: {result.num_windows}")
    
    print("\n--- Classification Metrics ---")
    c = result.classification
    print(f"  Accuracy:  {c.accuracy:.4f}")
    print(f"  Precision: {c.precision:.4f}")
    print(f"  Recall:    {c.recall:.4f}")
    print(f"  F1 Score:  {c.f1:.4f}")
    print(f"  AUC-ROC:   {c.auc_roc:.4f}")
    
    print("\n--- Trading Simulation ---")
    t = result.trading
    print(f"  Trades:       {t.num_trades}")
    print(f"  Mean Return:  {t.mean_return_pct:.2f}%")
    print(f"  Sharpe Ratio: {t.sharpe_ratio:.3f}")
    print(f"  Min/Max:      {t.min_return_pct:.2f}% / {t.max_return_pct:.2f}%")
