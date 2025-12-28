"""Ranking pipeline - end-to-end ranking-based stock prediction.

This pipeline implements Learning-to-Rank for cross-sectional stock prediction:
1. Load and prepare data (reuses existing data_loader, long_to_wide)
2. Compute forward returns (target for ranking)
3. Run rolling windows with LGBMRanker
4. Compute ranking metrics (IC, RankIC, ICIR)
5. Run portfolio backtest
6. Generate visualizations and reports
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import gc
import json
from typing import Optional, List

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER, CLOSE, TARGET
from config.settings import (
    NUM_ROLLING_WINDOWS,
    ROLLING_WINDOW_MOVEMENT_YEARS,
    TEST_PERIOD_YEARS,
    MS_PER_DAY,
    FORWARD_RETURN_DAYS,
    RETURN_TYPE,
    WINSORIZE_LIMITS,
    MIN_STOCKS_PER_TIMESTAMP,
    PORTFOLIO_TOP_N,
    PORTFOLIO_BOTTOM_N,
    TRANSACTION_COST_BPS,
    SLIPPAGE_BPS,
    MIN_STOCKS_FOR_IC,
    RANKER_N_ESTIMATORS,
    RANKER_LEARNING_RATE,
)

from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers
from core.target_builder import compute_forward_returns, FORWARD_RETURN
from core.splitter import split_by_timestamp, calculate_window_timestamps
from core.scaler import fit_scaler, transform_data
from core.preprocessor import preprocess_data, clip_extreme_values

from features.ratios import add_financial_ratios
from features.technical import add_technical_features
from features.cross_sectional import add_cross_sectional_features
from features.time_features import add_time_features

from learner.ranking import (
    LightGBMRankerWrapper,
    RankerConfig,
    build_group_from_timestamps,
    prepare_ranking_data,
    filter_min_stocks_per_timestamp,
)

from evaluation.ranking_metrics import RankingMetrics, compute_cross_sectional_ic_series
from evaluation.portfolio_simulator import (
    run_portfolio_backtest,
    compute_quintile_portfolio_returns,
    PortfolioConfig,
    BacktestResult,
)


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class RankingWindowResult:
    """Result from a single ranking window."""
    window_id: int
    predictions_df: pd.DataFrame  # timestamp, ticker, predicted_score, actual_return
    train_timestamps: int
    test_timestamps: int
    train_stocks_per_ts: float
    test_stocks_per_ts: float


@dataclass
class RankingPipelineResult:
    """Combined results from ranking pipeline."""
    metrics: RankingMetrics
    backtest: BacktestResult
    quintile_returns: pd.DataFrame
    predictions_df: pd.DataFrame
    window_summaries: List[dict]
    num_windows: int
    config: dict


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

def get_output_dir() -> Path:
    """Get output directory for ranking pipeline results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "output" / "runs" / f"ranking_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# DATA PREPARATION (reuses existing modules)
# =============================================================================

def prepare_wide_data(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long format to wide format with basic preparation.
    
    Reuses logic from single_window.py but simplified for ranking.
    """
    from config.settings import YEAR_2000_MS
    
    # Filter out data before year 2000
    df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS].copy()
    
    # Clean tickers
    df = clean_and_classify_tickers(df)
    df = add_macro_prefix(df)
    wide_df = long_to_wide(df)
    del df
    gc.collect()
    
    # Drop extremely sparse columns
    from core.preprocessor import drop_sparse_columns
    wide_df = drop_sparse_columns(wide_df, threshold=0.95)
    
    # Force float32 to save memory
    for col in wide_df.columns:
        if wide_df[col].dtype == 'float64':
            wide_df[col] = wide_df[col].astype('float32')
    
    return wide_df


def add_all_features(df: pd.DataFrame, global_time_min: int, global_time_max: int) -> pd.DataFrame:
    """Add all features to the DataFrame.
    
    Reuses existing feature engineering modules.
    """
    df = add_financial_ratios(df)
    df = add_technical_features(df)
    df = add_time_features(df, global_time_min, global_time_max)
    # Note: cross_sectional features are computed per-timestamp in training
    return df


def get_feature_columns_for_ranking(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns for ranking model.
    
    Excludes metadata columns (timestamp, ticker, target, forward_return).
    """
    excluded = {TIMESTAMP, TICKER, TARGET, FORWARD_RETURN, CLOSE, 
                'Open', 'High', 'Low', 'Volume'}
    
    feature_cols = [
        col for col in df.columns
        if col not in excluded
        and not col.startswith('MissingFlag_')  # Exclude raw missing flags
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    return feature_cols


# =============================================================================
# SINGLE RANKING WINDOW
# =============================================================================

def run_single_ranking_window(
    wide_df: pd.DataFrame,
    train_end_ts: int,
    test_end_ts: int,
    window_id: int,
    forward_return_days: int,
    return_type: str,
    winsorize_limits: Optional[tuple],
    min_stocks: int,
    ranker_config: RankerConfig,
    global_time_min: int,
    global_time_max: int,
) -> Optional[RankingWindowResult]:
    """Run ranking pipeline for a single train/test window.
    
    Args:
        wide_df: Wide format data.
        train_end_ts: End timestamp for training.
        test_end_ts: End timestamp for test.
        window_id: Window identifier.
        forward_return_days: Days for forward return calculation.
        return_type: 'simple' or 'log'.
        winsorize_limits: Limits for winsorizing returns.
        min_stocks: Minimum stocks per timestamp.
        ranker_config: Configuration for LGBMRanker.
        global_time_min: Min timestamp for time scaling.
        global_time_max: Max timestamp for time scaling.
    
    Returns:
        RankingWindowResult or None if insufficient data.
    """
    lookahead_ms = forward_return_days * MS_PER_DAY
    
    # Slice data with buffer for forward returns
    buffer_end = test_end_ts + lookahead_ms
    wide_slice = wide_df[wide_df[TIMESTAMP] < buffer_end].copy()
    
    # Split data
    split = split_by_timestamp(wide_slice, train_end_ts, test_end_ts)
    
    if split.train.empty or split.test.empty:
        return None
    
    # Compute forward returns for both train and test
    # Use full slice as price lookup to get future prices
    train_with_returns = compute_forward_returns(
        split.train,
        lookahead_days=forward_return_days,
        return_type=return_type,
        winsorize_limits=winsorize_limits,
        drop_na=True,
        price_lookup_df=wide_slice,
    )
    
    test_with_returns = compute_forward_returns(
        split.test,
        lookahead_days=forward_return_days,
        return_type=return_type,
        winsorize_limits=winsorize_limits,
        drop_na=True,
        price_lookup_df=wide_slice,
    )
    
    del wide_slice
    gc.collect()
    
    if train_with_returns.empty or test_with_returns.empty:
        return None
    
    # Filter timestamps with too few stocks
    train_with_returns = filter_min_stocks_per_timestamp(
        train_with_returns, min_stocks, TIMESTAMP
    )
    test_with_returns = filter_min_stocks_per_timestamp(
        test_with_returns, min_stocks, TIMESTAMP
    )
    
    if train_with_returns.empty or test_with_returns.empty:
        return None
    
    # Add features
    train_features = add_all_features(train_with_returns, global_time_min, global_time_max)
    test_features = add_all_features(test_with_returns, global_time_min, global_time_max)
    del train_with_returns, test_with_returns
    
    # Preprocess (handle NaN, infinities)
    train_features = preprocess_data(train_features, add_missing_flags=False)
    test_features = preprocess_data(test_features, add_missing_flags=False)
    
    # Get feature columns (intersection of train and test to ensure consistency)
    train_feature_cols = set(get_feature_columns_for_ranking(train_features))
    test_feature_cols = set(get_feature_columns_for_ranking(test_features))
    feature_cols = sorted(train_feature_cols & test_feature_cols)
    
    if not feature_cols:
        return None
    
    # Fit scaler on training data only
    scaler = fit_scaler(train_features)
    train_scaled = transform_data(train_features, scaler)
    test_scaled = transform_data(test_features, scaler)
    
    # Clip extreme values after scaling
    train_scaled = clip_extreme_values(train_scaled)
    test_scaled = clip_extreme_values(test_scaled)
    
    # Prepare data for ranking model
    X_train, y_train, groups_train = prepare_ranking_data(
        train_scaled,
        feature_cols=feature_cols,
        target_col=FORWARD_RETURN,
        timestamp_col=TIMESTAMP,
    )
    
    X_test, y_test, groups_test = prepare_ranking_data(
        test_scaled,
        feature_cols=feature_cols,
        target_col=FORWARD_RETURN,
        timestamp_col=TIMESTAMP,
    )
    
    # Sort test data to match X_test order (sorted by timestamp)
    test_sorted = test_scaled.sort_values(TIMESTAMP).reset_index(drop=True)
    
    # Train ranking model
    ranker = LightGBMRankerWrapper(ranker_config)
    ranker.fit(X_train, y_train, groups_train)
    
    # Predict
    predictions = ranker.predict(X_test)
    
    # Build predictions DataFrame
    predictions_df = pd.DataFrame({
        TIMESTAMP: test_sorted[TIMESTAMP].values,
        TICKER: test_sorted[TICKER].values,
        "predicted_score": predictions,
        "actual_return": y_test.values,
    })
    
    # Compute summary stats
    train_ts_counts = train_scaled.groupby(TIMESTAMP).size()
    test_ts_counts = test_sorted.groupby(TIMESTAMP).size()
    
    return RankingWindowResult(
        window_id=window_id,
        predictions_df=predictions_df,
        train_timestamps=len(train_ts_counts),
        test_timestamps=len(test_ts_counts),
        train_stocks_per_ts=train_ts_counts.mean(),
        test_stocks_per_ts=test_ts_counts.mean(),
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_ranking_pipeline(
    long_df: Optional[pd.DataFrame] = None,
    num_windows: int = NUM_ROLLING_WINDOWS,
    window_movement_years: float = ROLLING_WINDOW_MOVEMENT_YEARS,
    test_period_years: float = TEST_PERIOD_YEARS,
    forward_return_days: int = FORWARD_RETURN_DAYS,
    return_type: str = RETURN_TYPE,
    winsorize_limits: Optional[tuple] = WINSORIZE_LIMITS,
    min_stocks: int = MIN_STOCKS_PER_TIMESTAMP,
    portfolio_top_n: int = PORTFOLIO_TOP_N,
    portfolio_bottom_n: int = PORTFOLIO_BOTTOM_N,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    save_results: bool = True,
) -> RankingPipelineResult:
    """Run the full ranking pipeline across rolling windows.
    
    Args:
        long_df: Long format input data. If None, loads from default path.
        num_windows: Number of rolling windows.
        window_movement_years: How far back each window moves.
        test_period_years: Length of test period per window.
        forward_return_days: Days for forward return calculation.
        return_type: 'simple' or 'log' returns.
        winsorize_limits: Limits for winsorizing returns.
        min_stocks: Minimum stocks per timestamp.
        portfolio_top_n: Stocks for long portfolio.
        portfolio_bottom_n: Stocks for short portfolio.
        transaction_cost_bps: Transaction cost in basis points.
        slippage_bps: Slippage in basis points per trade.
        save_results: If True, save results to output directory.
    
    Returns:
        RankingPipelineResult with metrics, backtest, and predictions.
    """
    # Load data
    if long_df is None:
        print("Loading data...")
        long_df = load_long_data()
        print(f"Loaded {len(long_df):,} rows")
    
    # Convert to wide format
    print("Converting to wide format...")
    wide_df = prepare_wide_data(long_df)
    print(f"Wide format: {len(wide_df):,} rows, {len(wide_df.columns)} columns")
    
    del long_df
    gc.collect()
    
    if wide_df.empty:
        raise ValueError("No data after converting to wide format")
    
    # Get timestamp range
    data_min_ts = int(wide_df[TIMESTAMP].min())
    data_max_ts = int(wide_df[TIMESTAMP].max())
    
    # Calculate window timestamps
    window_timestamps = calculate_window_timestamps(
        data_max_ts,
        num_windows,
        window_movement_years,
        forward_return_days,
        test_period_years,
    )
    
    # Ranker configuration
    ranker_config = RankerConfig(
        n_estimators=RANKER_N_ESTIMATORS,
        learning_rate=RANKER_LEARNING_RATE,
    )
    
    # Run each window
    print(f"\nRunning {num_windows} ranking windows...")
    
    all_predictions: List[pd.DataFrame] = []
    window_summaries: List[dict] = []
    
    for window_id, (train_end_ts, test_end_ts) in enumerate(window_timestamps):
        print(f"\n--- Window {window_id + 1}/{num_windows} ---")
        
        result = run_single_ranking_window(
            wide_df=wide_df,
            train_end_ts=train_end_ts,
            test_end_ts=test_end_ts,
            window_id=window_id,
            forward_return_days=forward_return_days,
            return_type=return_type,
            winsorize_limits=winsorize_limits,
            min_stocks=min_stocks,
            ranker_config=ranker_config,
            global_time_min=data_min_ts,
            global_time_max=data_max_ts,
        )
        
        if result is None:
            print(f"  Skipping: insufficient data")
            continue
        
        # Add window_id to predictions
        result.predictions_df["window_id"] = window_id
        all_predictions.append(result.predictions_df)
        
        # Create summary
        summary = {
            "window_id": window_id,
            "train_timestamps": result.train_timestamps,
            "test_timestamps": result.test_timestamps,
            "train_stocks_per_ts": f"{result.train_stocks_per_ts:.1f}",
            "test_stocks_per_ts": f"{result.test_stocks_per_ts:.1f}",
        }
        window_summaries.append(summary)
        
        print(f"  Train: {result.train_timestamps} timestamps, {result.train_stocks_per_ts:.1f} stocks/ts")
        print(f"  Test:  {result.test_timestamps} timestamps, {result.test_stocks_per_ts:.1f} stocks/ts")
    
    if not all_predictions:
        raise ValueError("No windows completed successfully")
    
    # Combine all predictions
    print("\nCombining results...")
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Compute ranking metrics
    print("Computing ranking metrics...")
    metrics = RankingMetrics.from_predictions(
        combined_predictions,
        timestamp_col=TIMESTAMP,
        predicted_col="predicted_score",
        actual_col="actual_return",
        min_stocks=MIN_STOCKS_FOR_IC,
    )
    
    # Compute quintile returns
    quintile_returns = compute_quintile_portfolio_returns(
        combined_predictions,
        timestamp_col=TIMESTAMP,
        score_col="predicted_score",
        return_col="actual_return",
    )
    
    # Run portfolio backtest
    print("Running portfolio backtest...")
    portfolio_config = PortfolioConfig(
        top_n=portfolio_top_n,
        bottom_n=portfolio_bottom_n,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    backtest = run_portfolio_backtest(
        combined_predictions,
        portfolio_config,
        timestamp_col=TIMESTAMP,
        score_col="predicted_score",
        return_col="actual_return",
        return_horizon_days=forward_return_days,
    )
    
    # Build config dict
    config = {
        "num_windows": num_windows,
        "window_movement_years": window_movement_years,
        "test_period_years": test_period_years,
        "forward_return_days": forward_return_days,
        "return_type": return_type,
        "winsorize_limits": winsorize_limits,
        "min_stocks": min_stocks,
        "portfolio_top_n": portfolio_top_n,
        "portfolio_bottom_n": portfolio_bottom_n,
        "transaction_cost_bps": transaction_cost_bps,
        "ranker_n_estimators": ranker_config.n_estimators,
        "ranker_learning_rate": ranker_config.learning_rate,
    }
    
    result = RankingPipelineResult(
        metrics=metrics,
        backtest=backtest,
        quintile_returns=quintile_returns,
        predictions_df=combined_predictions,
        window_summaries=window_summaries,
        num_windows=len(all_predictions),
        config=config,
    )
    
    # Save results
    if save_results:
        output_dir = save_ranking_results(result)
        print(f"\nResults saved to: {output_dir}")
    
    return result


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_ranking_results(result: RankingPipelineResult) -> Path:
    """Save ranking pipeline results to output directory."""
    output_dir = get_output_dir()
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(result.config, f, indent=2)
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(result.metrics.to_dict(), f, indent=2)
    
    # Save predictions
    result.predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Save quintile returns
    result.quintile_returns.to_csv(output_dir / "quintile_returns.csv")
    
    # Save backtest results
    result.backtest.daily_returns.to_csv(output_dir / "daily_returns.csv")
    
    # Save window summaries
    with open(output_dir / "window_summaries.json", "w") as f:
        json.dump(result.window_summaries, f, indent=2)
    
    # Generate visualizations
    try:
        from evaluation.visualization import (
            plot_quintile_returns,
            plot_ic_series,
            plot_cumulative_returns,
            create_ranking_dashboard,
        )
        
        # Quintile chart
        plot_quintile_returns(
            result.quintile_returns,
            save_path=str(output_dir / "quintile_returns.png"),
        )
        
        # IC series
        plot_ic_series(
            result.metrics.ic_series,
            save_path=str(output_dir / "ic_series.png"),
        )
        
        # Cumulative returns
        plot_cumulative_returns(
            result.backtest.daily_returns,
            save_path=str(output_dir / "cumulative_returns.png"),
        )
        
        # Dashboard
        create_ranking_dashboard(
            result.metrics.ic_series,
            result.quintile_returns,
            result.backtest.daily_returns,
            save_path=str(output_dir / "dashboard.png"),
        )
        
        import matplotlib.pyplot as plt
        plt.close('all')
        
    except ImportError:
        print("Warning: matplotlib not available, skipping visualizations")
    
    return output_dir


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_ranking_summary(result: RankingPipelineResult) -> None:
    """Print summary of ranking pipeline results."""
    print("\n" + "=" * 60)
    print("RANKING PIPELINE RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nWindows completed: {result.num_windows}")
    
    print("\n--- Ranking Metrics ---")
    print(result.metrics.summary())
    
    print("\n--- Portfolio Backtest ---")
    print(result.backtest.summary())
