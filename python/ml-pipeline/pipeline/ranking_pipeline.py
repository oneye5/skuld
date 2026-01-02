"""Ranking pipeline - end-to-end ranking-based stock prediction.

This pipeline implements Learning-to-Rank for cross-sectional stock prediction:
1. Load and prepare data (reuses existing data_loader, long_to_wide)
2. Compute forward returns (target for ranking)
3. Run rolling windows with LGBMRanker
4. Compute ranking metrics (IC, RankIC, ICIR)
5. Run portfolio backtest
6. Generate visualizations and reports

Key safeguards against data leakage:
- Train/test split with strict temporal ordering
- Scalers fit on training data only
- Cross-sectional features computed per-timestamp after split
- Forward returns computed with proper lookahead handling
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


def _to_json_serializable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        # Convert both keys and values - handle numpy int keys
        return {
            (int(k) if isinstance(k, (np.integer, np.int32, np.int64)) else k): _to_json_serializable(v) 
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


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
    RANKER_NUM_LEAVES,
    RANKER_MAX_DEPTH,
    RANKER_MIN_CHILD_SAMPLES,
    RANKER_REG_ALPHA,
    RANKER_REG_LAMBDA,
    RANKER_SUBSAMPLE,
    RANKER_COLSAMPLE_BYTREE,
    RANKER_DEVICE,
    RANKER_EARLY_STOPPING_ROUNDS,
)

from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers
from core.target_builder import compute_forward_returns, FORWARD_RETURN
from core.splitter import split_by_timestamp, calculate_window_timestamps
from core.scaler import fit_scaler, transform_data
from core.preprocessor import preprocess_data, clip_extreme_values
from core.validation import (
    validate_wide_data,
    validate_no_lookahead,
    validate_groups_match_data,
    check_data_quality_report,
    ValidationError,
)
from core.experiment_tracking import create_experiment_manifest, compute_data_hash
from core.logging_config import (
    get_logger,
    log_timing,
    log_dataframe_info,
    log_metrics,
    log_window_start,
    log_window_result,
    log_pipeline_summary,
    log_config,
)
import traceback
import uuid

from features.ratios import add_financial_ratios
from features.technical import add_technical_features
from features.cross_sectional import add_cross_sectional_features
from features.feature_config import apply_experimental_features

from learner.ranking import (
    LightGBMRankerWrapper,
    RankerConfig,
    build_group_from_timestamps,
    prepare_ranking_data,
    filter_min_stocks_per_timestamp,
)

from evaluation.ranking_metrics import (
    RankingMetrics, 
    compute_cross_sectional_ic_series,
    ComprehensiveMetrics,
    compute_decile_returns,
)
from evaluation.portfolio_simulator import (
    run_portfolio_backtest,
    compute_quintile_portfolio_returns,
    PortfolioConfig,
    BacktestResult,
    run_random_baseline,
    RandomBaselineResult,
)

# Module logger
logger = get_logger(__name__)


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
    feature_importances: Optional[dict] = None  # Feature importances from this window
    cluster_map: Optional[dict] = None  # Cluster assignments from this window


@dataclass
class ClusterInfo:
    """Cluster information for the pipeline result."""
    cluster_map: dict[str, int]  # ticker -> cluster_id
    cluster_report: dict  # Full report from get_cluster_membership_report
    cluster_performance: Optional[pd.DataFrame] = None  # Performance by cluster


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
    # Extended metrics and data
    comprehensive_metrics: Optional['ComprehensiveMetrics'] = None
    feature_importances: Optional[dict] = None
    decile_returns: Optional[dict] = None
    # Random baseline comparison
    random_baseline: Optional['RandomBaselineResult'] = None
    # Cluster analysis (NZX-focused)
    cluster_info: Optional[ClusterInfo] = None


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

def get_output_dir(run_id: Optional[str] = None) -> Path:
    """Get output directory for ranking pipeline results.
    
    Args:
        run_id: Optional run identifier. If None, generates timestamp-based ID.
    
    Returns:
        Path to output directory.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "output" / "runs" / f"ranking_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _generate_run_id() -> str:
    """Generate a unique run identifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


# =============================================================================
# DATA PREPARATION (reuses existing modules)
# =============================================================================

def prepare_wide_data(
    long_df: pd.DataFrame | None = None,
    use_cache: bool = True,
    filter_anomalies: bool | None = None,
) -> pd.DataFrame:
    """Convert long format to wide format with basic preparation.
    
    Uses parquet caching for 20x speedup on repeated runs.
    
    Args:
        long_df: Long format data. If None and use_cache=True, loads from cache.
        use_cache: Whether to use parquet cache. Default True.
        filter_anomalies: Whether to detect and filter anomalous price data.
            None (default) uses the FILTER_ANOMALIES setting from config.
    
    Returns:
        Wide format DataFrame.
    """
    from core.preprocessor import (
        drop_sparse_columns, 
        detect_price_anomalies, 
        filter_anomalous_data,
        get_anomaly_summary,
    )
    from config.settings import FILTER_ANOMALIES, ANOMALY_RETURN_THRESHOLD
    
    if filter_anomalies is None:
        filter_anomalies = FILTER_ANOMALIES
    
    if use_cache:
        from core.data_cache import load_cached_wide_data
        # load_cached_wide_data handles the full transformation pipeline
        wide_df = load_cached_wide_data()
    else:
        # Original implementation for when cache is disabled
        from config.settings import YEAR_2000_MS
        
        if long_df is None:
            long_df = load_long_data()
        
        # Filter out data before year 2000
        df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS].copy()
        
        # Clean tickers
        df = clean_and_classify_tickers(df)
        df = add_macro_prefix(df)
        wide_df = long_to_wide(df)
        del df
        gc.collect()
    
    # Drop extremely sparse columns
    wide_df = drop_sparse_columns(wide_df, threshold=0.95)
    
    # Detect and filter price anomalies (unadjusted splits, ticker recycling, etc.)
    # This trims the OLD data before the discontinuity, keeping the newer series
    if filter_anomalies and CLOSE in wide_df.columns:
        logger = get_logger(__name__)
        logger.info("Detecting price anomalies...")
        
        wide_df = detect_price_anomalies(
            wide_df,
            price_col=CLOSE,
            return_threshold=ANOMALY_RETURN_THRESHOLD,
        )
        
        summary = get_anomaly_summary(wide_df)
        if summary.get('n_affected_tickers', 0) > 0:
            logger.warning(
                f"Found {summary['anomaly_rows']} anomaly points in {summary['n_affected_tickers']} tickers. "
                f"Trimming {summary['rows_to_trim']} rows ({summary['trim_pct']:.2f}%) of pre-anomaly data."
            )
            logger.info(f"Affected tickers: {summary['affected_tickers']}")
        
        wide_df, removed = filter_anomalous_data(wide_df, trim_before_anomaly=True)
        logger.info(f"Removed {len(removed)} rows (old data before price discontinuities)")
    
    # Force float32 to save memory
    for col in wide_df.columns:
        if wide_df[col].dtype == 'float64':
            wide_df[col] = wide_df[col].astype('float32')
    
    return wide_df


def prepare_wide_data_with_features(
    long_df: pd.DataFrame | None = None,
    use_cache: bool = True,
    experimental_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert long format to wide format WITH pre-computed features.
    
    Uses parquet caching for ~20x speedup on repeated runs.
    This combines prepare_wide_data() + add_all_features() with caching.
    
    Args:
        long_df: Long format data. If None, loads from default.
        use_cache: Whether to use parquet cache. Default True.
        experimental_features: List of experimental feature sets to apply.
    
    Returns:
        Wide format DataFrame with features computed.
    """
    if use_cache and long_df is None:
        from core.data_cache import load_cached_wide_data_with_features
        return load_cached_wide_data_with_features(
            force_refresh=False,
            experimental_features=experimental_features,
        )
    else:
        # Compute without cache
        wide_df = prepare_wide_data(long_df, use_cache=use_cache)
        return add_all_features(wide_df, experimental_features)


def add_all_features(
    df: pd.DataFrame, 
    experimental_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add all features to the DataFrame.
    
    Reuses existing feature engineering modules.
    
    Args:
        df: Input DataFrame.
        experimental_features: List of experimental feature sets to apply.
    """
    df = add_financial_ratios(df)
    df = add_technical_features(df)
    
    # Always add alpha factors (research-backed features)
    from features.alpha_factors import add_alpha_factors
    df = add_alpha_factors(df)
    
    # Apply additional experimental features if requested
    if experimental_features and "alpha_fast" in experimental_features:
        from features.alpha_factors_fast import add_alpha_factors_fast
        df = add_alpha_factors_fast(df)
    elif experimental_features:
        df = apply_experimental_features(df, experimental_features)
    
    # Note: cross_sectional features are computed per-timestamp in training
    return df


def get_feature_columns_for_ranking(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns for ranking model.
    
    Excludes metadata columns (timestamp, ticker, target, forward_return)
    and raw price/event columns that could cause data leakage.
    """
    excluded = {TIMESTAMP, TICKER, TARGET, FORWARD_RETURN, CLOSE, 
                'Open', 'High', 'Low', 'Volume',
                # Raw price and event columns - potential leakage sources
                'AdjClose',   # Raw price level - no cross-sectional meaning, used only for returns
                'Dividend',   # Point-in-time dividend - could encode future events
                'Split',      # Stock split indicator - could encode future events
                # Cluster features - Rank_InCluster uses raw Close price which implicitly
                # encodes dividend/split history (AdjClose vs Close divergence).
                # This creates spurious correlation with survival/returns.
                # Cluster ID also correlates with volatility regimes and survivorship.
                'Cluster',         # Raw cluster ID - correlates with volatility/survival
                'Rank_InCluster',  # Uses Close price which encodes div/split history
                }
    
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
) -> Optional[RankingWindowResult]:
    """Run ranking pipeline for a single train/test window.
    
    Note: Features should already be computed on wide_df before calling this.
    
    Args:
        wide_df: Wide format data WITH features already computed.
        train_end_ts: End timestamp for training.
        test_end_ts: End timestamp for test.
        window_id: Window identifier.
        forward_return_days: Days for forward return calculation.
        return_type: 'simple' or 'log'.
        winsorize_limits: Limits for winsorizing returns.
        min_stocks: Minimum stocks per timestamp.
        ranker_config: Configuration for LGBMRanker.
    
    Returns:
        RankingWindowResult or None if insufficient data.
    
    Raises:
        ValidationError: If data validation fails (e.g., lookahead bias detected).
    """
    lookahead_ms = forward_return_days * MS_PER_DAY
    
    logger.info(f"Window {window_id}: Starting processing")
    logger.debug(f"Window {window_id}: train_end_ts={train_end_ts}, test_end_ts={test_end_ts}")
    
    # Slice data with buffer for forward returns
    buffer_end = test_end_ts + lookahead_ms
    wide_slice = wide_df[wide_df[TIMESTAMP] < buffer_end].copy()
    logger.debug(f"Window {window_id}: Data slice has {len(wide_slice):,} rows")
    
    # Split data
    split = split_by_timestamp(wide_slice, train_end_ts, test_end_ts)
    
    if split.train.empty or split.test.empty:
        logger.warning(f"Window {window_id}: Empty train or test split")
        return None
    
    # VALIDATION: Ensure no lookahead bias in train/test split
    try:
        validate_no_lookahead(split.train, split.test, TIMESTAMP)
    except ValidationError as e:
        logger.error(f"Window {window_id}: Lookahead validation failed: {e}")
        raise
    
    # Compute forward returns for both train and test
    # Use full slice as price lookup to get future prices
    with log_timing(f"Window {window_id}: compute forward returns", logger):
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
        logger.warning(f"Window {window_id}: Empty data after forward return computation")
        return None
    
    # Filter timestamps with too few stocks
    train_with_returns = filter_min_stocks_per_timestamp(
        train_with_returns, min_stocks, TIMESTAMP
    )
    test_with_returns = filter_min_stocks_per_timestamp(
        test_with_returns, min_stocks, TIMESTAMP
    )
    
    if train_with_returns.empty or test_with_returns.empty:
        logger.warning(f"Window {window_id}: Insufficient stocks per timestamp")
        return None
    
    # Features are already computed on wide_df - just use them directly
    train_features = train_with_returns
    test_features = test_with_returns
    del train_with_returns, test_with_returns
    gc.collect()
    
    # ==========================================================================
    # CLUSTER FEATURES (computed per-window to prevent leakage)
    # Clusters are fit on TRAINING DATA ONLY, then applied to test
    # ==========================================================================
    from features.cluster_fast import compute_clusters_fast, add_cluster_features_fast
    
    # Compute clusters using only training period data
    # Use train_end_ts as the cutoff to ensure no future data is used
    train_only_df = wide_df[wide_df[TIMESTAMP] <= train_end_ts].copy()
    
    # Get cluster assignments (14 clusters for ~10% of tickers per cluster)
    cluster_map = compute_clusters_fast(
        train_only_df, 
        n_clusters=14,
        lookback_days=500,  # Use up to 2 years of training data
        min_obs=100,
    )
    logger.debug(f"Window {window_id}: Computed {len(set(cluster_map.values()))} clusters for {len(cluster_map)} tickers")
    
    # Apply cluster assignments to train and test
    train_features = add_cluster_features_fast(train_features, cluster_map)
    test_features = add_cluster_features_fast(test_features, cluster_map)
    
    del train_only_df
    gc.collect()
    
    # NOTE: Cross-sectional features disabled - they cause suspicious Sharpe inflation
    # from features.cross_sectional import add_cross_sectional_features
    # train_features = add_cross_sectional_features(train_features)
    # test_features = add_cross_sectional_features(test_features)
    
    # Preprocess (handle NaN, infinities)
    train_features = preprocess_data(train_features, add_missing_flags=False)
    test_features = preprocess_data(test_features, add_missing_flags=False)
    
    # Get feature columns (intersection of train and test to ensure consistency)
    train_feature_cols = set(get_feature_columns_for_ranking(train_features))
    test_feature_cols = set(get_feature_columns_for_ranking(test_features))
    feature_cols = sorted(train_feature_cols & test_feature_cols)
    
    # Log dropped features
    dropped_train = train_feature_cols - test_feature_cols
    dropped_test = test_feature_cols - train_feature_cols
    if dropped_train:
        print(f"  Warning: Dropped {len(dropped_train)} features present in train but not test: {list(dropped_train)[:5]}...")
    if dropped_test:
        print(f"  Warning: Dropped {len(dropped_test)} features present in test but not train: {list(dropped_test)[:5]}...")
    
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
    
    # Split training data for validation if early stopping is enabled
    X_val, y_val, groups_val = None, None, None
    if ranker_config.early_stopping_rounds is not None:
        # Use last 20% of timestamps for validation
        train_timestamps = train_scaled[TIMESTAMP].unique()
        n_val_ts = max(1, int(len(train_timestamps) * 0.2))
        val_ts_cutoff = np.sort(train_timestamps)[-n_val_ts]
        
        train_mask = train_scaled[TIMESTAMP] < val_ts_cutoff
        val_data = train_scaled[~train_mask].copy()
        train_scaled_for_fit = train_scaled[train_mask].copy()
        
        # Re-prepare data for the actual training split
        X_train, y_train, groups_train = prepare_ranking_data(
            train_scaled_for_fit,
            feature_cols=feature_cols,
            target_col=FORWARD_RETURN,
            timestamp_col=TIMESTAMP,
        )
        
        # Prepare validation data
        X_val, y_val, groups_val = prepare_ranking_data(
            val_data,
            feature_cols=feature_cols,
            target_col=FORWARD_RETURN,
            timestamp_col=TIMESTAMP,
        )
    
    # Train ranking model
    logger.info(f"Window {window_id}: Training ranker with {len(X_train):,} samples, {len(feature_cols)} features")
    ranker = LightGBMRankerWrapper(ranker_config)
    with log_timing(f"Window {window_id}: model training", logger):
        ranker.fit(X_train, y_train, groups_train, X_val, y_val, groups_val)
    
    # Predict
    logger.debug(f"Window {window_id}: Generating predictions for {len(X_test):,} test samples")
    predictions = ranker.predict(X_test)
    
    # Build predictions DataFrame
    predictions_df = pd.DataFrame({
        TIMESTAMP: test_sorted[TIMESTAMP].values,
        TICKER: test_sorted[TICKER].values,
        "predicted_score": predictions,
        "actual_return": y_test.values,
    })
    
    # Extract feature importances
    feature_importances = None
    if ranker.model is not None:
        try:
            importances = ranker.model.feature_importances_
            feature_importances = dict(zip(feature_cols, importances))
        except Exception as e:
            logger.warning(f"Window {window_id}: Could not extract feature importances: {e}")
    
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
        feature_importances=feature_importances,
        cluster_map=cluster_map,
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
    ranker_n_estimators: int = RANKER_N_ESTIMATORS,
    ranker_learning_rate: float = RANKER_LEARNING_RATE,
    ranker_num_leaves: int = RANKER_NUM_LEAVES,
    ranker_max_depth: int = RANKER_MAX_DEPTH,
    ranker_min_child_samples: int = RANKER_MIN_CHILD_SAMPLES,
    ranker_reg_alpha: float = RANKER_REG_ALPHA,
    ranker_reg_lambda: float = RANKER_REG_LAMBDA,
    ranker_subsample: float = RANKER_SUBSAMPLE,
    ranker_colsample_bytree: float = RANKER_COLSAMPLE_BYTREE,
    save_results: bool = True,
    experimental_features: Optional[List[str]] = None,
    ranker_config: Optional[dict] = None,
    top_n: Optional[int] = None,
    bottom_n: Optional[int] = None,
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
        ranker_n_estimators: Number of boosting iterations.
        ranker_learning_rate: Learning rate for the ranker.
        ranker_num_leaves: Maximum number of leaves per tree.
        save_results: If True, save results to output directory.
        experimental_features: List of experimental feature sets to apply.
        ranker_config: Optional dict to override ranker parameters.
        top_n: Alias for portfolio_top_n (for convenience).
        bottom_n: Alias for portfolio_bottom_n (for convenience).
    
    Returns:
        RankingPipelineResult with metrics, backtest, and predictions.
    """
    # Generate unique run ID for traceability
    run_id = _generate_run_id()
    pipeline_start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info(f"PIPELINE RUN: {run_id}")
    logger.info(f"Started at: {pipeline_start_time.isoformat()}")
    logger.info("=" * 60)
    
    # Handle convenience aliases
    if top_n is not None:
        portfolio_top_n = top_n
    if bottom_n is not None:
        portfolio_bottom_n = bottom_n
    
    # Handle ranker_config dict override
    if ranker_config is not None:
        ranker_n_estimators = ranker_config.get("n_estimators", ranker_n_estimators)
        ranker_learning_rate = ranker_config.get("learning_rate", ranker_learning_rate)
        ranker_num_leaves = ranker_config.get("num_leaves", ranker_num_leaves)
        ranker_max_depth = ranker_config.get("max_depth", ranker_max_depth)
        ranker_min_child_samples = ranker_config.get("min_child_samples", ranker_min_child_samples)
        ranker_subsample = ranker_config.get("subsample", ranker_subsample)
        ranker_colsample_bytree = ranker_config.get("colsample_bytree", ranker_colsample_bytree)
    
    # Log configuration for traceability
    run_config = {
        "run_id": run_id,
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
        "slippage_bps": slippage_bps,
        "ranker_n_estimators": ranker_n_estimators,
        "ranker_learning_rate": ranker_learning_rate,
    }
    log_config(run_config, logger)
    
    # Convert to wide format with features (uses cache for massive speedup)
    # First run: ~85s, subsequent runs: ~5s
    logger.info("Phase 1: Loading data with features (using cache if available)...")
    print("Loading data with features (using cache if available)...")
    with log_timing("data loading with features", logger):
        wide_df = prepare_wide_data_with_features(
            long_df, 
            use_cache=(long_df is None),
            experimental_features=experimental_features,
        )
    log_dataframe_info(wide_df, "Wide format data with features", logger)
    print(f"Wide format with features: {len(wide_df):,} rows, {len(wide_df.columns)} columns")
    
    if wide_df.empty:
        logger.error("No data after converting to wide format")
        raise ValueError("No data after converting to wide format")
    
    # Get max timestamp for window calculation
    data_max_ts = wide_df[TIMESTAMP].max()
    logger.debug(f"Data max timestamp: {data_max_ts} ({datetime.fromtimestamp(data_max_ts/1000).isoformat()})")
    
    # Calculate window timestamps
    window_timestamps = calculate_window_timestamps(
        data_max_ts,
        num_windows,
        window_movement_years,
        forward_return_days,
        test_period_years,
    )
    logger.info(f"Calculated {len(window_timestamps)} rolling windows")
    
    # Ranker configuration (use passed parameters)
    ranker_config = RankerConfig(
        n_estimators=ranker_n_estimators,
        learning_rate=ranker_learning_rate,
        num_leaves=ranker_num_leaves,
        max_depth=ranker_max_depth,
        min_child_samples=ranker_min_child_samples,
        reg_alpha=ranker_reg_alpha,
        reg_lambda=ranker_reg_lambda,
        subsample=ranker_subsample,
        colsample_bytree=ranker_colsample_bytree,
        device=RANKER_DEVICE,
        early_stopping_rounds=RANKER_EARLY_STOPPING_ROUNDS,
    )
    
    # Run each window
    logger.info(f"Phase 2: Running {num_windows} ranking windows...")
    print(f"\nRunning {num_windows} ranking windows...")
    
    all_predictions: List[pd.DataFrame] = []
    window_summaries: List[dict] = []
    windows_completed = 0
    windows_skipped = 0
    last_cluster_map: Optional[dict] = None  # Store cluster map from most recent window
    
    for window_id, (train_end_ts, test_end_ts) in enumerate(window_timestamps):
        log_window_start(window_id, num_windows)
        print(f"\n--- Window {window_id + 1}/{num_windows} ---")
        
        try:
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
            )
        except Exception as e:
            logger.error(f"Window {window_id}: Failed with error: {e}")
            logger.debug(f"Window {window_id}: Traceback:\n{traceback.format_exc()}")
            windows_skipped += 1
            continue
        
        if result is None:
            logger.warning(f"Window {window_id}: Skipped due to insufficient data")
            print(f"  Skipping: insufficient data")
            windows_skipped += 1
            continue
        
        windows_completed += 1
        
        # Store cluster map from this window (last one will be used for reporting)
        if result.cluster_map:
            last_cluster_map = result.cluster_map
        
        # Add window_id to predictions
        result.predictions_df["window_id"] = window_id
        all_predictions.append(result.predictions_df)
        
        # Compute per-window IC
        window_ic = compute_cross_sectional_ic_series(
            result.predictions_df,
            timestamp_col=TIMESTAMP,
            predicted_col="predicted_score",
            actual_col="actual_return",
            min_stocks=min_stocks,
        ).mean()
        
        # Create summary
        summary = {
            "window_id": window_id,
            "train_timestamps": result.train_timestamps,
            "test_timestamps": result.test_timestamps,
            "train_stocks_per_ts": f"{result.train_stocks_per_ts:.1f}",
            "test_stocks_per_ts": f"{result.test_stocks_per_ts:.1f}",
            "feature_importances": result.feature_importances,
            "ic": float(window_ic) if not pd.isna(window_ic) else 0.0,
        }
        window_summaries.append(summary)
        
        # Log window result with actual IC
        log_window_result(
            window_id, 
            result.train_timestamps, 
            result.test_timestamps, 
            ic=window_ic if not pd.isna(window_ic) else 0.0,
        )
        print(f"  Train: {result.train_timestamps} timestamps, {result.train_stocks_per_ts:.1f} stocks/ts")
        print(f"  Test:  {result.test_timestamps} timestamps, {result.test_stocks_per_ts:.1f} stocks/ts")
        print(f"  IC:    {window_ic:.4f}" if not pd.isna(window_ic) else "  IC:    N/A")
    
    logger.info(f"Windows completed: {windows_completed}/{num_windows}, skipped: {windows_skipped}")
    
    if not all_predictions:
        logger.error("No windows completed successfully")
        raise ValueError("No windows completed successfully")
    
    # Combine all predictions
    logger.info("Phase 4: Combining results and computing metrics...")
    print("\nCombining results...")
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    logger.info(f"Combined predictions: {len(combined_predictions):,} rows")
    
    # Aggregate feature importances across windows
    aggregated_importances = {}
    importance_count = 0
    for ws in window_summaries:
        if 'feature_importances' in ws and ws['feature_importances']:
            importance_count += 1
            for feature, importance in ws['feature_importances'].items():
                if feature not in aggregated_importances:
                    aggregated_importances[feature] = []
                aggregated_importances[feature].append(importance)
    
    # Average importances across windows
    if aggregated_importances:
        feature_importances = {
            f: np.mean(vals) for f, vals in aggregated_importances.items()
        }
        logger.debug(f"Aggregated feature importances from {importance_count} windows")
    else:
        feature_importances = None
    
    # Compute ranking metrics
    logger.info("Computing ranking metrics...")
    print("Computing ranking metrics...")
    metrics = RankingMetrics.from_predictions(
        combined_predictions,
        timestamp_col=TIMESTAMP,
        predicted_col="predicted_score",
        actual_col="actual_return",
        min_stocks=MIN_STOCKS_FOR_IC,
        forward_return_days=forward_return_days,
    )
    logger.info(f"Ranking metrics: IC={metrics.mean_ic:.4f}, ICIR={metrics.icir:.4f}")
    
    # Compute quintile returns
    quintile_returns = compute_quintile_portfolio_returns(
        combined_predictions,
        timestamp_col=TIMESTAMP,
        score_col="predicted_score",
        return_col="actual_return",
    )
    
    # Run portfolio backtest
    logger.info("Phase 5: Running portfolio backtest...")
    print("Running portfolio backtest...")
    portfolio_config = PortfolioConfig(
        top_n=portfolio_top_n,
        bottom_n=portfolio_bottom_n,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    
    # Prepare price data for accurate drawdown calculation with long holding periods
    # We need daily price data with columns: timestamp, ticker, Close
    # Convert wide_df (which has Close column) to a format compatible with compute_daily_portfolio_returns
    price_data_for_backtest = wide_df[[TIMESTAMP, TICKER, 'Close']].copy()
    print(f"Using {len(price_data_for_backtest):,} price records for continuous drawdown calculation")
    
    backtest = run_portfolio_backtest(
        combined_predictions,
        portfolio_config,
        timestamp_col=TIMESTAMP,
        score_col="predicted_score",
        return_col="actual_return",
        return_horizon_days=forward_return_days,
        price_data=price_data_for_backtest,
    )
    
    # Run random baseline for comparison
    print("Running random baseline (100 trials)...")
    random_baseline = run_random_baseline(
        combined_predictions,
        portfolio_config,
        timestamp_col=TIMESTAMP,
        ticker_col=TICKER,
        return_col="actual_return",
        return_horizon_days=forward_return_days,
        n_trials=100,
        model_sharpe=backtest.sharpe_ratio,
    )
    
    # Compute comprehensive metrics
    # IMPORTANT: Use true_daily_returns (continuous) for risk metrics when available,
    # not period returns (only 20 points for annual rebalancing)
    print("Computing comprehensive metrics...")
    comprehensive_metrics = None
    decile_returns = None
    
    # Choose the right returns series for risk metrics
    # - true_daily_returns: ~5000 daily observations, proper for Sortino/Calmar/Omega
    # - daily_returns: period returns (20 observations for annual), not enough for risk metrics
    returns_for_risk_metrics = backtest.true_daily_returns if backtest.true_daily_returns is not None and len(backtest.true_daily_returns) > 50 else backtest.daily_returns
    
    # Determine periods_per_year based on which returns we're using
    if backtest.true_daily_returns is not None and len(backtest.true_daily_returns) > 50:
        risk_periods_per_year = 252  # Daily returns
    else:
        risk_periods_per_year = max(1, 252 // forward_return_days)  # Period returns
    
    try:
        comprehensive_metrics = ComprehensiveMetrics.from_predictions_and_returns(
            combined_predictions,
            returns_for_risk_metrics,
            timestamp_col=TIMESTAMP,
            predicted_col="predicted_score",
            actual_col="actual_return",
            min_stocks=MIN_STOCKS_FOR_IC,
            forward_return_days=forward_return_days,  # Always use actual horizon for IC annualization
        )
        decile_returns = comprehensive_metrics.decile_returns
    except Exception as e:
        print(f"  Warning: Could not compute comprehensive metrics: {e}")
        logger.warning(f"Could not compute comprehensive metrics: {e}")
        # Fall back to just decile returns
        try:
            decile_returns = compute_decile_returns(
                combined_predictions["predicted_score"],
                combined_predictions["actual_return"]
            )
        except Exception as e2:
            logger.warning(f"Could not compute decile returns fallback: {e2}")
    
    # Build config dict (includes run_id for traceability)
    config = {
        "run_id": run_id,
        "started_at": pipeline_start_time.isoformat(),
        "num_windows": num_windows,
        "windows_completed": windows_completed,
        "windows_skipped": windows_skipped,
        "window_movement_years": window_movement_years,
        "test_period_years": test_period_years,
        "forward_return_days": forward_return_days,
        "return_type": return_type,
        "winsorize_limits": winsorize_limits,
        "min_stocks": min_stocks,
        "portfolio_top_n": portfolio_top_n,
        "portfolio_bottom_n": portfolio_bottom_n,
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "ranker_n_estimators": ranker_config.n_estimators,
        "ranker_learning_rate": ranker_config.learning_rate,
        "ranker_num_leaves": ranker_config.num_leaves,
        "ranker_max_depth": ranker_config.max_depth,
        "ranker_min_child_samples": ranker_config.min_child_samples,
        "ranker_subsample": ranker_config.subsample,
        "ranker_colsample_bytree": ranker_config.colsample_bytree,
    }
    
    # Calculate pipeline duration
    pipeline_end_time = datetime.now()
    pipeline_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
    config["completed_at"] = pipeline_end_time.isoformat()
    config["duration_seconds"] = pipeline_duration
    
    # Build cluster info from last window (most recent cluster assignments)
    cluster_info = None
    if last_cluster_map:
        print("Building cluster analysis report...")
        try:
            from features.ticker_clusters import (
                get_cluster_membership_report,
                compute_ticker_characteristics,
                get_cluster_performance_by_predictions,
            )
            
            # Get ticker characteristics for the report
            stats_df = compute_ticker_characteristics(wide_df, lookback_days=500)
            
            # Build full report
            cluster_report = get_cluster_membership_report(last_cluster_map, stats_df)
            
            # Compute performance by cluster
            cluster_perf = get_cluster_performance_by_predictions(
                combined_predictions,
                last_cluster_map,
                ticker_col=TICKER,
                actual_col="actual_return",
                predicted_col="predicted_score",
            )
            
            cluster_info = ClusterInfo(
                cluster_map=last_cluster_map,
                cluster_report=cluster_report,
                cluster_performance=cluster_perf,
            )
            logger.info(f"Cluster report: {len(last_cluster_map)} tickers in {len(cluster_report['clusters'])} clusters")
        except Exception as e:
            logger.warning(f"Could not build cluster report: {e}")
    
    # Log final summary
    log_pipeline_summary(
        mean_ic=metrics.mean_ic,
        icir=metrics.icir,
        sharpe=backtest.sharpe_ratio,
        n_windows=windows_completed,
    )
    logger.info(f"Pipeline completed in {pipeline_duration:.1f}s")
    
    result = RankingPipelineResult(
        metrics=metrics,
        backtest=backtest,
        quintile_returns=quintile_returns,
        predictions_df=combined_predictions,
        window_summaries=window_summaries,
        num_windows=len(all_predictions),
        config=config,
        comprehensive_metrics=comprehensive_metrics,
        feature_importances=feature_importances,
        decile_returns=decile_returns,
        random_baseline=random_baseline,
        cluster_info=cluster_info,
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
    # Use run_id from config if available, otherwise generate new
    run_id = result.config.get("run_id")
    output_dir = get_output_dir(run_id)
    
    logger.info(f"Saving results to: {output_dir}")
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(_to_json_serializable(result.config), f, indent=2)
    
    # Save metrics (basic)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(_to_json_serializable(result.metrics.to_dict()), f, indent=2)
    
    # Save comprehensive metrics if available
    if result.comprehensive_metrics is not None:
        with open(output_dir / "comprehensive_metrics.json", "w") as f:
            json.dump(_to_json_serializable(result.comprehensive_metrics.to_dict()), f, indent=2)
    
    # Save feature importances if available
    if result.feature_importances:
        # Sort by importance descending
        sorted_importances = dict(
            sorted(result.feature_importances.items(), key=lambda x: x[1], reverse=True)
        )
        with open(output_dir / "feature_importances.json", "w") as f:
            json.dump(_to_json_serializable(sorted_importances), f, indent=2)
    
    # Save predictions
    result.predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Save quintile returns
    result.quintile_returns.to_csv(output_dir / "quintile_returns.csv")
    
    # Save backtest results
    result.backtest.daily_returns.to_csv(output_dir / "daily_returns.csv")
    if result.backtest.pre_fee_daily_returns is not None:
        result.backtest.pre_fee_daily_returns.to_csv(output_dir / "daily_returns_pre_fee.csv")
    if result.backtest.turnover_series is not None:
        result.backtest.turnover_series.to_csv(output_dir / "turnover.csv")
    if result.backtest.true_daily_returns is not None:
        result.backtest.true_daily_returns.to_csv(output_dir / "true_daily_returns.csv")
        print(f"Saved {len(result.backtest.true_daily_returns)} continuous daily returns for accurate drawdown")
    
    # Save detailed backtest metrics (implementation-focused)
    backtest_metrics = {
        "returns": {
            "total_return_post_fee": result.backtest.total_return,
            "total_return_pre_fee": result.backtest.pre_fee_total_return,
            "annualized_return_post_fee": result.backtest.annualized_return_post_fee,
            "annualized_return_pre_fee": result.backtest.annualized_return_pre_fee,
            "annualized_volatility": result.backtest.annualized_volatility,
        },
        "risk": {
            "sharpe_ratio_post_fee": result.backtest.sharpe_ratio,
            "sharpe_ratio_pre_fee": result.backtest.pre_fee_sharpe_ratio,
            "calmar_ratio": result.backtest.calmar_ratio,
            "max_drawdown": result.backtest.max_drawdown,
        },
        "implementation": {
            "avg_turnover_per_rebalance": result.backtest.avg_turnover,
            "avg_cost_per_rebalance": result.backtest.avg_cost_per_rebalance,
            "total_cost_drag": result.backtest.total_cost_drag,
            "return_per_unit_turnover": result.backtest.return_per_unit_turnover,
            "num_rebalances": result.backtest.num_rebalances,
            "avg_holding_period_years": result.backtest.avg_holding_period_years,
        },
        "portfolio": {
            "long_positions": result.config["portfolio_top_n"],
            "short_positions": result.config["portfolio_bottom_n"],
            "transaction_cost_bps": result.config["transaction_cost_bps"],
            "slippage_bps": result.config.get("slippage_bps", 0),
        },
    }
    
    # Add annual statistics if available
    if result.backtest.annual_stats is not None:
        backtest_metrics["annual_statistics"] = result.backtest.annual_stats.to_dict()
    
    with open(output_dir / "backtest_metrics.json", "w") as f:
        json.dump(_to_json_serializable(backtest_metrics), f, indent=2)
    
    # Save random baseline results if available
    if result.random_baseline is not None:
        with open(output_dir / "random_baseline.json", "w") as f:
            json.dump(_to_json_serializable(result.random_baseline.to_dict()), f, indent=2)
    
    # Save window summaries (without feature importances to keep file small)
    window_summaries_clean = [
        {k: v for k, v in ws.items() if k != 'feature_importances'}
        for ws in result.window_summaries
    ]
    with open(output_dir / "window_summaries.json", "w") as f:
        json.dump(_to_json_serializable(window_summaries_clean), f, indent=2)
    
    # Save cluster information (NZX-focused analysis)
    if result.cluster_info is not None:
        cluster_dir = output_dir / "clusters"
        cluster_dir.mkdir(exist_ok=True)
        
        # Save cluster map
        with open(cluster_dir / "cluster_map.json", "w") as f:
            json.dump(_to_json_serializable(result.cluster_info.cluster_map), f, indent=2)
        
        # Save cluster report (membership and characteristics)
        with open(cluster_dir / "cluster_report.json", "w") as f:
            json.dump(_to_json_serializable(result.cluster_info.cluster_report), f, indent=2)
        
        # Save cluster performance
        if result.cluster_info.cluster_performance is not None:
            result.cluster_info.cluster_performance.to_csv(
                cluster_dir / "cluster_performance.csv", index=False
            )
        
        # Generate cluster text report
        from features.ticker_clusters import format_cluster_report_text
        cluster_report_text = format_cluster_report_text(result.cluster_info.cluster_report)
        with open(cluster_dir / "cluster_membership.txt", "w") as f:
            f.write(cluster_report_text)
        
        print(f"Saved cluster analysis to {cluster_dir}")
    
    # Generate visualizations using the extended generator
    try:
        from evaluation.visualization import generate_all_figures_extended
        
        print("Generating comprehensive visualizations...")
        
        # Prepare metrics dict for visualization
        metrics_dict = result.metrics.to_dict()
        if result.comprehensive_metrics:
            metrics_dict.update(result.comprehensive_metrics.to_dict())
        
        # Add backtest metrics to avoid recalculation in visualization
        # This ensures figures show the same metrics as console output
        metrics_dict.update({
            'sharpe_ratio': result.backtest.sharpe_ratio,
            'pre_fee_sharpe_ratio': result.backtest.pre_fee_sharpe_ratio,
            'annualized_return': result.backtest.annualized_return_post_fee,
            'annualized_volatility': result.backtest.annualized_volatility,
            'max_drawdown': result.backtest.max_drawdown,
            'calmar_ratio': result.backtest.calmar_ratio,
            'total_return': result.backtest.total_return,
        })
        
        # Use true_daily_returns (continuous) for visualization when available
        # This ensures drawdown and other plots reflect actual daily movements
        # rather than only rebalance-point movements
        returns_for_viz = result.backtest.true_daily_returns if (
            result.backtest.true_daily_returns is not None 
            and len(result.backtest.true_daily_returns) > 10
        ) else result.backtest.daily_returns
        
        if result.backtest.true_daily_returns is not None and len(result.backtest.true_daily_returns) > 10:
            print(f"  Using {len(returns_for_viz)} continuous daily returns for accurate visualization")
        else:
            print(f"  Using {len(returns_for_viz)} period returns for visualization")
        
        # Generate all figures including extended visualizations
        saved_figures = generate_all_figures_extended(
            predictions_df=result.predictions_df,
            ic_series=result.metrics.ic_series,
            rank_ic_series=result.metrics.rank_ic_series,
            quintile_df=result.quintile_returns,
            returns_series=returns_for_viz,
            output_dir=str(output_dir),
            feature_importances=result.feature_importances,
            metrics_dict=metrics_dict,
            decile_returns=result.decile_returns,
            turnover_series=result.backtest.turnover_series,
            pre_fee_returns_series=result.backtest.pre_fee_daily_returns,
            timestamp_col=TIMESTAMP,
            predicted_col="predicted_score",
            actual_col="actual_return",
        )
        
        # Generate cluster figures if cluster info available
        if result.cluster_info is not None:
            from evaluation.visualization import generate_cluster_figures
            cluster_figures = generate_cluster_figures(
                cluster_report=result.cluster_info.cluster_report,
                cluster_performance=result.cluster_info.cluster_performance,
                output_dir=str(output_dir),
            )
            saved_figures.update(cluster_figures)
        
        # Save figure manifest
        with open(output_dir / "figures_manifest.json", "w") as f:
            json.dump(_to_json_serializable(saved_figures), f, indent=2)
        
        import matplotlib.pyplot as plt
        plt.close('all')
        
    except ImportError as e:
        logger.warning(f"matplotlib not available, skipping visualizations: {e}")
        print("Warning: matplotlib not available, skipping visualizations")
    except Exception as e:
        print(f"Warning: Error generating visualizations: {e}")
        # Fall back to basic visualizations
        try:
            from evaluation.visualization import (
                plot_quintile_returns,
                plot_ic_series,
                plot_cumulative_returns,
                create_ranking_dashboard,
            )
            
            # Use true_daily_returns when available for accurate visualization
            returns_for_viz = result.backtest.true_daily_returns if (
                result.backtest.true_daily_returns is not None 
                and len(result.backtest.true_daily_returns) > 10
            ) else result.backtest.daily_returns
            
            # Quintile chart
            plot_quintile_returns(
                result.quintile_returns,
                save_path=str(output_dir / "figures" / "quintile_returns.png"),
            )
            
            # IC series
            plot_ic_series(
                result.metrics.ic_series,
                save_path=str(output_dir / "figures" / "ic_series.png"),
            )
            
            # Cumulative returns
            plot_cumulative_returns(
                returns_for_viz,
                save_path=str(output_dir / "figures" / "cumulative_returns.png"),
            )
            
            # Dashboard
            create_ranking_dashboard(
                result.metrics.ic_series,
                result.quintile_returns,
                returns_for_viz,
                save_path=str(output_dir / "figures" / "dashboard.png"),
            )
            
            import matplotlib.pyplot as plt
            plt.close('all')
            
        except Exception as e2:
            logger.warning(f"Could not generate fallback visualizations: {e2}")
    
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
    
    # Print random baseline comparison if available
    if result.random_baseline is not None:
        print("\n--- Random Baseline Comparison ---")
        print(result.random_baseline.summary())
        
        # Compute and display improvement over random
        model_sharpe = result.backtest.sharpe_ratio
        random_sharpe = result.random_baseline.mean_sharpe_post_fee
        if not np.isnan(model_sharpe) and not np.isnan(random_sharpe):
            improvement = model_sharpe - random_sharpe
            std_dev = result.random_baseline.std_sharpe_post_fee
            if std_dev > 0:
                z_score = improvement / std_dev
                print(f"\nModel vs Random:")
                print(f"  Sharpe improvement:      {improvement:+.2f}")
                print(f"  Z-score vs random:       {z_score:.2f}")
                if z_score > 2:
                    print(f"  Significance:            *** Highly significant (z > 2)")
                elif z_score > 1.5:
                    print(f"  Significance:            ** Moderately significant (z > 1.5)")
                elif z_score > 1:
                    print(f"  Significance:            * Marginally significant (z > 1)")
                else:
                    print(f"  Significance:            Not significantly better than random")
    
    # Print comprehensive metrics if available
    if result.comprehensive_metrics is not None:
        print("\n--- Extended Risk Metrics ---")
        cm = result.comprehensive_metrics
        print(f"Sortino Ratio:     {cm.sortino_ratio:.2f}" if not pd.isna(cm.sortino_ratio) else "Sortino Ratio:     N/A")
        print(f"Calmar Ratio:      {cm.calmar_ratio:.2f}" if not pd.isna(cm.calmar_ratio) else "Calmar Ratio:      N/A")
        print(f"Omega Ratio:       {cm.omega_ratio:.2f}" if not pd.isna(cm.omega_ratio) else "Omega Ratio:       N/A")
        print(f"Profit Factor:     {cm.win_loss_metrics.profit_factor:.2f}" if not pd.isna(cm.win_loss_metrics.profit_factor) else "Profit Factor:     N/A")
        print(f"Win Rate:          {cm.win_loss_metrics.win_rate:.2%}" if not pd.isna(cm.win_loss_metrics.win_rate) else "Win Rate:          N/A")
        print(f"Expectancy:        {cm.win_loss_metrics.expectancy:.4f}" if not pd.isna(cm.win_loss_metrics.expectancy) else "Expectancy:        N/A")
        
        print("\n--- IC Stability ---")
        print(f"IC Positive Rate:  {cm.ic_positive_rate:.2%}" if not pd.isna(cm.ic_positive_rate) else "IC Positive Rate:  N/A")
        print(f"IC Stability Score:{cm.ic_stability_score:.4f}" if not pd.isna(cm.ic_stability_score) else "IC Stability Score:N/A")
        
        print("\n--- Quintile Analysis ---")
        print(f"Is Monotonic:      {cm.quintile_monotonicity.get('is_monotonic', 'N/A')}")
        print(f"Decile Spread:     {cm.decile_spread:.4f}" if not pd.isna(cm.decile_spread) else "Decile Spread:     N/A")
        
        print("\n--- Statistical Significance ---")
        ic_sig = "Significant" if cm.ic_ttest_pvalue < 0.05 else "Not Significant"
        print(f"IC p-value:        {cm.ic_ttest_pvalue:.4f} ({ic_sig})" if not pd.isna(cm.ic_ttest_pvalue) else "IC p-value:        N/A")
    
    # Print cluster analysis if available
    if result.cluster_info is not None:
        print("\n--- NZX Cluster Analysis ---")
        report = result.cluster_info.cluster_report
        clusters = report.get('clusters', {})
        print(f"Total clusters:    {len(clusters)}")
        print(f"Total tickers:     {len(result.cluster_info.cluster_map)}")
        
        # Show brief cluster summary
        print("\nCluster Overview:")
        for cluster_id in sorted(clusters.keys()):
            c = clusters[cluster_id]
            chars = c.get('characteristics', {})
            vol_pct = chars.get('volatility', 0) * 100
            ret_pct = chars.get('mean_return', 0) * 100
            print(f"  C{cluster_id}: {c['label']:<20} ({c['n_stocks']:2} stocks, vol={vol_pct:5.1f}%, ret={ret_pct:+6.1f}%)")
        
        # Show model performance by cluster if available
        if result.cluster_info.cluster_performance is not None and len(result.cluster_info.cluster_performance) > 0:
            perf = result.cluster_info.cluster_performance
            print("\nModel Performance by Cluster:")
            for _, row in perf.iterrows():
                ic = row['pearson_ic']
                hr = row['hit_rate']
                n = row['n_predictions']
                print(f"  C{int(row['cluster'])}: IC={ic:+.3f}, Hit Rate={hr:.0%} (n={n})")
    
    # Print top features if available
    if result.feature_importances:
        print("\n--- Top 10 Features ---")
        sorted_features = sorted(result.feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, importance) in enumerate(sorted_features, 1):
            print(f"  {i:2}. {feature:<30} {importance:.4f}")
    
    # Print NZX-specific annual implementation notes
    config = result.config
    forward_days = config.get('forward_return_days', 365)
    if forward_days >= 200:
        print("\n--- Annual Strategy Implementation Notes ---")
        print(f"Rebalance frequency:   Annual ({forward_days}-day horizon)")
        print(f"Positions:             Long {config.get('portfolio_top_n', 10)}, Short {config.get('portfolio_bottom_n', 10)}")
        print(f"Transaction cost:      {config.get('transaction_cost_bps', 10)} bps round-trip")
        
        if result.backtest.annual_stats is not None:
            stats = result.backtest.annual_stats
            print(f"\nExpected Annual Range:")
            print(f"  5th percentile:      {stats.pct_5_annual_return:+.1%}")
            print(f"  Median:              {stats.median_annual_return:+.1%}")
            print(f"  95th percentile:     {stats.pct_95_annual_return:+.1%}")
            print(f"  Win probability:     {stats.pct_positive_years:.0%}")
    
    print("\n" + "=" * 60)

# =============================================================================
# PREDICTION API (for real-world use)
# =============================================================================

@dataclass
class PredictionResult:
    """Result from prediction pipeline."""
    predictions: pd.DataFrame  # timestamp, ticker, predicted_score, rank
    prediction_date: datetime
    forward_days: int
    n_stocks: int
    feature_columns: List[str]
    model_config: dict
    training_samples: int = 0
    
    @property
    def top_picks(self) -> pd.DataFrame:
        """Get top 10 ranked stocks."""
        return self.predictions.head(10)
    
    @property
    def bottom_picks(self) -> pd.DataFrame:
        """Get bottom 10 ranked stocks."""
        return self.predictions.tail(10)
    
    def get_stock_rank(self, ticker: str) -> Optional[dict]:
        """Get ranking info for a specific stock."""
        match = self.predictions[self.predictions[TICKER] == ticker]
        if match.empty:
            return None
        row = match.iloc[0]
        return {
            "ticker": ticker,
            "rank": int(row["rank"]),
            "score": float(row["predicted_score"]),
            "percentile": 100 * (1 - row["rank"] / self.n_stocks),
        }


def train_and_save_model(
    output_path: str | Path,
    forward_days: int = FORWARD_RETURN_DAYS,
    min_stocks: int = MIN_STOCKS_PER_TIMESTAMP,
) -> "ModelBundle":
    """Train a ranking model on all available data and save it.
    
    Use this to pre-train a model that can later be loaded for quick predictions.
    
    Args:
        output_path: Path to save the model bundle (.pkl file).
        forward_days: Forward return horizon in days.
        min_stocks: Minimum stocks per timestamp for training.
    
    Returns:
        ModelBundle containing the trained model and all components.
    
    Example:
        >>> bundle = train_and_save_model("models/ranking_model.pkl")
        >>> print(f"Model saved with {bundle.n_features} features")
    """
    from core.model_persistence import ModelBundle, save_model, compute_data_fingerprint
    from core.target_builder import compute_forward_returns, FORWARD_RETURN
    
    logger.info(f"Training model for {forward_days}-day horizon...")
    
    # Load and prepare data
    wide_df = prepare_wide_data_with_features(use_cache=True)
    
    # Get timestamps
    timestamps = sorted(wide_df[TIMESTAMP].unique())
    latest_ts = timestamps[-1]
    
    # Training cutoff: need buffer for forward returns
    buffer_ts = forward_days * MS_PER_DAY
    train_cutoff = latest_ts - buffer_ts
    train_timestamps = [ts for ts in timestamps if ts <= train_cutoff]
    
    if len(train_timestamps) < 10:
        raise ValueError(f"Insufficient training timestamps: {len(train_timestamps)}")
    
    # Get training data
    train_df = wide_df[wide_df[TIMESTAMP].isin(train_timestamps)].copy()
    
    # Compute forward returns
    train_with_returns = compute_forward_returns(
        train_df,
        lookahead_days=forward_days,
        return_type="simple",
        winsorize_limits=WINSORIZE_LIMITS,
        drop_na=True,
        price_lookup_df=wide_df,
    )
    
    # Filter
    train_with_returns = filter_min_stocks_per_timestamp(
        train_with_returns, min_stocks, TIMESTAMP
    )
    
    if train_with_returns.empty:
        raise ValueError("No valid training data after filtering")
    
    # Preprocess
    train_processed = preprocess_data(train_with_returns, add_missing_flags=False)
    feature_cols = get_feature_columns_for_ranking(train_processed)
    
    # Fit scaler
    scaler = fit_scaler(train_processed[feature_cols])
    train_scaled = transform_data(train_processed, scaler)
    
    # Prepare ranking data
    X_train, y_train, groups_train = prepare_ranking_data(
        train_scaled,
        feature_cols=feature_cols,
        target_col=FORWARD_RETURN,
        timestamp_col=TIMESTAMP,
    )
    
    # Train ranker
    ranker_config = RankerConfig(
        n_estimators=RANKER_N_ESTIMATORS,
        learning_rate=RANKER_LEARNING_RATE,
        num_leaves=RANKER_NUM_LEAVES,
        max_depth=RANKER_MAX_DEPTH,
        min_child_samples=RANKER_MIN_CHILD_SAMPLES,
        subsample=RANKER_SUBSAMPLE,
        colsample_bytree=RANKER_COLSAMPLE_BYTREE,
        device=RANKER_DEVICE,
    )
    
    ranker = LightGBMRankerWrapper(ranker_config)
    ranker.fit(X_train, y_train, groups_train)
    
    # Create bundle
    bundle = ModelBundle(
        ranker=ranker,
        scaler=scaler,
        feature_columns=feature_cols,
        config={
            "forward_return_days": forward_days,
            "n_estimators": ranker_config.n_estimators,
            "learning_rate": ranker_config.learning_rate,
            "training_samples": len(X_train),
            "training_timestamps": len(train_timestamps),
        },
        metadata={
            "created_at": datetime.now().isoformat(),
            "data_fingerprint": compute_data_fingerprint(train_df),
        },
    )
    
    # Save
    save_model(bundle, output_path)
    logger.info(f"Model saved to {output_path}")
    
    return bundle


def generate_predictions(
    model_bundle: "ModelBundle" = None,
    model_path: str | Path = None,
) -> PredictionResult:
    """Generate predictions for the most recent timestamp.
    
    Either provide a loaded ModelBundle or a path to load from.
    
    Args:
        model_bundle: Pre-loaded ModelBundle (takes precedence).
        model_path: Path to saved model file.
    
    Returns:
        PredictionResult with stock rankings.
    
    Example:
        >>> from core.model_persistence import load_model
        >>> bundle = load_model("models/ranking_model.pkl")
        >>> result = generate_predictions(model_bundle=bundle)
        >>> print(result.top_picks)
    """
    from core.model_persistence import load_model
    from core.scaler import transform_data
    
    if model_bundle is None and model_path is None:
        raise ValueError("Must provide either model_bundle or model_path")
    
    if model_bundle is None:
        model_bundle = load_model(model_path)
    
    # Load and prepare current data
    wide_df = prepare_wide_data_with_features(use_cache=True)
    
    # Get latest timestamp
    latest_ts = wide_df[TIMESTAMP].max()
    latest_date = datetime.fromtimestamp(latest_ts / 1000)
    
    predict_df = wide_df[wide_df[TIMESTAMP] == latest_ts].copy()
    
    # Preprocess
    predict_processed = preprocess_data(predict_df, add_missing_flags=False)
    
    # Check features
    available_features = [f for f in model_bundle.feature_columns 
                         if f in predict_processed.columns]
    
    if len(available_features) < len(model_bundle.feature_columns) * 0.8:
        logger.warning(
            f"Only {len(available_features)}/{len(model_bundle.feature_columns)} "
            "features available. Predictions may be less accurate."
        )
    
    # Scale
    predict_scaled = transform_data(predict_processed, model_bundle.scaler)
    
    # Predict
    X_predict = predict_scaled[available_features].values
    predictions = model_bundle.ranker.predict(
        pd.DataFrame(X_predict, columns=available_features)
    )
    
    # Build result
    result_df = pd.DataFrame({
        TIMESTAMP: predict_scaled[TIMESTAMP].values,
        TICKER: predict_scaled[TICKER].values,
        "predicted_score": predictions,
        "rank": pd.Series(predictions).rank(ascending=False, method="first").astype(int).values,
    })
    
    result_df = result_df.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    
    # Add price if available
    if CLOSE in predict_scaled.columns:
        close_map = predict_scaled.set_index(TICKER)[CLOSE].to_dict()
        result_df["close_price"] = result_df[TICKER].map(close_map)
    
    forward_days = model_bundle.config.get("forward_return_days", FORWARD_RETURN_DAYS)
    
    return PredictionResult(
        predictions=result_df,
        prediction_date=latest_date,
        forward_days=forward_days,
        n_stocks=len(result_df),
        feature_columns=available_features,
        model_config=model_bundle.config,
        training_samples=model_bundle.config.get("training_samples", 0),
    )