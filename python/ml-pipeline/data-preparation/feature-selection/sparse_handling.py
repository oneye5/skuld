"""Sophisticated sparse feature handling for financial/macro data.

Instead of simply dropping sparse columns, this module provides intelligent
strategies for handling sparsity:

1. **Forward-fill propagation** - Carry forward last known values (appropriate for 
   slowly-changing macro data)
2. **Feature aggregation** - Combine many sparse related features into fewer robust composites
3. **Missingness encoding** - Use missingness patterns as features themselves
4. **Adaptive thresholds** - Different sparsity tolerance for different feature types
5. **Representative selection** - From highly correlated sparse groups, keep best representative
"""

import pandas as pd
import numpy as np
from typing import Tuple
from dataclasses import dataclass, field

from config.column_names import TIMESTAMP, TICKER, TARGET, MACRO_PREFIX


@dataclass
class SparseHandlingResult:
    """Result of sparse feature handling.
    
    Attributes:
        kept_features: Features kept after processing.
        dropped_features: Features dropped entirely.
        aggregated_groups: Dict mapping aggregate feature name to source features.
        fill_rates: Dict of feature -> fill rate after forward-fill.
    """
    kept_features: list[str]
    dropped_features: list[str]
    aggregated_groups: dict[str, list[str]]
    fill_rates: dict[str, float]


@dataclass 
class SparseConfig:
    """Configuration for sparse feature handling.
    
    Attributes:
        macro_missing_threshold: Max missing ratio for macro features (more lenient).
        ticker_missing_threshold: Max missing ratio for ticker features.
        post_ffill_threshold: Max missing ratio AFTER forward-fill (stricter).
        aggregate_corr_threshold: Min correlation to aggregate features together.
        min_group_size: Minimum features to form an aggregate group.
    """
    macro_missing_threshold: float = 0.70  # More lenient for slowly-updating macro
    ticker_missing_threshold: float = 0.50  # Stricter for ticker features
    post_ffill_threshold: float = 0.30  # After ffill, should have most data
    aggregate_corr_threshold: float = 0.90  # High correlation for aggregation
    min_group_size: int = 3  # Need at least 3 features to aggregate


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature columns (excluding metadata and target)."""
    exclude_cols = {TIMESTAMP, TICKER, TARGET, "index"}
    return [col for col in df.columns if col not in exclude_cols]


def _is_macro_feature(col: str) -> bool:
    """Check if a column is a macro feature."""
    return col.startswith(MACRO_PREFIX)


def analyze_sparsity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sparsity patterns across features.
    
    Returns DataFrame with per-feature statistics useful for deciding
    handling strategy.
    """
    feature_cols = _get_feature_columns(df)
    
    stats = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        col_data = df[col]
        missing_ratio = col_data.isna().mean()
        
        # Count unique values (excluding NaN)
        n_unique = col_data.nunique(dropna=True)
        
        # Check if it's constant when present
        is_constant = n_unique <= 1
        
        # Estimate fill rate after forward-fill (approximate)
        # Group by ticker, check how many rows could be filled
        if TICKER in df.columns and missing_ratio > 0 and missing_ratio < 1:
            filled = df.groupby(TICKER)[col].ffill()
            post_ffill_missing = filled.isna().mean()
        else:
            post_ffill_missing = missing_ratio
        
        stats.append({
            'feature': col,
            'missing_ratio': missing_ratio,
            'post_ffill_missing': post_ffill_missing,
            'n_unique': n_unique,
            'is_constant': is_constant,
            'is_macro': _is_macro_feature(col),
            'dtype': str(col_data.dtype),
        })
    
    return pd.DataFrame(stats)


def forward_fill_sparse(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply forward-fill to sparse features within each ticker.
    
    For macro data that updates infrequently, forward-fill propagates
    the last known value forward in time. This is appropriate when
    the underlying value persists until updated.
    
    Args:
        df: DataFrame with sparse features.
        features: Specific features to forward-fill. If None, all numeric features.
    
    Returns:
        DataFrame with forward-filled values.
    """
    result = df.copy()
    
    if features is None:
        features = _get_feature_columns(result)
        features = [f for f in features if f in result.columns and 
                   result[f].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    if TICKER in result.columns:
        result = result.sort_values([TICKER, TIMESTAMP])
        result[features] = result.groupby(TICKER, sort=False)[features].ffill()
    else:
        result = result.sort_values(TIMESTAMP)
        result[features] = result[features].ffill()
    
    return result


def aggregate_correlated_sparse(
    df: pd.DataFrame,
    feature_group: list[str],
    agg_name: str,
    method: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate a group of correlated sparse features into one robust feature.
    
    When multiple sparse features measure similar concepts (e.g., different
    age groups of unemployment), combining them produces a more robust signal
    with better coverage.
    
    NOTE: Only use 'mean', 'median', or 'first_valid' methods. These are
    row-wise operations that don't require fitting, so they're safe from
    leakage. The 'pca_first' method is removed as it would require fitting
    on training data and transforming test data separately.
    
    Args:
        df: DataFrame with features.
        feature_group: List of feature names to aggregate.
        agg_name: Name for the aggregated feature.
        method: Aggregation method ('mean', 'median', 'first_valid').
    
    Returns:
        DataFrame with aggregated feature added (originals optionally kept).
    """
    result = df.copy()
    
    # Filter to features that exist
    valid_features = [f for f in feature_group if f in result.columns]
    if len(valid_features) < 2:
        return result
    
    subset = result[valid_features]
    
    if method == "mean":
        # Mean across features, ignoring NaN
        result[agg_name] = subset.mean(axis=1, skipna=True)
    elif method == "median":
        result[agg_name] = subset.median(axis=1, skipna=True)
    elif method == "first_valid":
        # Take first non-NaN value across features
        result[agg_name] = subset.bfill(axis=1).iloc[:, 0]
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Use 'mean', 'median', or 'first_valid'.")
    
    return result


def identify_sparse_groups(
    df: pd.DataFrame,
    prefix_pattern: str | None = None,
) -> dict[str, list[str]]:
    """
    Identify groups of sparse features that could be aggregated.
    
    Groups features by common prefix patterns or high correlation.
    
    Args:
        df: DataFrame with features.
        prefix_pattern: Regex pattern to group by (e.g., "NZ_Labor_UnemploymentRate").
    
    Returns:
        Dict mapping group name to list of feature names.
    """
    import re
    
    feature_cols = _get_feature_columns(df)
    groups = {}
    
    if prefix_pattern:
        pattern = re.compile(prefix_pattern)
        matching = [f for f in feature_cols if pattern.match(f)]
        if matching:
            groups[prefix_pattern] = matching
    else:
        # Auto-detect groups by common prefix
        # Look for patterns like "NZ_Labor_UnemploymentRate_Age15Plus_*"
        prefix_counts = {}
        for col in feature_cols:
            # Extract prefix (everything before last underscore + number)
            parts = col.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                prefix = parts[0]
                if prefix not in prefix_counts:
                    prefix_counts[prefix] = []
                prefix_counts[prefix].append(col)
        
        # Keep groups with 3+ members
        groups = {k: v for k, v in prefix_counts.items() if len(v) >= 3}
    
    return groups


def handle_sparse_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: SparseConfig | None = None,
    aggregate_groups: bool = True,
    drop_post_aggregate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, SparseHandlingResult]:
    """
    Comprehensive sparse feature handling pipeline.
    
    Steps:
    1. Forward-fill all features within ticker
    2. Identify feature groups for aggregation
    3. Create aggregate features from sparse groups
    4. Drop features still too sparse after processing
    5. Optionally drop original features after aggregation
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        config: Configuration for thresholds. Defaults to SparseConfig().
        aggregate_groups: Whether to create aggregate features.
        drop_post_aggregate: Whether to drop source features after aggregation.
    
    Returns:
        Tuple of (processed_train, processed_test, result).
    """
    if config is None:
        config = SparseConfig()
    
    # Step 1: Forward-fill both sets
    train_filled = forward_fill_sparse(train_df)
    test_filled = forward_fill_sparse(test_df)
    
    # Analyze post-fill sparsity on training data
    sparsity_stats = analyze_sparsity(train_filled)
    
    # Step 2: Identify groups for aggregation
    aggregated_groups = {}
    if aggregate_groups:
        groups = identify_sparse_groups(train_filled)
        
        for group_name, group_features in groups.items():
            # Check if group is sparse enough to warrant aggregation
            group_stats = sparsity_stats[sparsity_stats['feature'].isin(group_features)]
            avg_missing = group_stats['post_ffill_missing'].mean()
            
            if avg_missing > 0.1:  # Only aggregate if group is somewhat sparse
                agg_name = f"AGG_{group_name.replace('MACRO_', '')}"
                train_filled = aggregate_correlated_sparse(
                    train_filled, group_features, agg_name, method="mean"
                )
                test_filled = aggregate_correlated_sparse(
                    test_filled, group_features, agg_name, method="mean"
                )
                aggregated_groups[agg_name] = group_features
    
    # Step 3: Decide what to drop
    feature_cols = _get_feature_columns(train_filled)
    dropped = []
    kept = []
    fill_rates = {}
    
    for col in feature_cols:
        if col not in train_filled.columns:
            continue
        
        missing_ratio = train_filled[col].isna().mean()
        fill_rates[col] = 1 - missing_ratio
        is_macro = _is_macro_feature(col)
        
        # Check if this feature was aggregated
        was_aggregated = any(col in group for group in aggregated_groups.values())
        
        # Determine threshold
        if is_macro:
            threshold = config.macro_missing_threshold
        else:
            threshold = config.ticker_missing_threshold
        
        # Apply stricter post-ffill threshold
        threshold = min(threshold, config.post_ffill_threshold)
        
        # Decision logic
        if missing_ratio > threshold:
            dropped.append(col)
        elif was_aggregated and drop_post_aggregate:
            # Keep aggregate, drop source
            dropped.append(col)
        else:
            kept.append(col)
    
    # Step 4: Filter DataFrames
    metadata_cols = [col for col in [TIMESTAMP, TICKER] if col in train_filled.columns]
    target_cols = [TARGET] if TARGET in train_filled.columns else []
    
    # Include aggregate features in kept
    agg_features = list(aggregated_groups.keys())
    final_cols = metadata_cols + kept + agg_features + target_cols
    final_cols = [c for c in final_cols if c in train_filled.columns]
    
    train_out = train_filled[final_cols].copy()
    test_out = test_filled[[c for c in final_cols if c in test_filled.columns]].copy()
    
    result = SparseHandlingResult(
        kept_features=kept + agg_features,
        dropped_features=dropped,
        aggregated_groups=aggregated_groups,
        fill_rates=fill_rates,
    )
    
    return train_out, test_out, result


def select_representative_features(
    df: pd.DataFrame,
    feature_group: list[str],
    n_keep: int = 1,
    method: str = "least_missing",
) -> list[str]:
    """
    From a group of similar features, select the best representative(s).
    
    Useful when you have many versions of the same metric (e.g., different
    age groups) and want to keep only the most robust one.
    
    Args:
        df: DataFrame with features.
        feature_group: List of similar features.
        n_keep: Number of representatives to keep.
        method: Selection method ('least_missing', 'most_variance', 'median_corr').
    
    Returns:
        List of selected feature names.
    """
    valid_features = [f for f in feature_group if f in df.columns]
    if len(valid_features) <= n_keep:
        return valid_features
    
    if method == "least_missing":
        # Keep features with least missing data
        missing_rates = {f: df[f].isna().mean() for f in valid_features}
        sorted_features = sorted(missing_rates.keys(), key=lambda x: missing_rates[x])
        return sorted_features[:n_keep]
    
    elif method == "most_variance":
        # Keep features with most variance (after filling)
        filled = df[valid_features].ffill()
        variances = filled.var()
        sorted_features = variances.sort_values(ascending=False).index.tolist()
        return sorted_features[:n_keep]
    
    elif method == "median_corr":
        # Keep feature most correlated with others (central tendency)
        filled = df[valid_features].ffill().dropna()
        if len(filled) < 10:
            return valid_features[:n_keep]
        
        corr_matrix = filled.corr()
        median_corr = corr_matrix.median()
        sorted_features = median_corr.sort_values(ascending=False).index.tolist()
        return sorted_features[:n_keep]
    
    return valid_features[:n_keep]
