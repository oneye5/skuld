"""Validation utilities for data integrity and input checking.

This module provides:
- Decorators for validating function inputs
- Data quality validators for DataFrames
- Pipeline-specific validation functions
"""

from functools import wraps
from typing import List, Optional, Callable, Any
import pandas as pd
import numpy as np
import warnings

from config.columns import TIMESTAMP, TICKER, CLOSE


# =============================================================================
# VALIDATION EXCEPTIONS
# =============================================================================

class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class DataQualityWarning(UserWarning):
    """Warning for data quality issues that don't prevent execution."""
    pass


# =============================================================================
# DECORATOR-BASED VALIDATION
# =============================================================================

def validate_dataframe(
    required_cols: Optional[List[str]] = None,
    min_rows: int = 0,
    check_no_duplicates: Optional[List[str]] = None,
):
    """Decorator to validate DataFrame inputs.
    
    Args:
        required_cols: List of columns that must be present.
        min_rows: Minimum number of rows required.
        check_no_duplicates: Columns to check for duplicate combinations.
    
    Usage:
        @validate_dataframe(required_cols=[TIMESTAMP, TICKER], min_rows=10)
        def my_function(df: pd.DataFrame) -> pd.DataFrame:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find DataFrame argument (first positional or 'df' keyword)
            df = None
            if args and isinstance(args[0], pd.DataFrame):
                df = args[0]
            elif 'df' in kwargs and isinstance(kwargs['df'], pd.DataFrame):
                df = kwargs['df']
            
            if df is None:
                raise ValidationError(
                    f"{func.__name__}: Expected DataFrame as first argument or 'df' keyword"
                )
            
            # Check required columns
            if required_cols:
                missing = set(required_cols) - set(df.columns)
                if missing:
                    raise ValidationError(
                        f"{func.__name__}: Missing required columns: {missing}"
                    )
            
            # Check minimum rows
            if len(df) < min_rows:
                raise ValidationError(
                    f"{func.__name__}: DataFrame has {len(df)} rows, minimum is {min_rows}"
                )
            
            # Check for duplicates
            if check_no_duplicates:
                cols_to_check = [c for c in check_no_duplicates if c in df.columns]
                if cols_to_check:
                    duplicates = df.duplicated(subset=cols_to_check, keep=False)
                    if duplicates.any():
                        n_dups = duplicates.sum()
                        raise ValidationError(
                            f"{func.__name__}: Found {n_dups} duplicate rows on columns {cols_to_check}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_no_nan(columns: Optional[List[str]] = None):
    """Decorator to ensure specified columns have no NaN values.
    
    Args:
        columns: Columns to check. If None, checks all numeric columns.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            df = args[0] if args and isinstance(args[0], pd.DataFrame) else kwargs.get('df')
            
            if df is None:
                return func(*args, **kwargs)
            
            cols_to_check = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
            cols_present = [c for c in cols_to_check if c in df.columns]
            
            for col in cols_present:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    raise ValidationError(
                        f"{func.__name__}: Column '{col}' has {nan_count} NaN values"
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# DATA QUALITY VALIDATORS
# =============================================================================

def validate_wide_data(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> List[str]:
    """Validate wide format DataFrame for common issues.
    
    Checks:
    - Required columns present (timestamp, ticker)
    - Timestamps are non-negative
    - Close prices are positive (if present)
    - No duplicate (timestamp, ticker) pairs
    - Data is sorted by timestamp (per ticker)
    
    Args:
        df: Wide format DataFrame to validate.
        raise_on_error: If True, raises ValidationError on first issue.
                        If False, returns list of all issues found.
    
    Returns:
        List of validation issues (empty if all checks pass).
    
    Raises:
        ValidationError: If raise_on_error=True and validation fails.
    """
    issues = []
    
    # Check required columns
    required = [TIMESTAMP, TICKER]
    missing = set(required) - set(df.columns)
    if missing:
        issues.append(f"Missing required columns: {missing}")
    
    if issues and raise_on_error:
        raise ValidationError(issues[0])
    
    # Check timestamp validity
    if TIMESTAMP in df.columns:
        if (df[TIMESTAMP] < 0).any():
            issues.append(f"Found {(df[TIMESTAMP] < 0).sum()} negative timestamps")
        
        if df[TIMESTAMP].isna().any():
            issues.append(f"Found {df[TIMESTAMP].isna().sum()} NaN timestamps")
    
    # Check Close prices
    if CLOSE in df.columns:
        invalid_close = (df[CLOSE] <= 0) | df[CLOSE].isna()
        if invalid_close.any():
            n_invalid = invalid_close.sum()
            # Warning, not error - some macro data may not have Close
            warnings.warn(
                f"Found {n_invalid} rows with invalid Close price (<=0 or NaN)",
                DataQualityWarning
            )
    
    # Check for duplicate (timestamp, ticker) pairs
    if TIMESTAMP in df.columns and TICKER in df.columns:
        duplicates = df.duplicated(subset=[TIMESTAMP, TICKER], keep=False)
        if duplicates.any():
            n_dups = duplicates.sum()
            sample = df[duplicates][[TIMESTAMP, TICKER]].head(3).to_dict('records')
            issues.append(f"Found {n_dups} duplicate (timestamp, ticker) pairs. Examples: {sample}")
    
    # Check timestamp ordering per ticker
    if TIMESTAMP in df.columns and TICKER in df.columns:
        for ticker in df[TICKER].unique()[:10]:  # Sample check
            ticker_df = df[df[TICKER] == ticker]
            if not ticker_df[TIMESTAMP].is_monotonic_increasing:
                issues.append(f"Timestamps not sorted for ticker '{ticker}'")
                break  # One warning is enough
    
    if issues and raise_on_error:
        raise ValidationError("\n".join(issues))
    
    return issues


def validate_no_lookahead(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP,
) -> bool:
    """Validate that test data doesn't precede training data (no lookahead bias).
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        timestamp_col: Column containing timestamps.
    
    Returns:
        True if validation passes.
    
    Raises:
        ValidationError: If test timestamps overlap with or precede train timestamps.
    """
    if timestamp_col not in train_df.columns or timestamp_col not in test_df.columns:
        raise ValidationError(f"Timestamp column '{timestamp_col}' not found")
    
    train_max = train_df[timestamp_col].max()
    test_min = test_df[timestamp_col].min()
    
    if test_min <= train_max:
        raise ValidationError(
            f"Lookahead bias detected: test_min ({test_min}) <= train_max ({train_max}). "
            f"Test data must come strictly after training data."
        )
    
    return True


def validate_feature_columns(
    df: pd.DataFrame,
    feature_cols: List[str],
    allow_missing: bool = False,
) -> List[str]:
    """Validate feature columns exist and have valid data.
    
    Args:
        df: DataFrame to check.
        feature_cols: List of expected feature columns.
        allow_missing: If True, return only present columns instead of raising.
    
    Returns:
        List of valid feature columns.
    
    Raises:
        ValidationError: If required columns are missing (when allow_missing=False).
    """
    present_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = set(feature_cols) - set(present_cols)
    
    if missing_cols and not allow_missing:
        raise ValidationError(f"Missing feature columns: {missing_cols}")
    
    # Check for columns with all NaN
    all_nan_cols = [c for c in present_cols if df[c].isna().all()]
    if all_nan_cols:
        warnings.warn(
            f"Columns with all NaN values: {all_nan_cols}",
            DataQualityWarning
        )
        present_cols = [c for c in present_cols if c not in all_nan_cols]
    
    # Check for columns with zero variance
    zero_var_cols = []
    for col in present_cols:
        if df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
            if df[col].std() == 0:
                zero_var_cols.append(col)
    
    if zero_var_cols:
        warnings.warn(
            f"Columns with zero variance: {zero_var_cols}",
            DataQualityWarning
        )
    
    return present_cols


def validate_groups_match_data(
    X: pd.DataFrame,
    groups: List[int],
) -> bool:
    """Validate that group sizes match data size (required for LGBMRanker).
    
    Args:
        X: Feature DataFrame.
        groups: List of group sizes.
    
    Returns:
        True if validation passes.
    
    Raises:
        ValidationError: If sum(groups) != len(X).
    """
    total_group_size = sum(groups)
    data_size = len(X)
    
    if total_group_size != data_size:
        raise ValidationError(
            f"Group sizes don't match data: sum(groups)={total_group_size}, len(X)={data_size}"
        )
    
    return True


# =============================================================================
# PIPELINE-SPECIFIC VALIDATORS
# =============================================================================

def validate_ranking_config(
    forward_days: int,
    top_n: int,
    bottom_n: int,
    min_stocks: int,
) -> List[str]:
    """Validate ranking pipeline configuration parameters.
    
    Args:
        forward_days: Forward return horizon.
        top_n: Long portfolio size.
        bottom_n: Short portfolio size.
        min_stocks: Minimum stocks per timestamp.
    
    Returns:
        List of warnings (empty if all good).
    
    Raises:
        ValidationError: For invalid configurations.
    """
    warnings_list = []
    
    if forward_days <= 0:
        raise ValidationError(f"forward_days must be positive, got {forward_days}")
    
    if forward_days > 365:
        warnings_list.append(
            f"forward_days={forward_days} is long; predictions may become stale"
        )
    
    if top_n <= 0:
        raise ValidationError(f"top_n must be positive, got {top_n}")
    
    if bottom_n < 0:
        raise ValidationError(f"bottom_n cannot be negative, got {bottom_n}")
    
    if min_stocks < top_n + bottom_n:
        warnings_list.append(
            f"min_stocks ({min_stocks}) < top_n + bottom_n ({top_n + bottom_n}); "
            f"some timestamps may have insufficient stocks for portfolio construction"
        )
    
    return warnings_list


def check_data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a data quality report for a DataFrame.
    
    Args:
        df: DataFrame to analyze.
    
    Returns:
        Dictionary with quality metrics.
    """
    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    
    # Missing value analysis
    missing_pct = df.isnull().mean()
    report["columns_with_missing"] = (missing_pct > 0).sum()
    report["high_missing_columns"] = missing_pct[missing_pct > 0.5].index.tolist()
    
    # Numeric column stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report["n_numeric_columns"] = len(numeric_cols)
    
    # Check for infinities
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    report["columns_with_infinities"] = inf_counts
    
    # Timestamp range (if present)
    if TIMESTAMP in df.columns:
        report["timestamp_min"] = df[TIMESTAMP].min()
        report["timestamp_max"] = df[TIMESTAMP].max()
        report["n_unique_timestamps"] = df[TIMESTAMP].nunique()
    
    # Ticker info (if present)
    if TICKER in df.columns:
        report["n_unique_tickers"] = df[TICKER].nunique()
    
    return report
