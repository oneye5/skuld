"""Generic lag, moving average, and momentum feature generator.

This module provides a configurable system for generating temporal features
(lags, moving averages, momentum) based on configurations in config/lag_ma_config.py.

Usage:
    from features.lag_ma_features import add_lag_ma_features
    
    df = add_lag_ma_features(df)  # Uses all enabled configs
"""

import re
import numpy as np
import pandas as pd
from typing import List, Optional

from config.columns import TICKER, TIMESTAMP
from config.lag_ma_config import (
    FeatureLagMAConfig,
    get_enabled_configs,
    get_ticker_configs,
    get_macro_configs,
)


def _get_matching_columns(
    df: pd.DataFrame,
    pattern: str,
    exclude_generated: bool = True
) -> List[str]:
    """Find columns matching the regex pattern.
    
    Args:
        df: DataFrame to search
        pattern: Regex pattern to match (case-insensitive)
        exclude_generated: If True, exclude columns that look like generated
            lag/MA features (contain _Lag_, _MA_, _Mom_, etc.)
    
    Returns:
        List of matching column names
    """
    regex = re.compile(pattern, re.IGNORECASE)
    matches = [col for col in df.columns if regex.search(col)]
    
    if exclude_generated:
        # Exclude columns that are already generated features
        generated_patterns = ['_Lag_', '_MA_', '_Mom_', '_Spike', '_Vol_']
        matches = [
            col for col in matches
            if not any(p in col for p in generated_patterns)
        ]
    
    return matches


def _generate_output_name(
    feature: str,
    config: FeatureLagMAConfig,
    suffix: str
) -> str:
    """Generate output column name based on config.
    
    Args:
        feature: Original feature name
        config: Configuration with output_prefix
        suffix: Suffix to add (e.g., "Lag_7", "MA_14")
    
    Returns:
        Output column name
    """
    if config.output_prefix:
        # Use prefix: "Attn_Lag_7"
        return f"{config.output_prefix}_{suffix}"
    else:
        # Use original name: "Wiki_Views_Lag_7"
        return f"{feature}_{suffix}"


def _add_ticker_features(
    df: pd.DataFrame,
    config: FeatureLagMAConfig,
) -> pd.DataFrame:
    """Add lag/MA/momentum features for ticker-level data.
    
    Features are computed per-ticker using groupby.
    """
    if TICKER not in df.columns:
        return df
    
    result = df.copy()
    matching_cols = _get_matching_columns(df, config.feature_pattern)
    
    if not matching_cols:
        return df
    
    # Process each matching column
    for feature in matching_cols:
        if feature not in df.columns:
            continue
        
        # Lags
        for lag in config.lags:
            out_name = _generate_output_name(feature, config, f"Lag_{lag}")
            result[out_name] = result.groupby(TICKER)[feature].shift(lag)
        
        # Moving averages
        for window in config.mas:
            out_name = _generate_output_name(feature, config, f"MA_{window}")
            min_periods = max(1, int(window * config.min_periods_ratio))
            result[out_name] = result.groupby(TICKER)[feature].transform(
                lambda x: x.rolling(window, min_periods=min_periods).mean()
            )
        
        # Momentum (percent change)
        for window in config.momentum:
            out_name = _generate_output_name(feature, config, f"Mom_{window}")
            result[out_name] = result.groupby(TICKER)[feature].transform(
                lambda x: x.pct_change(periods=window, fill_method=None)
            )
        
        # Diff features (current - MA, or MA - MA)
        # These capture mean reversion and trend regime signals
        for w1, w2 in config.diffs:
            if w1 == 0:
                # Current value - MA_w2 (deviation from mean)
                ma_col = _generate_output_name(feature, config, f"MA_{w2}")
                diff_name = _generate_output_name(feature, config, f"Diff_0_{w2}")
                if ma_col in result.columns:
                    result[diff_name] = result[feature] - result[ma_col]
                else:
                    # Compute MA on the fly if not already computed
                    min_periods = max(1, int(w2 * config.min_periods_ratio))
                    ma_vals = result.groupby(TICKER)[feature].transform(
                        lambda x: x.rolling(w2, min_periods=min_periods).mean()
                    )
                    result[diff_name] = result[feature] - ma_vals
            else:
                # MA_w1 - MA_w2 (trend difference between horizons)
                ma1_col = _generate_output_name(feature, config, f"MA_{w1}")
                ma2_col = _generate_output_name(feature, config, f"MA_{w2}")
                diff_name = _generate_output_name(feature, config, f"Diff_{w1}_{w2}")
                
                # Get or compute MA_w1
                if ma1_col in result.columns:
                    ma1_vals = result[ma1_col]
                else:
                    min_periods = max(1, int(w1 * config.min_periods_ratio))
                    ma1_vals = result.groupby(TICKER)[feature].transform(
                        lambda x: x.rolling(w1, min_periods=min_periods).mean()
                    )
                
                # Get or compute MA_w2
                if ma2_col in result.columns:
                    ma2_vals = result[ma2_col]
                else:
                    min_periods = max(1, int(w2 * config.min_periods_ratio))
                    ma2_vals = result.groupby(TICKER)[feature].transform(
                        lambda x: x.rolling(w2, min_periods=min_periods).mean()
                    )
                
                result[diff_name] = ma1_vals - ma2_vals
        
        # Spike indicator
        if config.include_spike and config.mas:
            longest_ma = max(config.mas)
            ma_col = _generate_output_name(feature, config, f"MA_{longest_ma}")
            spike_name = _generate_output_name(feature, config, "Spike")
            if ma_col in result.columns:
                result[spike_name] = result[feature] / result[ma_col].replace(0, np.nan)
        
        # Volatility
        if config.include_volatility:
            vol_name = _generate_output_name(
                feature, config, f"Vol_{config.volatility_window}"
            )
            result[vol_name] = result.groupby(TICKER)[feature].transform(
                lambda x: x.pct_change(fill_method=None).rolling(
                    config.volatility_window, 
                    min_periods=max(1, config.volatility_window // 2)
                ).std()
            )
    
    return result


def _add_global_features(
    df: pd.DataFrame,
    config: FeatureLagMAConfig,
) -> pd.DataFrame:
    """Add lag/MA/momentum features for global (macro) data.
    
    Features are computed at timestamp level (no grouping).
    """
    result = df.copy()
    matching_cols = _get_matching_columns(df, config.feature_pattern)
    
    if not matching_cols:
        return df
    
    # Ensure sorted by timestamp for correct lag/MA computation
    if TIMESTAMP in df.columns:
        result = result.sort_values(TIMESTAMP)
    
    for feature in matching_cols:
        if feature not in df.columns:
            continue
        
        # For global features, use simpler naming (no prefix transformation)
        # Just append the suffix to avoid name collisions
        base_name = feature.replace("MACRO_", "").replace("^", "").replace(".", "_")
        
        # Lags
        for lag in config.lags:
            out_name = f"{feature}_L{lag}"
            result[out_name] = result[feature].shift(lag)
        
        # Moving averages
        for window in config.mas:
            out_name = f"{feature}_MA{window}"
            min_periods = max(1, int(window * config.min_periods_ratio))
            result[out_name] = result[feature].rolling(
                window, min_periods=min_periods
            ).mean()
        
        # Momentum
        for window in config.momentum:
            out_name = f"{feature}_Mom{window}"
            result[out_name] = result[feature].pct_change(
                periods=window, fill_method=None
            )
        
        # Diff features (current - MA, or MA - MA)
        for w1, w2 in config.diffs:
            if w1 == 0:
                # Current value - MA_w2
                ma_col = f"{feature}_MA{w2}"
                diff_name = f"{feature}_Diff_0_{w2}"
                if ma_col in result.columns:
                    result[diff_name] = result[feature] - result[ma_col]
                else:
                    min_periods = max(1, int(w2 * config.min_periods_ratio))
                    ma_vals = result[feature].rolling(w2, min_periods=min_periods).mean()
                    result[diff_name] = result[feature] - ma_vals
            else:
                # MA_w1 - MA_w2
                ma1_col = f"{feature}_MA{w1}"
                ma2_col = f"{feature}_MA{w2}"
                diff_name = f"{feature}_Diff_{w1}_{w2}"
                
                if ma1_col in result.columns:
                    ma1_vals = result[ma1_col]
                else:
                    min_periods = max(1, int(w1 * config.min_periods_ratio))
                    ma1_vals = result[feature].rolling(w1, min_periods=min_periods).mean()
                
                if ma2_col in result.columns:
                    ma2_vals = result[ma2_col]
                else:
                    min_periods = max(1, int(w2 * config.min_periods_ratio))
                    ma2_vals = result[feature].rolling(w2, min_periods=min_periods).mean()
                
                result[diff_name] = ma1_vals - ma2_vals
        
        # Spike
        if config.include_spike and config.mas:
            longest_ma = max(config.mas)
            ma_col = f"{feature}_MA{longest_ma}"
            if ma_col in result.columns:
                result[f"{feature}_Spike"] = (
                    result[feature] / result[ma_col].replace(0, np.nan)
                )
    
    return result


def add_lag_ma_features(
    df: pd.DataFrame,
    configs: Optional[List[FeatureLagMAConfig]] = None,
) -> pd.DataFrame:
    """Add lag, MA, and momentum features based on configuration.
    
    Args:
        df: Input DataFrame with TICKER and TIMESTAMP columns
        configs: List of configurations to apply. If None, uses all enabled
            configs from lag_ma_config.py.
    
    Returns:
        DataFrame with additional lag/MA/momentum features.
    """
    if configs is None:
        configs = get_enabled_configs()
    
    result = df.copy()
    
    for config in configs:
        if not config.enabled:
            continue
        
        if config.scope == "ticker":
            result = _add_ticker_features(result, config)
        elif config.scope == "global":
            result = _add_global_features(result, config)
    
    return result


def add_ticker_lag_ma_features(
    df: pd.DataFrame,
    configs: Optional[List[FeatureLagMAConfig]] = None,
) -> pd.DataFrame:
    """Add only ticker-level lag/MA features.
    
    Args:
        df: Input DataFrame
        configs: Specific configs to use, or None for defaults
    
    Returns:
        DataFrame with ticker-level temporal features.
    """
    if configs is None:
        configs = get_ticker_configs()
    
    result = df.copy()
    for config in configs:
        if config.enabled and config.scope == "ticker":
            result = _add_ticker_features(result, config)
    
    return result


def add_macro_lag_ma_features(
    df: pd.DataFrame,
    configs: Optional[List[FeatureLagMAConfig]] = None,
) -> pd.DataFrame:
    """Add only macro/global lag/MA features.
    
    Args:
        df: Input DataFrame
        configs: Specific configs to use, or None for defaults
    
    Returns:
        DataFrame with macro-level temporal features.
    """
    if configs is None:
        configs = get_macro_configs()
    
    result = df.copy()
    for config in configs:
        if config.enabled and config.scope == "global":
            result = _add_global_features(result, config)
    
    return result


def create_custom_config(
    feature_pattern: str,
    lags: Optional[List[int]] = None,
    mas: Optional[List[int]] = None,
    momentum: Optional[List[int]] = None,
    output_prefix: Optional[str] = None,
    scope: str = "ticker",
) -> FeatureLagMAConfig:
    """Create a custom lag/MA configuration on the fly.
    
    Useful for ad-hoc feature engineering experiments.
    
    Example:
        config = create_custom_config(
            feature_pattern="^Close$",
            lags=[1, 5, 10, 20],
            mas=[5, 20, 50],
            output_prefix="Price",
        )
        df = add_lag_ma_features(df, configs=[config])
    """
    return FeatureLagMAConfig(
        feature_pattern=feature_pattern,
        output_prefix=output_prefix,
        lags=lags or [],
        mas=mas or [],
        momentum=momentum or [],
        scope=scope,
        enabled=True,
    )
